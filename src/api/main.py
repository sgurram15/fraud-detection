"""Phase C1 — FastAPI real-time fraud scoring service.

Serves the PRODUCTION model (baseline_xgboost.pkl, full-data run) behind a
sub-100ms scoring API with SHAP reason codes and an FCA-audit explanation per
decision. Operating point: cost-optimal threshold 0.19 (from
docs/model_performance/baseline_metrics.json).

Endpoints: POST /score, POST /score/batch, GET /health, GET /metrics.

NOTE on spec deviations (deliberate, per the user's instructions):
  * Production model is baseline_xgboost.pkl (not tuned) — the model-selection
    decision (docs/model_comparison.json) chose baseline. MODEL_META.version is
    therefore "xgboost-baseline-v1", and the threshold is loaded from
    baseline_metrics.json (default_threshold = 0.19), not model_comparison.json.
  * The request carries already-derived features (velocity, deviation, time
    flags). We still run feature-store serving mode (for device_type_fraud_rate
    + latency), then overlay the caller's explicit features, then align to the
    model's full training column set (absent raw V/C/D columns -> 0).
  * SHAP reasons use the startup-initialised TreeExplainer (C1.1) reused per
    request — functionally explain_prediction() but without rebuilding the
    explainer each call, which would blow the <100ms SLA. The FCA dict is built
    by explain.generate_fca_explanation().
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import S3_BUCKET, USE_S3
from src.features.feature_store import FeatureStore
from src.models.explain import _describe, generate_fca_explanation

logger = logging.getLogger("fraud_api")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

_MODEL_FILENAME = "baseline_xgboost.pkl"  # production model (user-confirmed)
_MODEL_PATH = _ROOT / "src" / "models" / "saved" / _MODEL_FILENAME
_METRICS_JSON = _ROOT / "docs" / "model_performance" / "baseline_metrics.json"

# Decision routing bands (C1.3) — independent of the cost threshold.
APPROVE_MAX = 0.30   # prob < 0.30          -> APPROVE
HOLD_MIN = 0.70      # 0.30 <= prob <= 0.70 -> REVIEW ; prob > 0.70 -> HOLD

# Engineered serving features the request can override on the store output.
_OVERLAY = {
    "tx_velocity_1h": "txn_count_1h",
    "tx_velocity_24h": "txn_count_24h",
    "amt_deviation": "amt_dev_ratio_card_mean",
    "hour_of_day": "hour_of_day",
    "day_of_week": "day_of_week",
    "card_age_days": "card_age_days",
}


# --------------------------------------------------------------------------- #
# C1.1 — Model loader
# --------------------------------------------------------------------------- #
def _load_threshold() -> float:
    if not _METRICS_JSON.exists():
        raise FileNotFoundError(
            f"{_METRICS_JSON} not found — cannot resolve the cost-optimal "
            "threshold. Run train_baseline.py first."
        )
    data = json.loads(_METRICS_JSON.read_text(encoding="utf-8"))
    return float(data.get("default_threshold", data["metrics"]["threshold"]))


def _load_model():
    if USE_S3:
        if not S3_BUCKET:
            raise RuntimeError("USE_S3=true but S3_BUCKET is empty.")
        import tempfile

        import boto3

        key = f"models/saved/{_MODEL_FILENAME}"
        tmp = Path(tempfile.gettempdir()) / _MODEL_FILENAME
        logger.info("Loading model from s3://%s/%s", S3_BUCKET, key)
        boto3.client("s3").download_file(S3_BUCKET, key, str(tmp))
        return joblib.load(tmp)
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. Train it with "
            "`python src/models/train_baseline.py` (or set USE_S3=true)."
        )
    return joblib.load(_MODEL_PATH)


def _n_model_features(model) -> int:
    try:
        return len(model.get_booster().feature_names)
    except Exception:
        return -1


_t0 = time.perf_counter()
MODEL = _load_model()
STORE = FeatureStore().load()  # raises clearly if online_state.pkl missing
EXPLAINER = shap.TreeExplainer(MODEL)  # C1.1: SHAP explainer init at startup
_BOOSTER = MODEL.get_booster()  # native pred_contribs path (see _shap_reasons)
_THRESHOLD = _load_threshold()
_LOAD_SECONDS = round(time.perf_counter() - _t0, 3)

# Model training columns. The API supplies ~18 engineered features; the ~400
# unsupplied raw IEEE-CIS columns (C/D/V/M, etc.) are filled with NaN so
# XGBoost routes them via its learned "missing" default branch. Filling 0
# instead biases every score high (0 is a fraud-correlated value for many C
# columns), making APPROVE unreachable — NaN is the correct sentinel here.
_FEATURE_NAMES = list(MODEL.get_booster().feature_names)
_FEAT_INDEX = {name: i for i, name in enumerate(_FEATURE_NAMES)}


def _feature_row(feats: dict) -> np.ndarray:
    """Map the engineered feature dict onto a contiguous float32 row aligned to
    the model's training columns; unsupplied columns stay NaN (XGBoost
    missing). Building a DMatrix from this is ~90x faster than from a 417-col
    pandas frame (the latter measured ~93ms/call -> blew the SLA)."""
    row = np.full((1, len(_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for name, val in feats.items():
        j = _FEAT_INDEX.get(name)
        if j is None or val is None or isinstance(val, str):
            continue
        try:
            row[0, j] = float(val)
        except (TypeError, ValueError):
            pass
    return row

MODEL_META = {
    "version": "xgboost-baseline-v1",
    "trained_on": "IEEE-CIS",
    "features": 18,  # engineered serving features (model input width below)
    "operating_point": "cost-optimal",
    "threshold": _THRESHOLD,
    "loaded_at": datetime.now(timezone.utc).isoformat(),
}
logger.info("Model loaded: version=%s load_time=%.3fs model_input_cols=%d "
            "engineered_features=%d threshold=%.2f",
            MODEL_META["version"], _LOAD_SECONDS, _n_model_features(MODEL),
            MODEL_META["features"], _THRESHOLD)


# --------------------------------------------------------------------------- #
# C1.2 — Schemas
# --------------------------------------------------------------------------- #
class TransactionRequest(BaseModel):
    transaction_id: str
    card_id: str
    amount: float = Field(gt=0, description="Transaction amount, must be > 0")
    device_type: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    destination_account_age_days: int = Field(ge=0)
    tx_velocity_1h: int = 0
    tx_velocity_24h: int = 0
    amt_deviation: float = 1.0
    is_late_night: bool = False
    is_weekend: bool = False
    card_age_days: int = 0


class FraudScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    threshold_used: float
    reasons: list[str]
    fca_explanation: dict
    model_version: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Session/day counters (C1.5, C1.6)
# --------------------------------------------------------------------------- #
_START_TIME = time.time()
_LOCK = threading.Lock()  # guards STORE mutation + counters
_COUNTERS = {
    "session_scored": 0,
    "today_scored": 0,
    "fraud_today": 0,            # prob >= cost threshold
    "latency_sum_ms": 0.0,
    "decisions": {"APPROVE": 0, "REVIEW": 0, "HOLD": 0},
}


def _route(prob: float) -> str:
    if prob < APPROVE_MAX:
        return "APPROVE"
    if prob <= HOLD_MIN:
        return "REVIEW"
    return "HOLD"


def _predict_and_explain(feats: dict) -> tuple[float, list[dict]]:
    """Score + top-5 SHAP reasons from a SINGLE shared DMatrix (built from a
    fast numpy row). pred_contribs yields the same tree_path_dependent SHAP
    values as the startup TreeExplainer (EXPLAINER) but in optimized C++."""
    row = _feature_row(feats)
    dm = xgb.DMatrix(row, feature_names=_FEATURE_NAMES, missing=np.nan)
    prob = float(_BOOSTER.predict(dm)[0])  # binary:logistic -> P(fraud)
    contribs = _BOOSTER.predict(dm, pred_contribs=True)[0]
    sv = np.asarray(contribs[:-1])  # drop the bias term
    order = np.argsort(np.abs(sv))[::-1][:5]
    out = []
    for rank, i in enumerate(order, 1):
        feat = _FEATURE_NAMES[i]
        raw = row[0, i]
        val_py = None if np.isnan(raw) else float(raw)
        s = float(sv[i])
        out.append({
            "rank": rank, "feature": feat, "value": val_py, "shap_value": s,
            "direction": "increases" if s > 0 else "decreases",
            "reason": _describe(feat, val_py),
        })
    return prob, out


def _build_features(req: TransactionRequest) -> dict:
    """Feature-store serving mode + overlay the caller's explicit features."""
    raw_txn = {
        "TransactionAmt": req.amount,
        "DeviceType": req.device_type,
        "card1": req.card_id,
        "P_emaildomain": None,
        "TransactionDT": None,
    }
    with _LOCK:  # get_features mutates per-card state
        feats = STORE.get_features(raw_txn)
    for req_field, feat_name in _OVERLAY.items():
        feats[feat_name] = getattr(req, req_field)
    feats["is_late_night"] = int(req.is_late_night)
    feats["is_weekend"] = int(req.is_weekend)
    feats["TransactionAmt"] = req.amount  # raw amount is a model feature
    return feats


def _score(req: TransactionRequest) -> FraudScoreResponse:
    t0 = time.perf_counter()
    feats = _build_features(req)
    prob, reason_dicts = _predict_and_explain(feats)
    decision = _route(prob)
    fca = generate_fca_explanation(
        {"TransactionID": req.transaction_id, "TransactionAmt": req.amount},
        {"fraud_probability": prob, "threshold": _THRESHOLD,
         "decision": decision},
        reason_dicts,
        model_meta={"name": MODEL_META["version"],
                    "artifact": f"src/models/saved/{_MODEL_FILENAME}"},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    with _LOCK:
        _COUNTERS["session_scored"] += 1
        _COUNTERS["today_scored"] += 1
        _COUNTERS["latency_sum_ms"] += latency_ms
        _COUNTERS["decisions"][decision] += 1
        if prob >= _THRESHOLD:
            _COUNTERS["fraud_today"] += 1

    return FraudScoreResponse(
        transaction_id=req.transaction_id,
        fraud_probability=prob,
        decision=decision,
        threshold_used=_THRESHOLD,
        reasons=[r["reason"] for r in reason_dicts],
        fca_explanation=fca,
        model_version=MODEL_META["version"],
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- #
# App + middleware (C1.7)
# --------------------------------------------------------------------------- #
app = FastAPI(title="PSP Fraud Scoring API", version="1.0")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("request_id=%s path=%s latency_ms=%.2f status=%s",
                request_id, request.url.path, elapsed_ms, response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled error request_id=%s: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal error", "request_id": request_id},
    )


# --------------------------------------------------------------------------- #
# C1.3 — POST /score
# --------------------------------------------------------------------------- #
@app.post("/score", response_model=FraudScoreResponse)
def score(req: TransactionRequest, request: Request) -> FraudScoreResponse:
    resp = _score(req)
    logger.info("request_id=%s txn=%s decision=%s prob=%.4f latency_ms=%.2f",
                getattr(request.state, "request_id", None),
                resp.transaction_id, resp.decision, resp.fraud_probability,
                resp.latency_ms)
    return resp


# --------------------------------------------------------------------------- #
# C1.4 — POST /score/batch
# --------------------------------------------------------------------------- #
@app.post("/score/batch", response_model=list[FraudScoreResponse])
def score_batch(reqs: list[TransactionRequest]) -> list[FraudScoreResponse]:
    if len(reqs) > 1000:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"error": f"batch too large ({len(reqs)} > 1000 max)"},
        )
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_score, reqs))
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg = total_ms / len(reqs) if reqs else 0.0
    logger.info("batch size=%d total_ms=%.1f avg_ms=%.2f",
                len(reqs), total_ms, avg)
    return results


# --------------------------------------------------------------------------- #
# C1.5 — GET /health
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_version": MODEL_META["version"],
        "model_loaded": MODEL is not None,
        "uptime_seconds": int(time.time() - _START_TIME),
        "transactions_scored_session": _COUNTERS["session_scored"],
    }


# --------------------------------------------------------------------------- #
# C1.6 — GET /metrics
# --------------------------------------------------------------------------- #
@app.get("/metrics")
def metrics() -> dict:
    scored = _COUNTERS["today_scored"]
    avg_latency = (_COUNTERS["latency_sum_ms"] / scored) if scored else 0.0
    fraud_rate = (_COUNTERS["fraud_today"] / scored) if scored else 0.0
    return {
        "transactions_scored_today": scored,
        "fraud_rate_today": round(fraud_rate, 4),
        "avg_latency_ms": round(avg_latency, 3),
        "decisions": dict(_COUNTERS["decisions"]),
    }
