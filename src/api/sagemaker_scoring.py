"""C4.1 — SageMaker inference handlers.

The four-function SageMaker contract (model_fn / input_fn / predict_fn /
output_fn) for the real-time endpoint that replaces the local FastAPI service
in production. It mirrors the serving logic in src/api/main.py exactly — same
production model (baseline_xgboost.pkl, user-confirmed), same cost-optimal
threshold (0.19), same NaN-fill feature alignment, same SHAP reason codes and
FCA audit record, same APPROVE/REVIEW/HOLD bands — but loads its artifacts from
the SageMaker ``model_dir`` instead of repo-relative paths.

This module is imported by the SageMaker XGBoost/sklearn inference container; it
does NOT import src.api.main (which loads the model at import time from local
paths). It does reuse the project's FeatureStore and FCA explanation builder so
online/offline feature parity is preserved.

deploy_endpoint.py (C4.2) packages alongside this file:
  baseline_xgboost.pkl   online_state.pkl   baseline_metrics.json
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.features.feature_store import FeatureStore, _CardState
from src.models.explain import _describe, generate_fca_explanation

_MODEL_FILENAME = "baseline_xgboost.pkl"
_STATE_FILENAME = "online_state.pkl"
_METRICS_FILENAME = "baseline_metrics.json"
_MODEL_VERSION = "xgboost-baseline-v1"

# Decision bands (identical to src/api/main.py).
APPROVE_MAX = 0.30
HOLD_MIN = 0.70

# Engineered serving features the request overrides on the store output.
_OVERLAY = {
    "tx_velocity_1h": "txn_count_1h",
    "tx_velocity_24h": "txn_count_24h",
    "amt_deviation": "amt_dev_ratio_card_mean",
    "hour_of_day": "hour_of_day",
    "day_of_week": "day_of_week",
    "card_age_days": "card_age_days",
}


def _load_feature_store(state_path: Path) -> FeatureStore:
    """Reconstruct a serving FeatureStore from a packaged state pickle (the
    repo-relative FeatureStore.load() path does not exist in the container)."""
    store = FeatureStore()
    with open(state_path, "rb") as fh:
        blob = pickle.load(fh)
    store._state = defaultdict(_CardState, blob["state"])
    store._device_rates = blob["device_rates"]
    store._global_prior = blob["global_prior"]
    store._max_dt = blob["max_dt"]
    return store


# --------------------------------------------------------------------------- #
# SageMaker contract
# --------------------------------------------------------------------------- #
def model_fn(model_dir: str) -> dict:
    """Load the model + feature store + SHAP explainer + threshold once at
    container start. Returned object is passed to predict_fn per request."""
    import joblib

    md = Path(model_dir)
    model = joblib.load(md / _MODEL_FILENAME)
    store = _load_feature_store(md / _STATE_FILENAME)

    threshold = 0.19
    metrics_file = md / _METRICS_FILENAME
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        threshold = float(data.get("default_threshold",
                                   data.get("metrics", {}).get("threshold",
                                                               0.19)))

    booster = model.get_booster()
    feature_names = list(booster.feature_names)
    return {
        "model": model,
        "booster": booster,
        "store": store,
        "threshold": threshold,
        "feature_names": feature_names,
        "feat_index": {n: i for i, n in enumerate(feature_names)},
    }


def input_fn(request_body, content_type: str = "application/json") -> dict:
    """Parse the incoming transaction JSON into a dict."""
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def _feature_row(feats: dict, feature_names: list, feat_index: dict) -> np.ndarray:
    """Engineered features -> model-aligned float32 row; unsupplied columns
    stay NaN (XGBoost native missing). Identical policy to src/api/main.py."""
    row = np.full((1, len(feature_names)), np.nan, dtype=np.float32)
    for name, val in feats.items():
        j = feat_index.get(name)
        if j is None or val is None or isinstance(val, str):
            continue
        try:
            row[0, j] = float(val)
        except (TypeError, ValueError):
            pass
    return row


def _route(prob: float) -> str:
    if prob < APPROVE_MAX:
        return "APPROVE"
    if prob <= HOLD_MIN:
        return "REVIEW"
    return "HOLD"


def predict_fn(input_data: dict, model: dict) -> dict:
    """Feature-store serving + score + SHAP reasons + decision band."""
    store = model["store"]
    raw_txn = {
        "TransactionAmt": input_data.get("amount"),
        "DeviceType": input_data.get("device_type"),
        "card1": input_data.get("card_id"),
        "P_emaildomain": None,
        "TransactionDT": None,
    }
    feats = store.get_features(raw_txn)
    for req_field, feat_name in _OVERLAY.items():
        if req_field in input_data:
            feats[feat_name] = input_data[req_field]
    feats["is_late_night"] = int(bool(input_data.get("is_late_night", False)))
    feats["is_weekend"] = int(bool(input_data.get("is_weekend", False)))
    feats["TransactionAmt"] = input_data.get("amount")

    names, idx = model["feature_names"], model["feat_index"]
    row = _feature_row(feats, names, idx)
    dm = xgb.DMatrix(row, feature_names=names, missing=np.nan)
    booster = model["booster"]
    prob = float(booster.predict(dm)[0])
    contribs = booster.predict(dm, pred_contribs=True)[0]
    sv = np.asarray(contribs[:-1])
    order = np.argsort(np.abs(sv))[::-1][:5]
    reasons = []
    for rank, i in enumerate(order, 1):
        raw = row[0, i]
        val_py = None if np.isnan(raw) else float(raw)
        s = float(sv[i])
        reasons.append({
            "rank": rank, "feature": names[i], "value": val_py,
            "shap_value": s, "direction": "increases" if s > 0 else "decreases",
            "reason": _describe(names[i], val_py),
        })
    return {"input": input_data, "prob": prob, "reasons": reasons,
            "threshold": model["threshold"]}


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Format the prediction as the FraudScoreResponse JSON (matches the
    FastAPI contract so clients are endpoint-agnostic)."""
    inp = prediction["input"]
    prob = prediction["prob"]
    threshold = prediction["threshold"]
    decision = _route(prob)
    fca = generate_fca_explanation(
        {"TransactionID": inp.get("transaction_id"),
         "TransactionAmt": inp.get("amount")},
        {"fraud_probability": prob, "threshold": threshold,
         "decision": decision},
        prediction["reasons"],
        model_meta={"name": _MODEL_VERSION,
                    "artifact": f"src/models/saved/{_MODEL_FILENAME}"},
    )
    body = {
        "transaction_id": inp.get("transaction_id"),
        "fraud_probability": prob,
        "decision": decision,
        "threshold_used": threshold,
        "reasons": [r["reason"] for r in prediction["reasons"]],
        "fca_explanation": fca,
        "model_version": _MODEL_VERSION,
    }
    return json.dumps(body, default=str)
