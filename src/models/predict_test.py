"""Score the held-out TEST split and save per-transaction predictions.

Reproduces the project's verified, leakage-safe test split (the same 80/20
stratified split with ``random_state=42`` used by ``validate_model.py``),
scores it with the production model, and writes per-row predictions to
``data/processed/test_predictions.csv``:

    TransactionID, fraud_probability, predicted_fraud, decision, actual_is_fraud

Design choices kept consistent with the rest of the project:
* **Data path mirrors validate_model.py** (load_data -> build_features -> _xy
  -> stratified 80/20 split), so the test partition is exactly the one the
  reported metrics were computed on.
* **Median imputation fitted on TRAIN only** (matching the training pipeline),
  so probabilities reproduce the documented test performance — no leakage.
* **Decision bands match the serving API** (src/api/main.py): APPROVE below the
  cost-optimal threshold (baseline_metrics.json ``default_threshold`` = 0.19),
  REVIEW up to 0.70, HOLD above.

Set ``FRAUD_SAMPLE_N=all`` for the FULL held-out test set (the EC2 run does
this); the local default is bounded so an ad-hoc run stays quick.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.features.build_features import build_features
from src.features.handle_imbalance import _xy
from src.models.train_baseline import DEFAULT_SAMPLE_N, RANDOM_STATE, load_data

logger = logging.getLogger(__name__)

_MODEL_PATH = _ROOT / "src" / "models" / "saved" / "baseline_xgboost.pkl"
_METRICS_JSON = _ROOT / "docs" / "model_performance" / "baseline_metrics.json"
_OUT = _ROOT / "data" / "processed" / "test_predictions.csv"

HOLD_MIN = 0.70  # prob > 0.70 -> HOLD (matches src/api/main.py)


def _threshold() -> float:
    """Cost-optimal threshold from baseline_metrics.json (fallback 0.19)."""
    if _METRICS_JSON.exists():
        d = json.loads(_METRICS_JSON.read_text(encoding="utf-8"))
        return float(d.get("default_threshold",
                           d.get("metrics", {}).get("threshold", 0.19)))
    return 0.19


def _route(prob: float, thr: float) -> str:
    if prob < thr:
        return "APPROVE"
    if prob <= HOLD_MIN:
        return "REVIEW"
    return "HOLD"


def _sample_n() -> int | None:
    env = os.environ.get("FRAUD_SAMPLE_N")
    if env and env.strip().lower() in ("all", "0", "full", "none"):
        return None
    if env:
        return int(env)
    return min(DEFAULT_SAMPLE_N, 40000)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sample_n = _sample_n()
    logger.info("Building test split (sample=%s)",
                "ALL" if sample_n is None else f"{sample_n:,}")

    df = load_data(sample_n)
    enriched = build_features(df)
    X, y = _xy(enriched)
    ids = (enriched["TransactionID"] if "TransactionID" in enriched
           else pd.Series(np.arange(len(X)), index=X.index))

    idx = np.arange(len(X))
    tr_i, te_i = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_te = X.iloc[tr_i], X.iloc[te_i]
    y_te = y.iloc[te_i]
    ids_te = ids.iloc[te_i]

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{_MODEL_PATH} not found; run train_baseline.py first."
        )
    model = joblib.load(_MODEL_PATH)
    cols = getattr(model.get_booster(), "feature_names", None)

    # Median imputer fitted on TRAIN only (matches the training pipeline).
    X_tr_a = X_tr.reindex(columns=cols, fill_value=0) if cols else X_tr
    X_te_a = X_te.reindex(columns=cols, fill_value=0) if cols else X_te
    imp = SimpleImputer(strategy="median").fit(X_tr_a)
    proba = model.predict_proba(imp.transform(X_te_a))[:, 1]

    thr = _threshold()
    preds = pd.DataFrame({
        "TransactionID": ids_te.to_numpy(),
        "fraud_probability": np.round(proba, 6),
        "predicted_fraud": (proba >= thr).astype(int),
        "decision": [_route(float(p), thr) for p in proba],
        "actual_is_fraud": y_te.to_numpy().astype(int),
        "threshold": thr,
    })

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(_OUT, index=False)

    # Summary so the run prints what it produced.
    n = len(preds)
    tp = int(((preds.predicted_fraud == 1) & (preds.actual_is_fraud == 1)).sum())
    fp = int(((preds.predicted_fraud == 1) & (preds.actual_is_fraud == 0)).sum())
    fn = int(((preds.predicted_fraud == 0) & (preds.actual_is_fraud == 1)).sum())
    tn = int(((preds.predicted_fraud == 0) & (preds.actual_is_fraud == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    logger.info("Wrote %d predictions -> %s", n, _OUT)
    print("\n" + "=" * 56)
    print("TEST PREDICTIONS")
    print("=" * 56)
    print(f"  rows           : {n:,}")
    print(f"  threshold      : {thr}")
    print(f"  decisions      : {preds.decision.value_counts().to_dict()}")
    print(f"  confusion @thr : tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"  recall         : {recall:.4f}")
    print(f"  precision      : {precision:.4f}")
    print(f"  saved          : {_OUT}")


if __name__ == "__main__":
    main()
