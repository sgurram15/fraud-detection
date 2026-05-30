"""SHAP explainability for the fraud-detection model.

Public API:
  * ``explain_prediction(model, transaction_features) -> list[dict]``
      Top-5 reasons (in plain English) why one transaction was flagged, with
      direction (increases vs decreases fraud probability).
  * ``explain_batch(model, df)``
      Importance + summary + per-feature dependence plots, written under
      ``docs/model_performance/``.
  * ``generate_fca_explanation(transaction, prediction, shap_reasons) -> dict``
      Structured, JSON-serialisable explanation suitable for an immutable
      audit trail (the evidence stored for every automated decision).

Run as a script to load the tuned model, call ``explain_batch`` on the test
set, and save the plots + an example FCA audit record.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import USE_S3
from src.config import model_path as _model_root

logger = logging.getLogger(__name__)

_PERF_DIR = _ROOT / "docs" / "model_performance"
_IMPORTANCE_PNG = _PERF_DIR / "shap_importance.png"
_SUMMARY_PNG = _PERF_DIR / "shap_summary.png"
_DEPENDENCE_PNG = _PERF_DIR / "shap_dependence_{i}_{feat}.png"
_EXAMPLE_AUDIT = _PERF_DIR / "fca_explanation_example.json"

# ---------------------------------------------------------------------------
# Human-readable feature descriptions
# ---------------------------------------------------------------------------
#
# Maps the model's feature name to a callable that turns the row's value into
# a one-line plain-English description an analyst can read. Unknown features
# fall back to ``"<pretty name> = <value>"``.

_HOURS_AMPM = {0: "12am"} | {h: f"{h}am" for h in range(1, 12)} \
    | {12: "12pm"} | {h: f"{h - 12}pm" for h in range(13, 24)}


def _fmt_hour(v: float) -> str:
    try:
        h = int(round(float(v)))
        return f"Transaction at {_HOURS_AMPM.get(h, f'{h}:00')}"
    except Exception:
        return f"Transaction hour = {v}"


def _fmt_dow(v: float) -> str:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    try:
        return f"{days[int(v)]} transaction"
    except Exception:
        return f"day_of_week = {v}"


def _fmt_signed_amount(v: float) -> str:
    direction = "above" if v >= 0 else "below"
    return f"Transaction is £{abs(v):,.0f} {direction} this card's average"


HUMAN_READABLE: dict[str, Any] = {
    # Velocity
    "txn_count_1h": lambda v: f"{int(v)} transaction(s) by this card in the last hour",
    "tx_velocity_1h": lambda v: f"{int(v)} transaction(s) by this card in the last hour",
    "txn_count_24h": lambda v: f"{int(v)} transaction(s) by this card in the last 24 hours",
    "txn_value_24h": lambda v: f"£{v:,.0f} transacted by this card in the last 24 hours",
    # Deviation
    "amt_dev_ratio_card_mean": lambda v: f"Transaction is {v:.1f}x this card's average amount",
    "amt_deviation": lambda v: f"Transaction is {v:.1f}x this card's average amount",
    "amt_dev_from_card_mean": _fmt_signed_amount,
    "is_largest_for_card": lambda v: ("Largest amount this card has ever transacted"
                                       if int(v) == 1 else "Not the card's largest amount"),
    "amt_no_history": lambda v: ("No prior history for this card"
                                  if int(v) == 1 else "Card has prior history"),
    # Time
    "hour_of_day": _fmt_hour,
    "day_of_week": _fmt_dow,
    "is_weekend": lambda v: ("Weekend transaction" if int(v) == 1
                              else "Weekday transaction"),
    "is_late_night": lambda v: ("Transaction during late-night hours (midnight-5am)"
                                 if int(v) == 1 else "Not a late-night transaction"),
    # Account / identity
    "card_age_days": lambda v: f"Card first seen {int(v)} day(s) ago",
    "p_email_is_free": lambda v: ("Purchaser uses a free email provider"
                                   if int(v) == 1 else
                                   ("Purchaser uses a non-free email" if int(v) == 0
                                    else "Purchaser email unknown")),
    "p_email_was_missing": lambda v: ("Purchaser email domain missing"
                                       if int(v) == 1 else "Purchaser email present"),
    "amt_was_missing": lambda v: ("Transaction amount missing"
                                   if int(v) == 1 else "Transaction amount present"),
    "card_amt_avg_30d": lambda v: f"Card's 30-day average amount: £{v:,.0f}",
    "device_type_fraud_rate": lambda v: f"Device type historical fraud rate {v:.1%}",
    # Raw transaction
    "TransactionAmt": lambda v: f"Transaction amount £{v:,.2f}",
    "dist1": lambda v: f"Address-distance metric (dist1) = {v:.0f}",
}


def _describe(feature: str, value: Any) -> str:
    fmt = HUMAN_READABLE.get(feature)
    if fmt is not None:
        try:
            return fmt(value)
        except Exception:  # noqa: BLE001
            pass
    pretty = feature.replace("_", " ")
    if isinstance(value, float):
        return f"{pretty} = {value:.4g}"
    return f"{pretty} = {value}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _align(X: pd.DataFrame, model) -> pd.DataFrame:
    """Reindex X to the columns/order the model was trained on."""
    try:
        names = model.get_booster().feature_names
    except Exception:
        names = None
    if names:
        return X.reindex(columns=names, fill_value=0)
    return X


def _to_frame(transaction_features) -> pd.DataFrame:
    """Coerce a dict / Series / 1-row DataFrame into a 1-row DataFrame."""
    if isinstance(transaction_features, pd.DataFrame):
        return transaction_features.head(1).copy()
    if isinstance(transaction_features, pd.Series):
        return transaction_features.to_frame().T
    if isinstance(transaction_features, dict):
        return pd.DataFrame([transaction_features])
    raise TypeError(
        "transaction_features must be a dict, pandas Series, or 1-row DataFrame"
    )


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for an XGBoost classifier as an (n, n_features) array."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # Newer shap returns a list[array] for some classifiers; binary XGB
    # already returns the array directly. Normalise to ndarray.
    if isinstance(sv, list):
        sv = sv[-1]  # positive-class contributions
    return np.asarray(sv)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def explain_prediction(model, transaction_features) -> list[dict]:
    """Top-5 reasons the transaction was flagged, in plain English.

    Each reason is a dict with:
      feature, value, shap_value, direction ('increases'|'decreases'),
      reason (str), rank (1..5).
    """
    X = _align(_to_frame(transaction_features), model)
    sv = _shap_values(model, X)[0]  # 1-D
    order = np.argsort(np.abs(sv))[::-1][:5]

    out = []
    for rank, i in enumerate(order, 1):
        feat = X.columns[i]
        val = X.iloc[0, i]
        try:
            val_py = val.item() if hasattr(val, "item") else val
        except Exception:
            val_py = val
        s = float(sv[i])
        out.append({
            "rank": rank,
            "feature": feat,
            "value": val_py,
            "shap_value": s,
            "direction": "increases" if s > 0 else "decreases",
            "reason": _describe(feat, val_py),
        })
    return out


def explain_batch(model, df: pd.DataFrame, sample: int = 2000) -> dict:
    """Importance + summary + top-3 dependence plots.

    For large frames, randomly subsamples to ``sample`` rows for plot
    readability (TreeExplainer is fast, but beeswarm rendering is the bottleneck).
    Returns paths of the saved plots and the ordered top features.
    """
    _PERF_DIR.mkdir(parents=True, exist_ok=True)
    X = _align(df, model)
    if len(X) > sample:
        X = X.sample(sample, random_state=42)
    sv = _shap_values(model, X)

    # Mean |SHAP| per feature -> ranking.
    mean_abs = np.abs(sv).mean(axis=0)
    rank = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    top_features = rank.head(20).index.tolist()

    # Importance (bar).
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=20)
    plt.gcf().set_size_inches(8, 7)
    plt.tight_layout()
    plt.savefig(_IMPORTANCE_PNG, dpi=110, bbox_inches="tight")
    plt.close("all")

    # Summary (beeswarm).
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.gcf().set_size_inches(8, 7)
    plt.tight_layout()
    plt.savefig(_SUMMARY_PNG, dpi=110, bbox_inches="tight")
    plt.close("all")

    # Dependence plots for the top 3.
    dependence_paths = []
    for i, feat in enumerate(top_features[:3], 1):
        shap.dependence_plot(feat, sv, X, show=False)
        plt.gcf().set_size_inches(7, 5)
        plt.tight_layout()
        path = Path(str(_DEPENDENCE_PNG).format(i=i, feat=feat))
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close("all")
        dependence_paths.append(str(path))

    return {
        "n_rows_used": len(X),
        "top_features_by_mean_abs_shap": rank.head(20).to_dict(),
        "importance_plot": str(_IMPORTANCE_PNG),
        "summary_plot": str(_SUMMARY_PNG),
        "dependence_plots": dependence_paths,
    }


def _confidence(prob: float, threshold: float) -> str:
    d = abs(prob - threshold)
    if d > 0.30:
        return "HIGH"
    if d > 0.10:
        return "MEDIUM"
    return "LOW"


# Production model identity — used when a caller does not supply its own. The
# live serving path (src/api/main.py) and the CLI below both pass the actual
# model they loaded, so the audit record never mislabels which model decided.
DEFAULT_MODEL_META = {
    "name": "xgboost-baseline-v1",
    "artifact": "src/models/saved/baseline_xgboost.pkl",
}


def generate_fca_explanation(transaction: dict, prediction, shap_reasons,
                             model_meta: dict | None = None) -> dict:
    """Structured audit record for one automated decision.

    ``prediction`` may be a float probability or a dict with any of
    ``fraud_probability``, ``threshold``, ``decision``. ``model_meta`` records
    which model produced the decision (name + artifact); it defaults to the
    production model. The output is JSON-serialisable (numpy types coerced to
    Python).
    """
    if isinstance(prediction, dict):
        prob = float(prediction.get("fraud_probability",
                                     prediction.get("score", 0.0)))
        threshold = float(prediction.get("threshold", 0.5))
        decision = prediction.get("decision")
    else:
        prob = float(prediction)
        threshold = 0.5
        decision = None

    if decision is None:
        decision = "BLOCK" if prob >= threshold else "ALLOW"

    txn_id = transaction.get("TransactionID")
    txn_id = int(txn_id) if isinstance(txn_id, (int, np.integer)) else txn_id

    # Coerce numpy scalars in reasons to plain Python for JSON safety.
    safe_reasons = []
    for r in shap_reasons:
        v = r.get("value")
        if isinstance(v, (np.integer, np.floating)):
            v = v.item()
        safe_reasons.append({**r, "value": v})

    # Flat feature -> contribution map for frontend bar charts. Same SHAP values
    # as top_reasons, but indexed by feature name for direct lookup.
    shap_values = {r["feature"]: float(r["shap_value"]) for r in safe_reasons}

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_id": txn_id,
        "model": dict(model_meta) if model_meta else dict(DEFAULT_MODEL_META),
        "fraud_probability": prob,
        "threshold": threshold,
        "decision": decision,
        "confidence_level": _confidence(prob, threshold),
        "explanation_method": "SHAP TreeExplainer (tree_path_dependent)",
        "top_reasons": safe_reasons,
        "shap_values": shap_values,
        "governance": {
            "purpose": "FCA-aligned evidence for an automated payment decision",
            "consumer_duty_review_eligible": decision != "ALLOW",
            "human_oversight_recommended": decision != "ALLOW",
        },
    }


# ---------------------------------------------------------------------------
# CLI: run on the test set, save plots, demo a single explanation
# ---------------------------------------------------------------------------
def _build_test_xy(sample_n: int):
    """Reuse the project's verified data pipeline so feature order matches."""
    from src.features.build_features import build_features
    from src.features.handle_imbalance import _xy
    from src.models.train_baseline import RANDOM_STATE, load_data
    from sklearn.model_selection import train_test_split

    df = load_data(sample_n)
    enriched = build_features(df)
    X, y = _xy(enriched)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Keep raw transaction context for the FCA demo.
    raw_test = enriched.loc[X_test.index]
    return X_test, y_test, raw_test


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    import joblib

    if USE_S3:
        raise NotImplementedError(
            "explain.py loads a saved model via joblib (local-FS only); run "
            "with USE_S3=false. The library functions are model-object-driven "
            "and path-agnostic."
        )
    tuned = _ROOT / _model_root() / "tuned_xgboost.pkl"
    baseline = _ROOT / _model_root() / "baseline_xgboost.pkl"
    if tuned.exists():
        model_path = tuned
    elif baseline.exists():
        model_path = baseline
    else:
        raise FileNotFoundError(
            "No saved model in src/models/saved/. Train one first "
            "(e.g. python src/models/train_baseline.py)."
        )
    logger.info("Loading %s", model_path.name)
    model = joblib.load(model_path)

    import os
    sample_n = int(os.environ.get("FRAUD_SAMPLE_N", "30000"))
    X_test, y_test, raw_test = _build_test_xy(sample_n)
    logger.info("Test set %d rows; computing SHAP ...", len(X_test))

    out = explain_batch(model, X_test, sample=min(2000, len(X_test)))
    logger.info("Saved: %s, %s, %d dependence plots",
                Path(out["importance_plot"]).name,
                Path(out["summary_plot"]).name,
                len(out["dependence_plots"]))

    # Pick the highest-scoring real test transaction for the demo.
    proba = model.predict_proba(_align(X_test, model))[:, 1]
    idx = int(np.argmax(proba))
    txn_row = X_test.iloc[[idx]]
    reasons = explain_prediction(model, txn_row)
    raw = raw_test.iloc[idx].to_dict()
    fca = generate_fca_explanation(
        transaction={"TransactionID": raw.get("TransactionID")},
        prediction={"fraud_probability": float(proba[idx]),
                    "threshold": 0.5},
        shap_reasons=reasons,
        model_meta={"name": model_path.stem,
                    "artifact": f"src/models/saved/{model_path.name}"},
    )
    _EXAMPLE_AUDIT.write_text(json.dumps(fca, indent=2, default=str))
    logger.info("Saved example FCA audit record -> %s",
                _EXAMPLE_AUDIT.name)

    print("\n=== Highest-scoring test transaction ===")
    print(f"TransactionID={raw.get('TransactionID')}  "
          f"P(fraud)={proba[idx]:.4f}  decision={fca['decision']}  "
          f"confidence={fca['confidence_level']}")
    print("Top 5 reasons:")
    for r in reasons:
        sign = "+" if r["shap_value"] > 0 else "-"
        print(f"  {r['rank']}. [{sign}] {r['reason']}  "
              f"(shap={r['shap_value']:+.3f})")
    print(f"\nTop features in batch: "
          f"{list(out['top_features_by_mean_abs_shap'])[:5]}")


if __name__ == "__main__":
    main()
