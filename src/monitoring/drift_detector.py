"""C3.1 — Data drift detector (Evidently).

Compares the engineered feature distributions the model was trained on
(reference) against recently scored live traffic (current) and reports which
features have drifted, recommending a retrain when too many have.

  reference : the engineered training table materialised by the feature store
              during training (data/processed/feature_store/
              training_features.parquet).
  current   : the engineered feature vectors retained in the streaming audit
              trail (data/audit/{date}/*.json), last $DRIFT_WINDOW_DAYS days.

Outputs:
  docs/monitoring/drift_report.html    (full Evidently report)
  docs/monitoring/drift_summary.json   (machine-readable summary)

Target drift (isFraud) is only meaningful once the fraud team confirms
outcomes; live decisions are not ground truth, so target drift is skipped here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger("drift")

_REF_PARQUET = (_ROOT / "data" / "processed" / "feature_store"
                / "training_features.parquet")
_AUDIT_DIR = _ROOT / "data" / "audit"
_OUT_DIR = _ROOT / "docs" / "monitoring"
_HTML_PATH = _OUT_DIR / "drift_report.html"
_JSON_PATH = _OUT_DIR / "drift_summary.json"

# The engineered serving features (feature-store output) shared by the
# reference table and the audit records. These are what we monitor for drift.
ENGINEERED_FEATURES = [
    "amt_was_missing", "txn_count_1h", "txn_count_24h", "txn_value_24h",
    "amt_no_history", "amt_dev_from_card_mean", "amt_dev_ratio_card_mean",
    "is_largest_for_card", "hour_of_day", "day_of_week", "is_weekend",
    "is_late_night", "card_age_days", "p_email_was_missing", "p_email_is_free",
    "card_amt_avg_30d", "device_type_fraud_rate",
]

DRIFT_WINDOW_DAYS = int(os.getenv("DRIFT_WINDOW_DAYS", "7"))

# Evidently auto-selects a drift method per column and encodes it (and its
# threshold) in the metric name, e.g.
#   ValueDrift(column=hour_of_day,method=Wasserstein distance (normed),threshold=0.1)
#   ValueDrift(column=card1,method=chi-square p_value,threshold=0.05)
# For p-value methods a column drifts when value < threshold; for distance
# methods (Jensen-Shannon, Wasserstein, PSI, ...) when value > threshold.
_VALUE_DRIFT_RE = re.compile(
    r"ValueDrift\(column=(?P<col>.+?),method=(?P<method>.+),"
    r"threshold=(?P<thr>[0-9.eE+-]+)\)$"
)


def _column_drifted(method: str, value: float, threshold: float) -> bool:
    if "p_value" in method:
        return value < threshold
    return value > threshold  # distance metric: larger == more drift


def load_reference(n_sample: int | None = 5000) -> pd.DataFrame:
    """Engineered feature distributions from the training run."""
    if not _REF_PARQUET.exists():
        raise FileNotFoundError(
            f"Reference features not found at {_REF_PARQUET}. Run the feature "
            "store training pass (src/features/feature_store.py) first."
        )
    df = pd.read_parquet(_REF_PARQUET, columns=ENGINEERED_FEATURES)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=42).reset_index(drop=True)
    logger.info("Reference: %d rows x %d features", len(df), df.shape[1])
    return df


def load_current(window_days: int = DRIFT_WINDOW_DAYS) -> pd.DataFrame:
    """Engineered feature vectors from the last `window_days` of audit records."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=window_days)
    rows: list[dict] = []
    if _AUDIT_DIR.exists():
        for day_dir in sorted(_AUDIT_DIR.glob("*")):
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                continue
            for f in day_dir.glob("*.json"):
                if f.name == "audit_summary.json":
                    continue
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                feats = rec.get("scored", {}).get("features")
                if feats:
                    rows.append(feats)
    if not rows:
        raise ValueError(
            f"No scored audit records with features in the last {window_days} "
            f"days under {_AUDIT_DIR}. Run the streaming pipeline first."
        )
    df = pd.DataFrame(rows)
    # Keep only the monitored features that are actually present.
    cols = [c for c in ENGINEERED_FEATURES if c in df.columns]
    logger.info("Current: %d rows x %d features", len(df), len(cols))
    return df[cols]


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 save: bool = True) -> dict:
    """Run the Evidently data-drift report on the columns common to both
    frames and return a JSON-serialisable summary. Generic so the C3.4 test can
    drive it with synthetic frames."""
    columns = [c for c in reference.columns if c in current.columns]
    if not columns:
        raise ValueError("reference and current share no comparable columns")
    ref = reference[columns].apply(pd.to_numeric, errors="coerce")
    cur = current[columns].apply(pd.to_numeric, errors="coerce")

    data_def = DataDefinition(numerical_columns=columns)
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    report = Report([DataDriftPreset()])
    with warnings.catch_warnings():
        # Evidently's internal correlation step divides by a zero std on
        # constant columns; the resulting RuntimeWarnings are harmless noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        snapshot = report.run(cur_ds, ref_ds)  # (current, reference)
    result = snapshot.dict()

    drifted: list[str] = []
    drift_share = 0.0
    for metric in result.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        m = _VALUE_DRIFT_RE.match(name)
        if m and isinstance(value, (int, float)):
            if _column_drifted(m["method"], float(value), float(m["thr"])):
                drifted.append(m["col"])
        elif name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drift_share = float(value.get("share", 0.0))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(columns),
        "n_drifted": len(drifted),
        "drift_share": round(drift_share, 4),
        "drifted_features": sorted(drifted),
        # "dataset drift" when at least half the features drifted (Evidently's
        # default DriftedColumnsCount share threshold).
        "dataset_drift": drift_share >= 0.5,
    }

    if save:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            snapshot.save_html(str(_HTML_PATH))
            summary["html_report"] = str(_HTML_PATH)
        except Exception as exc:  # HTML render is non-critical
            logger.warning("could not save HTML report: %s", exc)
        _JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["json_summary"] = str(_JSON_PATH)
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    reference = load_reference()
    current = load_current()
    summary = detect_drift(reference, current)
    if summary["drifted_features"]:
        print(f"DRIFT DETECTED: {summary['drifted_features']} "
              f"({summary['n_drifted']}/{summary['n_features']} features) "
              "— RETRAINING RECOMMENDED")
    else:
        print("NO DRIFT DETECTED — model performance stable")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
