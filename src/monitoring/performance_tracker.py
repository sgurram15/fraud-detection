"""C3.2 — Live performance tracker.

Reads the streaming audit trail and, for any decision the fraud team has since
confirmed (an ``outcome`` field: 1 = fraud, 0 = legitimate, written back onto
the audit record post-decision), computes rolling performance over the last
$PERF_WINDOW_DAYS days:

  recall, precision, false-positive rate, false-negative rate, an AUC estimate,
  and an illustrative daily £ loss.

Metrics are compared against the model's documented test-set baseline
(docs/model_performance/baseline_metrics.json). Any metric that has degraded by
more than 3% (absolute) relative to baseline is flagged. The report is written
to docs/monitoring/performance_report.json.

On real traffic before any outcomes are confirmed there are no labels, so the
tracker honestly reports ``insufficient_labels`` rather than inventing metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from sklearn.metrics import roc_auc_score

logger = logging.getLogger("performance")

_AUDIT_DIR = _ROOT / "data" / "audit"
_BASELINE = _ROOT / "docs" / "model_performance" / "baseline_metrics.json"
_OUT_DIR = _ROOT / "docs" / "monitoring"
_REPORT_PATH = _OUT_DIR / "performance_report.json"

PERF_WINDOW_DAYS = int(os.getenv("PERF_WINDOW_DAYS", "7"))
_DEGRADE_THRESHOLD = 0.03  # absolute; flag a >3% drop vs baseline

# Illustrative cost assumptions (see docs/model_card.md — recalibrate before
# production with the client's real fraud-loss figures).
_FN_COST_GBP = 125.0
_FP_COST_GBP = 25.0


def load_labeled_records(window_days: int = PERF_WINDOW_DAYS) -> list[dict]:
    """Audit records in-window that carry a confirmed ``outcome`` label."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=window_days)
    out: list[dict] = []
    if not _AUDIT_DIR.exists():
        return out
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
            scored = rec.get("scored", {})
            outcome = rec.get("outcome", scored.get("outcome"))
            prob = scored.get("fraud_probability")
            thr = scored.get("threshold_used")
            if outcome in (0, 1) and prob is not None and thr is not None:
                out.append({"outcome": int(outcome), "prob": float(prob),
                            "threshold": float(thr)})
    return out


def compute_metrics(records: list[dict]) -> dict:
    """Confusion-matrix metrics from labeled records (predicted positive when
    fraud_probability >= the threshold used at decision time)."""
    tp = fp = fn = tn = 0
    for r in records:
        pred = 1 if r["prob"] >= r["threshold"] else 0
        actual = r["outcome"]
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    labels = [r["outcome"] for r in records]
    probs = [r["prob"] for r in records]
    auc = (roc_auc_score(labels, probs)
           if len(set(labels)) == 2 else None)  # AUC needs both classes

    return {
        "n_labeled": len(records),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "auc_estimate": round(auc, 4) if auc is not None else None,
        "daily_loss_gbp_estimate": round(fn * _FN_COST_GBP + fp * _FP_COST_GBP,
                                         2),
    }


def _load_baseline() -> dict:
    if not _BASELINE.exists():
        return {}
    return json.loads(_BASELINE.read_text(encoding="utf-8")).get("metrics", {})


def compare_to_baseline(metrics: dict, baseline: dict) -> dict:
    """Flag metrics that degraded by more than 3% (absolute) vs baseline."""
    flags: list[str] = []
    deltas: dict[str, float] = {}
    # Higher-is-better: recall, precision, auc. Lower-is-better: FPR, FNR.
    for key, better_high in (("recall", True), ("precision", True),
                             ("false_positive_rate", False),
                             ("false_negative_rate", False)):
        base = baseline.get(key)
        cur = metrics.get(key)
        if base is None or cur is None:
            continue
        delta = cur - base
        deltas[key] = round(delta, 4)
        degraded = (-delta if better_high else delta) > _DEGRADE_THRESHOLD
        if degraded:
            flags.append(key)
    base_auc = baseline.get("auc_roc")
    if base_auc is not None and metrics.get("auc_estimate") is not None:
        d = metrics["auc_estimate"] - base_auc
        deltas["auc"] = round(d, 4)
        if -d > _DEGRADE_THRESHOLD:
            flags.append("auc")
    return {"deltas_vs_baseline": deltas, "degraded_metrics": flags}


def track_performance(window_days: int = PERF_WINDOW_DAYS,
                      save: bool = True) -> dict:
    records = load_labeled_records(window_days)
    if not records:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "status": "insufficient_labels",
            "n_labeled": 0,
            "note": ("No confirmed fraud outcomes in the audit trail yet. Live "
                     "performance cannot be computed until the fraud team "
                     "writes outcome labels back onto decisions."),
        }
    else:
        metrics = compute_metrics(records)
        comparison = compare_to_baseline(metrics, _load_baseline())
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "status": ("degraded" if comparison["degraded_metrics"]
                       else "stable"),
            **metrics,
            **comparison,
        }
    if save:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(_REPORT_PATH)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    report = track_performance()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
