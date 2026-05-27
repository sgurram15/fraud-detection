"""C3.3 — Monitoring runner.

Runs the drift detector (C3.1) and the performance tracker (C3.2), combines
their output into one report, and prints a single verdict:

  HEALTHY             no feature drift and no confirmed performance degradation
  WARNING             some features drifted, or labels insufficient to confirm
  RETRAINING_REQUIRED dataset-level drift, or a metric degraded > 3% vs baseline

In production this runs weekly on an AWS EventBridge schedule (documented in
docs/production_architecture.md, Version 2).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.monitoring import drift_detector, performance_tracker

logger = logging.getLogger("monitoring")

_OUT_DIR = _ROOT / "docs" / "monitoring"
_COMBINED_PATH = _OUT_DIR / "monitoring_report.json"


def _run_drift() -> dict:
    try:
        reference = drift_detector.load_reference()
        current = drift_detector.load_current()
        return drift_detector.detect_drift(reference, current)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("drift detection skipped: %s", exc)
        return {"status": "unavailable", "reason": str(exc),
                "n_drifted": 0, "dataset_drift": False, "drifted_features": []}


def _verdict(drift: dict, performance: dict) -> str:
    if drift.get("dataset_drift") or performance.get("status") == "degraded":
        return "RETRAINING_REQUIRED"
    if (drift.get("n_drifted", 0) > 0
            or drift.get("status") == "unavailable"
            or performance.get("status") == "insufficient_labels"):
        return "WARNING"
    return "HEALTHY"


def run() -> dict:
    drift = _run_drift()
    performance = performance_tracker.track_performance()
    verdict = _verdict(drift, performance)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "drift": drift,
        "performance": performance,
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _COMBINED_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    report = run()
    drift, perf = report["drift"], report["performance"]
    print("=" * 64)
    print("FRAUD DETECTION — MODEL MONITORING")
    print("=" * 64)
    if drift.get("status") == "unavailable":
        print(f"Drift:       UNAVAILABLE ({drift['reason']})")
    else:
        print(f"Drift:       {drift['n_drifted']}/{drift['n_features']} "
              f"features drifted (share {drift['drift_share']}); "
              f"dataset_drift={drift['dataset_drift']}")
        if drift["drifted_features"]:
            print(f"             drifted: {drift['drifted_features']}")
    print(f"Performance: {perf['status']} "
          f"(labeled records: {perf.get('n_labeled', 0)})")
    if perf.get("degraded_metrics"):
        print(f"             degraded: {perf['degraded_metrics']}")
    print("-" * 64)
    print(f"VERDICT: {report['verdict']}")
    print(f"\nSaved {_COMBINED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
