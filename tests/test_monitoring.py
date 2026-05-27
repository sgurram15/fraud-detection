"""C3.4 — Monitoring tests.

Drives the drift detector (C3.1) with synthetic data:
  * a "current" set drawn from the SAME distribution as reference -> expect NO
    dataset drift,
  * a "current" set with shifted feature distributions -> expect drift IS
    detected.

Also smoke-tests the performance tracker on synthetic labeled records.

Run: python tests/test_monitoring.py   (PASS/FAIL per test, non-zero on fail).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.monitoring import drift_detector, performance_tracker

RNG = np.random.default_rng(7)
_FEATURES = ["txn_count_1h", "txn_count_24h", "amt_dev_ratio_card_mean",
             "card_age_days", "hour_of_day", "device_type_fraud_rate"]


def _normal(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "txn_count_1h": RNG.poisson(1.0, n),
        "txn_count_24h": RNG.poisson(5.0, n),
        "amt_dev_ratio_card_mean": RNG.normal(1.0, 0.3, n),
        "card_age_days": RNG.normal(300, 60, n).clip(0),
        "hour_of_day": RNG.integers(0, 24, n),
        "device_type_fraud_rate": RNG.normal(0.035, 0.01, n).clip(0),
    })


def _drifted(n: int) -> pd.DataFrame:
    # Shift every feature well away from the reference distribution.
    return pd.DataFrame({
        "txn_count_1h": RNG.poisson(8.0, n),
        "txn_count_24h": RNG.poisson(40.0, n),
        "amt_dev_ratio_card_mean": RNG.normal(5.0, 1.0, n),
        "card_age_days": RNG.normal(20, 10, n).clip(0),
        "hour_of_day": RNG.integers(0, 5, n),
        "device_type_fraud_rate": RNG.normal(0.25, 0.05, n).clip(0),
    })


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_no_drift_on_normal() -> bool:
    reference = _normal(1000)
    current = _normal(200)
    summary = drift_detector.detect_drift(reference, current, save=False)
    return _result(
        "no drift detected on same-distribution data",
        not summary["dataset_drift"],
        f"dataset_drift={summary['dataset_drift']} "
        f"drifted={summary['drifted_features']}")


def test_drift_on_shifted() -> bool:
    reference = _normal(1000)
    current = _drifted(250)  # 250 drifted (mission: 200 normal + 50 drifted set)
    summary = drift_detector.detect_drift(reference, current, save=False)
    return _result(
        "drift detected on shifted data",
        summary["dataset_drift"] and summary["n_drifted"] >= 1,
        f"dataset_drift={summary['dataset_drift']} "
        f"n_drifted={summary['n_drifted']}")


def test_mixed_set_detects_drift() -> bool:
    # Mission C3.4: 200 normal + 50 drifted in the current set.
    reference = _normal(1000)
    current = pd.concat([_normal(200), _drifted(50)], ignore_index=True)
    summary = drift_detector.detect_drift(reference, current, save=False)
    return _result(
        "drift detected on 200-normal + 50-drifted set",
        summary["n_drifted"] >= 1,
        f"n_drifted={summary['n_drifted']} "
        f"drifted={summary['drifted_features']}")


def test_performance_metrics() -> bool:
    # Perfect separation -> recall/precision 1.0, AUC 1.0.
    records = ([{"outcome": 1, "prob": 0.9, "threshold": 0.19}] * 30
               + [{"outcome": 0, "prob": 0.05, "threshold": 0.19}] * 70)
    m = performance_tracker.compute_metrics(records)
    ok = (m["recall"] == 1.0 and m["precision"] == 1.0
          and m["auc_estimate"] == 1.0 and m["fp"] == 0 and m["fn"] == 0)
    return _result("performance metrics on labeled records", ok, f"metrics={m}")


def main() -> int:
    print("=" * 64)
    print("Monitoring test suite")
    print("=" * 64)
    results = [
        test_no_drift_on_normal(),
        test_drift_on_shifted(),
        test_mixed_set_detects_drift(),
        test_performance_metrics(),
    ]
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
