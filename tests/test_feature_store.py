"""Smoke/acceptance tests for src/features/feature_store.py.

Runnable directly: ``python tests/test_feature_store.py``. Prints PASS/FAIL
per test with a reason on failure, and exits non-zero if anything fails.

Covers:
  1. Training mode  -- 1000 rows through fit_batch; all engineered features
     present in the output table.
  2. Serving mode   -- one hardcoded transaction dict; features printed.
  3. Serving latency -- must be < 100 ms (production SLA).
"""

from __future__ import annotations

import glob
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Make `src.*` importable when run as a plain script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.features.feature_store import FeatureStore

# Keep test output clean; the feature pipeline is chatty at INFO.
logging.basicConfig(level=logging.WARNING)

# The full set of features get_features() must return / fit_batch must add.
EXPECTED_FEATURES = {
    "card_uid", "amt_was_missing", "txn_count_1h", "txn_count_24h",
    "txn_value_24h", "amt_no_history", "amt_dev_from_card_mean",
    "amt_dev_ratio_card_mean", "is_largest_for_card", "hour_of_day",
    "day_of_week", "is_weekend", "is_late_night", "card_age_days",
    "p_email_was_missing", "p_email_is_free", "card_amt_avg_30d",
    "device_type_fraud_rate",
}

SERVING_SLA_MS = 100.0

# A single hardcoded transaction (realistic IEEE-CIS shape).
HARDCODED_TXN = {
    "TransactionID": 9_999_001,
    "TransactionDT": 8_640_000,        # ~100 days past the reference epoch
    "TransactionAmt": 249.95,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 321.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 325.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "DeviceType": "mobile",
}


def _load_sample(n: int = 1000) -> pd.DataFrame:
    txn = glob.glob(
        str(_ROOT / "data" / "raw" / "**" / "train_transaction.csv"),
        recursive=True,
    )
    if not txn:
        raise FileNotFoundError(
            "train_transaction.csv not found under data/raw -- run "
            "data/download_data.py first."
        )
    df = pd.read_csv(txn[0], nrows=n)
    ident = glob.glob(
        str(_ROOT / "data" / "raw" / "**" / "train_identity.csv"),
        recursive=True,
    )
    if ident:
        ids = pd.read_csv(ident[0])
        df = df.merge(ids, on="TransactionID", how="left")
    return df


def _result(name: str, passed: bool, reason: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_training_mode(store: FeatureStore, sample: pd.DataFrame) -> bool:
    name = "training_mode (1000 rows, all features present)"
    try:
        enriched = store.fit_batch(sample)
    except Exception as exc:  # noqa: BLE001
        return _result(name, False, f"fit_batch raised: {exc!r}")

    if len(enriched) != len(sample):
        return _result(
            name, False,
            f"row count changed: {len(sample)} -> {len(enriched)}",
        )
    missing = sorted(EXPECTED_FEATURES - set(enriched.columns))
    if missing:
        return _result(name, False, f"missing features: {missing}")
    return _result(name, True)


def test_serving_mode(store: FeatureStore) -> bool:
    name = "serving_mode (hardcoded transaction)"
    try:
        feats = store.get_features(HARDCODED_TXN)
    except Exception as exc:  # noqa: BLE001
        return _result(name, False, f"get_features raised: {exc!r}")

    missing = sorted(EXPECTED_FEATURES - set(feats))
    print("  returned features:")
    for k in sorted(feats):
        print(f"    {k:28s} = {feats[k]}")
    if missing:
        return _result(name, False, f"missing keys: {missing}")
    return _result(name, True)


def test_serving_latency(store: FeatureStore) -> bool:
    name = f"serving_latency (< {SERVING_SLA_MS:.0f} ms SLA)"
    # Warm one call, then time a representative single inference.
    store.get_features(HARDCODED_TXN)
    t0 = time.perf_counter()
    store.get_features(HARDCODED_TXN)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  measured serving time: {elapsed_ms:.3f} ms")
    if elapsed_ms >= SERVING_SLA_MS:
        return _result(
            name, False,
            f"{elapsed_ms:.3f} ms exceeds {SERVING_SLA_MS:.0f} ms SLA",
        )
    return _result(name, True)


def main() -> int:
    print("=" * 64)
    print("feature_store test suite")
    print("=" * 64)
    try:
        sample = _load_sample(1000)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] setup -- {exc}")
        return 1

    store = FeatureStore()
    results = [
        test_training_mode(store, sample),
        test_serving_mode(store),
        test_serving_latency(store),
    ]

    print("-" * 64)
    passed = sum(results)
    print(f"{passed}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
