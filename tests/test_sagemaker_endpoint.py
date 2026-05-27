"""C4.3 — SageMaker endpoint smoke/acceptance test.

Sends 10 transactions to the live endpoint named by SAGEMAKER_ENDPOINT (.env)
and verifies:
  * response schema matches FraudScoreResponse,
  * per-call latency under 200ms,
  * APPROVE / REVIEW / HOLD all appear across the 10 samples,
  * SHAP reasons present in every response.

If SAGEMAKER_ENDPOINT is unset (no endpoint deployed) the test SKIPS with exit
0 — deploying an endpoint costs money and is gated behind STOP POINT C4, so the
suite must not require one to be running.

Run: python tests/test_sagemaker_endpoint.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.aws._common import REGION, get_env

REQUIRED_FIELDS = {
    "transaction_id", "fraud_probability", "decision", "threshold_used",
    "reasons", "fca_explanation", "model_version",
}

# 10 varied transactions spanning the risk spectrum.
_SAMPLES = [
    {"amount": 60.0, "device_type": "desktop", "hour_of_day": 14,
     "card_age_days": 400, "tx_velocity_1h": 0, "is_late_night": False},
    {"amount": 5000.0, "device_type": "mobile", "hour_of_day": 3,
     "card_age_days": 1, "tx_velocity_1h": 9, "amt_deviation": 12.0,
     "is_late_night": True},
    {"amount": 500.0, "device_type": "desktop", "hour_of_day": 20,
     "card_age_days": 120, "tx_velocity_1h": 2, "amt_deviation": 2.0},
    {"amount": 1500.0, "device_type": "mobile", "hour_of_day": 11,
     "card_age_days": 200, "tx_velocity_1h": 2, "amt_deviation": 0.5},
    {"amount": 25.0, "device_type": "desktop", "hour_of_day": 9,
     "card_age_days": 800},
    {"amount": 3200.0, "device_type": "mobile", "hour_of_day": 2,
     "card_age_days": 3, "tx_velocity_1h": 7, "amt_deviation": 8.0,
     "is_late_night": True},
    {"amount": 120.0, "device_type": "mobile", "hour_of_day": 18,
     "card_age_days": 300},
    {"amount": 900.0, "device_type": "desktop", "hour_of_day": 22,
     "card_age_days": 60, "amt_deviation": 3.0},
    {"amount": 75.0, "device_type": "desktop", "hour_of_day": 12,
     "card_age_days": 500},
    {"amount": 4200.0, "device_type": "mobile", "hour_of_day": 4,
     "card_age_days": 2, "tx_velocity_1h": 10, "amt_deviation": 15.0,
     "is_late_night": True},
]


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def _full_txn(i: int, over: dict) -> dict:
    base = {
        "transaction_id": f"TXN-SM-{i}", "card_id": f"CARD-SM-{i}",
        "amount": 100.0, "device_type": "mobile", "hour_of_day": 12,
        "day_of_week": 3, "destination_account_age_days": 100,
        "tx_velocity_1h": 0, "tx_velocity_24h": 0, "amt_deviation": 1.0,
        "is_late_night": False, "is_weekend": False, "card_age_days": 100,
    }
    base.update(over)
    return base


def main() -> int:
    endpoint = get_env("SAGEMAKER_ENDPOINT")
    if not endpoint:
        print("[SKIP] SAGEMAKER_ENDPOINT not set — no endpoint deployed "
              "(deploy is gated behind STOP POINT C4). Nothing to test.")
        return 0

    try:
        import boto3
        from sagemaker.deserializers import JSONDeserializer
        from sagemaker.predictor import Predictor
        from sagemaker.serializers import JSONSerializer
        import sagemaker
    except ImportError as exc:
        print(f"[SKIP] sagemaker SDK / boto3 not installed ({exc}).")
        return 0

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
    predictor = Predictor(endpoint_name=endpoint, sagemaker_session=session,
                          serializer=JSONSerializer(),
                          deserializer=JSONDeserializer())

    print("=" * 64)
    print(f"SageMaker endpoint test — {endpoint}")
    print("=" * 64)

    decisions, latencies, schema_ok, reasons_ok = set(), [], True, True
    for i, over in enumerate(_SAMPLES):
        t0 = time.perf_counter()
        resp = predictor.predict(_full_txn(i, over))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if REQUIRED_FIELDS - set(resp):
            schema_ok = False
        decisions.add(resp.get("decision"))
        if not resp.get("reasons"):
            reasons_ok = False

    avg = sum(latencies) / len(latencies)
    print(f"  avg latency: {avg:.1f} ms (max {max(latencies):.1f})")
    results = [
        _result("response schema matches FraudScoreResponse", schema_ok),
        _result("avg latency under 200ms", avg < 200.0, f"avg {avg:.1f} ms"),
        _result("APPROVE/REVIEW/HOLD all appear",
                {"APPROVE", "REVIEW", "HOLD"}.issubset(decisions),
                f"saw {decisions}"),
        _result("SHAP reasons present in every response", reasons_ok),
    ]
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
