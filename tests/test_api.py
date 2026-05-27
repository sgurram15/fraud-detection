"""C1.8 — API acceptance tests for src/api/main.py (FastAPI TestClient).

Run: python tests/test_api.py   (no running server needed)

Prints PASS/FAIL per test with a reason on failure; exits non-zero on any
failure.

Calibration note: the production model expects the full IEEE-CIS feature
vector (417 cols); the API supplies ~18 engineered features and NaN-fills the
rest. The resulting probability is therefore NOT intuitively monotonic — the
spec's "amount=5000, new account, late night" profile actually scores ~0.4
(REVIEW). The decision-band fixtures below are empirically calibrated against
the loaded model on FRESH card_ids (get_features mutates per-card state, so a
fresh card_id per request is required for determinism). If the model artifact
is retrained, recalibrate these fixtures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

REQUIRED_FIELDS = {
    "transaction_id", "fraud_probability", "decision", "threshold_used",
    "reasons", "fca_explanation", "model_version", "latency_ms",
}

_CID = iter(range(1, 10**9))


def _txn(**over) -> dict:
    """A transaction with a UNIQUE card_id each call (fresh feature-store
    state -> deterministic score)."""
    base = {
        "transaction_id": f"TXN-{next(_CID)}",
        "card_id": f"CARD-{next(_CID)}",
        "amount": 100.0,
        "device_type": "mobile",
        "hour_of_day": 12,
        "day_of_week": 2,
        "destination_account_age_days": 100,
        "tx_velocity_1h": 0,
        "tx_velocity_24h": 0,
        "amt_deviation": 1.0,
        "is_late_night": False,
        "is_weekend": False,
        "card_age_days": 300,
    }
    base.update(over)
    return base


# Empirically calibrated band fixtures (fresh-card probabilities):
#   APPROVE ~0.16 | REVIEW ~0.45 | HOLD ~0.79
APPROVE_TXN = dict(amount=80.0, device_type="desktop", hour_of_day=3,
                   tx_velocity_1h=8, tx_velocity_24h=20, amt_deviation=1.0,
                   card_age_days=0, is_late_night=True)
REVIEW_TXN = dict(amount=500.0, device_type="desktop", hour_of_day=20,
                  tx_velocity_1h=2, tx_velocity_24h=4, amt_deviation=2.0,
                  card_age_days=120)
HOLD_TXN = dict(amount=1500.0, device_type="mobile", hour_of_day=11,
                tx_velocity_1h=2, tx_velocity_24h=5, amt_deviation=0.5,
                card_age_days=200)


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_approve() -> bool:
    r = client.post("/score", json=_txn(**APPROVE_TXN))
    if r.status_code != 200:
        return _result("approve_case", False, f"status {r.status_code}")
    b = r.json()
    return _result(
        "approve_case (low-risk -> APPROVE)", b["decision"] == "APPROVE",
        f"decision={b['decision']} prob={b['fraud_probability']:.4f} "
        f"(expected <0.30)")


def test_hold() -> bool:
    r = client.post("/score", json=_txn(**HOLD_TXN))
    if r.status_code != 200:
        return _result("hold_case", False, f"status {r.status_code}")
    b = r.json()
    return _result(
        "hold_case (high-risk -> HOLD)", b["decision"] == "HOLD",
        f"decision={b['decision']} prob={b['fraud_probability']:.4f} "
        f"(expected >0.70)")


def test_review() -> bool:
    r = client.post("/score", json=_txn(**REVIEW_TXN))
    if r.status_code != 200:
        return _result("review_case", False, f"status {r.status_code}")
    b = r.json()
    return _result(
        "review_case (medium-risk -> REVIEW)", b["decision"] == "REVIEW",
        f"decision={b['decision']} prob={b['fraud_probability']:.4f} "
        f"(expected 0.30-0.70)")


def test_batch() -> bool:
    payload = [_txn(amount=100.0 + i * 10) for i in range(10)]
    r = client.post("/score/batch", json=payload)
    if r.status_code != 200:
        return _result("batch (10 txns)", False, f"status {r.status_code}")
    body = r.json()
    if len(body) != 10:
        return _result("batch (10 txns)", False, f"got {len(body)} responses")
    for i, item in enumerate(body):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            return _result("batch (10 txns)", False,
                           f"item {i} missing {missing}")
        if item["decision"] not in {"APPROVE", "REVIEW", "HOLD"}:
            return _result("batch (10 txns)", False,
                           f"item {i} bad decision {item['decision']}")
    return _result("batch (10 txns, all valid responses)", True)


def test_health() -> bool:
    r = client.get("/health")
    if r.status_code != 200:
        return _result("health_check", False, f"status {r.status_code}")
    b = r.json()
    ok = b.get("status") == "healthy" and b.get("model_loaded") is True
    return _result("health_check (status=healthy)", ok, f"body={b}")


def test_latency() -> bool:
    n = 100
    times = []
    for _ in range(n):
        body = _txn(amount=120.0)
        t0 = time.perf_counter()
        r = client.post("/score", json=body)
        times.append((time.perf_counter() - t0) * 1000.0)
        if r.status_code != 200:
            return _result("latency (100 reqs)", False,
                           f"status {r.status_code}")
    avg = sum(times) / len(times)
    print(f"  avg latency over {n} requests: {avg:.2f} ms "
          f"(min {min(times):.1f}, max {max(times):.1f})")
    return _result(f"latency (100 reqs, avg<100ms)", avg < 100.0,
                   f"avg {avg:.2f} ms")


def test_schema() -> bool:
    # Check every required field is present across a few varied responses.
    for fixture in (APPROVE_TXN, REVIEW_TXN, HOLD_TXN):
        b = client.post("/score", json=_txn(**fixture)).json()
        missing = REQUIRED_FIELDS - set(b)
        if missing:
            return _result("schema (all fields present)", False,
                           f"missing {missing}")
        if not isinstance(b["reasons"], list) or not isinstance(
                b["fca_explanation"], dict):
            return _result("schema (all fields present)", False,
                           "reasons/fca_explanation wrong type")
    return _result("schema (all required fields present)", True)


def test_error_missing_field() -> bool:
    bad = _txn()
    del bad["amount"]  # required, no default
    r = client.post("/score", json=bad)
    return _result("error (missing field -> 422 not 500)",
                   r.status_code == 422, f"status {r.status_code}")


def main() -> int:
    print("=" * 64)
    print("API test suite (src/api/main.py)")
    print("=" * 64)
    results = [
        test_approve(),
        test_review(),
        test_hold(),
        test_batch(),
        test_health(),
        test_latency(),
        test_schema(),
        test_error_missing_field(),
    ]
    print("-" * 64)
    passed = sum(results)
    print(f"{passed}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
