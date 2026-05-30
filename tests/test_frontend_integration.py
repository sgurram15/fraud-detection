"""Frontend integration test — verifies the live API response shape that
src/frontend/index.html relies on.

Run: python tests/test_frontend_integration.py

Uses FastAPI TestClient (no running server). Scores 5 transactions calibrated
to collectively touch APPROVE / REVIEW / HOLD, and asserts every field the
frontend reads. Prints PASS/FAIL per field per transaction; non-zero exit on
any failure.

Calibration mirrors tests/test_api.py: the production model expects the full
IEEE-CIS column set; the API NaN-fills unsupplied columns so scores are not
intuitively monotonic. Fixtures are tuned on fresh card_ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

REQUIRED_FIELDS = [
    "transaction_id", "fraud_probability", "decision", "threshold_used",
    "reasons", "fca_explanation", "model_version", "latency_ms",
]

_CID = iter(range(1, 10**9))


def _txn(**over) -> dict:
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


# Calibrated fixtures (same as tests/test_api.py): APPROVE ~0.16, REVIEW ~0.45,
# HOLD ~0.79. Two extra REVIEW-band txns round out to 5 total.
FIXTURES = [
    ("APPROVE", dict(amount=80.0, device_type="desktop", hour_of_day=3,
                     tx_velocity_1h=8, tx_velocity_24h=20, amt_deviation=1.0,
                     card_age_days=0, is_late_night=True)),
    ("REVIEW",  dict(amount=500.0, device_type="desktop", hour_of_day=20,
                     tx_velocity_1h=2, tx_velocity_24h=4, amt_deviation=2.0,
                     card_age_days=120)),
    ("HOLD",    dict(amount=1500.0, device_type="mobile", hour_of_day=11,
                     tx_velocity_1h=2, tx_velocity_24h=5, amt_deviation=0.5,
                     card_age_days=200)),
    ("REVIEW",  dict(amount=250.0, device_type="mobile", hour_of_day=14,
                     tx_velocity_1h=1, tx_velocity_24h=3, amt_deviation=1.5,
                     card_age_days=60)),
    ("APPROVE", dict(amount=45.0, device_type="desktop", hour_of_day=10,
                     tx_velocity_1h=0, tx_velocity_24h=1, amt_deviation=0.8,
                     card_age_days=500)),
]


_PASS = "[PASS]"
_FAIL = "[FAIL]"


class Recorder:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, label: str, ok: bool, reason: str = "") -> bool:
        line = f"  {_PASS if ok else _FAIL} {label}"
        if not ok and reason:
            line += f"  -- {reason}"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        return ok


def _verify_response(rec: Recorder, body: dict) -> None:
    # Schema completeness
    missing = [f for f in REQUIRED_FIELDS if f not in body]
    rec.check(f"all 8 required fields present",
              not missing, f"missing {missing}")

    # fraud_probability: float in [0, 1]
    p = body.get("fraud_probability")
    rec.check("fraud_probability is float in [0,1]",
              isinstance(p, (int, float)) and 0.0 <= float(p) <= 1.0,
              f"got {p!r}")

    # decision
    d = body.get("decision")
    rec.check("decision in {APPROVE, REVIEW, HOLD}",
              d in {"APPROVE", "REVIEW", "HOLD"},
              f"got {d!r}")

    # reasons: non-empty list of str
    r = body.get("reasons")
    rec.check("reasons is non-empty list[str]",
              isinstance(r, list) and len(r) > 0
              and all(isinstance(x, str) and x for x in r),
              f"got {type(r).__name__} len={len(r) if isinstance(r, list) else 'n/a'}")

    # fca_explanation: non-empty dict
    fca = body.get("fca_explanation")
    rec.check("fca_explanation is non-empty dict",
              isinstance(fca, dict) and len(fca) > 0,
              f"got {type(fca).__name__}")

    # shap_values inside fca_explanation: dict[str, float]
    sv = fca.get("shap_values") if isinstance(fca, dict) else None
    sv_ok = (isinstance(sv, dict) and len(sv) > 0
             and all(isinstance(k, str) and isinstance(v, (int, float))
                     for k, v in sv.items()))
    rec.check("fca_explanation.shap_values is dict[str,float]",
              sv_ok,
              f"got {type(sv).__name__}"
              + (f" len={len(sv)}" if isinstance(sv, dict) else ""))

    # latency_ms: positive float
    lm = body.get("latency_ms")
    rec.check("latency_ms is positive float",
              isinstance(lm, (int, float)) and float(lm) > 0.0,
              f"got {lm!r}")

    # model_version: non-empty string
    mv = body.get("model_version")
    rec.check("model_version is non-empty str",
              isinstance(mv, str) and len(mv) > 0,
              f"got {mv!r}")


def main() -> int:
    print("=" * 72)
    print("Frontend integration test (FastAPI TestClient + src/api/main.py)")
    print("=" * 72)
    rec = Recorder()
    decisions_seen: set[str] = set()

    for i, (expected_band, over) in enumerate(FIXTURES, 1):
        payload = _txn(**over)
        print(f"\n--- txn {i}/5  (target band: {expected_band}, "
              f"txn={payload['transaction_id']}) ---")
        resp = client.post("/score", json=payload)
        ok_http = rec.check(f"HTTP 200", resp.status_code == 200,
                            f"status {resp.status_code}")
        if not ok_http:
            continue
        body = resp.json()
        decisions_seen.add(body.get("decision", ""))
        print(f"  decision={body.get('decision')} "
              f"prob={body.get('fraud_probability'):.4f}")
        _verify_response(rec, body)

    print()
    print("-" * 72)
    bands_ok = decisions_seen >= {"APPROVE", "REVIEW", "HOLD"}
    rec.check("all three bands covered by 5 fixtures (APPROVE+REVIEW+HOLD)",
              bands_ok, f"got {sorted(decisions_seen)}")
    print("-" * 72)
    print(f"{rec.passed} passed, {rec.failed} failed")
    return 0 if rec.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
