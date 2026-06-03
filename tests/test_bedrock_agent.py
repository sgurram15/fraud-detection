"""C6.3 — Bedrock agent test.

All six tests run WITHOUT requiring live Bedrock access — they exercise the
deterministic local agent and the Bedrock-disabled fallback path. (A separate
live check was done under C6.1 / the mission log.)

Run: python tests/test_bedrock_agent.py
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.streaming import fraud_agent

_REASONS = ["high velocity", "new account", "late night",
            "amount deviation", "device mismatch"]


def _txn(**over) -> dict:
    base = {
        "transaction_id": "TXN-T", "amount": 1000.0, "device_type": "mobile",
        "hour_of_day": 12, "destination_account_age_days": 180,
        "tx_velocity_1h": 0, "amt_deviation": 1.0,
    }
    base.update(over)
    return base


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_block() -> bool:
    r = fraud_agent.local_agent(
        _txn(amount=3000, destination_account_age_days=2), 0.95, _REASONS)
    ok = (r["decision"] == "BLOCK" and r["confidence"] == "HIGH"
          and r["alert_fraud_team"] is True and r["draft_sar"] is True)
    return _result("local_agent BLOCK on score=0.95/dest_age=2/£3000", ok,
                   str(r))


def test_hold_step_up() -> bool:
    r = fraud_agent.local_agent(
        _txn(amount=2000, destination_account_age_days=3), 0.82, _REASONS)
    ok = (r["decision"] == "HOLD_AND_STEP_UP" and r["confidence"] == "HIGH"
          and r["alert_fraud_team"] is False)
    return _result("local_agent HOLD_AND_STEP_UP on score=0.82/dest_age=3",
                   ok, str(r))


def test_monitor() -> bool:
    r = fraud_agent.local_agent(
        _txn(amount=500, destination_account_age_days=180, hour_of_day=14,
             tx_velocity_1h=2), 0.72, _REASONS)
    ok = r["decision"] == "MONITOR"
    return _result("local_agent MONITOR on score=0.72/established account",
                   ok, str(r))


def test_all_fields_present() -> bool:
    required = ["decision", "confidence", "reasoning", "fca_narrative",
                "alert_fraud_team", "draft_sar", "estimated_liability_gbp",
                "source", "model"]
    txns = [
        (_txn(amount=3000, destination_account_age_days=2), 0.96),
        (_txn(amount=2000, destination_account_age_days=3), 0.82),
        (_txn(amount=500, hour_of_day=14, tx_velocity_1h=2), 0.72),
        (_txn(amount=1500, hour_of_day=3), 0.78),
        (_txn(amount=800, tx_velocity_1h=7), 0.85),
    ]
    ok = True
    for txn, score in txns:
        r = fraud_agent.local_agent(txn, score, _REASONS)
        missing = [f for f in required if f not in r]
        if missing:
            ok = False
            print(f"    missing {missing} for score={score}")
    return _result("all 9 required fields present in every response", ok)


def test_reasoning_english() -> bool:
    txns = [
        (_txn(amount=3000, destination_account_age_days=2), 0.96),
        (_txn(amount=2000, destination_account_age_days=3), 0.82),
        (_txn(amount=500, hour_of_day=14, tx_velocity_1h=2), 0.72),
        (_txn(amount=1500, hour_of_day=3), 0.78),
        (_txn(amount=800, tx_velocity_1h=7), 0.85),
    ]
    ok = True
    for txn, score in txns:
        r = fraud_agent.local_agent(txn, score, _REASONS)
        if len(r["reasoning"]) <= 20:
            ok = False
            print(f"    reasoning too short for score={score}")
        if "Fraud probability" not in r["fca_narrative"]:
            ok = False
            print(f"    fca_narrative missing 'Fraud probability' "
                  f"for score={score}")
    return _result("reasoning >20 chars; fca_narrative has 'Fraud probability'",
                   ok)


def test_bedrock_disabled() -> bool:
    import src.streaming.fraud_agent as fa

    # Ensure Bedrock is disabled and prove no boto3 call is attempted by
    # making any boto3 import inside call_bedrock_agent blow up loudly.
    prev = {k: os.environ.get(k) for k in ("AWS_BEDROCK_ENABLED",)}
    os.environ["AWS_BEDROCK_ENABLED"] = "false"
    called = {"bedrock": False}
    orig = fa.call_bedrock_agent

    def _tripwire(*a, **k):
        called["bedrock"] = True
        return orig(*a, **k)

    fa.call_bedrock_agent = _tripwire
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = fa.process_flagged_transaction(
                _txn(amount=3000, destination_account_age_days=2), 0.95,
                _REASONS)
        ok = (r["source"] == "local_rules" and called["bedrock"] is False)
    finally:
        fa.call_bedrock_agent = orig
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return _result("process_flagged_transaction uses local_rules, no Bedrock "
                   "call, when AWS_BEDROCK_ENABLED=false", ok, str(r))


def main() -> int:
    print("=" * 64)
    print("Bedrock agent test suite (C6.3)")
    print("=" * 64)
    results = [
        test_block(), test_hold_step_up(), test_monitor(),
        test_all_fields_present(), test_reasoning_english(),
        test_bedrock_disabled(),
    ]
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
