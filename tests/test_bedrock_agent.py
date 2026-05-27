"""C6.3 — Bedrock agent test.

Two layers:
  * ALWAYS: validate the prompt builder and the response parser/validator
    against canned model output (no AWS needed) — this is the logic that turns
    a Bedrock response into an auditable verdict.
  * LIVE (only when AWS_BEDROCK_ENABLED=true and boto3 is available): send 5
    hardcoded flagged transactions to the real Bedrock agent and verify each
    verdict parses, has a valid decision, non-empty reasoning + fca_narrative.

Token usage is logged per call by bedrock_reason (see logs).

Run: python tests/test_bedrock_agent.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.streaming import fraud_agent

_VALID_RESPONSE = (
    'Here is my assessment:\n'
    '{"decision": "BLOCK", "confidence": "HIGH", '
    '"reasoning": "New account moving £5000 at 3am with high velocity.", '
    '"fca_narrative": "Automated screening blocked this transaction because '
    'the fraud probability and behavioural signals indicated a high risk.", '
    '"alert_fraud_team": true, "draft_sar": true}'
)


def _flagged(i: int, prob: float, **over) -> dict:
    base = {
        "transaction_id": f"TXN-BR-{i}", "card_id": f"CARD-BR-{i}",
        "amount": 4000.0, "device_type": "mobile", "hour_of_day": 3,
        "card_age_days": 1, "tx_velocity_1h": 8, "tx_velocity_24h": 15,
        "score_result": {"decision": "HOLD", "fraud_probability": prob,
                         "threshold_used": 0.19,
                         "reasons": ["high velocity", "new account",
                                     "late night"]},
    }
    base.update(over)
    return base


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_prompt_nonempty() -> bool:
    p = fraud_agent.build_prompt(_flagged(0, 0.95))
    ok = isinstance(p, str) and "JSON" in p and "TXN-BR-0" in p
    return _result("prompt builder produces a JSON-instructed prompt", ok)


def test_parse_valid() -> bool:
    v = fraud_agent.parse_bedrock_response(_VALID_RESPONSE)
    ok = (v["decision"] == "BLOCK" and v["confidence"] == "HIGH"
          and v["reasoning"] and v["fca_narrative"]
          and v["agent_mode"] == "bedrock")
    return _result("parser extracts + validates a valid verdict", ok, str(v))


def test_parse_rejects_bad() -> bool:
    bad_cases = [
        "no json here",
        '{"decision": "WAT", "confidence": "HIGH", "reasoning": "x", '
        '"fca_narrative": "y"}',                       # bad decision
        '{"decision": "BLOCK", "confidence": "HIGH", "reasoning": "", '
        '"fca_narrative": "y"}',                       # empty reasoning
    ]
    all_rejected = True
    for case in bad_cases:
        try:
            fraud_agent.parse_bedrock_response(case)
            all_rejected = False
        except ValueError:
            pass
    return _result("parser rejects malformed / invalid responses",
                   all_rejected)


def test_live_bedrock() -> bool | None:
    if os.getenv("AWS_BEDROCK_ENABLED", "").lower() != "true":
        print("[SKIP] AWS_BEDROCK_ENABLED!=true — live Bedrock test skipped.")
        return None
    try:
        import boto3  # noqa: F401
    except ImportError:
        print("[SKIP] boto3 not installed — live Bedrock test skipped.")
        return None

    probs = [0.95, 0.82, 0.74, 0.88, 0.97]
    ok = True
    for i, p in enumerate(probs):
        try:
            v = fraud_agent.bedrock_reason(_flagged(i, p))
            if (v["decision"] not in fraud_agent._VALID_DECISIONS
                    or not v["reasoning"] or not v["fca_narrative"]):
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"  live call {i} failed: {exc!r}")
            ok = False
    return _result("5 live Bedrock verdicts valid", ok)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    print("=" * 64)
    print("Bedrock agent test suite")
    print("=" * 64)
    results = [test_prompt_nonempty(), test_parse_valid(),
               test_parse_rejects_bad()]
    live = test_live_bedrock()
    if live is not None:
        results.append(live)
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
