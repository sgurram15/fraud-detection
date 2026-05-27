"""C2.5 — Fraud reasoning agent (local Bedrock simulation).

Subscribes to ``transactions_flagged`` (HOLD decisions, score > 0.70) and
produces a structured reasoning verdict, then publishes a complete audit record
to ``audit_log``.

Two modes, selected by $AWS_BEDROCK_ENABLED:
  * unset / false  -> local rule-based reasoner (no network, deterministic).
  * true           -> Amazon Bedrock Claude Sonnet via boto3 (wired in C6.2).

Both modes emit the same verdict shape:
  {decision, confidence, reasoning, fca_narrative, alert_fraud_team, draft_sar}
where decision is one of HOLD_AND_STEP_UP | BLOCK | MONITOR.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.streaming.stream_bus import (
    AUDIT_LOG,
    SHUTDOWN,
    StreamBus,
    TRANSACTIONS_FLAGGED,
)

logger = logging.getLogger("agent")

_NEW_ACCOUNT_DAYS = 30  # card age below this counts as a "new account"


def _is_new_account(event: dict) -> bool:
    age = event.get("card_age_days", 0) or 0
    return age < _NEW_ACCOUNT_DAYS


def local_reason(event: dict) -> dict:
    """Rule-based stand-in for the Bedrock agent (mission C2.5 rules)."""
    score = float(event.get("score_result", {}).get("fraud_probability", 0.0))
    amount = float(event.get("amount", 0.0))
    new_account = _is_new_account(event)
    top_reasons = event.get("score_result", {}).get("reasons", [])[:3]
    reason_str = "; ".join(top_reasons) if top_reasons else "model risk signals"

    if score > 0.90:
        decision, confidence = "BLOCK", "HIGH"
        reasoning = (
            f"Fraud probability {score:.2f} exceeds the 0.90 block threshold. "
            f"Transaction of £{amount:,.2f} blocked outright. Drivers: "
            f"{reason_str}."
        )
        alert, sar = True, True
    elif new_account:
        decision, confidence = "HOLD_AND_STEP_UP", "HIGH"
        reasoning = (
            f"Fraud probability {score:.2f} on a new account "
            f"(card age < {_NEW_ACCOUNT_DAYS} days) for £{amount:,.2f}. "
            f"Holding for step-up authentication. Drivers: {reason_str}."
        )
        alert, sar = True, False
    else:
        decision, confidence = "HOLD_AND_STEP_UP", "MEDIUM"
        reasoning = (
            f"Fraud probability {score:.2f} on an established card for "
            f"£{amount:,.2f}. Step-up authentication requested before "
            f"settlement. Drivers: {reason_str}."
        )
        alert, sar = False, False

    fca_narrative = (
        f"Automated screening assigned a fraud probability of {score:.2f} "
        f"(operating threshold "
        f"{event.get('score_result', {}).get('threshold_used', 'n/a')}). "
        f"Decision: {decision} (confidence {confidence}). The principal "
        f"contributing factors were: {reason_str}. This explanation is "
        f"retained for the customer and for FCA audit (Consumer Duty / "
        f"UK GDPR Art. 22)."
    )
    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "fca_narrative": fca_narrative,
        "alert_fraud_team": alert,
        "draft_sar": sar,
        "agent_mode": "local",
    }


def bedrock_reason(event: dict) -> dict:
    """Amazon Bedrock Claude Sonnet path. Fully wired in C6.2 — until then it
    raises so callers fall back to :func:`local_reason` rather than silently
    returning a fake verdict."""
    raise NotImplementedError(
        "Bedrock agent path is implemented in Phase C6.2. Unset "
        "AWS_BEDROCK_ENABLED to use the local rule-based reasoner."
    )


def reason(event: dict) -> dict:
    if os.getenv("AWS_BEDROCK_ENABLED", "").lower() == "true":
        try:
            return bedrock_reason(event)
        except Exception as exc:
            logger.warning("Bedrock path failed (%s); using local reasoner",
                           exc)
    verdict = local_reason(event)
    if verdict["agent_mode"] == "local":
        logger.debug("[LOCAL AGENT] Decision made without Bedrock")
    return verdict


async def run(bus: StreamBus, metrics: dict | None = None) -> dict:
    """Consume flagged transactions, reason, write enriched audit record.

    Updates ``metrics["agent"]`` live (per-verdict counts) when provided."""
    handled = 0
    async for event in bus.subscribe(TRANSACTIONS_FLAGGED):
        if event is SHUTDOWN:
            bus.done(TRANSACTIONS_FLAGGED)
            break
        try:
            verdict = reason(event)
            if metrics is not None:
                metrics["agent"][verdict["decision"]] += 1
            await bus.publish(AUDIT_LOG, {
                "transaction_id": event["transaction_id"],
                "stage": "agent",
                "decision": event.get("score_result", {}).get("decision"),
                "fraud_probability": event.get("score_result", {})
                .get("fraud_probability"),
                "agent_verdict": verdict,
                "fca_explanation": event.get("score_result", {})
                .get("fca_explanation"),
            })
            handled += 1
            logger.debug("[%s] agent %s -> %s (%s)", time.strftime("%H:%M:%S"),
                         event["transaction_id"], verdict["decision"],
                         verdict["confidence"])
        finally:
            bus.done(TRANSACTIONS_FLAGGED)
    return {"agent_decisions": handled}
