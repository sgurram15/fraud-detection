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

import json
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

# Bedrock config (C6.2). Region eu-west-2 (FCA residency); the EU cross-region
# inference profile is the default (Claude 4.x models are profile-only — the
# bare model id is not on-demand invocable in eu-west-2). Override with
# $BEDROCK_MODEL_ID. Verified working: eu.anthropic.claude-sonnet-4-5-...,
# anthropic.claude-3-7-sonnet-20250219-v1:0.
_BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
_BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
_BEDROCK_MAX_RETRIES = 5
_VALID_DECISIONS = {"HOLD_AND_STEP_UP", "BLOCK", "MONITOR"}
_VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


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


def build_prompt(event: dict) -> str:
    """Compose the fraud-reasoning prompt for one flagged transaction."""
    sr = event.get("score_result", {})
    reasons = sr.get("reasons", [])[:5]
    return (
        "You are a fraud analyst for a UK Payment Service Provider, operating "
        "under FCA Consumer Duty and UK GDPR Art. 22 (right to an explanation "
        "of automated decisions). A transaction has been flagged by the "
        "scoring model. Decide the action.\n\n"
        f"Transaction: id={event.get('transaction_id')}, "
        f"amount=£{event.get('amount')}, device={event.get('device_type')}, "
        f"hour={event.get('hour_of_day')}, card_age_days="
        f"{event.get('card_age_days')}, velocity_1h="
        f"{event.get('tx_velocity_1h')}, velocity_24h="
        f"{event.get('tx_velocity_24h')}.\n"
        f"Model fraud probability: {sr.get('fraud_probability')} "
        f"(threshold {sr.get('threshold_used')}).\n"
        f"Top model reason codes: {reasons}\n\n"
        "Respond with ONLY a JSON object, no prose, with exactly these keys:\n"
        '{"decision": "HOLD_AND_STEP_UP|BLOCK|MONITOR", '
        '"confidence": "HIGH|MEDIUM|LOW", '
        '"reasoning": "plain-English paragraph for the analyst", '
        '"fca_narrative": "explanation suitable for the FCA audit log", '
        '"alert_fraud_team": true|false, "draft_sar": true|false}'
    )


def parse_bedrock_response(text: str) -> dict:
    """Extract and validate the agent verdict JSON from the model's text.
    Raises ValueError if it is missing required keys or invalid values."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in Bedrock response")
    verdict = json.loads(text[start:end + 1])
    if verdict.get("decision") not in _VALID_DECISIONS:
        raise ValueError(f"invalid decision: {verdict.get('decision')!r}")
    if verdict.get("confidence") not in _VALID_CONFIDENCE:
        raise ValueError(f"invalid confidence: {verdict.get('confidence')!r}")
    for key in ("reasoning", "fca_narrative"):
        if not str(verdict.get(key, "")).strip():
            raise ValueError(f"empty {key}")
    verdict.setdefault("alert_fraud_team", verdict["decision"] != "MONITOR")
    verdict.setdefault("draft_sar", verdict["decision"] == "BLOCK")
    verdict["agent_mode"] = "bedrock"
    return verdict


def bedrock_reason(event: dict) -> dict:
    """Amazon Bedrock Claude Sonnet path (C6.2): boto3 bedrock-runtime Converse
    API with exponential backoff on throttling and per-call token logging."""
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("bedrock-runtime", region_name=_BEDROCK_REGION)
    messages = [{"role": "user",
                 "content": [{"text": build_prompt(event)}]}]
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, _BEDROCK_MAX_RETRIES + 1):
        try:
            resp = client.converse(
                modelId=_BEDROCK_MODEL_ID,
                messages=messages,
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
            logger.info("Bedrock %s tokens in=%s out=%s total=%s (txn=%s)",
                        _BEDROCK_MODEL_ID, usage.get("inputTokens"),
                        usage.get("outputTokens"), usage.get("totalTokens"),
                        event.get("transaction_id"))
            return parse_bedrock_response(text)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            last_exc = exc
            if code in ("ThrottlingException", "TooManyRequestsException",
                        "ServiceUnavailableException") and \
                    attempt < _BEDROCK_MAX_RETRIES:
                logger.warning("Bedrock throttled (%s); retry %d in %.1fs",
                               code, attempt, delay)
                time.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            raise
    raise RuntimeError(f"Bedrock failed after {_BEDROCK_MAX_RETRIES} attempts: "
                       f"{last_exc}")


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
