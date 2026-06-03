"""C2.5 / C6.2 — Fraud reasoning agent.

Subscribes to ``transactions_flagged`` (HOLD decisions, score > 0.70) and
produces a structured reasoning verdict, then publishes a complete audit record
to ``audit_log``.

Public API (C6 spec):
  * ``call_bedrock_agent(transaction, score, shap_reasons)`` — Amazon Bedrock
    Claude Sonnet path via boto3 ``invoke_model``.
  * ``local_agent(transaction, score, shap_reasons)`` — deterministic
    rule-based fallback (no network), used for local dev and tests.
  * ``process_flagged_transaction(transaction, score, shap_reasons)`` — main
    entry point: tries Bedrock when ``AWS_BEDROCK_ENABLED=true`` (with
    exponential backoff on throttling), otherwise / on failure uses the local
    agent. Always returns a valid decision dict.

Verdict shape (both paths):
  {decision, confidence, reasoning, fca_narrative, alert_fraud_team,
   draft_sar, estimated_liability_gbp, source, model}
where decision is one of HOLD_AND_STEP_UP | BLOCK | MONITOR.

NOTE on the model id: the mission spec named ``claude-sonnet-4-20250514``, but
live verification (2026-05-29) showed Claude 4.x is *inference-profile-only* in
eu-west-2 — the bare id is not invocable there. The default below is the
verified EU inference profile; override with $BEDROCK_MODEL_ID.
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

# Region eu-west-2 (FCA residency). EU cross-region inference profile is the
# default — see module docstring. Override with $BEDROCK_MODEL_ID.
_BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
_BEDROCK_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

_VALID_DECISIONS = ["HOLD_AND_STEP_UP", "BLOCK", "MONITOR"]
_REQUIRED_FIELDS = [
    "decision", "confidence", "reasoning",
    "fca_narrative", "alert_fraud_team", "draft_sar",
]


def call_bedrock_agent(transaction: dict, score: float,
                       shap_reasons: list) -> dict:
    """Calls Amazon Bedrock Claude Sonnet and returns a structured decision
    dict. Raises on any error so the caller can fall back to local_agent()."""

    prompt = f"""You are a fraud detection agent for an
FCA-regulated UK payment platform operating under PS21/3,
Consumer Duty, and PSR APP fraud reimbursement rules.

TRANSACTION FLAGGED — fraud probability: {score:.3f}

Transaction details:
- Amount: £{transaction.get('amount', 0):,.2f}
- Destination account age: {transaction.get('destination_account_age_days', 0)} days
- Card velocity last hour: {transaction.get('tx_velocity_1h', 0)} transactions
- Amount deviation from average: {transaction.get('amt_deviation', 1.0):.1f}x
- Hour of transaction: {transaction.get('hour_of_day', 12):02d}:00
- Device type: {transaction.get('device_type', 'unknown')}

Top SHAP signals (fraud drivers):
{chr(10).join(f"- {r}" for r in shap_reasons[:5])}

FCA context:
- PSR reimbursement liability applies if fraud confirmed
- Consumer Duty requires explainable automated decisions
- PS21/3 requires this decision to be audit-logged

Based on this evidence, choose ONE decision:
- HOLD_AND_STEP_UP: Hold payment, send step-up
  authentication to customer. Use when fraud is likely
  but customer confirmation could resolve it.
- BLOCK: Block payment immediately, alert fraud team.
  Use when fraud probability is very high or pattern
  matches known APP scam typology.
- MONITOR: Allow payment but flag for analyst review.
  Use when risk is elevated but not sufficient to hold.

Respond ONLY with valid JSON — no preamble, no explanation
outside the JSON:
{{
  "decision": "HOLD_AND_STEP_UP|BLOCK|MONITOR",
  "confidence": "HIGH|MEDIUM|LOW",
  "reasoning": "2-3 sentences plain English explaining
                the decision for the fraud analyst",
  "fca_narrative": "1-2 sentences suitable for FCA audit
                    log — factual, no speculation",
  "alert_fraud_team": true|false,
  "draft_sar": true|false,
  "estimated_liability_gbp": number
}}"""

    import json

    import boto3

    client = boto3.client('bedrock-runtime', region_name=_BEDROCK_REGION)

    response = client.invoke_model(
        modelId=_BEDROCK_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )

    body = json.loads(response['body'].read())
    usage = body.get('usage', {})
    logger.info("Bedrock %s tokens in=%s out=%s (txn=%s)", _BEDROCK_MODEL_ID,
                usage.get('input_tokens'), usage.get('output_tokens'),
                transaction.get('transaction_id'))
    text = body['content'][0]['text'].strip()

    # Strip markdown fences if present
    if text.startswith('```'):
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]

    result = json.loads(text)

    # Validate required fields
    for field in _REQUIRED_FIELDS:
        if field not in result:
            raise ValueError(f"Missing field: {field}")

    # Validate decision value
    if result['decision'] not in _VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {result['decision']}")

    result.setdefault('estimated_liability_gbp',
                      round(float(transaction.get('amount', 0)) * 0.5, 2))
    result['source'] = 'bedrock'
    result['model'] = _BEDROCK_MODEL_ID
    return result


def local_agent(transaction: dict, score: float,
                shap_reasons: list) -> dict:
    """Rule-based fallback when Bedrock is not available.
    Deterministic — same input always produces same output.
    Used for local development and testing."""
    amount = transaction.get('amount', 0)
    dest_age = transaction.get('destination_account_age_days', 365)
    velocity = transaction.get('tx_velocity_1h', 0)
    hour = transaction.get('hour_of_day', 12)

    # Decision logic mirroring Bedrock reasoning
    if score >= 0.90:
        decision = 'BLOCK'
        confidence = 'HIGH'
        reasoning = (
            f"Fraud probability {score:.1%} exceeds block "
            f"threshold. Amount £{amount:,.2f} to "
            f"{dest_age}-day-old account with velocity "
            f"{velocity}/hr strongly matches APP fraud pattern."
        )
        alert = True
        sar = score >= 0.95

    elif score >= 0.70 and dest_age <= 7:
        decision = 'HOLD_AND_STEP_UP'
        confidence = 'HIGH'
        reasoning = (
            f"New destination account ({dest_age} days) "
            f"combined with {score:.1%} fraud probability "
            f"warrants step-up authentication before release."
        )
        alert = False
        sar = False

    elif score >= 0.70 and (hour <= 5 or velocity >= 5):
        decision = 'HOLD_AND_STEP_UP'
        confidence = 'MEDIUM'
        reasoning = (
            f"Elevated fraud probability {score:.1%} "
            f"with late-night timing or high velocity "
            f"warrants customer confirmation."
        )
        alert = False
        sar = False

    else:
        decision = 'MONITOR'
        confidence = 'MEDIUM'
        reasoning = (
            f"Score {score:.1%} above threshold but "
            f"pattern does not meet hold criteria. "
            f"Flagged for analyst review."
        )
        alert = False
        sar = False

    fca_narrative = (
        f"Automated decision: {decision}. "
        f"Fraud probability: {score:.3f}. "
        f"Primary signals: {'; '.join(shap_reasons[:3])}. "
        f"Rule-based local agent (Bedrock unavailable)."
    )

    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'fca_narrative': fca_narrative,
        'alert_fraud_team': alert,
        'draft_sar': sar,
        'estimated_liability_gbp': round(amount * 0.5, 2),
        'source': 'local_rules',
        'model': 'rule_based_v1',
    }


def process_flagged_transaction(transaction: dict,
                                score: float,
                                shap_reasons: list) -> dict:
    """Main entry point. Tries Bedrock first, falls back to local.
    Always returns a valid decision dict.
    Handles exponential backoff on Bedrock throttling."""

    use_bedrock = os.getenv('AWS_BEDROCK_ENABLED',
                            'false').lower() == 'true'

    if use_bedrock:
        for attempt in range(3):
            try:
                result = call_bedrock_agent(
                    transaction, score, shap_reasons
                )
                return result
            except Exception as e:  # noqa: BLE001
                if 'ThrottlingException' in str(e):
                    wait = (2 ** attempt) * 1.0
                    time.sleep(wait)
                    continue
                print(f"Bedrock error: {e} — using local agent")
                break

    return local_agent(transaction, score, shap_reasons)


def _agent_transaction(event: dict) -> dict:
    """Build the flat transaction dict the agent functions expect from a
    streaming event. destination_account_age falls back to card age (the IEEE
    dataset has no shipping/destination address — see model card)."""
    txn = dict(event)
    if "destination_account_age_days" not in txn:
        txn["destination_account_age_days"] = event.get("card_age_days", 365)
    return txn


async def run(bus: StreamBus, metrics: dict | None = None) -> dict:
    """Consume flagged transactions, reason via process_flagged_transaction(),
    write an enriched audit record to ``audit_log``.

    Updates ``metrics["agent"]`` live (per-verdict counts) when provided."""
    import time as _time
    from datetime import datetime, timezone

    handled = 0
    async for event in bus.subscribe(TRANSACTIONS_FLAGGED):
        if event is SHUTDOWN:
            bus.done(TRANSACTIONS_FLAGGED)
            break
        try:
            sr = event.get("score_result", {})
            score = float(sr.get("fraud_probability", 0.0))
            shap_reasons = sr.get("reasons", [])
            t0 = _time.perf_counter()
            verdict = process_flagged_transaction(
                _agent_transaction(event), score, shap_reasons)
            processing_ms = round((_time.perf_counter() - t0) * 1000.0, 3)
            if metrics is not None:
                metrics["agent"][verdict["decision"]] += 1
            # Audit record = full decision dict + the C6.2 required fields.
            await bus.publish(AUDIT_LOG, {
                **verdict,
                "stage": "agent",
                "transaction_id": event["transaction_id"],
                "fraud_probability": score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": processing_ms,
                "fca_explanation": sr.get("fca_explanation"),
            })
            handled += 1
            logger.debug("[%s] agent %s -> %s (%s) via %s",
                         time.strftime("%H:%M:%S"), event["transaction_id"],
                         verdict["decision"], verdict["confidence"],
                         verdict["source"])
        finally:
            bus.done(TRANSACTIONS_FLAGGED)
    return {"agent_decisions": handled}
