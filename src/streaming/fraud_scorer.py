"""C2.4 — Fraud scorer consumer.

Subscribes to ``transactions_enriched``, calls ``POST /score`` on the FastAPI
service, and routes each transaction by decision:

    APPROVE -> transactions_approved
    REVIEW  -> transactions_scored
    HOLD    -> transactions_flagged

and writes an audit record to ``audit_log`` for every transaction.

Transport: by default the scorer talks to the imported app in-process via an
httpx ASGI transport (no separate server, works in tests). Set $FRAUD_API_URL
(e.g. http://localhost:8000) to call a real running uvicorn / SageMaker-style
HTTP endpoint instead.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.streaming.stream_bus import (
    AUDIT_LOG,
    SHUTDOWN,
    StreamBus,
    TRANSACTIONS_APPROVED,
    TRANSACTIONS_ENRICHED,
    TRANSACTIONS_FLAGGED,
    TRANSACTIONS_SCORED,
)

logger = logging.getLogger("scorer")

_DECISION_TOPIC = {
    "APPROVE": TRANSACTIONS_APPROVED,
    "REVIEW": TRANSACTIONS_SCORED,
    "HOLD": TRANSACTIONS_FLAGGED,
}

# Fields the /score endpoint (TransactionRequest) accepts.
_REQUEST_FIELDS = (
    "transaction_id", "card_id", "amount", "device_type", "hour_of_day",
    "day_of_week", "destination_account_age_days", "tx_velocity_1h",
    "tx_velocity_24h", "amt_deviation", "is_late_night", "is_weekend",
    "card_age_days",
)


def _to_request(event: dict) -> dict:
    return {k: event[k] for k in _REQUEST_FIELDS if k in event}


def _make_client() -> httpx.AsyncClient:
    base = os.getenv("FRAUD_API_URL")
    if base:
        logger.info("Scorer using HTTP endpoint %s", base)
        return httpx.AsyncClient(base_url=base, timeout=10.0)
    # In-process ASGI: import here so a real-HTTP run need not load the model.
    from src.api.main import app
    logger.info("Scorer using in-process ASGI transport")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://fraud-api",
        timeout=10.0,
    )


async def run(bus: StreamBus, metrics: dict | None = None) -> dict:
    """Consume enriched, score via API, route by decision. Returns stats.

    If ``metrics`` is given it is updated live (decision counts + latency sum)
    so an external dashboard can read progress without waiting for shutdown."""
    decisions: Counter = Counter()
    scored = 0
    client = _make_client()
    try:
        async for event in bus.subscribe(TRANSACTIONS_ENRICHED):
            if event is SHUTDOWN:
                bus.done(TRANSACTIONS_ENRICHED)
                break
            try:
                resp = await client.post("/score", json=_to_request(event))
                resp.raise_for_status()
                body = resp.json()
                decision = body["decision"]
                prob = body["fraud_probability"]
                enriched = dict(event)
                enriched["score_result"] = body
                await bus.publish(_DECISION_TOPIC[decision], enriched)
                # Every transaction gets an audit record (HOLDs get a richer
                # one after the agent stage; this is the baseline trail).
                await bus.publish(AUDIT_LOG, {
                    "transaction_id": event["transaction_id"],
                    "stage": "scored",
                    "decision": decision,
                    "fraud_probability": prob,
                    "threshold_used": body["threshold_used"],
                    "reasons": body["reasons"],
                    "fca_explanation": body["fca_explanation"],
                    "model_version": body["model_version"],
                    "latency_ms": body["latency_ms"],
                    # Engineered feature vector retained so the monitoring
                    # layer (C3) can measure feature drift on live traffic.
                    "features": {k: v for k, v
                                 in event.get("_features", {}).items()
                                 if k != "card_uid"},
                })
                decisions[decision] += 1
                scored += 1
                if metrics is not None:
                    metrics["scored"] = scored
                    metrics["decisions"][decision] += 1
                    metrics["latency_sum_ms"] += float(body["latency_ms"])
                logger.debug("[%s] %s -> %s (%.3f)", time.strftime("%H:%M:%S"),
                             event["transaction_id"], decision, prob)
            except Exception:
                logger.exception("scoring failed for %s",
                                 event.get("transaction_id"))
            finally:
                bus.done(TRANSACTIONS_ENRICHED)
    finally:
        await client.aclose()
    return {"scored": scored, "decisions": dict(decisions)}
