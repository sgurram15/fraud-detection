"""C2.3 — Feature enrichment consumer.

Subscribes to ``transactions_inbound``, runs the feature store in serving mode
on each event's raw fields, folds the resulting history-dependent features
(velocity, deviation, card age) back onto the event, and republishes to
``transactions_enriched``.

This enricher owns its OWN FeatureStore instance — separate from the one inside
the scoring API — so replaying state here does not double-count velocity in the
API's own serving pass.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.features.feature_store import FeatureStore
from src.streaming.stream_bus import (
    SHUTDOWN,
    StreamBus,
    TRANSACTIONS_ENRICHED,
    TRANSACTIONS_INBOUND,
)

logger = logging.getLogger("enricher")


def enrich_event(event: dict, store: FeatureStore) -> dict:
    """Compute serving features and overlay the request-facing ones."""
    feats = store.get_features(event.get("_raw", {}))
    event = dict(event)
    event["tx_velocity_1h"] = int(feats["txn_count_1h"])
    event["tx_velocity_24h"] = int(feats["txn_count_24h"])
    event["amt_deviation"] = float(feats["amt_dev_ratio_card_mean"])
    event["card_age_days"] = int(feats["card_age_days"])
    # No true shipping/destination account age in IEEE-CIS; proxy with card age
    # so the field is populated and internally consistent (documented limit).
    event["destination_account_age_days"] = int(feats["card_age_days"])
    event["_features"] = feats  # carried for observability; not sent to API
    return event


async def run(bus: StreamBus) -> dict:
    """Consume inbound, enrich, publish enriched. Returns stats on shutdown."""
    store = FeatureStore().load()
    enriched = 0
    latency_sum = 0.0
    async for event in bus.subscribe(TRANSACTIONS_INBOUND):
        if event is SHUTDOWN:
            bus.done(TRANSACTIONS_INBOUND)
            break
        try:
            t0 = time.perf_counter()
            out = enrich_event(event, store)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latency_sum += elapsed_ms
            await bus.publish(TRANSACTIONS_ENRICHED, out)
            enriched += 1
            logger.debug("[%s] Enriched %s (%.1fms)", time.strftime("%H:%M:%S"),
                         event.get("transaction_id"), elapsed_ms)
        finally:
            bus.done(TRANSACTIONS_INBOUND)
    return {"enriched": enriched,
            "avg_enrich_ms": round(latency_sum / enriched, 3) if enriched
            else 0.0}
