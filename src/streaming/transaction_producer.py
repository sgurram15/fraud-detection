"""C2.2 — Transaction producer.

Reads raw IEEE-CIS transactions and publishes each as a JSON-serialisable dict
to ``transactions_inbound`` at a fixed rate (default 10 TPS), partitioned by
card_id. Each emitted event carries the raw fields the feature store needs
(``_raw``) plus the base request fields the scoring API expects; the enricher
(C2.3) fills in the history-dependent fields (velocity, deviation, card age).

Source resolution (first existing wins):
  1. $TXN_SOURCE
  2. data/raw/test_transaction.csv               (mission's expected path)
  3. data/raw/ieee-fraud-detection/test_transaction.csv  (Kaggle layout here)
Only $TXN_LIMIT rows (default 1000) are read — the full file is ~613 MB.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.features.build_features import CARD_ID_COLS, REFERENCE_DATETIME
from src.streaming.stream_bus import StreamBus, TRANSACTIONS_INBOUND

logger = logging.getLogger("producer")

TRANSACTIONS_PER_SECOND = int(os.getenv("TRANSACTIONS_PER_SECOND", "10"))
TXN_LIMIT = int(os.getenv("TXN_LIMIT", "1000"))

# Raw columns we actually read (avoid loading all ~400 IEEE columns).
_USECOLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "P_emaildomain", *CARD_ID_COLS,
]
# DeviceType lives in the identity file; the transaction file has none, so we
# assign one deterministically per card to give the device_type_fraud_rate
# feature a real (and varied) value. Keys match the trained device-rate map.
_DEVICE_TYPES = ("desktop", "mobile")


def _resolve_source() -> Path:
    candidates = [
        os.getenv("TXN_SOURCE"),
        str(_ROOT / "data" / "raw" / "test_transaction.csv"),
        str(_ROOT / "data" / "raw" / "ieee-fraud-detection"
            / "test_transaction.csv"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "No transaction source found. Set $TXN_SOURCE or place "
        "test_transaction.csv under data/raw/. Run data/download_data.py."
    )


def load_transactions(limit: int = TXN_LIMIT) -> list[dict]:
    """Read and map raw rows to inbound transaction events."""
    src = _resolve_source()
    usecols = [c for c in _USECOLS]  # pandas tolerates a subset via callable
    df = pd.read_csv(
        src, nrows=limit,
        usecols=lambda c: c in usecols,  # missing CARD_ID_COLS are skipped
    )
    logger.info("Loaded %d transactions from %s", len(df), src)

    events: list[dict] = []
    for i, row in enumerate(df.itertuples(index=False)):
        d = row._asdict()
        dt = float(d.get("TransactionDT") or 0.0)
        when = REFERENCE_DATETIME + timedelta(seconds=dt)
        amount = float(d.get("TransactionAmt") or 0.0)
        device_type = _DEVICE_TYPES[i % len(_DEVICE_TYPES)]
        # card_id: composite of the available card columns (matches the
        # feature store's card_uid construction).
        card_id = "|".join(
            str(d[c]) if d.get(c) is not None and pd.notna(d.get(c)) else "NA"
            for c in CARD_ID_COLS if c in d
        ) or f"CARD-{i}"
        txn_id = f"TXN-{int(d.get('TransactionID', i))}"

        events.append({
            "transaction_id": txn_id,
            "card_id": card_id,
            "amount": round(amount, 2),
            "device_type": device_type,
            "hour_of_day": int(when.hour),
            "day_of_week": int(when.weekday()),
            "is_late_night": bool(0 <= when.hour <= 4),
            "is_weekend": bool(when.weekday() >= 5),
            # Filled by the enricher from feature-store state:
            "destination_account_age_days": 0,
            "tx_velocity_1h": 0,
            "tx_velocity_24h": 0,
            "amt_deviation": 1.0,
            "card_age_days": 0,
            # Raw fields the feature store consumes directly:
            "_raw": {
                "TransactionAmt": amount,
                "TransactionDT": dt,
                "DeviceType": device_type,
                "P_emaildomain": d.get("P_emaildomain"),
                **{c: d.get(c) for c in CARD_ID_COLS if c in d},
            },
        })
    return events


async def run(bus: StreamBus, limit: int = TXN_LIMIT,
              tps: int = TRANSACTIONS_PER_SECOND) -> dict:
    """Publish events to ``transactions_inbound`` at ``tps``. Returns stats."""
    events = load_transactions(limit)
    interval = 1.0 / tps if tps > 0 else 0.0
    published = errors = 0
    t0 = time.perf_counter()
    for ev in events:
        try:
            await bus.publish(TRANSACTIONS_INBOUND, ev)
            published += 1
            logger.debug("[%s] Published %s £%.2f",
                         time.strftime("%H:%M:%S"),
                         ev["transaction_id"], ev["amount"])
        except Exception as exc:  # never let one bad row kill the producer
            errors += 1
            logger.warning("publish failed for %s: %s",
                           ev.get("transaction_id"), exc)
        if interval:
            await asyncio.sleep(interval)
    elapsed = time.perf_counter() - t0
    stats = {"published": published, "errors": errors,
             "elapsed_seconds": round(elapsed, 2),
             "tps": round(published / elapsed, 2) if elapsed else 0.0}
    logger.info("Producer done: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    async def _main() -> None:
        bus = StreamBus()
        await run(bus, limit=min(TXN_LIMIT, 50))
        print(bus.get_stats())

    asyncio.run(_main())
