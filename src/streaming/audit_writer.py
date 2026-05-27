"""C2.6 — Audit log writer.

Subscribes to ``audit_log`` and writes an immutable-style audit trail. A
transaction may produce two records (a ``scored`` record always, plus an
``agent`` record when it was HELD); both are merged by stage into a single
per-transaction JSON file so the file is the complete decision evidence.

  Local:        data/audit/{YYYY-MM-DD}/{transaction_id}.json
  USE_S3=true:  s3://$S3_BUCKET/audit/{YYYY-MM-DD}/{transaction_id}.json

A running ``audit_summary.json`` counts totals by decision type, by agent
confidence, and the fraud rate.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import S3_BUCKET, USE_S3
from src.streaming.stream_bus import AUDIT_LOG, SHUTDOWN, StreamBus

logger = logging.getLogger("audit")

_AUDIT_DIR = _ROOT / "data" / "audit"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AuditWriter:
    def __init__(self) -> None:
        self._decisions: Counter = Counter()
        self._confidence: Counter = Counter()
        self._fraud = 0
        self._total = 0
        self._seen: set[str] = set()

    def _day_dir(self) -> Path:
        d = _AUDIT_DIR / _today()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, record: dict) -> None:
        txn_id = record["transaction_id"]
        stage = record.get("stage", "scored")

        if USE_S3:
            # S3 audit sink is a documented follow-up (uses the same key
            # layout); local FS is the PoC sink.
            logger.warning("USE_S3=true: S3 audit sink not wired; writing "
                           "locally as a fallback for %s", txn_id)

        path = self._day_dir() / f"{txn_id}.json"
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.setdefault("transaction_id", txn_id)
        existing[stage] = record
        path.write_text(json.dumps(existing, indent=2, default=str),
                        encoding="utf-8")

        # Summary: count each transaction once (on its scored record); add
        # agent confidence when the agent stage arrives.
        if stage == "scored":
            if txn_id not in self._seen:
                self._seen.add(txn_id)
                self._total += 1
                self._decisions[record.get("decision", "UNKNOWN")] += 1
                thr = record.get("threshold_used")
                prob = record.get("fraud_probability")
                if thr is not None and prob is not None and prob >= thr:
                    self._fraud += 1
        elif stage == "agent":
            conf = record.get("agent_verdict", {}).get("confidence")
            if conf:
                self._confidence[conf] += 1
        logger.debug("Audit written %s (%s)", txn_id, stage)

    def summary(self) -> dict:
        return {
            "total_decisions": self._total,
            "by_decision_type": dict(self._decisions),
            "by_confidence": dict(self._confidence),
            "fraud_rate": round(self._fraud / self._total, 4)
            if self._total else 0.0,
        }

    def flush_summary(self) -> Path:
        path = self._day_dir() / "audit_summary.json"
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path


async def run(bus: StreamBus) -> dict:
    """Consume audit records, persist them, return the final summary."""
    writer = AuditWriter()
    async for record in bus.subscribe(AUDIT_LOG):
        if record is SHUTDOWN:
            bus.done(AUDIT_LOG)
            break
        try:
            writer.write(record)
        finally:
            bus.done(AUDIT_LOG)
    writer.flush_summary()
    return writer.summary()
