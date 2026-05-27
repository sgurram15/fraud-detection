"""C2.1 — Local in-process stream bus (Kafka stand-in).

An ``asyncio.Queue`` per topic that mimics the Kafka topic semantics the
production pipeline uses, so every other streaming component
(``publish``/``subscribe``) is written once and runs unchanged locally or
against Amazon MSK. The MSK swap is C5.3: when ``KAFKA_BOOTSTRAP_SERVERS`` is
set the bus would back each topic with a confluent-kafka producer/consumer
instead of a queue — that path is intentionally stubbed here (the client is not
installed and not yet tested) and raises a clear error rather than pretending
to work.

Shutdown model: producers call :meth:`publish`; consumers call
:meth:`subscribe` and MUST call :meth:`done` after handling each message. The
orchestrator drains the pipeline by awaiting :meth:`join` on each topic in
order, which is what guarantees "no transactions lost" before tasks are
cancelled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator

logger = logging.getLogger("stream_bus")

# Topic names (mirrored, with hyphens, as MSK topics in C5.2).
TRANSACTIONS_INBOUND = "transactions_inbound"
TRANSACTIONS_ENRICHED = "transactions_enriched"
TRANSACTIONS_SCORED = "transactions_scored"
TRANSACTIONS_FLAGGED = "transactions_flagged"
TRANSACTIONS_APPROVED = "transactions_approved"
AUDIT_LOG = "audit_log"

TOPICS = (
    TRANSACTIONS_INBOUND,
    TRANSACTIONS_ENRICHED,
    TRANSACTIONS_SCORED,
    TRANSACTIONS_FLAGGED,
    TRANSACTIONS_APPROVED,
    AUDIT_LOG,
)

# Sentinel a consumer breaks its subscribe loop on. The orchestrator sends one
# per consumer after the upstream topic has fully drained (see run_pipeline).
SHUTDOWN = object()


class StreamBus:
    """In-process pub/sub over one ``asyncio.Queue`` per topic."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {
            t: asyncio.Queue() for t in TOPICS
        }
        self._published: dict[str, int] = {t: 0 for t in TOPICS}
        self._consumed: dict[str, int] = {t: 0 for t in TOPICS}
        self._t0 = time.perf_counter()

    def _topic(self, topic: str) -> asyncio.Queue:
        try:
            return self._queues[topic]
        except KeyError:
            raise KeyError(
                f"unknown topic {topic!r}; known topics: {', '.join(TOPICS)}"
            ) from None

    async def publish(self, topic: str, message: dict) -> None:
        """Publish one message to a topic. Partitioning is by ``card_id`` in
        production (Kafka); locally a single queue preserves per-topic order."""
        await self._topic(topic).put(message)
        self._published[topic] += 1

    async def subscribe(self, topic: str) -> AsyncGenerator[dict, None]:
        """Yield messages from a topic forever. The caller MUST call
        :meth:`done` after handling each yielded message so :meth:`join` can
        tell when the topic is fully drained."""
        q = self._topic(topic)
        while True:
            msg = await q.get()
            if msg is not SHUTDOWN:  # sentinel is control, not a real message
                self._consumed[topic] += 1
            yield msg

    def done(self, topic: str) -> None:
        """Mark the most recently consumed message on ``topic`` as handled."""
        self._topic(topic).task_done()

    async def send_shutdown(self, topic: str) -> None:
        """Enqueue the SHUTDOWN sentinel so the topic's consumer can break its
        loop cleanly. Not counted in published stats."""
        await self._topic(topic).put(SHUTDOWN)

    async def join(self, topic: str) -> None:
        """Block until every message published to ``topic`` has been handled."""
        await self._topic(topic).join()

    def pending(self, topic: str) -> int:
        return self._topic(topic).qsize()

    def get_stats(self) -> dict:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        total_pub = sum(self._published.values())
        return {
            "elapsed_seconds": round(elapsed, 2),
            "throughput_msg_per_s": round(total_pub / elapsed, 2),
            "topics": {
                t: {
                    "published": self._published[t],
                    "consumed": self._consumed[t],
                    "pending": self.pending(t),
                }
                for t in TOPICS
            },
        }


# --------------------------------------------------------------------------- #
# C5.3 — Amazon MSK backend (kafka-python)
# --------------------------------------------------------------------------- #
# Internal underscore topic names -> hyphenated MSK topics created by
# setup_msk_topics.py. Note transactions_scored (the REVIEW sink) maps to
# transactions-review.
_MSK_TOPIC = {
    TRANSACTIONS_INBOUND: "transactions-inbound",
    TRANSACTIONS_ENRICHED: "transactions-enriched",
    TRANSACTIONS_SCORED: "transactions-review",
    TRANSACTIONS_FLAGGED: "transactions-flagged",
    TRANSACTIONS_APPROVED: "transactions-approved",
    AUDIT_LOG: "audit-log",
}
_SHUTDOWN_MARKER = {"__stream_bus_shutdown__": True}


class KafkaStreamBus:
    """MSK-backed bus exposing the SAME interface as :class:`StreamBus`, so no
    other streaming file changes (C5.3's "transparent swap").

    Backed by kafka-python with SASL_SSL + OAUTHBEARER IAM auth. Blocking
    producer/consumer calls are run in a thread so they don't stall the event
    loop. ``subscribe`` re-materialises the :data:`SHUTDOWN` sentinel from a
    serialised marker message, so consumers' ``is SHUTDOWN`` checks work
    unchanged.

    UNVERIFIED: there is no cluster to test against. The data plane
    (publish/subscribe/get_stats) is faithful; the drain coordination
    (join/done/shutdown) is best-effort and assumes Kafka's per-partition
    ordering. In production the pipeline components run as long-lived consumers
    rather than self-draining via join() — see run_pipeline_on_ec2.py (C5.4).
    """

    def __init__(self, bootstrap: str | None = None,
                 group_id: str = "fraud-detection-pipeline") -> None:
        self._bootstrap = (bootstrap
                           or os.environ["KAFKA_BOOTSTRAP_SERVERS"]).split(",")
        self._group_id = group_id
        self._producer = None
        self._consumers: dict[str, object] = {}
        self._published: dict[str, int] = {t: 0 for t in TOPICS}
        self._consumed: dict[str, int] = {t: 0 for t in TOPICS}
        self._t0 = time.perf_counter()

    # --- client construction (lazy; needs kafka-python + IAM signer) ---
    def _token(self) -> str:
        from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
        import os as _os
        tok, _ = MSKAuthTokenProvider.generate_auth_token(
            _os.getenv("AWS_DEFAULT_REGION", "eu-west-2"))
        return tok

    def _auth(self) -> dict:
        provider = type("P", (), {"token": lambda _self: self._token()})()
        return {"security_protocol": "SASL_SSL",
                "sasl_mechanism": "OAUTHBEARER",
                "sasl_oauth_token_provider": provider}

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
                **self._auth())
        return self._producer

    def _get_consumer(self, topic: str):
        if topic not in self._consumers:
            from kafka import KafkaConsumer
            self._consumers[topic] = KafkaConsumer(
                _MSK_TOPIC[topic],
                bootstrap_servers=self._bootstrap,
                group_id=self._group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda b: json.loads(b.decode()),
                **self._auth())
        return self._consumers[topic]

    async def publish(self, topic: str, message: dict) -> None:
        producer = self._get_producer()
        await asyncio.to_thread(producer.send, _MSK_TOPIC[topic], message)
        self._published[topic] += 1

    async def subscribe(self, topic: str) -> AsyncGenerator[dict, None]:
        consumer = self._get_consumer(topic)
        while True:
            batch = await asyncio.to_thread(consumer.poll, 1000)  # ms
            for _tp, records in batch.items():
                for rec in records:
                    val = rec.value
                    if isinstance(val, dict) and val.get(
                            "__stream_bus_shutdown__"):
                        yield SHUTDOWN  # re-materialise the sentinel object
                        continue
                    self._consumed[topic] += 1
                    yield val

    def done(self, topic: str) -> None:
        # Best-effort offset commit (Kafka analogue of task_done).
        consumer = self._consumers.get(topic)
        if consumer is not None:
            try:
                consumer.commit()
            except Exception:  # noqa: BLE001 — commit is best-effort here
                pass

    async def send_shutdown(self, topic: str) -> None:
        # Publish the marker on enough partitions that every consumer sees one.
        producer = self._get_producer()
        for _ in range(4):  # matches the 4-partition topics in C5.2
            await asyncio.to_thread(producer.send, _MSK_TOPIC[topic],
                                    _SHUTDOWN_MARKER)
        await asyncio.to_thread(producer.flush)

    async def join(self, topic: str) -> None:
        # Kafka has no queue.join(); flush the producer so published messages
        # are delivered. True consumer-side drain is not observable here.
        if self._producer is not None:
            await asyncio.to_thread(self._producer.flush)

    def pending(self, topic: str) -> int:
        return 0  # not observable without an admin lag query

    def get_stats(self) -> dict:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        total_pub = sum(self._published.values())
        return {
            "backend": "msk",
            "elapsed_seconds": round(elapsed, 2),
            "throughput_msg_per_s": round(total_pub / elapsed, 2),
            "topics": {t: {"published": self._published[t],
                           "consumed": self._consumed[t], "pending": 0}
                       for t in TOPICS},
        }


def create_bus():
    """Return the MSK-backed bus when KAFKA_BOOTSTRAP_SERVERS is set, else the
    local asyncio bus. This is the only line that changes between local and
    production — every component talks to the same publish/subscribe API."""
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        logger.info("Using MSK (Kafka) stream bus")
        return KafkaStreamBus()
    return StreamBus()
