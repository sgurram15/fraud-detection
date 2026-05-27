"""C2.7 — Pipeline orchestrator.

Runs the full local streaming pipeline as concurrent asyncio tasks:

    producer -> enricher -> scorer -> (approved | review | flagged)
                                   \\-> audit_log <- agent (flagged)
    audit_writer <- audit_log

and prints a live dashboard every 2 seconds. On completion it drains every
stage in pipeline order (so no transaction is lost), prints a final summary,
and writes docs/pipeline_run_report.json.

Run:  python src/streaming/run_pipeline.py            (defaults: TXN_LIMIT, 10 TPS)
      TXN_LIMIT=500 python src/streaming/run_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.streaming import (
    audit_writer,
    feature_enricher,
    fraud_agent,
    fraud_scorer,
    transaction_producer,
)
from src.streaming.stream_bus import (
    AUDIT_LOG,
    TRANSACTIONS_APPROVED,
    TRANSACTIONS_ENRICHED,
    TRANSACTIONS_FLAGGED,
    TRANSACTIONS_INBOUND,
    TRANSACTIONS_SCORED,
    create_bus,
)

logger = logging.getLogger("pipeline")

_REPORT_PATH = _ROOT / "docs" / "pipeline_run_report.json"

# Illustrative cost assumptions (see docs/model_card.md — must be recalibrated
# with the client's real fraud-loss data before production).
_AVG_FRAUD_LOSS_GBP = 125.0   # value of a prevented fraud (FN cost)
_FALSE_POS_COST_GBP = 25.0    # cost of a blocked legitimate txn (FP cost)


def _new_metrics() -> dict:
    return {
        "scored": 0,
        "decisions": Counter({"APPROVE": 0, "REVIEW": 0, "HOLD": 0}),
        "latency_sum_ms": 0.0,
        "agent": Counter({"HOLD_AND_STEP_UP": 0, "BLOCK": 0, "MONITOR": 0}),
    }


def _render_dashboard(metrics: dict, elapsed: float) -> str:
    scored = metrics["scored"]
    d = metrics["decisions"]
    approve, review, hold = d["APPROVE"], d["REVIEW"], d["HOLD"]
    tps = scored / elapsed if elapsed > 0 else 0.0
    avg_lat = metrics["latency_sum_ms"] / scored if scored else 0.0
    pct = lambda n: (100.0 * n / scored) if scored else 0.0
    ag = metrics["agent"]
    saving = hold * _AVG_FRAUD_LOSS_GBP
    fp_cost = review * _FALSE_POS_COST_GBP
    w = 52

    def line(text: str) -> str:
        return "║  " + text.ljust(w - 4) + "║"

    sep = "╠" + "═" * (w - 2) + "╣"
    return "\n".join([
        "╔" + "═" * (w - 2) + "╗",
        "║" + "FRAUD DETECTION PIPELINE — LIVE".center(w - 2) + "║",
        sep,
        line(f"Transactions processed:  {scored:,}"),
        line(f"Throughput:              {tps:.1f} TPS"),
        line(f"Avg end-to-end latency:  {avg_lat:.0f}ms"),
        sep,
        line(f"APPROVE:  {approve:>5,}  ({pct(approve):.1f}%)"),
        line(f"REVIEW:   {review:>5,}  ({pct(review):.1f}%)"),
        line(f"HOLD:     {hold:>5,}  ({pct(hold):.1f}%)"),
        sep,
        line("Agent decisions (HOLD transactions):"),
        line(f"  HOLD_AND_STEP_UP:  {ag['HOLD_AND_STEP_UP']}"),
        line(f"  BLOCK:             {ag['BLOCK']}"),
        line(f"  MONITOR:           {ag['MONITOR']}"),
        sep,
        line(f"Est. daily fraud saving:   £{saving:,.0f}"),
        line(f"Est. daily false pos cost: £{fp_cost:,.0f}"),
        "╚" + "═" * (w - 2) + "╝",
    ])


async def _dashboard_loop(metrics: dict, bus: StreamBus, t0: float,
                          stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        print("\n" + _render_dashboard(metrics, time.perf_counter() - t0),
              flush=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_pipeline(limit: int | None = None, tps: int | None = None,
                       dashboard: bool = True,
                       dashboard_interval: float = 2.0) -> dict:
    bus = create_bus()  # local asyncio bus, or MSK when KAFKA_BOOTSTRAP_SERVERS set
    metrics = _new_metrics()
    limit = limit if limit is not None else transaction_producer.TXN_LIMIT
    tps = tps if tps is not None else transaction_producer.TRANSACTIONS_PER_SECOND
    t0 = time.perf_counter()

    # Long-running consumers (break on their SHUTDOWN sentinel).
    enricher_task = asyncio.create_task(feature_enricher.run(bus))
    scorer_task = asyncio.create_task(fraud_scorer.run(bus, metrics))
    agent_task = asyncio.create_task(fraud_agent.run(bus, metrics))
    audit_task = asyncio.create_task(audit_writer.run(bus))

    stop = asyncio.Event()
    dash_task = (asyncio.create_task(
        _dashboard_loop(metrics, bus, t0, stop, dashboard_interval))
        if dashboard else None)

    # Producer runs to completion, then we drain each stage in order.
    producer_stats = await transaction_producer.run(bus, limit=limit, tps=tps)

    await bus.join(TRANSACTIONS_INBOUND)      # all enriched
    await bus.send_shutdown(TRANSACTIONS_INBOUND)
    enricher_stats = await enricher_task

    await bus.join(TRANSACTIONS_ENRICHED)     # all scored + routed
    await bus.send_shutdown(TRANSACTIONS_ENRICHED)
    scorer_stats = await scorer_task

    await bus.join(TRANSACTIONS_FLAGGED)      # all agent-reasoned
    await bus.send_shutdown(TRANSACTIONS_FLAGGED)
    agent_stats = await agent_task

    await bus.join(AUDIT_LOG)                 # all audit records written
    await bus.send_shutdown(AUDIT_LOG)
    audit_summary = await audit_task

    stop.set()
    if dash_task:
        await dash_task

    elapsed = time.perf_counter() - t0
    scored = metrics["scored"]
    bus_stats = bus.get_stats()
    report = {
        "limit": limit,
        "target_tps": tps,
        "elapsed_seconds": round(elapsed, 2),
        "producer": producer_stats,
        "enricher": enricher_stats,
        "scorer": scorer_stats,
        "agent": agent_stats,
        "audit_summary": audit_summary,
        "decisions": dict(metrics["decisions"]),
        "agent_decisions": dict(metrics["agent"]),
        "avg_latency_ms": round(metrics["latency_sum_ms"] / scored, 3)
        if scored else 0.0,
        "terminal_counts": {
            "approved": bus_stats["topics"][TRANSACTIONS_APPROVED]["published"],
            "review": bus_stats["topics"][TRANSACTIONS_SCORED]["published"],
            "flagged": bus_stats["topics"][TRANSACTIONS_FLAGGED]["published"],
        },
        "bus_stats": bus_stats,
    }

    if dashboard:
        print("\n" + _render_dashboard(metrics, elapsed))
        print("\nFinal report:")
        print(json.dumps({k: report[k] for k in (
            "elapsed_seconds", "decisions", "agent_decisions",
            "avg_latency_ms", "audit_summary")}, indent=2))

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(report, indent=2, default=str),
                            encoding="utf-8")
    if dashboard:
        print(f"\nSaved {_REPORT_PATH}")
    return report


if __name__ == "__main__":
    # The dashboard uses box-drawing glyphs; force UTF-8 so a Windows console
    # (cp1252) or a redirected pipe doesn't raise UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    # Quieten the very chatty per-transaction loggers for a clean dashboard.
    for noisy in ("src.features.feature_store", "fraud_api", "httpx",
                  "enricher", "scorer", "agent", "audit", "producer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run_pipeline())
