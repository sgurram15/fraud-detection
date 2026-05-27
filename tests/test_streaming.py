"""C2.8 — Streaming pipeline acceptance tests.

Runs the full local pipeline on 100 transactions (no inter-message delay) and
verifies the end-to-end invariants:

  * every transaction reaches a terminal topic (approved / review / flagged)
  * no transactions are lost between topics (published == consumed per stage)
  * an audit record is written for every transaction
  * an agent decision is present for every HOLD transaction
  * average end-to-end latency is under 500ms
  * the run completes with no unhandled exceptions

Run: python tests/test_streaming.py   (prints PASS/FAIL per test, non-zero exit
on any failure).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.streaming import run_pipeline
from src.streaming.stream_bus import (
    AUDIT_LOG,
    TRANSACTIONS_ENRICHED,
    TRANSACTIONS_FLAGGED,
    TRANSACTIONS_INBOUND,
)

N = 100
_AUDIT_TODAY = (_ROOT / "data" / "audit"
                / datetime.now(timezone.utc).strftime("%Y-%m-%d"))


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def main() -> int:
    print("=" * 64)
    print(f"Streaming pipeline test suite ({N} transactions)")
    print("=" * 64)

    # Clean today's audit dir so file/summary counts reflect THIS run only.
    if _AUDIT_TODAY.exists():
        shutil.rmtree(_AUDIT_TODAY)

    # Keep the run quiet; we assert on the returned report.
    for noisy in ("src.features.feature_store", "fraud_api", "httpx",
                  "enricher", "scorer", "agent", "audit", "producer",
                  "pipeline"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    results: list[bool] = []
    no_exception = True
    report: dict = {}
    try:
        report = asyncio.run(run_pipeline.run_pipeline(
            limit=N, tps=0, dashboard=False))
    except Exception as exc:  # noqa: BLE001
        no_exception = False
        results.append(_result("no unhandled exceptions during run", False,
                                repr(exc)))

    if report:
        produced = report["producer"]["published"]
        scored = report["scorer"]["scored"]
        decisions = report["decisions"]
        terminal = report["terminal_counts"]
        terminal_total = sum(terminal.values())
        bus_topics = report["bus_stats"]["topics"]

        # 1. Every transaction reaches a terminal topic.
        results.append(_result(
            "all transactions reach a terminal topic",
            terminal_total == scored == produced,
            f"produced={produced} scored={scored} terminal={terminal_total}"))

        # 2. No transactions lost between topics (published == consumed).
        lost = {
            t: (bus_topics[t]["published"], bus_topics[t]["consumed"])
            for t in (TRANSACTIONS_INBOUND, TRANSACTIONS_ENRICHED,
                      TRANSACTIONS_FLAGGED, AUDIT_LOG)
            if bus_topics[t]["published"] != bus_topics[t]["consumed"]
        }
        results.append(_result("no transactions lost between topics",
                               not lost, f"unbalanced topics: {lost}"))

        # 3. Audit record written for every transaction.
        audit_total = report["audit_summary"]["total_decisions"]
        json_files = ([p for p in _AUDIT_TODAY.glob("*.json")
                       if p.name != "audit_summary.json"]
                      if _AUDIT_TODAY.exists() else [])
        results.append(_result(
            "audit record written for every transaction",
            audit_total == produced and len(json_files) == produced,
            f"audit_total={audit_total} files={len(json_files)} "
            f"produced={produced}"))

        # 4. Agent decision present for every HOLD transaction.
        agent_decisions = report["agent"]["agent_decisions"]
        results.append(_result(
            "agent decision present for every HOLD transaction",
            agent_decisions == decisions["HOLD"],
            f"agent_decisions={agent_decisions} HOLD={decisions['HOLD']}"))

        # 5. Average end-to-end latency under 500ms.
        avg_lat = report["avg_latency_ms"]
        print(f"  avg end-to-end latency: {avg_lat:.1f} ms")
        results.append(_result("avg latency under 500ms", avg_lat < 500.0,
                               f"avg {avg_lat:.1f} ms"))

        # 6. No unhandled exceptions (run returned a report).
        results.append(_result("no unhandled exceptions during run",
                               no_exception))

    print("-" * 64)
    passed = sum(results)
    print(f"{passed}/{len(results)} tests passed")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
