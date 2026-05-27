"""C7.3 — CloudWatch publisher test.

ALWAYS (no AWS): the metric-data builder, the report->metrics mapper, and the
stdout fallback when credentials are absent.

LIVE (opt-in: CLOUDWATCH_LIVE_TEST=true + credentials): publish 10 test metric
points, read them back, and verify the FraudDetectionPOC dashboard exists. This
is gated behind an explicit flag so a normal local run never calls AWS.

Run: python tests/test_cloudwatch.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.monitoring import cloudwatch_publisher as cw


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_build_metric_data() -> bool:
    metrics = {"TransactionsProcessed": 500, "FraudRate": 1.2,
               "EndToEndLatencyMs": 27.0, "DailyFraudSaving_GBP": 296445,
               "AgentDecisions_Hold": 6, "AgentDecisions_Block": 0,
               "FalsePositiveRate": 75.2, "UnknownMetric": 1, "Missing": None}
    data = cw.build_metric_data(metrics)
    names = {d["MetricName"] for d in data}
    units_ok = all(d["Unit"] == cw._UNITS[d["MetricName"]] for d in data)
    ok = (len(data) == 7 and "UnknownMetric" not in names
          and "Missing" not in names and units_ok)
    return _result("build_metric_data: 7 known metrics, correct units", ok,
                   f"emitted {sorted(names)}")


def test_metrics_from_report() -> bool:
    report = {
        "decisions": {"APPROVE": 100, "REVIEW": 380, "HOLD": 20},
        "agent_decisions": {"HOLD_AND_STEP_UP": 18, "BLOCK": 2, "MONITOR": 0},
        "avg_latency_ms": 27.0,
        "audit_summary": {"fraud_rate": 0.04},
        "est_daily_saving_gbp": 296445,
    }
    m = cw.metrics_from_report(report)
    ok = (m["TransactionsProcessed"] == 500
          and abs(m["FraudRate"] - 4.0) < 1e-6
          and abs(m["FalsePositiveRate"] - 76.0) < 1e-6  # 380/500*100
          and m["AgentDecisions_Block"] == 2
          and m["DailyFraudSaving_GBP"] == 296445)
    return _result("metrics_from_report maps a pipeline report", ok, str(m))


def test_stdout_fallback() -> bool:
    pub = cw.CloudWatchPublisher()
    pub._available = False  # force the no-credentials path
    sent = pub.publish({"TransactionsProcessed": 10})
    return _result("publish falls back to stdout without credentials",
                   sent is False)


def test_live() -> bool | None:
    if os.getenv("CLOUDWATCH_LIVE_TEST", "").lower() != "true":
        print("[SKIP] CLOUDWATCH_LIVE_TEST!=true — live CloudWatch test "
              "skipped (avoids unsolicited AWS calls).")
        return None
    try:
        import boto3
    except ImportError:
        print("[SKIP] boto3 not installed.")
        return None

    pub = cw.CloudWatchPublisher()
    if not pub.available():
        print("[SKIP] no AWS credentials available.")
        return None

    client = boto3.client("cloudwatch", region_name=cw.REGION)
    for i in range(10):
        pub.publish({"TransactionsProcessed": 100 + i})
        time.sleep(0.2)
    # Read back (metrics take a moment to be queryable).
    time.sleep(5)
    from datetime import datetime, timedelta, timezone
    stats = client.get_metric_statistics(
        Namespace=cw.NAMESPACE, MetricName="TransactionsProcessed",
        StartTime=datetime.now(timezone.utc) - timedelta(minutes=5),
        EndTime=datetime.now(timezone.utc), Period=60, Statistics=["Sum"])
    has_data = len(stats.get("Datapoints", [])) > 0
    try:
        client.get_dashboard(DashboardName="FraudDetectionPOC")
        dash_ok = True
    except Exception:  # noqa: BLE001
        dash_ok = False
    return _result("live: metrics published+readable and dashboard exists",
                   has_data and dash_ok,
                   f"datapoints={len(stats.get('Datapoints', []))} "
                   f"dashboard={dash_ok}")


def main() -> int:
    print("=" * 64)
    print("CloudWatch publisher test suite")
    print("=" * 64)
    results = [test_build_metric_data(), test_metrics_from_report(),
               test_stdout_fallback()]
    live = test_live()
    if live is not None:
        results.append(live)
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
