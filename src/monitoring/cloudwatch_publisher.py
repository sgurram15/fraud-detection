"""C7.1 — CloudWatch custom-metrics publisher.

Publishes the pipeline's operational metrics to the CloudWatch namespace
``FraudDetection/Pipeline``. Intended to be called every ~60s while the
pipeline runs (see :func:`publish_periodic`). When AWS credentials / boto3 are
unavailable it logs the metrics to stdout instead, so it is safe to wire into
the local pipeline.

Metrics (name -> unit):
  TransactionsProcessed   Count
  FraudRate               Percent
  FalsePositiveRate       Percent
  EndToEndLatencyMs       Milliseconds
  AgentDecisions_Hold     Count
  AgentDecisions_Block    Count
  DailyFraudSaving_GBP    None
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("cloudwatch")

NAMESPACE = "FraudDetection/Pipeline"
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

# metric name -> CloudWatch unit
_UNITS = {
    "TransactionsProcessed": "Count",
    "FraudRate": "Percent",
    "FalsePositiveRate": "Percent",
    "EndToEndLatencyMs": "Milliseconds",
    "AgentDecisions_Hold": "Count",
    "AgentDecisions_Block": "Count",
    "DailyFraudSaving_GBP": "None",
}


def build_metric_data(metrics: dict) -> list[dict]:
    """Map a flat {metric_name: value} dict to CloudWatch MetricData. Only
    known metric names are emitted; unknowns are ignored."""
    out: list[dict] = []
    for name, unit in _UNITS.items():
        if name in metrics and metrics[name] is not None:
            out.append({
                "MetricName": name,
                "Value": float(metrics[name]),
                "Unit": unit,
            })
    return out


def metrics_from_report(report: dict) -> dict:
    """Derive the publishable metrics from a run_pipeline report dict."""
    decisions = report.get("decisions", {})
    agent = report.get("agent_decisions", {})
    processed = sum(decisions.values())
    review = decisions.get("REVIEW", 0)
    audit = report.get("audit_summary", {})
    return {
        "TransactionsProcessed": processed,
        "FraudRate": (audit.get("fraud_rate", 0.0) * 100.0),
        # Without confirmed labels, REVIEW share is a proxy for FP pressure.
        "FalsePositiveRate": (review / processed * 100.0) if processed else 0.0,
        "EndToEndLatencyMs": report.get("avg_latency_ms", 0.0),
        "AgentDecisions_Hold": agent.get("HOLD_AND_STEP_UP", 0),
        "AgentDecisions_Block": agent.get("BLOCK", 0),
        "DailyFraudSaving_GBP": report.get("est_daily_saving_gbp", 0.0),
    }


class CloudWatchPublisher:
    def __init__(self, region: str = REGION) -> None:
        self.region = region
        self._client = None
        self._available: bool | None = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("cloudwatch", region_name=self.region)
        return self._client

    def available(self) -> bool:
        """True if boto3 + credentials are usable; cached after first check."""
        if self._available is None:
            try:
                import boto3
                self._available = (
                    boto3.Session().get_credentials() is not None)
            except ImportError:
                self._available = False
        return self._available

    def publish(self, metrics: dict) -> bool:
        """Publish metrics. Returns True if sent to CloudWatch, False if it
        fell back to stdout logging."""
        data = build_metric_data(metrics)
        if not self.available():
            logger.info("[CloudWatch stdout-only] %s: %s", NAMESPACE,
                        {d["MetricName"]: d["Value"] for d in data})
            return False
        self._get_client().put_metric_data(Namespace=NAMESPACE,
                                           MetricData=data)
        logger.info("Published %d metrics to %s", len(data), NAMESPACE)
        return True


def publish_periodic(get_metrics, interval: int = 60, stop=None) -> None:
    """Publish every `interval` seconds until `stop` (a threading.Event or any
    object with is_set()) is set. `get_metrics` is a no-arg callable returning
    the current metrics dict."""
    pub = CloudWatchPublisher()
    while not (stop and stop.is_set()):
        try:
            pub.publish(get_metrics())
        except Exception:  # noqa: BLE001 — never let telemetry kill the run
            logger.exception("metric publish failed")
        if stop is not None:
            stop.wait(interval)  # type: ignore[attr-defined]
        else:
            time.sleep(interval)


def main() -> int:
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    report_path = _ROOT / "docs" / "pipeline_run_report.json"
    if not report_path.exists():
        print("No pipeline_run_report.json — run the pipeline first.")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pub = CloudWatchPublisher()
    sent = pub.publish(metrics_from_report(report))
    print("Published to CloudWatch." if sent
          else "Logged to stdout (no AWS credentials).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
