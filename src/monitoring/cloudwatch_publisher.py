"""C7.1 — CloudWatch custom-metrics publisher.

Publishes the pipeline's operational metrics to the CloudWatch namespace
``FraudDetection/Pipeline``. Designed to be called every ~60s while the
pipeline runs. When AWS credentials / boto3 are unavailable it logs the metrics
to stdout instead and returns False — it never raises, so it is safe to wire
into the local pipeline (monitoring must never crash the pipeline).

Public API (C7 spec):
  * ``publish_metrics(metrics: dict) -> bool``
  * ``publish_pipeline_health(status, component, detail='') -> bool``
  * ``_log_locally(metrics: dict)`` — stdout fallback
  * ``metrics_from_report(report: dict) -> dict`` — adapt a run_pipeline report
"""

from __future__ import annotations

import os
from datetime import datetime

NAMESPACE = "FraudDetection/Pipeline"
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")


def publish_metrics(metrics: dict) -> bool:
    """Publishes pipeline metrics to CloudWatch.

    metrics dict keys:
      transactions_processed: int
      fraud_rate: float (0-1)
      false_positive_rate: float (0-1)
      avg_latency_ms: float
      decisions_hold: int
      decisions_block: int
      decisions_approve: int
      decisions_review: int
      daily_fraud_saving_gbp: float
      model_version: str

    Returns True if published, False if CloudWatch unavailable.
    Falls back to stdout logging silently."""

    metric_data = [
        {
            'MetricName': 'TransactionsProcessed',
            'Value': metrics.get('transactions_processed', 0),
            'Unit': 'Count',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'FraudRate',
            'Value': metrics.get('fraud_rate', 0) * 100,
            'Unit': 'Percent',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'FalsePositiveRate',
            'Value': metrics.get('false_positive_rate', 0) * 100,
            'Unit': 'Percent',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'AvgLatencyMs',
            'Value': metrics.get('avg_latency_ms', 0),
            'Unit': 'Milliseconds',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'DecisionsHold',
            'Value': metrics.get('decisions_hold', 0),
            'Unit': 'Count',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'DecisionsBlock',
            'Value': metrics.get('decisions_block', 0),
            'Unit': 'Count',
            'Timestamp': datetime.utcnow(),
        },
        {
            'MetricName': 'DailyFraudSavingGBP',
            'Value': metrics.get('daily_fraud_saving_gbp', 0),
            'Unit': 'None',
            'Timestamp': datetime.utcnow(),
        },
    ]

    try:
        import boto3
        client = boto3.client('cloudwatch', region_name=REGION)
        client.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=metric_data,
        )
        return True
    except Exception as e:  # noqa: BLE001
        # Silent fallback — log to stdout, never crash pipeline
        print(f"[CloudWatch] Unavailable ({e}) — "
              f"metrics logged locally only")
        _log_locally(metrics)
        return False


def _log_locally(metrics: dict):
    """Stdout fallback when CloudWatch unavailable."""
    ts = datetime.utcnow().strftime('%H:%M:%S')
    print(f"[{ts}] METRICS | "
          f"txns={metrics.get('transactions_processed', 0)} | "
          f"fraud_rate={metrics.get('fraud_rate', 0):.1%} | "
          f"fpr={metrics.get('false_positive_rate', 0):.1%} | "
          f"latency={metrics.get('avg_latency_ms', 0):.0f}ms | "
          f"hold={metrics.get('decisions_hold', 0)} | "
          f"saving=£{metrics.get('daily_fraud_saving_gbp', 0):,.0f}")


def publish_pipeline_health(status: str,
                            component: str,
                            detail: str = '') -> bool:
    """Publishes pipeline component health events.
    status: HEALTHY | DEGRADED | DOWN
    component: kafka | features | model | agent | audit"""
    try:
        import boto3
        client = boto3.client('cloudwatch', region_name=REGION)
        client.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{
                'MetricName': f'ComponentHealth_{component}',
                'Value': 1 if status == 'HEALTHY' else 0,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Status', 'Value': status},
                    {'Name': 'Component', 'Value': component},
                ],
                'Timestamp': datetime.utcnow(),
            }],
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[CloudWatch] Health publish failed: {e}")
        return False


def metrics_from_report(report: dict) -> dict:
    """Derive the publish_metrics() dict from a run_pipeline report. Without
    confirmed labels, the REVIEW share is used as a proxy for FP pressure."""
    decisions = report.get("decisions", {})
    agent = report.get("agent_decisions", {})
    processed = sum(decisions.values())
    review = decisions.get("REVIEW", 0)
    audit = report.get("audit_summary", {})
    return {
        "transactions_processed": processed,
        "fraud_rate": audit.get("fraud_rate", 0.0),
        "false_positive_rate": (review / processed) if processed else 0.0,
        "avg_latency_ms": report.get("avg_latency_ms", 0.0),
        "decisions_hold": agent.get("HOLD_AND_STEP_UP", 0),
        "decisions_block": agent.get("BLOCK", 0),
        "decisions_approve": decisions.get("APPROVE", 0),
        "decisions_review": review,
        "daily_fraud_saving_gbp": report.get("est_daily_saving_gbp", 0.0),
        "model_version": report.get("model_version", "xgboost-baseline-v1"),
    }


def main() -> int:
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    report_path = root / "docs" / "pipeline_run_report.json"
    if not report_path.exists():
        print("No pipeline_run_report.json — run the pipeline first.")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sent = publish_metrics(metrics_from_report(report))
    print("Published to CloudWatch." if sent
          else "Logged to stdout (no AWS credentials).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
