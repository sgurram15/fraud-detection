"""C7.2 — Create CloudWatch alarms + dashboard for the fraud pipeline.

Creates an SNS topic (subscribing ALERT_EMAIL from .env), four alarms on the
FraudDetection/Pipeline metrics, and a dashboard 'FraudDetectionPOC'.

Alarms:
  * FraudDetection-HighFPR        FalsePositiveRate > 5% (2x 5-min periods)
  * FraudDetection-HighLatency    AvgLatencyMs > 1000ms (latency SLA breach)
  * FraudDetection-ZeroFraudRate  FraudRate <= 0 for 1h (model likely failed)
  * FraudDetection-ExcessiveBlocks  DecisionsBlock > 100 / 1h (abnormal volume)

Cost is minimal (~£0.10/month for the alarms + custom metrics). Makes no AWS
calls until you confirm at the prompt (or pass --yes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import (
    REGION,
    confirm,
    get_boto3,
    get_env,
    load_dotenv_into_environ,
)

load_dotenv_into_environ()  # make .env AWS creds available to boto3

NAMESPACE = "FraudDetection/Pipeline"
DASHBOARD_NAME = "FraudDetectionPOC"
SNS_TOPIC_NAME = "fraud-detection-alerts"


def _ensure_sns(boto3, email: str | None) -> str:
    sns = boto3.client("sns", region_name=REGION)
    arn = sns.create_topic(Name=SNS_TOPIC_NAME,
                           Tags=[{"Key": "Project",
                                  "Value": "fraud-detection-poc"}])["TopicArn"]
    if email:
        sns.subscribe(TopicArn=arn, Protocol="email", Endpoint=email)
        print(f"  SNS subscription requested for {email} "
              "(confirm via the email link).")
    return arn


def _alarms(topic_arn: str) -> list[dict]:
    common = {"Namespace": NAMESPACE, "ActionsEnabled": True,
              "AlarmActions": [topic_arn]}
    return [
        {**common, "AlarmName": "FraudDetection-HighFPR",
         "MetricName": "FalsePositiveRate", "Statistic": "Average",
         "Period": 300, "EvaluationPeriods": 2, "Threshold": 5.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "TreatMissingData": "notBreaching",
         "AlarmDescription": "FPR exceeds 5% commercial constraint"},
        {**common, "AlarmName": "FraudDetection-HighLatency",
         "MetricName": "AvgLatencyMs", "Statistic": "Average",
         "Period": 300, "EvaluationPeriods": 1, "Threshold": 1000.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "AlarmDescription": "End-to-end latency exceeds 1000ms SLA"},
        {**common, "AlarmName": "FraudDetection-ZeroFraudRate",
         "MetricName": "FraudRate", "Statistic": "Maximum",
         "Period": 300, "EvaluationPeriods": 12, "Threshold": 0.0,
         "ComparisonOperator": "LessThanOrEqualToThreshold",
         "TreatMissingData": "notBreaching",
         "AlarmDescription": "No fraud detected for 1 hour"},
        {**common, "AlarmName": "FraudDetection-ExcessiveBlocks",
         "MetricName": "DecisionsBlock", "Statistic": "Sum",
         "Period": 3600, "EvaluationPeriods": 1, "Threshold": 100.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "AlarmDescription": "More than 100 blocks in one hour"},
    ]


def _dashboard_body() -> str:
    def widget(title, metrics, stat="Average", annotations=None):
        props = {"region": REGION, "title": title, "stat": stat,
                 "metrics": metrics}
        if annotations:
            props["annotations"] = {"horizontal": annotations}
        return {"type": "metric", "width": 12, "height": 6,
                "properties": props}
    return json.dumps({"widgets": [
        widget("Transactions processed", [[NAMESPACE, "TransactionsProcessed"]],
               "Sum"),
        widget("Fraud rate (%)", [[NAMESPACE, "FraudRate"]]),
        widget("False positive rate (%)", [[NAMESPACE, "FalsePositiveRate"]],
               annotations=[{"label": "5% cap", "value": 5.0}]),
        widget("Avg latency (ms)", [[NAMESPACE, "AvgLatencyMs"]],
               annotations=[{"label": "1000ms SLA", "value": 1000.0}]),
        widget("Decision distribution",
               [[NAMESPACE, "DecisionsHold"], [NAMESPACE, "DecisionsBlock"]],
               "Sum"),
        widget("Daily fraud saving (£)", [[NAMESPACE, "DailyFraudSavingGBP"]]),
    ]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    email = get_env("ALERT_EMAIL")
    email_display = email or "(none set — alarms will have no subscriber)"
    print(f"About to create CloudWatch alarms + '{DASHBOARD_NAME}' dashboard "
          f"in {REGION}")
    print(f"  alert email: {email_display}")
    print("  cost: ~£0.10/month")
    if not args.yes and not confirm("Proceed?"):
        print("Aborted — no AWS resources created.")
        return 0

    boto3 = get_boto3()
    topic_arn = _ensure_sns(boto3, email)
    cw = boto3.client("cloudwatch", region_name=REGION)
    for alarm in _alarms(topic_arn):
        cw.put_metric_alarm(**alarm)
        print(f"  alarm: {alarm['AlarmName']}")
    cw.put_dashboard(DashboardName=DASHBOARD_NAME,
                     DashboardBody=_dashboard_body())
    print(f"  dashboard: {DASHBOARD_NAME}")
    print("\nCLOUDWATCH ALARMS CREATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
