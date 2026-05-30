"""C7.2 — Create CloudWatch alarms + dashboard for the fraud pipeline.

Creates an SNS topic (subscribing ALERT_EMAIL from .env), four alarms on the
FraudDetection/Pipeline metrics, and a dashboard 'FraudDetectionPOC'.

Alarms:
  * FalsePositiveRate > 5%            (commercial constraint breached)
  * EndToEndLatencyMs > 1000ms        (latency SLA breach)
  * FraudRate == 0 for > 1 hour       (model/feature pipeline likely failed)
  * AgentDecisions_Block > 100 / 1h   (abnormally high block volume)

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
        {**common, "AlarmName": "FraudDetection-FPR-High",
         "MetricName": "FalsePositiveRate", "Statistic": "Average",
         "Period": 300, "EvaluationPeriods": 1, "Threshold": 5.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "AlarmDescription": "False-positive rate above the 5% commercial cap"},
        {**common, "AlarmName": "FraudDetection-Latency-High",
         "MetricName": "EndToEndLatencyMs", "Statistic": "Average",
         "Period": 300, "EvaluationPeriods": 1, "Threshold": 1000.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "AlarmDescription": "End-to-end latency over 1000ms"},
        {**common, "AlarmName": "FraudDetection-FraudRate-Zero",
         "MetricName": "FraudRate", "Statistic": "Maximum",
         "Period": 3600, "EvaluationPeriods": 1, "Threshold": 0.0,
         "ComparisonOperator": "LessThanOrEqualToThreshold",
         "TreatMissingData": "breaching",
         "AlarmDescription": "Fraud rate flat at 0 for >1h — model may have "
                             "failed"},
        {**common, "AlarmName": "FraudDetection-Blocks-High",
         "MetricName": "AgentDecisions_Block", "Statistic": "Sum",
         "Period": 3600, "EvaluationPeriods": 1, "Threshold": 100.0,
         "ComparisonOperator": "GreaterThanThreshold",
         "AlarmDescription": "More than 100 agent BLOCK decisions in 1h"},
    ]


def _dashboard_body() -> str:
    def widget(title, metric, stat="Average"):
        return {"type": "metric", "width": 12, "height": 6,
                "properties": {"region": REGION, "title": title, "stat": stat,
                               "metrics": [[NAMESPACE, metric]]}}
    return json.dumps({"widgets": [
        widget("Transactions / period", "TransactionsProcessed", "Sum"),
        widget("Fraud rate (%)", "FraudRate"),
        widget("Latency P50/P95 (ms)", "EndToEndLatencyMs"),
        widget("Agent BLOCK decisions", "AgentDecisions_Block", "Sum"),
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
