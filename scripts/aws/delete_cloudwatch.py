"""C7 teardown — delete the CloudWatch alarms, dashboard, and SNS topic.

Removes everything setup_cloudwatch.py creates: the four FraudDetection alarms,
the FraudDetectionPOC dashboard, and the fraud-detection-alerts SNS topic.
Best-effort and idempotent (tolerant of already-absent resources).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import (
    REGION,
    confirm,
    get_boto3,
    load_dotenv_into_environ,
)

load_dotenv_into_environ()  # make .env AWS creds available to boto3

DASHBOARD_NAME = "FraudDetectionPOC"
SNS_TOPIC_NAME = "fraud-detection-alerts"
ALARM_NAMES = [
    "FraudDetection-HighFPR",
    "FraudDetection-HighLatency",
    "FraudDetection-ZeroFraudRate",
    "FraudDetection-ExcessiveBlocks",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes and not confirm(
            "Delete CloudWatch alarms + dashboard + SNS topic?"):
        print("Aborted.")
        return 0

    boto3 = get_boto3()
    cw = boto3.client("cloudwatch", region_name=REGION)
    sns = boto3.client("sns", region_name=REGION)

    try:
        cw.delete_alarms(AlarmNames=ALARM_NAMES)
        print(f"Deleted alarms: {', '.join(ALARM_NAMES)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (alarms: {exc})")

    try:
        cw.delete_dashboards(DashboardNames=[DASHBOARD_NAME])
        print(f"Deleted dashboard: {DASHBOARD_NAME}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (dashboard: {exc})")

    # Find the SNS topic by name (its ARN ends with the topic name).
    try:
        arn = next((t["TopicArn"] for t in
                    sns.list_topics().get("Topics", [])
                    if t["TopicArn"].rsplit(":", 1)[-1] == SNS_TOPIC_NAME),
                   None)
        if arn:
            sns.delete_topic(TopicArn=arn)
            print(f"Deleted SNS topic: {SNS_TOPIC_NAME}")
        else:
            print(f"  (SNS topic {SNS_TOPIC_NAME} not found)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (SNS topic: {exc})")

    print("\nCLOUDWATCH RESOURCES DELETED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
