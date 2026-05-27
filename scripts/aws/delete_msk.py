"""C5.5 — Delete the MSK cluster.

Reads MSK_CLUSTER_ARN from .env, shows the cluster state and rough cost
incurred, asks for confirmation, then deletes the cluster. MSK bills per broker
per hour (kafka.t3.small x2 ≈ £0.14/hour) for as long as the cluster exists —
always run this when done.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import REGION, confirm, get_boto3, get_env

_HOURLY_GBP = 0.14


def main() -> int:
    arn = get_env("MSK_CLUSTER_ARN")
    if not arn:
        print("MSK_CLUSTER_ARN not set in .env — nothing to delete.")
        return 0

    boto3 = get_boto3()
    kafka = boto3.client("kafka", region_name=REGION)

    try:
        info = kafka.describe_cluster(ClusterArn=arn)["ClusterInfo"]
        state = info["State"]
        created = info.get("CreationTime")
        print(f"Cluster:  {info.get('ClusterName')}")
        print(f"State:    {state}")
        print(f"Created:  {created}")
        if isinstance(created, datetime):
            hrs = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            print(f"Est. cost so far: ≈ £{hrs * _HOURLY_GBP:,.2f} "
                  f"({hrs:.1f}h x £{_HOURLY_GBP}/h)")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not describe cluster: {exc}")

    if not confirm("Delete MSK cluster?"):
        print("Aborted — cluster left running (still billing).")
        return 0

    kafka.delete_cluster(ClusterArn=arn)
    print(f"\nMSK CLUSTER DELETED — {arn}. No further charges.")
    print("Tip: remove MSK_CLUSTER_ARN + MSK_BOOTSTRAP_SERVERS from .env once "
          "confirmed gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
