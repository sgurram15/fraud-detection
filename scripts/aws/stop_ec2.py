"""A7 — Stop running EC2 instances tagged fraud-detection-training.

Lists them, asks for confirmation, stops, confirms stopped, and prints the
money saved (a stopped instance no longer accrues compute charges).
"""

from __future__ import annotations

from _common import REGION, confirm, get_boto3

HOURLY_GBP = 0.08


def main() -> None:
    boto3 = get_boto3()
    ec2 = boto3.client("ec2", region_name=REGION)

    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["fraud-detection-training"]},
        {"Name": "instance-state-name", "Values": ["running", "pending"]},
    ])
    ids = [i["InstanceId"]
           for r in resp["Reservations"] for i in r["Instances"]]

    if not ids:
        print("No running fraud-detection-training instances. Nothing to do.")
        return

    print("Running instances:")
    for i in ids:
        print(f"  {i}")
    if not confirm(f"Stop {len(ids)} instance(s)?"):
        print("Aborted — instances still running (still billing).")
        return

    ec2.stop_instances(InstanceIds=ids)
    print("Stopping ... waiting for stopped state.")
    ec2.get_waiter("instance_stopped").wait(InstanceIds=ids)

    print(f"\nStopped {len(ids)} instance(s). Confirmed.")
    print(f"Money saved: ~£{HOURLY_GBP * len(ids):.2f}/hour from now "
          f"(~£{HOURLY_GBP * len(ids) * 24:.2f}/day if it had run on).")


if __name__ == "__main__":
    main()
