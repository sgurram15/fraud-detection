"""A7 — TERMINATE the EC2 training instance.

Terminate (not just stop): a terminated instance cannot be accidentally
restarted and cannot incur further charges. This is irreversible — the
instance and its instance-store data are destroyed — so we confirm first.

Reads EC2_INSTANCE_ID from .env; falls back to the
Name=fraud-detection-training tag if that is unset.
"""

from __future__ import annotations

import os
import sys

from _common import REGION, confirm, get_boto3

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is not installed. Install with:\n"
          "    pip install python-dotenv", file=sys.stderr)
    sys.exit(1)


def _resolve_instance_ids(ec2) -> list[str]:
    iid = os.environ.get("EC2_INSTANCE_ID")
    if iid:
        return [iid]
    print("EC2_INSTANCE_ID not in .env; falling back to tag lookup.")
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["fraud-detection-training"]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    return [i["InstanceId"]
            for r in resp["Reservations"] for i in r["Instances"]]


def main() -> None:
    load_dotenv()
    boto3 = get_boto3()
    ec2 = boto3.client("ec2", region_name=REGION)

    ids = _resolve_instance_ids(ec2)
    if not ids:
        print("No fraud-detection-training instance found. Nothing to do.")
        return

    print("Instance(s) to TERMINATE (irreversible):")
    for i in ids:
        print(f"  {i}")
    if not confirm(f"TERMINATE {len(ids)} instance(s)? This destroys them"):
        print("Aborted — instance(s) NOT terminated (may still be billing).")
        return

    ec2.terminate_instances(InstanceIds=ids)
    print("Terminating ... waiting for terminated state.")
    ec2.get_waiter("instance_terminated").wait(InstanceIds=ids)

    # Confirm each really reached 'terminated'.
    resp = ec2.describe_instances(InstanceIds=ids)
    states = {i["InstanceId"]: i["State"]["Name"]
              for r in resp["Reservations"] for i in r["Instances"]}
    print(f"States: {states}")
    if all(s == "terminated" for s in states.values()):
        print("\nINSTANCE TERMINATED — No further charges")
    else:
        print("\nWARNING: not all instances reached 'terminated'. Re-check "
              "the EC2 console.")


if __name__ == "__main__":
    main()
