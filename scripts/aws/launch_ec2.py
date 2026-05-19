"""A6 — Launch a t3.large EC2 instance for training.

Amazon Linux 2023, eu-west-2, auto-bootstraps Python 3.11 + git + repo +
requirements. Tagged Name=fraud-detection-training, Project=fraud-detection-poc.

COST: ~£0.08/hour while RUNNING. Requires AWS_SETUP.md STOP 1-3 complete.
"""

from __future__ import annotations

import os
import sys

from _common import PROJECT_TAG, REGION, confirm, get_boto3

INSTANCE_TYPE = "t3.large"  # 2 vCPU, 8 GB RAM
HOURLY_GBP = 0.08
REPO_URL = os.getenv(
    "REPO_URL", "https://github.com/sgurram15/fraud-detection.git"
)

# Amazon Linux 2023 SSM public parameter (region-resolved at run time).
AL2023_SSM = ("/aws/service/ami-amazon-linux-latest/"
              "al2023-ami-kernel-default-x86_64")

USER_DATA = f"""#!/bin/bash
set -e
dnf install -y python3.11 python3.11-pip git
cd /home/ec2-user
git clone {REPO_URL} fraud-detection || true
cd fraud-detection
python3.11 -m pip install -r requirements.txt
echo READY > /home/ec2-user/BOOTSTRAP_DONE
"""


def main() -> None:
    boto3 = get_boto3()
    ec2 = boto3.client("ec2", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)

    print(f"This launches a {INSTANCE_TYPE} in {REGION} at "
          f"~£{HOURLY_GBP}/hour. It bills until stopped.")
    if not confirm("Launch instance?"):
        print("Aborted.")
        return

    ami = ssm.get_parameter(Name=AL2023_SSM)["Parameter"]["Value"]
    print(f"Using Amazon Linux 2023 AMI: {ami}")

    resp = ec2.run_instances(
        ImageId=ami, InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1, UserData=USER_DATA,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "fraud-detection-training"},
                     PROJECT_TAG],
        }],
    )
    inst = resp["Instances"][0]
    iid = inst["InstanceId"]
    print(f"Launched {iid}. Waiting for running state ...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    desc = ec2.describe_instances(InstanceIds=[iid])
    ip = (desc["Reservations"][0]["Instances"][0]
          .get("PublicIpAddress", "(no public IP)"))

    print(f"\nInstance ID: {iid}")
    print(f"Public IP:   {ip}")
    print(f"Cost:        ~£{HOURLY_GBP}/hour while running")
    print("Bootstrap writes /home/ec2-user/BOOTSTRAP_DONE (READY) when done.")
    print("\n*** REMEMBER TO STOP THIS INSTANCE WHEN DONE ***")
    print("    python scripts/aws/stop_ec2.py")


if __name__ == "__main__":
    main()
