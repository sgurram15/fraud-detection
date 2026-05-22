"""A6 — Launch a fully-provisioned t3.large EC2 training instance.

Order of operations (each step idempotent — safe to re-run):
  1. Load creds + config from .env (python-dotenv).
  2. Create/reuse SSH key pair (fraud-detection-key) -> ~/.ssh/*.pem (chmod 400).
  3. Create/reuse security group (fraud-detection-sg): inbound SSH 22 from the
     launcher's current public IP only.
  4. Create/reuse IAM role + instance profile (fraud-detection-ec2-role /
     -profile) granting the instance S3 + SageMaker + CloudWatch agent access.
  5. Resolve the latest AL2023 AMI (SSM parameter, DescribeImages fallback).
  6. Launch ONE t3.large with the key/SG/profile + bootstrap user-data.
  7. Wait for running (progress every 15s).
  8. Record EC2_* values back into .env.
  9. Print a summary.

COST: ~£0.08/hour while RUNNING. Run stop_ec2.py when done.

PERMISSION PREREQUISITES (the spec'd IAM user S3+EC2+SageMaker may NOT have
all of these — add them if a step returns AccessDenied):
  * Step 4 needs iam:CreateRole, iam:AttachRolePolicy, iam:CreateInstanceProfile,
    iam:AddRoleToInstanceProfile, iam:PassRole.
  * Step 5 (pure SSM path) needs ssm:GetParameter; we fall back to
    ec2:DescribeImages (covered by AmazonEC2FullAccess) if SSM is denied.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from _common import PROJECT_TAG, REGION, append_dotenv, confirm, get_boto3

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is not installed. Install with:\n"
          "    pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

INSTANCE_TYPE = "t3.large"  # 2 vCPU, 8 GB RAM
HOURLY_GBP = 0.08
SG_NAME = "fraud-detection-sg"
KEY_NAME = "fraud-detection-key"
KEY_PATH = Path.home() / ".ssh" / f"{KEY_NAME}.pem"
ROLE_NAME = "fraud-detection-ec2-role"
PROFILE_NAME = "fraud-detection-ec2-profile"

SSM_AMI_PARAM = ("/aws/service/ami-amazon-linux-latest/"
                 "al2023-ami-kernel-default-x86_64")
AL2023_NAME_GLOB = "al2023-ami-2023*-kernel-*-x86_64"  # DescribeImages fallback

REQUIRED_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION", "S3_BUCKET")

MANAGED_POLICIES = (
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
    "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess",
)

_IAM_DENIED_HELP = """IAM permissions missing. Run this once in AWS console or as root:
 aws iam attach-user-policy \\
   --user-name fraud-detection-dev \\
   --policy-arn arn:aws:iam::aws:policy/IAMFullAccess"""
EC2_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}

USER_DATA = """#!/bin/bash
set -e
exec > /var/log/bootstrap.log 2>&1
echo "Bootstrap started at $(date)"
yum update -y
yum install -y python3.11 python3.11-pip git
pip3.11 install --upgrade pip
pip3.11 install pandas numpy scikit-learn xgboost shap fastapi
pip3.11 install uvicorn mlflow boto3 imbalanced-learn evidently
pip3.11 install kafka-python python-dotenv paramiko tqdm
echo "Bootstrap completed at $(date)"
echo "READY" > /home/ec2-user/BOOTSTRAP_DONE
chown ec2-user:ec2-user /home/ec2-user/BOOTSTRAP_DONE
"""


def _require_env() -> str:
    load_dotenv()
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"Missing required .env keys: {', '.join(missing)}.\n"
              "Populate .env with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
              "AWS_DEFAULT_REGION (eu-west-2) and S3_BUCKET, then re-run.",
              file=sys.stderr)
        sys.exit(1)
    region = os.environ["AWS_DEFAULT_REGION"]
    if region != REGION:
        print(f"WARNING: AWS_DEFAULT_REGION={region} but FCA data residency "
              f"requires {REGION}. Proceeding with {region} as set.")
    return region


def _my_ip() -> str:
    return urlopen("https://api.ipify.org", timeout=10).read().decode().strip()


def _ensure_key_pair(ec2) -> None:
    from botocore.exceptions import ClientError

    exists = True
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "InvalidKeyPair.NotFound":
            exists = False
        else:
            raise

    if exists:
        if KEY_PATH.exists():
            print(f"Reusing key pair {KEY_NAME} ({KEY_PATH}).")
            return
        raise RuntimeError(
            f"Key pair '{KEY_NAME}' exists in AWS but {KEY_PATH} is missing "
            "locally — the private key cannot be re-downloaded. Delete the AWS "
            f"key pair (`aws ec2 delete-key-pair --key-name {KEY_NAME} "
            f"--region {REGION}`) and re-run, or restore the .pem."
        )

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    material = ec2.create_key_pair(KeyName=KEY_NAME)["KeyMaterial"]
    KEY_PATH.write_text(material, encoding="utf-8")
    os.chmod(KEY_PATH, stat.S_IRUSR)  # 400 (read-only owner)
    print(f"Created key pair {KEY_NAME}; saved to {KEY_PATH} (chmod 400).")


def _default_vpc_id(ec2) -> str | None:
    vpcs = ec2.describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}]
    )["Vpcs"]
    return vpcs[0]["VpcId"] if vpcs else None


def _ensure_security_group(ec2, my_ip: str) -> str:
    from botocore.exceptions import ClientError

    try:
        sgs = ec2.describe_security_groups(
            GroupNames=[SG_NAME]
        )["SecurityGroups"]
        sg_id = sgs[0]["GroupId"]
        print(f"Reusing security group {SG_NAME} ({sg_id}).")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "InvalidGroup.NotFound":
            raise
        vpc_id = _default_vpc_id(ec2)
        kwargs = {"GroupName": SG_NAME, "Description": "Fraud detection training"}
        if vpc_id:
            kwargs["VpcId"] = vpc_id
        sg_id = ec2.create_security_group(**kwargs)["GroupId"]
        ec2.create_tags(Resources=[sg_id], Tags=[
            {"Key": "Name", "Value": SG_NAME}, PROJECT_TAG,
        ])
        print(f"Created security group {SG_NAME} ({sg_id}).")

    cidr = f"{my_ip}/32"
    # Drop any stale SSH rules whose CIDR no longer matches the current IP.
    perms = ec2.describe_security_groups(
        GroupIds=[sg_id])["SecurityGroups"][0].get("IpPermissions", [])
    have_current = False
    for p in perms:
        if p.get("FromPort") == 22 and p.get("ToPort") == 22:
            for rng in p.get("IpRanges", []):
                if rng.get("CidrIp") == cidr:
                    have_current = True
                else:
                    ec2.revoke_security_group_ingress(
                        GroupId=sg_id, IpPermissions=[{
                            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                            "IpRanges": [{"CidrIp": rng["CidrIp"]}],
                        }])
                    print(f"Revoked stale SSH rule for {rng['CidrIp']}.")
    if have_current:
        print(f"SSH 22 from {cidr} already authorized.")
    else:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": cidr,
                              "Description": "SSH from launcher IP"}],
            }])
        print(f"Authorized inbound SSH 22 from {cidr} (no other inbound).")
    return sg_id


def _ensure_instance_profile(iam) -> str:
    from botocore.exceptions import ClientError

    try:
        try:
            iam.get_role(RoleName=ROLE_NAME)
            print(f"Reusing IAM role {ROLE_NAME}.")
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "NoSuchEntity":
                raise
            iam.create_role(
                RoleName=ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(EC2_TRUST_POLICY),
                Description="Fraud detection EC2 training role",
                Tags=[PROJECT_TAG],
            )
            print(f"Created IAM role {ROLE_NAME}.")

        for arn in MANAGED_POLICIES:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=arn)
        print("Attached policies: "
              f"{', '.join(a.split('/')[-1] for a in MANAGED_POLICIES)}.")

        try:
            iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
            print(f"Reusing instance profile {PROFILE_NAME}.")
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "NoSuchEntity":
                raise
            iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
            print(f"Created instance profile {PROFILE_NAME}.")

        prof = iam.get_instance_profile(
            InstanceProfileName=PROFILE_NAME)["InstanceProfile"]
        if ROLE_NAME not in [r["RoleName"] for r in prof["Roles"]]:
            iam.add_role_to_instance_profile(
                InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
            print(f"Added role {ROLE_NAME} to instance profile "
                  f"{PROFILE_NAME}.")
    except ClientError as exc:
        if "AccessDenied" in exc.response["Error"]["Code"]:
            print(f"\n{_IAM_DENIED_HELP}", file=sys.stderr)
            sys.exit(1)
        raise

    print("Waiting 10s for IAM propagation ...")
    time.sleep(10)
    return PROFILE_NAME


def _resolve_ami(ec2, ssm) -> str:
    from botocore.exceptions import ClientError

    try:
        ami = ssm.get_parameter(Name=SSM_AMI_PARAM)["Parameter"]["Value"]
        print(f"Resolved AL2023 AMI via SSM: {ami}")
        return ami
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(f"  SSM GetParameter failed ({code}); falling back to "
              "ec2:DescribeImages (still dynamic, never hardcoded).",
              file=sys.stderr)
        images = ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": [AL2023_NAME_GLOB]},
                {"Name": "architecture", "Values": ["x86_64"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "image-type", "Values": ["machine"]},
            ],
        )["Images"]
        images = [i for i in images if "minimal" not in i["Name"]]
        if not images:
            raise RuntimeError(
                f"No AL2023 AMI matched '{AL2023_NAME_GLOB}' in {REGION}.")
        ami = max(images, key=lambda i: i["CreationDate"])["ImageId"]
        print(f"Resolved AL2023 AMI via DescribeImages: {ami}")
        return ami


def _wait_running(ec2, iid: str) -> str:
    start = time.time()
    while True:
        inst = ec2.describe_instances(
            InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        if state == "running":
            return inst.get("PublicIpAddress", "(no public IP)")
        print(f"  ... state={state} ({int(time.time() - start)}s elapsed)")
        time.sleep(15)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emojis on Windows cp1252
    except (AttributeError, ValueError):
        pass

    region = _require_env()
    bucket = os.environ["S3_BUCKET"]
    boto3 = get_boto3()
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    iam = boto3.client("iam")  # IAM is global

    print(f"This launches a {INSTANCE_TYPE} in {region} at "
          f"~£{HOURLY_GBP}/hour. It bills until stopped/terminated.")
    if not confirm("Launch instance?"):
        print("Aborted.")
        return

    _ensure_key_pair(ec2)
    my_ip = _my_ip()
    print(f"Launcher public IP (SSH allow-list): {my_ip}")
    sg_id = _ensure_security_group(ec2, my_ip)
    _ensure_instance_profile(iam)
    ami = _resolve_ami(ec2, ssm)

    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": PROFILE_NAME},
        UserData=USER_DATA,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "fraud-detection-training"},
                     PROJECT_TAG],
        }],
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"Launched {iid}. Waiting for running state ...")
    ip = _wait_running(ec2, iid)

    append_dotenv({
        "EC2_INSTANCE_ID": iid,
        "EC2_PUBLIC_IP": ip,
        "EC2_KEY_NAME": KEY_NAME,
        "EC2_INSTANCE_PROFILE": PROFILE_NAME,
    })

    print()
    print(f"✅ Instance ID: {iid}")
    print(f"✅ Public IP: {ip}")
    print(f"✅ Key pair: {KEY_NAME}")
    print(f"✅ IAM profile: {PROFILE_NAME} (S3 + SageMaker access)")
    print("✅ Bootstrap running — wait 7-10 minutes before deploying")
    print(f"\U0001F4B0 Cost: ~£{HOURLY_GBP}/hour — run stop_ec2.py when done")
    print(f"\nUsing bucket: {bucket}")
    print("Next step: python scripts/aws/deploy_to_ec2.py")


if __name__ == "__main__":
    main()
