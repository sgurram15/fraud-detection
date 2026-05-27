"""C5.1 — Create the Amazon MSK (managed Kafka) cluster.

Creates a 2-broker kafka.t3.small cluster in eu-west-2 (FCA data residency),
waits for it to become ACTIVE (15-20 min), reads the bootstrap broker string,
and writes MSK_CLUSTER_ARN + MSK_BOOTSTRAP_SERVERS to .env.

PRECONDITION — STOP POINT C5 (the human must have confirmed the budget):
  kafka.t3.small x2 brokers ≈ £0.14/hour. Cluster creation takes 15-20 min.
  ~2 hours of testing ≈ £0.28. RUN scripts/aws/delete_msk.py WHEN DONE.

Networking: by default the script auto-discovers two subnets (distinct AZs) and
the default security group of the default VPC. Override with MSK_SUBNET_IDS
(comma-separated) and MSK_SECURITY_GROUP_ID in .env for a non-default VPC.

Makes no AWS calls until you confirm at the prompt (or pass --yes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import (
    REGION,
    append_dotenv,
    confirm,
    get_boto3,
    get_env,
)

CLUSTER_NAME = "fraud-detection-msk"
KAFKA_VERSION = "3.5.1"
BROKER_TYPE = "kafka.t3.small"
NUM_BROKERS = 2
STORAGE_GB = 20
POLL_SECONDS = 60


def _discover_network(boto3) -> tuple[list[str], str]:
    """Two subnets in distinct AZs + a security group from the default VPC."""
    override = get_env("MSK_SUBNET_IDS")
    sg_override = get_env("MSK_SECURITY_GROUP_ID")
    if override and sg_override:
        return [s.strip() for s in override.split(",") if s.strip()], sg_override

    ec2 = boto3.client("ec2", region_name=REGION)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise RuntimeError("No default VPC found; set MSK_SUBNET_IDS and "
                           "MSK_SECURITY_GROUP_ID in .env.")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    by_az: dict[str, str] = {}
    for s in subnets:
        by_az.setdefault(s["AvailabilityZone"], s["SubnetId"])
    chosen = list(by_az.values())[:NUM_BROKERS]
    if len(chosen) < NUM_BROKERS:
        raise RuntimeError(f"Need {NUM_BROKERS} subnets in distinct AZs; "
                           f"found {len(chosen)} in default VPC {vpc_id}.")
    sgs = ec2.describe_security_groups(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]},
        {"Name": "group-name", "Values": ["default"]}])["SecurityGroups"]
    sg_id = sg_override or (sgs[0]["GroupId"] if sgs else None)
    if not sg_id:
        raise RuntimeError("No default security group found; set "
                           "MSK_SECURITY_GROUP_ID in .env.")
    return chosen, sg_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    print(f"About to create MSK cluster '{CLUSTER_NAME}'")
    print(f"  region:  {REGION}")
    print(f"  brokers: {NUM_BROKERS} x {BROKER_TYPE} ({STORAGE_GB}GB each)")
    print(f"  kafka:   {KAFKA_VERSION}")
    print(f"  cost:    ≈ £0.14/hour total; ACTIVE in ~15-20 min")
    print("  REMEMBER TO DELETE WHEN DONE (scripts/aws/delete_msk.py)")
    if not args.yes and not confirm("Proceed and incur cost?"):
        print("Aborted — no AWS resources created.")
        return 0

    boto3 = get_boto3()
    subnets, sg_id = _discover_network(boto3)
    print(f"Using subnets {subnets} and security group {sg_id}")

    kafka = boto3.client("kafka", region_name=REGION)
    resp = kafka.create_cluster(
        ClusterName=CLUSTER_NAME,
        KafkaVersion=KAFKA_VERSION,
        NumberOfBrokerNodes=NUM_BROKERS,
        BrokerNodeGroupInfo={
            "InstanceType": BROKER_TYPE,
            "ClientSubnets": subnets,
            "SecurityGroups": [sg_id],
            "StorageInfo": {"EbsStorageInfo": {"VolumeSize": STORAGE_GB}},
        },
        # TLS in-transit; SASL/IAM is the recommended MSK auth (stream_bus
        # C5.3 connects with SASL_SSL).
        EncryptionInfo={"EncryptionInTransit": {"ClientBroker": "TLS"}},
        ClientAuthentication={"Sasl": {"Iam": {"Enabled": True}}},
        Tags={"Project": "fraud-detection-poc"},
    )
    arn = resp["ClusterArn"]
    print(f"Cluster creation started: {arn}")

    while True:
        state = kafka.describe_cluster(ClusterArn=arn)["ClusterInfo"][
            "State"]
        print(f"  state={state}")
        if state == "ACTIVE":
            break
        if state in ("FAILED", "DELETING"):
            print(f"Cluster entered {state} — aborting.", file=sys.stderr)
            return 1
        time.sleep(POLL_SECONDS)

    brokers = kafka.get_bootstrap_brokers(ClusterArn=arn)
    bootstrap = (brokers.get("BootstrapBrokerStringSaslIam")
                 or brokers.get("BootstrapBrokerStringTls", ""))
    append_dotenv({"MSK_CLUSTER_ARN": arn,
                   "MSK_BOOTSTRAP_SERVERS": bootstrap})
    print(f"\nMSK CLUSTER ACTIVE — {bootstrap}")
    print("Saved MSK_CLUSTER_ARN + MSK_BOOTSTRAP_SERVERS to .env")
    print("REMEMBER TO DELETE CLUSTER WHEN DONE — "
          "python scripts/aws/delete_msk.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
