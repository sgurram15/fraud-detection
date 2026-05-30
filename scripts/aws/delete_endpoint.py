"""C4.4 — Delete the SageMaker real-time endpoint.

Deletes the endpoint, its endpoint config, and the model so inference charges
stop. SageMaker bills ml.m5.large (~£0.13/hour) for as long as the endpoint
exists, whether or not it receives traffic — always run this when done.

Reads SAGEMAKER_ENDPOINT from .env. Asks for confirmation before deleting.
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
    get_env,
    load_dotenv_into_environ,
)

load_dotenv_into_environ()  # make .env AWS creds available to boto3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    endpoint = get_env("SAGEMAKER_ENDPOINT")
    if not endpoint:
        print("SAGEMAKER_ENDPOINT is not set in .env — nothing to delete.")
        return 0

    boto3 = get_boto3()
    sm = boto3.client("sagemaker", region_name=REGION)

    try:
        desc = sm.describe_endpoint(EndpointName=endpoint)
        print(f"Endpoint:  {endpoint}")
        print(f"Status:    {desc['EndpointStatus']}")
        print(f"Created:   {desc.get('CreationTime')}")
        print("Est. cost: ml.m5.large ≈ £0.13/hour while InService")
        config_name = desc.get("EndpointConfigName", endpoint)
    except sm.exceptions.ClientError as exc:
        print(f"Could not describe endpoint {endpoint}: {exc}")
        config_name = endpoint

    if not args.yes and not confirm(
            f"Delete endpoint {endpoint} (and its config + model)?"):
        print("Aborted — endpoint left running (still billing).")
        return 0

    # Delete endpoint, then its config, then the backing model. Each is
    # tolerant of an already-absent resource.
    for label, fn in (
        ("endpoint", lambda: sm.delete_endpoint(EndpointName=endpoint)),
        ("endpoint config",
         lambda: sm.delete_endpoint_config(EndpointConfigName=config_name)),
        ("model", lambda: sm.delete_model(ModelName=endpoint)),
    ):
        try:
            fn()
            print(f"Deleted {label}.")
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            print(f"  ({label}: {exc})")

    print(f"\nENDPOINT DELETED — {endpoint}. No further inference charges.")
    print("Tip: remove SAGEMAKER_ENDPOINT from .env once confirmed gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
