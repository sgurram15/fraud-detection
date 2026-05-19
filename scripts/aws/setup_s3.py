"""A4 — Create the project S3 bucket (private, versioned, encrypted).

Cost: £0 to create; ~£0.023/GB/month to store. Requires AWS_SETUP.md
STOP 2 (IAM user + credentials) complete.
"""

from __future__ import annotations

import random
import sys

from _common import BUCKET_PREFIX, PROJECT_TAG, REGION, get_boto3

FOLDERS = ["data/raw/", "data/processed/", "models/saved/", "mlruns/", "docs/"]


def main() -> None:
    boto3 = get_boto3()
    s3 = boto3.client("s3", region_name=REGION)

    bucket = f"{BUCKET_PREFIX}{random.randint(100000, 999999)}"
    print(f"Creating bucket {bucket} in {REGION} ...")
    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": REGION},
    )
    print("  bucket created")

    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    print("  block ALL public access: ON")

    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    print("  versioning: enabled")

    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault":
                       {"SSEAlgorithm": "AES256"}}]
        },
    )
    print("  encryption: SSE-S3")

    s3.put_bucket_tagging(
        Bucket=bucket, Tagging={"TagSet": [PROJECT_TAG]}
    )
    for f in FOLDERS:
        s3.put_object(Bucket=bucket, Key=f)
        print(f"  folder created: {f}")

    pab = s3.get_public_access_block(Bucket=bucket)
    private = all(pab["PublicAccessBlockConfiguration"].values())
    print(f"\nVerified private: {private}")
    if not private:
        print("ERROR: bucket is not fully private!", file=sys.stderr)
        sys.exit(1)

    print(f"\nDONE. Bucket: {bucket}")
    print("Set this for the pipeline:  export S3_BUCKET=" + bucket)
    print("Cost: £0 to create, ~£0.023/GB/month to store.")


if __name__ == "__main__":
    main()
