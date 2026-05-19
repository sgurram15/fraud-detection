"""Shared helpers for the AWS scripts.

All AWS resources MUST be in eu-west-2 (FCA data residency) and tagged
Project=fraud-detection-poc. boto3 is imported lazily with a clear message so
the scripts can be created/inspected without AWS deps installed.
"""

from __future__ import annotations

import sys

REGION = "eu-west-2"  # London — FCA data residency. Do not change.
PROJECT_TAG = {"Key": "Project", "Value": "fraud-detection-poc"}
BUCKET_PREFIX = "fraud-detection-poc-"


def get_boto3():
    try:
        import boto3  # noqa: F401

        return boto3
    except ImportError:
        print(
            "boto3 is not installed. Install with:\n"
            "    pip install boto3\n"
            "and complete AWS_SETUP.md STOP points 1-3 first.",
            file=sys.stderr,
        )
        sys.exit(1)


def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")
