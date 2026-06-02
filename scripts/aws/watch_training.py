"""Monitor the detached EC2 training run from S3 (works from anywhere).

Prints logs/status.json (status, current step, elapsed, last resource line) and
the tail of logs/training.log. Pass a line count to change the tail length, or
``--follow`` to refresh every 20s.

    python scripts/aws/watch_training.py            # status + last 40 log lines
    python scripts/aws/watch_training.py 100        # last 100 lines
    python scripts/aws/watch_training.py --follow    # live, refresh every 20s
"""

from __future__ import annotations

import sys
import time

from _common import REGION, get_boto3, get_env


def _show(s3, bucket: str, n: int) -> str | None:
    status = None
    try:
        body = s3.get_object(Bucket=bucket,
                             Key="logs/status.json")["Body"].read().decode()
        status = body
        print("=== status.json ===")
        print(body)
    except Exception as exc:  # noqa: BLE001
        print(f"(no status.json yet: {exc})")
    try:
        body = s3.get_object(Bucket=bucket, Key="logs/training.log")[
            "Body"].read().decode(errors="replace")
        lines = body.splitlines()
        print(f"\n=== training.log (last {n} of {len(lines)} lines) ===")
        print("\n".join(lines[-n:]))
    except Exception as exc:  # noqa: BLE001
        print(f"(no training.log yet: {exc})")
    return status


def main() -> None:
    bucket = get_env("S3_BUCKET")
    if not bucket:
        sys.exit("S3_BUCKET not in .env.")
    args = sys.argv[1:]
    follow = "--follow" in args
    nums = [a for a in args if a.isdigit()]
    n = int(nums[0]) if nums else 40

    boto3 = get_boto3()
    s3 = boto3.client("s3", region_name=REGION)

    if not follow:
        _show(s3, bucket, n)
        return
    print("Following (Ctrl-C to stop) — refresh every 20s\n")
    while True:
        print("\033[2J\033[H", end="")  # clear screen
        status = _show(s3, bucket, n)
        if status and ('"status": "DONE"' in status
                       or '"status": "FAILED"' in status
                       or '"status": "ERROR"' in status):
            print("\n[run finished — stopping follow]")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
