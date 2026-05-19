"""A5 — Upload data/raw/*.csv to s3://<bucket>/data/raw/.

Progress bar for large files, MD5 checksum verification, total size + cost
estimate. Usage: S3_BUCKET=<bucket> python scripts/aws/upload_data.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from _common import REGION, get_boto3

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "raw"
GB = 1024 ** 3
S3_GB_MONTH_GBP = 0.023


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(name: str, total: int):
    state = {"sent": 0}

    def cb(n: int) -> None:
        state["sent"] += n
        pct = 100 * state["sent"] / total if total else 100
        sys.stdout.write(f"\r  {name}: {pct:5.1f}%")
        sys.stdout.flush()

    return cb


def main() -> None:
    bucket = os.getenv("S3_BUCKET", "")
    if not bucket:
        print("Set S3_BUCKET (run setup_s3.py first).", file=sys.stderr)
        sys.exit(1)
    boto3 = get_boto3()
    s3 = boto3.client("s3", region_name=REGION)

    csvs = sorted(_RAW.glob("**/*.csv"))
    if not csvs:
        print(f"No CSVs under {_RAW}. Run data/download_data.py first.")
        sys.exit(1)

    total_bytes = 0
    for f in csvs:
        size = f.stat().st_size
        total_bytes += size
        key = f"data/raw/{f.name}"
        print(f"Uploading {f.name} ({size/GB:.3f} GB) -> s3://{bucket}/{key}")
        s3.upload_file(str(f), bucket, key,
                       Callback=_progress(f.name, size))
        print()
        head = s3.head_object(Bucket=bucket, Key=key)
        etag = head["ETag"].strip('"')
        if "-" not in etag:  # single-part upload => ETag == MD5
            local = _md5(f)
            ok = local == etag
            print(f"  checksum {'OK' if ok else 'MISMATCH'} ({etag})")
        else:
            print(f"  multipart upload; ETag={etag} (size-verified)")

    gb = total_bytes / GB
    print(f"\nDONE. Uploaded {len(csvs)} files, {gb:.3f} GB total.")
    print(f"Estimated S3 storage cost: £{gb * S3_GB_MONTH_GBP:.2f}/month "
          f"(£{S3_GB_MONTH_GBP}/GB/month).")


if __name__ == "__main__":
    main()
