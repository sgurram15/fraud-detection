"""Run the full training pipeline on EC2 (option B) and pull artifacts local.

Flow (all streamed back to the local terminal):
  Step 1  Download raw IEEE-CIS CSVs from S3 to the EC2 local disk.
  Step 2  Train locally with USE_S3=false on the FULL dataset
          (build_features -> handle_imbalance -> train_baseline ->
          compare_models).
  Step 3  Upload the model + metrics artifacts from EC2 back to S3.
Then download those artifacts from S3 to the local repo and print metrics.

This is the option-B resolution of the earlier USE_S3 blocker: the pipeline
scripts are local-FS only, so we stage data on EC2, run with USE_S3=false, and
move artifacts via explicit S3 download/upload.

Remote python is ``python3.11`` (the bootstrap installs deps there; bare
``python`` on AL2023 is 3.9 and would not see the installed packages).

NOTE: step 2 runs compare_models.py, which needs BOTH baseline_xgboost.pkl and
tuned_xgboost.pkl. A fresh clone + train_baseline only produces the baseline,
so compare_models is run non-fatally (its failure is logged but does not abort
the upload). To get a real comparison, also run tune_model.py on EC2 (heavy
hyperparameter search) or stage an existing tuned_xgboost.pkl first.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from _common import REGION, REPO_ROOT, get_boto3, ssh_connect, ssh_run_stream

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is not installed. Install with:\n"
          "    pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

KEY_NAME = "fraud-detection-key"
KEY_PATH = Path.home() / ".ssh" / f"{KEY_NAME}.pem"
SSH_USER = "ec2-user"
REMOTE_REPO = "/home/ec2-user/fraud-detection"

# (s3_key, local_repo_relative_path) artifacts to retrieve after training.
DOWNLOADS = (
    ("models/saved/baseline_xgboost.pkl",
     "src/models/saved/baseline_xgboost.pkl"),
    ("docs/model_performance/baseline_metrics.json",
     "docs/model_performance/baseline_metrics.json"),
    ("docs/model_performance/model_comparison.json",
     "docs/model_performance/model_comparison.json"),
)

# --- Step 1: download raw data S3 -> EC2 local disk (python3.11 heredoc) ---
_STEP1 = r"""
echo "=== STEP 1: download raw data from S3 to EC2 ==="
python3.11 - <<'PYEOF'
import boto3, os
s3 = boto3.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2'))
bucket = os.environ['S3_BUCKET']
files = [
    'train_transaction.csv',
    'train_identity.csv',
    'test_transaction.csv',
    'test_identity.csv',
]
os.makedirs('data/raw', exist_ok=True)
for f in files:
    print(f'Downloading {f}...')
    s3.download_file(bucket, f'data/raw/{f}', f'data/raw/{f}')
    print(f'{f} done')
print('All data downloaded')
PYEOF
"""

# --- Step 2: train locally with USE_S3=false on the full dataset ----------
_STEP2 = r"""
echo "=== STEP 2: train locally (USE_S3=false, full data) ==="
export USE_S3=false
export FRAUD_SAMPLE_N=0
python3.11 src/features/build_features.py
python3.11 src/features/handle_imbalance.py
python3.11 src/models/train_baseline.py
python3.11 src/models/compare_models.py || echo "WARN: compare_models failed (needs tuned_xgboost.pkl); continuing to artifact upload"
"""

# --- Step 3: upload artifacts EC2 -> S3 (python3.11 heredoc) --------------
_STEP3 = r"""
echo "=== STEP 3: upload artifacts to S3 ==="
python3.11 - <<'PYEOF'
import boto3, os
s3 = boto3.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2'))
bucket = os.environ['S3_BUCKET']
artifacts = [
    ('src/models/saved/baseline_xgboost.pkl',
     'models/saved/baseline_xgboost.pkl'),
    ('docs/model_performance/baseline_metrics.json',
     'docs/model_performance/baseline_metrics.json'),
    ('docs/model_performance/model_comparison.json',
     'docs/model_performance/model_comparison.json'),
]
for local, remote in artifacts:
    if os.path.exists(local):
        print(f'Uploading {local}...')
        s3.upload_file(local, bucket, remote)
        print(f'Uploaded to s3://{bucket}/{remote}')
    else:
        print(f'WARNING: {local} not found -- training may have failed')
print('Artifact upload complete')
PYEOF
"""


def _remote_script(bucket: str) -> str:
    header = (
        "set -e\n"
        f"cd {REMOTE_REPO}\n"
        f"export S3_BUCKET={bucket}\n"
        f"export AWS_DEFAULT_REGION={REGION}\n"
    )
    # Normalise to LF: if this file is checked out with CRLF (Windows git
    # autocrlf), the heredoc delimiters would arrive as "PYEOF\r" and break
    # on the Linux instance.
    return (header + _STEP1 + _STEP2 + _STEP3).replace("\r\n", "\n")


def _download_artifacts(bucket: str) -> Path | None:
    from botocore.exceptions import ClientError

    s3 = get_boto3().client("s3", region_name=REGION)
    metrics_local = None
    for key, rel in DOWNLOADS:
        dest = REPO_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"  s3://{bucket}/{key} -> {rel}")
            s3.download_file(bucket, key, str(dest))
            if rel.endswith("baseline_metrics.json"):
                metrics_local = dest
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            print(f"  WARNING: could not download {key} ({code}) — skipping.")
    return metrics_local


def _print_metrics(metrics_path: Path) -> None:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    m = data.get("metrics", {})
    print("\nFinal baseline metrics (from baseline_metrics.json):")
    for k in ("auc_roc", "auc_pr", "precision", "recall",
              "false_positive_rate", "threshold"):
        if k in m:
            print(f"  {k:22s} {m[k]:.4f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    load_dotenv()
    ip = os.environ.get("EC2_PUBLIC_IP")
    bucket = os.environ.get("S3_BUCKET")
    if not ip:
        sys.exit("EC2_PUBLIC_IP not in .env. Run launch_ec2.py first.")
    if not bucket:
        sys.exit("S3_BUCKET not in .env.")
    if not KEY_PATH.exists():
        sys.exit(f"SSH key {KEY_PATH} not found. Run launch_ec2.py first.")

    client = ssh_connect(ip, KEY_PATH, SSH_USER)
    try:
        rc = ssh_run_stream(
            client, _remote_script(bucket),
            "EC2 pipeline: download data -> train (USE_S3=false) -> upload")
        if rc != 0:
            sys.exit(f"EC2 pipeline failed with exit code {rc}.")
    finally:
        client.close()

    print("\nDownloading artifacts from S3 to local ...")
    metrics_path = _download_artifacts(bucket)
    if metrics_path and metrics_path.exists():
        _print_metrics(metrics_path)

    print("\nTRAINING COMPLETE — RUN stop_ec2.py NOW")


if __name__ == "__main__":
    main()
