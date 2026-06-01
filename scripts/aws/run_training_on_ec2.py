"""A9 — Run the full training pipeline on the EC2 instance.

SSHes into the running fraud-detection-training instance, runs the pipeline
in order, streams stdout/stderr back live, then uploads model artefacts to
S3 and prints TRAINING COMPLETE with a metrics summary.

Prereqs: instance launched (launch_ec2.py) and bootstrapped (READY);
SSH key path in env SSH_KEY; S3_BUCKET set. paramiko required:
    pip install paramiko
"""

from __future__ import annotations

import os
import sys

from _common import REGION, get_boto3

# Full-dataset one-shot pipeline. FRAUD_SAMPLE_N=all overrides the local 100k
# default (train_baseline.DEFAULT_SAMPLE_N) so the EC2 run trains, tunes and
# validates on the COMPLETE IEEE-CIS dataset — matching the production model.
_FULL = "FRAUD_SAMPLE_N=all"
PIPELINE = [
    "python src/features/build_features.py",
    "python src/features/handle_imbalance.py",
    f"{_FULL} python src/models/train_baseline.py",
    f"{_FULL} python src/models/tune_model.py",
    f"{_FULL} python src/models/validate_model.py",
    f"{_FULL} python src/models/predict_test.py",  # test-set predictions
]
REMOTE_DIR = "/home/ec2-user/fraud-detection"

# Map each remote artefact -> its S3 key, matching the bucket's existing layout
# (models/saved/, docs/model_performance/). Test predictions go under
# data/processed/ (the requested location).
ARTIFACTS = {
    "src/models/saved/baseline_xgboost.pkl":
        "models/saved/baseline_xgboost.pkl",
    "src/models/saved/tuned_xgboost.pkl":
        "models/saved/tuned_xgboost.pkl",
    "docs/model_performance/validation_report.json":
        "docs/model_performance/validation_report.json",
    "docs/model_performance/baseline_metrics.json":
        "docs/model_performance/baseline_metrics.json",
    "data/processed/test_predictions.csv":
        "data/processed/test_predictions.csv",
}


def _instance_ip(boto3):
    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["fraud-detection-training"]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ])
    for r in resp["Reservations"]:
        for i in r["Instances"]:
            return i.get("PublicIpAddress")
    return None


def main() -> None:
    bucket = os.getenv("S3_BUCKET", "")
    key_path = os.getenv("SSH_KEY", "")
    if not bucket or not key_path:
        print("Set S3_BUCKET and SSH_KEY env vars.", file=sys.stderr)
        sys.exit(1)
    try:
        import paramiko
    except ImportError:
        print("paramiko not installed: pip install paramiko", file=sys.stderr)
        sys.exit(1)

    boto3 = get_boto3()
    ip = _instance_ip(boto3)
    if not ip:
        print("No running fraud-detection-training instance found.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Connecting to ec2-user@{ip} ...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username="ec2-user",
                key_filename=os.path.expanduser(key_path))

    # Step 0: pull the raw CSVs (uploaded by upload_data.py) from S3 onto the
    # instance's local disk. The pipeline reads the local filesystem, and
    # deploy_to_ec2.py only clones code (data is gitignored), so without this
    # build_features.py would fail with "train_transaction.csv not found". Uses
    # boto3 (installed by bootstrap) + the instance-profile credentials, so it
    # needs no AWS CLI and no access keys on the box.
    data_sync = (
        f'''python3.11 -c "import boto3, os; '''
        f'''s3 = boto3.client('s3', region_name='{REGION}'); b = '{bucket}'; '''
        f'''keys = [o['Key'] for o in s3.list_objects_v2(Bucket=b, Prefix='data/raw/').get('Contents', []) if not o['Key'].endswith('/')]; '''
        f'''[os.makedirs(os.path.dirname(k) or '.', exist_ok=True) for k in keys]; '''
        f'''[s3.download_file(b, k, k) for k in keys]; '''
        f'''print('SYNCED', len(keys), 'files from s3://' + b + '/data/raw/')"'''
    )

    for cmd in [data_sync, *PIPELINE]:
        remote_cmd = f"cd {REMOTE_DIR} && {cmd}"
        print(f"\n=== {cmd} ===")
        _, stdout, stderr = ssh.exec_command(remote_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            sys.stdout.write(line)
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            err = stderr.read().decode(errors="replace")
            print(f"STEP FAILED (rc={rc}):\n{err}", file=sys.stderr)
            ssh.close()
            sys.exit(rc)

    # Upload artefacts to S3.
    print("\nUploading artefacts to S3 ...")
    sftp = ssh.open_sftp()
    s3 = boto3.client("s3", region_name=REGION)
    import tempfile

    for rel, key in ARTIFACTS.items():
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local = tmp.name
        try:
            sftp.get(f"{REMOTE_DIR}/{rel}", local)
            s3.upload_file(local, bucket, key)
            print(f"  uploaded s3://{bucket}/{key}")
        except FileNotFoundError:
            print(f"  (skipped, not produced: {rel})")
        finally:
            os.unlink(local)

    _, out, _ = ssh.exec_command(
        f"cat {REMOTE_DIR}/docs/model_performance/validation_report.json "
        "2>/dev/null | head -40"
    )
    summary = out.read().decode(errors="replace")
    ssh.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(summary or "(no validation_report.json found)")
    print("\nRemember: python scripts/aws/stop_ec2.py")


if __name__ == "__main__":
    main()
