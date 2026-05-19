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

PIPELINE = [
    "python src/features/build_features.py",
    "python src/features/handle_imbalance.py",
    "python src/models/train_baseline.py",
    "python src/models/tune_model.py",
    "python src/models/validate_model.py",
]
REMOTE_DIR = "/home/ec2-user/fraud-detection"
ARTIFACTS = ["src/models/saved/baseline_xgboost.pkl",
             "src/models/saved/tuned_xgboost.pkl",
             "docs/model_performance/validation_report.json"]


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

    for cmd in PIPELINE:
        full = f"cd {REMOTE_DIR} && {cmd}"
        print(f"\n=== {cmd} ===")
        _, stdout, stderr = ssh.exec_command(full, get_pty=True)
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

    for rel in ARTIFACTS:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local = tmp.name
        try:
            sftp.get(f"{REMOTE_DIR}/{rel}", local)
            s3.upload_file(local, bucket, f"models/{rel}")
            print(f"  uploaded s3://{bucket}/models/{rel}")
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
