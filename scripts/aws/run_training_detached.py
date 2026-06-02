"""Launch the FULL training run DETACHED on EC2, with S3-streamed observability.

Unlike run_training_on_ec2.py (a live SSH stream that dies if your laptop
sleeps and persists no logs), this launches remote_train.py detached on the
instance. The run logs to s3://<bucket>/logs/ (training.log, status.json,
metrics.log) every ~30s — so you can monitor from anywhere and diagnose any
failure even after the box is gone.

Flow: git pull on the box (latest committed pipeline code) -> sftp the freshest
remote_train.py -> launch it detached (nohup/setsid) -> return immediately.

Reads EC2_PUBLIC_IP + S3_BUCKET from .env; SSH key ~/.ssh/fraud-detection-key.pem.

    python scripts/aws/run_training_detached.py
    python scripts/aws/watch_training.py          # monitor progress
    python scripts/aws/stop_ec2.py                # terminate when DONE
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from _common import get_env, ssh_connect

KEY_PATH = Path.home() / ".ssh" / "fraud-detection-key.pem"
SSH_USER = "ec2-user"
REPO = "/home/ec2-user/fraud-detection"


def _run(client, cmd: str, timeout: int = 180) -> str:
    _i, o, e = client.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    return out


def main() -> None:
    ip = get_env("EC2_PUBLIC_IP")
    bucket = get_env("S3_BUCKET")
    if not ip:
        sys.exit("EC2_PUBLIC_IP not in .env — run launch_ec2.py first.")
    if not bucket:
        sys.exit("S3_BUCKET not in .env.")
    if not KEY_PATH.exists():
        sys.exit(f"SSH key {KEY_PATH} not found — run launch_ec2.py first.")

    local_runner = Path(__file__).resolve().parent / "remote_train.py"
    print(f"Target {ip}; bucket {bucket}.")

    client = ssh_connect(ip, KEY_PATH, SSH_USER)
    try:
        print("\n[1/3] git pull (latest committed pipeline code) ...")
        _run(client, f"cd {REPO} && git pull --ff-only || "
                     "echo '(git pull skipped/failed — using existing clone)'")

        print("\n[2/3] uploading freshest remote_train.py ...")
        sftp = client.open_sftp()
        sftp.put(str(local_runner), f"{REPO}/scripts/aws/remote_train.py")
        sftp.close()
        print("  uploaded scripts/aws/remote_train.py")

        print("\n[3/3] launching detached run (survives SSH/laptop disconnect) ...")
        launch = (
            f"cd {REPO} && S3_BUCKET={bucket} FRAUD_SAMPLE_N=all "
            f"nohup setsid python3.11 scripts/aws/remote_train.py "
            f">/home/ec2-user/remote_train.boot.log 2>&1 & echo LAUNCHED pid $!"
        )
        _run(client, launch)
        time.sleep(5)
        alive = _run(client, "pgrep -af remote_train.py | grep -v grep | "
                             "head -2 || echo 'NOT RUNNING — check "
                             "remote_train.boot.log'")
    finally:
        client.close()

    print("\n" + "=" * 60)
    print("DETACHED TRAINING LAUNCHED" if "remote_train" in alive
          else "LAUNCH UNCERTAIN — check the boot log")
    print("=" * 60)
    print("Monitor (works from anywhere, survives laptop sleep):")
    print("  python scripts/aws/watch_training.py")
    print(f"  aws s3 cp s3://{bucket}/logs/status.json -")
    print(f"  aws s3 cp s3://{bucket}/logs/training.log -")
    print(f"\nOn success, artifacts -> s3://{bucket}/models|docs|data/processed/")
    print("Terminate the box when status=DONE -> python scripts/aws/stop_ec2.py")


if __name__ == "__main__":
    main()
