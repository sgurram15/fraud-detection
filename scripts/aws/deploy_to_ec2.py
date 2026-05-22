"""Deploy the repo onto the running EC2 training instance.

Waits for the user-data bootstrap to finish (polls for BOOTSTRAP_DONE over
SSH, 15-min timeout), then clones the repo, writes the EC2-side .env
(USE_S3=true) and installs requirements. Reads EC2_PUBLIC_IP, S3_BUCKET,
EC2_INSTANCE_ID from .env. Uses ~/.ssh/fraud-detection-key.pem.

Run after launch_ec2.py:
    python scripts/aws/deploy_to_ec2.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from _common import REGION, ssh_connect, ssh_run_stream

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is not installed. Install with:\n"
          "    pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

KEY_NAME = "fraud-detection-key"
KEY_PATH = Path.home() / ".ssh" / f"{KEY_NAME}.pem"
SSH_USER = "ec2-user"
REPO_URL = "https://github.com/sgurram15/fraud-detection.git"
POLL_SECONDS = 30
TIMEOUT_SECONDS = 900  # 15 minutes
READY_FILE = "/home/ec2-user/BOOTSTRAP_DONE"


def _wait_for_bootstrap(ip: str) -> None:
    start = time.time()
    deadline = start + TIMEOUT_SECONDS
    while time.time() < deadline:
        elapsed_m = (time.time() - start) / 60
        print(f"Waiting for bootstrap... ({elapsed_m:.0f}m elapsed)")
        try:
            client = ssh_connect(ip, KEY_PATH, SSH_USER)
        except Exception as exc:  # noqa: BLE001 - sshd not up yet is expected
            print(f"  ssh not ready ({type(exc).__name__}); "
                  f"retry in {POLL_SECONDS}s ...")
            time.sleep(POLL_SECONDS)
            continue
        try:
            _in, out, _err = client.exec_command(
                f"test -f {READY_FILE} && echo PRESENT || echo ABSENT")
            present = out.read().decode().strip() == "PRESENT"
        finally:
            client.close()
        if present:
            print("  bootstrap complete (BOOTSTRAP_DONE present).")
            return
        time.sleep(POLL_SECONDS)
    raise TimeoutError(
        f"Timed out after {TIMEOUT_SECONDS // 60} min waiting for "
        f"{READY_FILE}. The user-data install may still be running or failed "
        "— check `sudo cat /var/log/bootstrap.log` on the instance.")


def main() -> None:
    load_dotenv()
    ip = os.environ.get("EC2_PUBLIC_IP")
    bucket = os.environ.get("S3_BUCKET")
    iid = os.environ.get("EC2_INSTANCE_ID")
    if not ip:
        sys.exit("EC2_PUBLIC_IP not in .env. Run launch_ec2.py first.")
    if not bucket:
        sys.exit("S3_BUCKET not in .env.")
    if not KEY_PATH.exists():
        sys.exit(f"SSH key {KEY_PATH} not found. Run launch_ec2.py first.")
    print(f"Target instance {iid or '(id unknown)'} at {ip}.")

    _wait_for_bootstrap(ip)

    # EC2-side .env (printf with explicit newlines).
    write_env = (
        "printf 'USE_S3=true\\nS3_BUCKET=%s\\nAWS_DEFAULT_REGION=%s\\n' > .env"
        % (bucket, REGION)
    )
    steps = [
        ("cd /home/ec2-user && git clone {url} fraud-detection || "
         "(cd fraud-detection && git pull --ff-only)".format(url=REPO_URL),
         "git clone"),
        ("cd /home/ec2-user/fraud-detection && " + write_env,
         "write EC2 .env (USE_S3=true)"),
        ("cd /home/ec2-user/fraud-detection && "
         "pip3.11 install -r requirements.txt --quiet && echo INSTALLED",
         "pip install -r requirements.txt"),
        ("echo DEPLOY COMPLETE", "marker"),
    ]

    client = ssh_connect(ip, KEY_PATH, SSH_USER)
    try:
        for cmd, label in steps:
            rc = ssh_run_stream(client, cmd, label)
            if rc != 0:
                sys.exit(f"Step '{label}' failed with exit code {rc}.")
        # Verify the EC2-side .env exists.
        _in, out, _err = client.exec_command(
            "test -f /home/ec2-user/fraud-detection/.env && echo OK || echo MISSING")
        if out.read().decode().strip() != "OK":
            sys.exit("Verification FAILED: fraud-detection/.env not found on "
                     "the instance.")
    finally:
        client.close()

    print("\nDEPLOYMENT COMPLETE — ready to train")


if __name__ == "__main__":
    main()
