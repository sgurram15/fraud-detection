"""C5.4 — Run the streaming pipeline on EC2 against MSK.

SSHes into the EC2 instance (EC2_PUBLIC_IP from .env), exports the production
environment (USE_S3, S3_BUCKET, KAFKA_BOOTSTRAP_SERVERS, optionally
AWS_BEDROCK_ENABLED), runs ``python src/streaming/run_pipeline.py`` there, and
streams the live dashboard back to the local terminal.

Because KAFKA_BOOTSTRAP_SERVERS is set on the remote, run_pipeline's
create_bus() transparently uses the MSK-backed bus — no code changes, just env.

Prereqs: instance running + bootstrapped; MSK cluster ACTIVE with topics
created; SSH key path in env SSH_KEY (or EC2_KEY path). paramiko required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import (
    get_env,
    ssh_connect,
    ssh_run_stream,
)

REMOTE_DIR = "/home/ec2-user/fraud-detection"


def main() -> int:
    ip = get_env("EC2_PUBLIC_IP")
    bucket = get_env("S3_BUCKET")
    bootstrap = get_env("MSK_BOOTSTRAP_SERVERS")
    key_path = get_env("SSH_KEY") or get_env("EC2_KEY")
    bedrock = get_env("AWS_BEDROCK_ENABLED")

    missing = [n for n, v in (("EC2_PUBLIC_IP", ip), ("S3_BUCKET", bucket),
                              ("MSK_BOOTSTRAP_SERVERS", bootstrap),
                              ("SSH_KEY", key_path)) if not v]
    if missing:
        print(f"Missing required config: {', '.join(missing)} "
              "(set in .env / env). Run setup_msk.py + setup_msk_topics.py "
              "first.", file=sys.stderr)
        return 1

    # Remote environment for the run. Bedrock only if explicitly enabled.
    exports = [
        "export USE_S3=true",
        f"export S3_BUCKET={bucket}",
        f"export KAFKA_BOOTSTRAP_SERVERS={bootstrap}",
    ]
    if bedrock and bedrock.lower() == "true":
        exports.append("export AWS_BEDROCK_ENABLED=true")
    env_prefix = " && ".join(exports)
    command = (f"cd {REMOTE_DIR} && {env_prefix} && "
               "PYTHONIOENCODING=utf-8 python src/streaming/run_pipeline.py")

    print(f"Connecting to ec2-user@{ip} ...")
    client = ssh_connect(ip, Path(key_path).expanduser())
    try:
        rc = ssh_run_stream(client, command, label="run_pipeline.py (MSK)")
    finally:
        client.close()

    print("\n" + "=" * 60)
    if rc == 0:
        print("PIPELINE RUN COMPLETE (MSK)")
    else:
        print(f"PIPELINE RUN FAILED (rc={rc})")
    print("Remember: python scripts/aws/delete_msk.py + stop_ec2.py when done.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
