"""Shared helpers for the AWS scripts.

All AWS resources MUST be in eu-west-2 (FCA data residency) and tagged
Project=fraud-detection-poc. boto3 is imported lazily with a clear message so
the scripts can be created/inspected without AWS deps installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REGION = "eu-west-2"  # London — FCA data residency. Do not change.
PROJECT_TAG = {"Key": "Project", "Value": "fraud-detection-poc"}
BUCKET_PREFIX = "fraud-detection-poc-"

# Repo-root .env (gitignored). Single source of truth for the AWS scripts'
# runtime config (S3_BUCKET, EC2_INSTANCE_ID, EC2_PUBLIC_IP, optional creds).
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


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


def get_paramiko():
    try:
        import paramiko  # noqa: F401

        return paramiko
    except ImportError:
        print(
            "paramiko is not installed. Install with:\n"
            "    pip install paramiko",
            file=sys.stderr,
        )
        sys.exit(1)


def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def read_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a KEY=VALUE .env file into a dict (quotes stripped). Returns {}
    if the file is absent."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def load_dotenv_into_environ(path: Path = ENV_PATH) -> None:
    """Populate os.environ from .env WITHOUT overwriting existing process env
    (so real shell/AWS env vars and ~/.aws still win)."""
    for k, v in read_dotenv(path).items():
        os.environ.setdefault(k, v)


def get_env(key: str, default: str | None = None) -> str | None:
    """Process env first, then .env file."""
    if key in os.environ:
        return os.environ[key]
    return read_dotenv().get(key, default)


def ssh_connect(ip: str, key_path: Path, user: str = "ec2-user",
                timeout: int = 15):
    """Open a paramiko SSH client to ip using key_path. AutoAddPolicy: we
    trust an instance we just launched (for a hardened setup, pin the host
    key from the console fingerprint instead)."""
    paramiko = get_paramiko()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=ip, username=user,
                   key_filename=str(key_path), timeout=timeout)
    return client


def ssh_run_stream(client, command: str, label: str | None = None) -> int:
    """Run a remote command, streaming combined stdout/stderr live to the
    local terminal. Returns the remote exit code."""
    if label:
        print(f"\n$ {label}")
    _stdin, stdout, _stderr = client.exec_command(command, get_pty=True)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    return stdout.channel.recv_exit_status()


def append_dotenv(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Update-or-append each KEY=VALUE in .env, preserving other lines.
    Creates the file if absent."""
    lines = (path.read_text(encoding="utf-8").splitlines()
             if path.exists() else [])
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.partition("=")[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
