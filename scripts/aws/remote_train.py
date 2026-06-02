"""On-instance training runner with full observability (runs ON the EC2 box).

Launched DETACHED (nohup/setsid) by run_training_detached.py so it survives
SSH/laptop disconnects. It runs the full pipeline on the FULL dataset and gives
real observability:

* All step stdout/stderr -> training.log (one file, append).
* A background shipper uploads, every ~30s, to s3://<bucket>/logs/:
    - training.log    (full output so far)
    - status.json     (status, current step, elapsed, last resource snapshot)
    - metrics.log     (free -m + df -h history -> spot OOM / disk-full)
* On step failure it captures `sudo dmesg` (OOM-killer evidence) into the log
  and flags "OOM DETECTED" if found.
* On success it uploads model + metrics + predictions to S3.

So from anywhere (even after the box is gone) you can read exactly what
happened: `aws s3 cp s3://<bucket>/logs/training.log -`.

Env: S3_BUCKET (required), FRAUD_SAMPLE_N (default 'all'). Run with python3.11.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time

import boto3

ROOT = "/home/ec2-user/fraud-detection"
REGION = "eu-west-2"
BUCKET = os.environ.get("S3_BUCKET", "")
SAMPLE = os.environ.get("FRAUD_SAMPLE_N", "all")
LOG = os.path.join(ROOT, "training.log")
METRICS = os.path.join(ROOT, "metrics.log")
SHIP_EVERY_S = 30

# Step 0 downloads the raw CSVs from S3 to the box (the pipeline reads the
# local filesystem; the git clone has no data). boto3 + instance-profile creds,
# so no AWS CLI / keys needed. Passed as a list -> no shell quoting.
_SYNC_SCRIPT = (
    "import boto3, os;"
    "b=os.environ['S3_BUCKET'];"
    "s3=boto3.client('s3', region_name='eu-west-2');"
    "ks=[o['Key'] for o in s3.list_objects_v2(Bucket=b, Prefix='data/raw/')"
    ".get('Contents', []) if not o['Key'].endswith('/')];"
    "[os.makedirs(os.path.dirname(k) or '.', exist_ok=True) for k in ks];"
    "[s3.download_file(b, k, k) for k in ks];"
    "print('synced', len(ks), 'files from s3://'+b+'/data/raw/')"
)

STEPS = [
    ("sync_data", ["python3.11", "-c", _SYNC_SCRIPT]),
    ("build_features", ["python3.11", "src/features/build_features.py"]),
    ("handle_imbalance", ["python3.11", "src/features/handle_imbalance.py"]),
    ("train_baseline", ["python3.11", "src/models/train_baseline.py"]),
    ("tune_model", ["python3.11", "src/models/tune_model.py"]),
    ("validate_model", ["python3.11", "src/models/validate_model.py"]),
    ("predict_test", ["python3.11", "src/models/predict_test.py"]),
]

# local repo-relative path -> S3 key (matches the bucket's existing layout)
ARTIFACTS = {
    "src/models/saved/baseline_xgboost.pkl": "models/saved/baseline_xgboost.pkl",
    "src/models/saved/tuned_xgboost.pkl": "models/saved/tuned_xgboost.pkl",
    "docs/model_performance/validation_report.json":
        "docs/model_performance/validation_report.json",
    "docs/model_performance/baseline_metrics.json":
        "docs/model_performance/baseline_metrics.json",
    "docs/model_performance/tuning_results.json":
        "docs/model_performance/tuning_results.json",
    "data/processed/test_predictions.csv":
        "data/processed/test_predictions.csv",
}

# Which artifacts each step produces -> uploaded to S3 the moment that step
# finishes (checkpointing), so partial results survive a later-step failure.
STEP_ARTIFACTS = {
    "train_baseline": [
        ("src/models/saved/baseline_xgboost.pkl",
         "models/saved/baseline_xgboost.pkl"),
        ("docs/model_performance/baseline_metrics.json",
         "docs/model_performance/baseline_metrics.json"),
    ],
    "tune_model": [
        ("src/models/saved/tuned_xgboost.pkl",
         "models/saved/tuned_xgboost.pkl"),
        ("docs/model_performance/tuning_results.json",
         "docs/model_performance/tuning_results.json"),
    ],
    "validate_model": [
        ("docs/model_performance/validation_report.json",
         "docs/model_performance/validation_report.json"),
    ],
    "predict_test": [
        ("data/processed/test_predictions.csv",
         "data/processed/test_predictions.csv"),
    ],
}

s3 = boto3.client("s3", region_name=REGION)
_stop = threading.Event()
state = {
    "status": "STARTING", "step": None,
    "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "updated_utc": None, "host": socket.gethostname(),
    "sample": SAMPLE, "last_resources": None,
}


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def log(msg: str) -> None:
    line = f"[{_now()}] {msg}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def _resources() -> str:
    out = []
    for cmd in (["free", "-m"], ["df", "-h", "/"]):
        try:
            out.append(subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=10).stdout.strip())
        except Exception as exc:  # noqa: BLE001
            out.append(f"({cmd[0]} err: {exc})")
    return "\n".join(out)


def _ship_once() -> None:
    state["updated_utc"] = _now()
    try:
        s3.upload_file(LOG, BUCKET, "logs/training.log")
    except Exception:  # noqa: BLE001 - log may not exist yet
        pass
    try:
        s3.upload_file(METRICS, BUCKET, "logs/metrics.log")
    except Exception:  # noqa: BLE001
        pass
    try:
        s3.put_object(Bucket=BUCKET, Key="logs/status.json",
                      Body=json.dumps(state, indent=2).encode())
    except Exception:  # noqa: BLE001
        pass


def _shipper() -> None:
    while not _stop.is_set():
        res = _resources()
        state["last_resources"] = res.splitlines()[1] if res else None
        with open(METRICS, "a", encoding="utf-8") as f:
            f.write(f"\n[{_now()}]\n{res}\n")
        _ship_once()
        _stop.wait(SHIP_EVERY_S)


def _capture_dmesg() -> None:
    try:
        dm = subprocess.run(["sudo", "dmesg"], capture_output=True,
                            text=True, timeout=20).stdout
        log("[DMESG tail]\n" + "\n".join(dm.splitlines()[-40:]))
        low = dm.lower()
        if ("out of memory" in low or "oom-kill" in low
                or "killed process" in low):
            log("*** OOM DETECTED in dmesg — the run was killed for memory. "
                "Use a larger instance (more RAM) or reduce the workload. ***")
            state["likely_cause"] = "OOM"
    except Exception as exc:  # noqa: BLE001
        log(f"(dmesg unavailable: {exc})")


def _upload_artifacts(items) -> list:
    """Upload (rel_path, s3_key) pairs that exist on disk. Idempotent."""
    done = []
    for rel, key in items:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            try:
                s3.upload_file(path, BUCKET, key)
                done.append(key)
                log(f"  uploaded s3://{BUCKET}/{key}")
            except Exception as exc:  # noqa: BLE001
                log(f"  UPLOAD FAILED {key}: {exc!r}")
    return done


def _maybe_terminate() -> None:
    """If SELF_TERMINATE=1, schedule an OS shutdown (the launcher sets the
    instance's shutdown-behavior to 'terminate') with a short grace for the
    final S3 uploads. Makes the detached run fully hands-off: it stops its own
    bill on completion OR failure."""
    if os.environ.get("SELF_TERMINATE", "").strip().lower() not in (
            "1", "true", "yes"):
        return
    log("SELF_TERMINATE=1 -> scheduling shutdown in 3 min "
        "(terminates the instance; grace for final uploads).")
    try:
        subprocess.run(["sudo", "shutdown", "-h", "+3"], timeout=15)
    except Exception as exc:  # noqa: BLE001
        log(f"(self-terminate failed: {exc})")


def main() -> None:
    if not BUCKET:
        print("S3_BUCKET env var required", file=sys.stderr)
        sys.exit(2)
    os.chdir(ROOT)
    open(LOG, "w").close()
    open(METRICS, "w").close()

    env = dict(os.environ)
    env["FRAUD_SAMPLE_N"] = SAMPLE
    env["PYTHONUNBUFFERED"] = "1"

    threading.Thread(target=_shipper, daemon=True).start()
    log(f"=== TRAINING RUN START (sample={SAMPLE}, host={state['host']}) ===")
    log("initial resources:\n" + _resources())

    for name, cmd in STEPS:
        state["status"], state["step"] = "RUNNING", name
        _ship_once()
        log(f"--- STEP START: {name} :: {' '.join(cmd)} ---")
        t0 = time.time()
        with open(LOG, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=env, cwd=ROOT)
            rc = proc.wait()
        mins = (time.time() - t0) / 60.0
        if rc != 0:
            log(f"--- STEP FAILED: {name} rc={rc} after {mins:.1f} min ---")
            _capture_dmesg()
            state.update(status="FAILED", failed_step=name, rc=rc)
            _stop.set()
            _ship_once()
            _maybe_terminate()
            sys.exit(rc)
        log(f"--- STEP DONE: {name} in {mins:.1f} min ---")
        # Checkpoint: push this step's artifacts to S3 immediately so partial
        # results survive even if a later step fails.
        step_arts = STEP_ARTIFACTS.get(name)
        if step_arts:
            log(f"checkpoint: uploading {name} artifacts ...")
            _upload_artifacts(step_arts)
            _ship_once()

    log("=== final artifact sweep to S3 ===")
    uploaded = _upload_artifacts(list(ARTIFACTS.items()))

    state.update(status="DONE", step=None, uploaded=uploaded)
    log("=== TRAINING RUN COMPLETE ===")
    _stop.set()
    _ship_once()
    _maybe_terminate()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: {exc!r}")
        state["status"] = "ERROR"
        _capture_dmesg()
        _stop.set()
        _ship_once()
        _maybe_terminate()
        raise
