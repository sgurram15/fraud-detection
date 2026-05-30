"""C4.2 — Deploy the fraud model as a SageMaker real-time endpoint.

Packages the production model + feature-store state + scoring handlers, uploads
them to S3, and creates a SageMaker model -> endpoint config -> endpoint
(ml.m5.large, 1 instance), waits for InService, smoke-tests it, and records the
endpoint name in .env as SAGEMAKER_ENDPOINT.

PRECONDITION — STOP POINT C4 (the human must have confirmed):
  1. AWS credentials are configured (aws configure / env / ~/.aws)
  2. A SageMaker execution role exists (set SAGEMAKER_ROLE_ARN in .env)
  3. S3_BUCKET in .env is correct

COST: ml.m5.large ≈ £0.13/hour while the endpoint exists, billed whether or not
it receives traffic. RUN scripts/aws/delete_endpoint.py WHEN DONE.

This script makes no AWS calls until you confirm at the prompt (or pass --yes).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import (
    PROJECT_TAG,
    REGION,
    REPO_ROOT,
    append_dotenv,
    confirm,
    get_boto3,
    get_env,
    load_dotenv_into_environ,
)

load_dotenv_into_environ()  # make .env AWS creds available to boto3

ENDPOINT_NAME = "fraud-scoring-endpoint"
INSTANCE_TYPE = "ml.m5.large"
S3_MODEL_KEY = "models/sagemaker/model.tar.gz"

# Artifacts packaged into model.tar.gz (loaded by sagemaker_scoring.model_fn).
_MODEL_PKL = REPO_ROOT / "src" / "models" / "saved" / "baseline_xgboost.pkl"
_STATE_PKL = (REPO_ROOT / "data" / "processed" / "feature_store"
              / "online_state.pkl")
_METRICS = REPO_ROOT / "docs" / "model_performance" / "baseline_metrics.json"

# Source files (preserving the src/ package layout) the inference container
# needs on its path so sagemaker_scoring's `from src...` imports resolve.
_SOURCE_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/api/__init__.py",
    "src/api/sagemaker_scoring.py",
    "src/features/__init__.py",
    "src/features/feature_store.py",
    "src/features/build_features.py",
    "src/models/__init__.py",
    "src/models/explain.py",
]


def _get_sagemaker():
    try:
        import sagemaker  # noqa: F401

        return sagemaker
    except ImportError:
        print("The SageMaker SDK is not installed. Install with:\n"
              "    pip install sagemaker\n", file=sys.stderr)
        sys.exit(1)


def _build_model_tarball(dest: Path) -> Path:
    """Bundle the model artifacts into model.tar.gz at the archive root."""
    for f in (_MODEL_PKL, _STATE_PKL, _METRICS):
        if not f.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {f}. Train the model / build the "
                "feature store first.")
    tar_path = dest / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in (_MODEL_PKL, _STATE_PKL, _METRICS):
            tar.add(f, arcname=f.name)
    return tar_path


def _build_source_dir(dest: Path) -> Path:
    """Copy the minimal src/ files needed by the handler, preserving layout."""
    src_root = dest / "code"
    for rel in _SOURCE_FILES:
        src = REPO_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"source file missing: {src}")
        out = src_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out)
    return src_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive cost confirmation")
    args = parser.parse_args()

    bucket = get_env("S3_BUCKET")
    role = get_env("SAGEMAKER_ROLE_ARN")
    if not bucket:
        print("S3_BUCKET is not set in .env (STOP POINT C4.3).", file=sys.stderr)
        return 1
    if not role:
        print("SAGEMAKER_ROLE_ARN is not set in .env. Create a SageMaker "
              "execution role and add its ARN.", file=sys.stderr)
        return 1

    print(f"About to create SageMaker endpoint '{ENDPOINT_NAME}'")
    print(f"  region:   {REGION}")
    print(f"  bucket:   s3://{bucket}/{S3_MODEL_KEY}")
    print(f"  instance: {INSTANCE_TYPE}  (≈ £0.13/hour while it exists)")
    print("  REMEMBER TO DELETE THE ENDPOINT WHEN DONE "
          "(scripts/aws/delete_endpoint.py)")
    if not args.yes and not confirm("Proceed and incur cost?"):
        print("Aborted — no AWS resources created.")
        return 0

    sagemaker = _get_sagemaker()
    boto3 = get_boto3()
    from sagemaker.sklearn.model import SKLearnModel

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_path = _build_model_tarball(tmp_path)
        source_dir = _build_source_dir(tmp_path)

        model_data = session.upload_data(
            str(tar_path), bucket=bucket,
            key_prefix=S3_MODEL_KEY.rsplit("/", 1)[0])
        print(f"Uploaded model artifacts -> {model_data}")

        model = SKLearnModel(
            model_data=model_data,
            role=role,
            entry_point="src/api/sagemaker_scoring.py",
            source_dir=str(source_dir),
            framework_version="1.2-1",
            py_version="py3",
            sagemaker_session=session,
            name=ENDPOINT_NAME,
        )
        print("Deploying endpoint (this can take several minutes) ...")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=ENDPOINT_NAME,
            tags=[PROJECT_TAG],
        )

    # Tag/verify InService and smoke-test.
    sm = boto3.client("sagemaker", region_name=REGION)
    state = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"Endpoint status: {state}")

    sample = {
        "transaction_id": "TXN-DEPLOY-TEST", "card_id": "CARD-TEST",
        "amount": 1500.0, "device_type": "mobile", "hour_of_day": 3,
        "day_of_week": 6, "destination_account_age_days": 2,
        "tx_velocity_1h": 5, "tx_velocity_24h": 12, "amt_deviation": 5.0,
        "is_late_night": True, "is_weekend": True, "card_age_days": 1,
    }
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    result = predictor.predict(sample)
    print("Smoke test response:")
    print(json.dumps(result, indent=2)[:600])

    append_dotenv({"SAGEMAKER_ENDPOINT": ENDPOINT_NAME})
    print(f"\nENDPOINT LIVE — {ENDPOINT_NAME} (saved to .env as "
          "SAGEMAKER_ENDPOINT)")
    print("REMEMBER TO DELETE ENDPOINT WHEN DONE — "
          "python scripts/aws/delete_endpoint.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
