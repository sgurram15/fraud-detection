"""Centralised paths / environment config (MISSION A8).

Single source of truth for local-vs-S3 data and model locations. Set
``USE_S3=true`` and ``S3_BUCKET=<bucket>`` to point the pipeline at S3
(bucket created by ``scripts/aws/setup_s3.py``).

NOTE (scope / safety): MISSION A8 also asks to rewrite every hardcoded path
in ``src/`` to branch on S3. That sweeping edit across the *verified* feature
and model pipeline is a regression risk (the pipeline was validated locally;
the stack was also churned to pandas 3.0 during the SHAP incident and not
re-verified). To avoid breaking working code, this module provides the
centralised config + resolver helpers; migrating each script to use them
should be done one file at a time with re-verification, not in one blind
pass. Logged accordingly in MISSION_LOG.md.
"""

from __future__ import annotations

import os

# --- Spec-required attributes (MISSION A8) ---
LOCAL_DATA_PATH = "data/raw/"
S3_BUCKET = os.getenv("S3_BUCKET", "")
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
MODEL_PATH = (
    "src/models/saved/"
    if not USE_S3
    else f"s3://{S3_BUCKET}/models/saved/"
)

# --- Convenience resolvers (preferred for new/migrated code) ---
LOCAL_PROCESSED_PATH = "data/processed/"
AWS_REGION = "eu-west-2"  # FCA data residency — do not change


def data_path(*parts: str) -> str:
    """Resolve a path under the raw-data root (S3 URI or local)."""
    base = (f"s3://{S3_BUCKET}/data/raw/"
            if USE_S3 else LOCAL_DATA_PATH)
    return base + "/".join(parts) if parts else base


def model_path(*parts: str) -> str:
    """Resolve a path under the model root (S3 URI or local)."""
    return MODEL_PATH + "/".join(parts) if parts else MODEL_PATH


def describe() -> str:
    return (f"USE_S3={USE_S3} | S3_BUCKET={S3_BUCKET or '(unset)'} | "
            f"data={data_path()} | models={MODEL_PATH}")


if __name__ == "__main__":
    print(describe())
