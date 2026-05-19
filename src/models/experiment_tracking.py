"""MLflow experiment tracking for the PSP fraud-detection models.

Stores runs in a local SQLite tracking backend at ``<repo>/mlruns/mlflow.db``
and groups them under the ``psp-fraud-detection`` experiment.

Public API:
  * ``start_run(run_name, params) -> ActiveRun``
  * ``log_results(run, metrics, model, feature_names)``
  * ``compare_runs()``

Rationale (FCA model-governance): every model that can influence a payment
decision must be reproducible and its performance claims traceable to the
exact params, data and code that produced them -- this module is that lineage.
"""

from __future__ import annotations

import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "psp-fraud-detection"
_ROOT = Path(__file__).resolve().parents[2]
_MLRUNS_DIR = _ROOT / "mlruns"
_DB_PATH = _MLRUNS_DIR / "mlflow.db"
_ARTIFACT_DIR = _MLRUNS_DIR / "artifacts"
_TRACKED_LIBS = ("mlflow", "scikit-learn", "xgboost", "pandas", "numpy",
                  "shap", "imbalanced-learn")

_AUC_CANDIDATES = ("auc_roc", "roc_auc", "auc", "roc_auc_score")

# MLflow 3.x deprecates the filesystem tracking store; the recommended local
# backend is a database. We use SQLite at mlruns/mlflow.db. In the Version 2
# production architecture (see docs/production_architecture.md) this local
# store is replaced by Amazon SageMaker Experiments, which provides managed
# experiment tracking, automatic model versioning, and SageMaker Pipelines
# integration for automated retraining.
_TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def _tracking_uri() -> str:
    # Create mlruns/ before MLflow connects, else SQLite cannot create the DB.
    _MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return _TRACKING_URI


def init_tracking() -> str:
    """Point MLflow at the local SQLite store and ensure the experiment
    exists. Idempotent -- safe to call repeatedly."""
    mlflow.set_tracking_uri(_tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        # Pin a deterministic local artifact root (SQLite stores only
        # metadata; artifacts still need a filesystem location).
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=_ARTIFACT_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.get_tracking_uri()


def _system_info() -> dict[str, str]:
    info = {
        "sys.python_version": platform.python_version(),
        "sys.platform": platform.platform(),
        "sys.timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for lib in _TRACKED_LIBS:
        try:
            info[f"lib.{lib}"] = version(lib)
        except PackageNotFoundError:
            info[f"lib.{lib}"] = "not-installed"
    return info


def start_run(run_name: str, params: dict):
    """Start an MLflow run, log hyperparameters + system info, return the run.

    The run is left *open* so the caller can train; pass the returned object
    to ``log_results`` which logs outputs and closes it.
    """
    init_tracking()
    active = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    mlflow.set_tags(_system_info())
    mlflow.set_tag("run_name", run_name)
    return active


def _feature_importances(model, feature_names) -> pd.DataFrame | None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        booster = getattr(model, "get_booster", None)
        if booster is not None:
            score = model.get_booster().get_score(importance_type="gain")
            imp = [score.get(f, 0.0) for f in feature_names]
        else:
            return None
    n = min(len(feature_names), len(imp))
    return (
        pd.DataFrame(
            {"feature": list(feature_names)[:n], "importance": list(imp)[:n]}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _log_model_artifact(client: MlflowClient, run_id: str, model) -> None:
    """Save the model in the best available MLflow flavour and attach it.

    Uses a temp dir + log_artifacts so it works regardless of which run is
    currently active.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mdir = Path(tmp) / "model"
        try:
            import mlflow.xgboost

            mlflow.xgboost.save_model(model, str(mdir))
        except Exception:
            try:
                import mlflow.sklearn

                mlflow.sklearn.save_model(model, str(mdir))
            except Exception:
                import joblib

                mdir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, mdir / "model.joblib")
        client.log_artifacts(run_id, str(mdir), artifact_path="model")


def log_results(run, metrics: dict, model, feature_names) -> None:
    """Log metrics, the trained model, feature importance, and tag the run."""
    client = MlflowClient()
    run_id = run.info.run_id

    for key, val in (metrics or {}).items():
        try:
            client.log_metric(run_id, key, float(val))
        except (TypeError, ValueError):
            client.set_tag(run_id, f"metric_skipped.{key}", str(val))

    _log_model_artifact(client, run_id, model)

    fi = _feature_importances(model, feature_names)
    if fi is not None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "feature_importance.csv"
            fi.to_csv(p, index=False)
            client.log_artifact(run_id, str(p),
                                 artifact_path="feature_importance")

    client.set_tag(run_id, "training-data", "IEEE-CIS")
    client.set_tag(run_id, "model-type", "XGBoost")

    # Close the run if it is the active one.
    act = mlflow.active_run()
    if act is not None and act.info.run_id == run_id:
        mlflow.end_run()


def _find_auc_column(cols) -> str | None:
    for cand in _AUC_CANDIDATES:
        if f"metrics.{cand}" in cols:
            return f"metrics.{cand}"
    for c in cols:  # any metric mentioning auc
        if c.startswith("metrics.") and "auc" in c.lower():
            return c
    return None


def compare_runs() -> pd.DataFrame:
    """Print all runs (params + metrics side by side); highlight best AUC-ROC."""
    init_tracking()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    if runs.empty:
        print(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
        return runs

    id_cols = ["run_id", "tags.run_name"]
    id_cols = [c for c in id_cols if c in runs.columns]
    param_cols = sorted(c for c in runs.columns if c.startswith("params."))
    metric_cols = sorted(c for c in runs.columns if c.startswith("metrics."))
    table = runs[id_cols + param_cols + metric_cols].copy()
    if "run_id" in table:
        table["run_id"] = table["run_id"].str.slice(0, 8)

    auc_col = _find_auc_column(runs.columns)
    best_idx = None
    if auc_col is not None and runs[auc_col].notna().any():
        best_idx = runs[auc_col].astype(float).idxmax()
        table["best"] = ""
        table.loc[best_idx, "best"] = "  <== BEST AUC-ROC"

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("=" * 80)
    print(f"MLflow runs - experiment '{EXPERIMENT_NAME}'  ({len(table)} runs)")
    print("=" * 80)
    print(table.to_string(index=False))
    if best_idx is not None:
        rid = runs.loc[best_idx, "run_id"][:8]
        print(f"\nBest run: {rid}  {auc_col} = "
              f"{float(runs.loc[best_idx, auc_col]):.5f}")
    else:
        print("\n(No AUC-ROC metric logged yet; cannot rank by AUC.)")
    return table


if __name__ == "__main__":
    # Verification: init store, create experiment, do a tiny real run,
    # then compare. Proves MLflow initialises without errors end-to-end.
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    print("Tracking URI:", init_tracking())
    X, y = make_classification(n_samples=400, n_features=8, weights=[0.96],
                               random_state=42)
    fnames = [f"f{i}" for i in range(X.shape[1])]

    try:
        from xgboost import XGBClassifier

        mdl = XGBClassifier(n_estimators=30, max_depth=3, eval_metric="auc")
        params = {"model": "xgboost", "n_estimators": 30, "max_depth": 3}
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(n_estimators=30, random_state=42)
        params = {"model": "random_forest", "n_estimators": 30}

    run = start_run("verification-smoke", params)
    mdl.fit(X, y)
    auc = roc_auc_score(y, mdl.predict_proba(X)[:, 1])
    log_results(run, {"auc_roc": auc, "n_samples": len(y)}, mdl, fnames)
    print(f"Logged smoke run (auc_roc={auc:.4f}).")
    compare_runs()
    print("\nMLflow initialised and round-tripped without errors.")
