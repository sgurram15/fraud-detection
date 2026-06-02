"""Hyperparameter tuning for the fraud XGBoost model.

Grid search over fraud-relevant params, optimising **recall at the 95%-
precision operating point** -- the real PSP business constraint (catch as
much fraud as possible while only 5% of flagged transactions are legitimate).

Leakage discipline (critical):
  * Features are engineered once; the grid search uses StratifiedKFold(k=5).
  * Within EACH fold, median imputation is fit on the fold-train only and
    SMOTE+undersampling is applied to the fold-train only. The validation
    fold keeps the real fraud distribution and is never resampled or fitted
    on. Resampling the whole set before CV would leak synthetic neighbours
    across folds and inflate the score.
  * The held-out test set (split off before tuning) is used only for the
    final baseline-vs-tuned comparison, at a threshold chosen on validation.

Search mode (set ``SEARCH_MODE`` below):
  * ``"random"``    -- RandomizedSearchCV over RANDOM_N_ITER combos on a
    RANDOM_SAMPLE_N row sample. Directional, < ~30 min on a laptop.
  * ``"full_grid"`` -- exhaustive GridSearchCV (3^5 = 243 combos x 5 folds =
    1,215 fits, ~2-4 h). Intended for the AWS/SageMaker path
    (docs/production_architecture.md).

Both modes drive an imbalanced-learn Pipeline (impute -> SMOTE -> undersample
-> XGB) inside StratifiedKFold(k=5), so resampling/imputation are fit per
fold on the training split only and the validation fold keeps the real fraud
distribution -- the search classes themselves never see leaked data.

Training-data caveat: Baseline and tuned models are compared on the same test
set at the same operating point. This is a fair artifact comparison but not a
controlled ablation -- training sets are not identical. Full controlled
comparison should be run on SageMaker with identical data splits and full
grid search.

``TUNE_SMOKE=1`` forces a tiny, fast configuration to verify the pipeline.
``FRAUD_SAMPLE_N`` overrides the sample size for either mode.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import model_path
from src.features.build_features import build_features
from src.features.handle_imbalance import _xy
from src.models.experiment_tracking import init_tracking, log_results, start_run
from src.models.train_baseline import (
    COST_FN_DEFAULT,
    COST_FP_DEFAULT,
    DAILY_TXNS,
    DEFAULT_SAMPLE_N,
    RANDOM_STATE,
    load_data,
)

logger = logging.getLogger(__name__)

# A8: model root from src/config (USE_S3 refused transitively via the
# train_baseline import, which is local-FS only for load/joblib).
_MODEL_DIR = _ROOT / model_path()
_TUNED_PATH = _MODEL_DIR / "tuned_xgboost.pkl"
_BASELINE_PATH = _MODEL_DIR / "baseline_xgboost.pkl"
_PERF_DIR = _ROOT / "docs" / "model_performance"
_TUNE_JSON = _PERF_DIR / "tuning_results.json"

TARGET_PRECISION = 0.95
TUNE_N_ESTIMATORS = 300
N_SPLITS = 5

# ---- Search configuration -------------------------------------------------
SEARCH_MODE = "random"   # Options: "random" (local), "full_grid" (AWS/SageMaker)
RANDOM_N_ITER = 20       # Number of random combinations to try
RANDOM_SAMPLE_N = 40000  # Row sample for local run

PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],   # higher = less overfit on rare fraud
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}
def recall_at_precision(y_true, y_prob, target: float = TARGET_PRECISION):
    """Highest recall achievable while precision >= target, with its
    threshold. Returns (recall, threshold, precision_at_threshold)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # prec/rec have len = len(thr)+1; the last point (rec=0) has no threshold.
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) == 0:
        return 0.0, 1.0, float(prec[:-1].max() if len(prec) > 1 else 0.0)
    best = ok[np.argmax(rec[:-1][ok])]
    return float(rec[best]), float(thr[best]), float(prec[best])


def _resampler():
    # Combined strategy (mirrors handle_imbalance "combined").
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),
        ("under", RandomUnderSampler(
            sampling_strategy=1.0, random_state=RANDOM_STATE)),
    ])


def _search_pipeline() -> Pipeline:
    """imbalanced-learn Pipeline used as the search estimator.

    Putting impute -> SMOTE -> undersample -> XGB *inside* the estimator means
    sklearn's *SearchCV refits it per CV fold, so imputation/resampling are
    learned on each fold's training split only and the validation fold keeps
    the real fraud distribution. This is the leakage-safe way to use
    RandomizedSearchCV / GridSearchCV with resampling.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),
        ("under", RandomUnderSampler(
            sampling_strategy=1.0, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(
            n_estimators=TUNE_N_ESTIMATORS, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def _recall_at_p95_scorer(estimator, X, y) -> float:
    """Callable scorer: recall at the 95%-precision operating point."""
    proba = estimator.predict_proba(X)[:, 1]
    return recall_at_precision(y, proba)[0]


@contextlib.contextmanager
def _tqdm_joblib(total: int, desc: str):
    """Route joblib batch completions into a tqdm bar (live count + ETA).
    No-op fallback if tqdm/joblib internals are unavailable, so the verbose
    per-fit logs still provide step-by-step progress."""
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, desc=desc, unit="fit",
                   dynamic_ncols=True, mininterval=5.0)
        base = joblib.parallel.BatchCompletionCallBack

        class _Cb(base):
            def __call__(self, *a, **k):
                bar.update(n=self.batch_size)
                return super().__call__(*a, **k)

        joblib.parallel.BatchCompletionCallBack = _Cb
    except Exception:  # noqa: BLE001
        logger.info("Progress bar unavailable; relying on verbose fit logs.")
        yield None
        return
    try:
        yield bar
    finally:
        joblib.parallel.BatchCompletionCallBack = base
        bar.close()


def run_search(X, y, mode: str, n_iter: int):
    """Run RandomizedSearchCV (mode='random') or GridSearchCV
    (mode='full_grid'). Returns (best_params, best_score, results_list).
    Emits a start banner, per-fit logs (verbose=2) and a live progress bar/ETA.
    """
    pipe = _search_pipeline()
    dist = {f"xgb__{k}": v for k, v in PARAM_GRID.items()}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    common = dict(estimator=pipe, scoring=_recall_at_p95_scorer, cv=cv,
                  n_jobs=1, refit=False, error_score=0.0, verbose=2)
    if mode == "full_grid":
        search = GridSearchCV(param_grid=dist, **common)
        n_combos = len(ParameterGrid(dist))
    else:
        search = RandomizedSearchCV(
            param_distributions=dist, n_iter=n_iter,
            random_state=RANDOM_STATE, **common,
        )
        n_combos = n_iter
    total_fits = n_combos * N_SPLITS
    ncpu = os.cpu_count() or 1

    print("=" * 60, flush=True)
    print("HYPERPARAMETER TUNING STARTED", flush=True)
    print(f"  combos x folds : {n_combos} x {N_SPLITS} = {total_fits} fits",
          flush=True)
    print(f"  rows           : {len(X):,}", flush=True)
    print(f"  CPU cores      : {ncpu} (each XGBoost fit uses all cores)",
          flush=True)
    print("  progress       : one '[CV] END' line per fit + live ETA bar",
          flush=True)
    print("=" * 60, flush=True)
    logger.info("TRAINING STARTED: %d fits, %d rows, %d CPU cores",
                total_fits, len(X), ncpu)
    if ncpu < 2:
        logger.warning("Only %d CPU core(s) detected — tuning will be slow.",
                       ncpu)

    t0 = time.perf_counter()
    with _tqdm_joblib(total_fits, "CV fits"):
        search.fit(X, y)
    mins = (time.perf_counter() - t0) / 60.0
    logger.info("Tuning finished: %d fits in %.1f min (avg %.1fs/fit)",
                total_fits, mins, mins * 60 / max(1, total_fits))

    cvres = search.cv_results_
    results = []
    for params, mean, std in zip(cvres["params"],
                                 cvres["mean_test_score"],
                                 cvres["std_test_score"]):
        clean = {k.replace("xgb__", ""): v for k, v in params.items()}
        results.append({
            **clean,
            "mean_recall_at_p95": float(mean),
            "std_recall_at_p95": float(std),
        })
    results.sort(key=lambda r: r["mean_recall_at_p95"], reverse=True)
    best = {k: results[0][k] for k in PARAM_GRID}
    return best, results[0]["mean_recall_at_p95"], results


def _fit_final(params: dict, X_tr, y_tr, X_va, y_va):
    """Refit on train, choose the 95%-precision threshold on the real
    (non-resampled) validation slice."""
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_va_i = imp.transform(X_va)
    X_rs, y_rs = _resampler().fit_resample(X_tr_i, y_tr)
    model = XGBClassifier(
        **params, n_estimators=TUNE_N_ESTIMATORS, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_rs, y_rs)
    _, thr, _ = recall_at_precision(
        y_va, model.predict_proba(X_va_i)[:, 1]
    )
    return model, imp, thr


def _evaluate(model, imp, threshold, X_test, y_test, cost_fn, cost_fp):
    proba = model.predict_proba(imp.transform(X_test))[:, 1]
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    n = len(y_test)
    return {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "daily_loss_gbp": round(
            (fn / n) * DAILY_TXNS * cost_fn
            + (fp / n) * DAILY_TXNS * cost_fp, 2
        ),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _align(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align columns to a saved model's expected feature order if available."""
    try:
        names = model.get_booster().feature_names
    except Exception:
        names = None
    if names:
        return X.reindex(columns=names, fill_value=0)
    return X


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _PERF_DIR.mkdir(parents=True, exist_ok=True)

    smoke = os.environ.get("TUNE_SMOKE", "").strip() in ("1", "true", "yes")
    mode = SEARCH_MODE
    n_iter = RANDOM_N_ITER
    if smoke:  # fast pipeline verification, overrides everything
        mode, n_iter = "random", 4

    env = os.environ.get("FRAUD_SAMPLE_N")
    if env and env.strip().lower() in ("all", "0", "full", "none"):
        sample_n = None
    elif env:
        sample_n = int(env)
    elif smoke:
        sample_n = 8000
    elif mode == "random":
        sample_n = RANDOM_SAMPLE_N
    else:  # full_grid
        sample_n = DEFAULT_SAMPLE_N

    if mode == "random":
        n_combos = min(n_iter, int(np.prod([len(v) for v in
                                            PARAM_GRID.values()])))
        note = (f"LOCAL MODE: Randomised search over {n_iter} combinations. "
                "Full 243-combination grid search available via SageMaker "
                "— see docs/production_architecture.md")
    else:
        n_combos = int(np.prod([len(v) for v in PARAM_GRID.values()]))
        note = ("FULL GRID MODE: Estimated runtime 2-4 hours. Recommended "
                "to run on SageMaker ml.m5.4xlarge or larger.")
    print(note)
    logger.info("Search=%s | %d combos x %d folds = %d fits | sample=%s",
                mode, n_combos, N_SPLITS, n_combos * N_SPLITS,
                "ALL" if sample_n is None else f"{sample_n:,}")

    df = load_data(sample_n)
    enriched = build_features(df)
    X, y = _xy(enriched)
    del df, enriched

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train,
    )

    init_tracking()
    best, best_cv, results = run_search(X_train, y_train, mode, n_iter)

    # Log every evaluated combination to MLflow as a separate run.
    for i, r in enumerate(results, 1):
        params = {k: r[k] for k in PARAM_GRID}
        run = start_run(f"tune-{i:03d}", params)
        mlflow.log_metric("mean_recall_at_p95", r["mean_recall_at_p95"])
        mlflow.log_metric("std_recall_at_p95", r["std_recall_at_p95"])
        mlflow.set_tag("phase", "hyperparameter-tuning")
        mlflow.set_tag("search_mode", mode)
        if mlflow.active_run() and mlflow.active_run().info.run_id == \
                run.info.run_id:
            mlflow.end_run()

    logger.info("Best params: %s (CV recall@P95=%.4f)", best, best_cv)

    cost_fn, cost_fp = COST_FN_DEFAULT, COST_FP_DEFAULT
    tuned_model, tuned_imp, tuned_thr = _fit_final(
        best, X_tr, y_tr, X_val, y_val
    )
    tuned_eval = _evaluate(tuned_model, tuned_imp, tuned_thr,
                           X_test, y_test, cost_fn, cost_fp)
    joblib.dump(tuned_model, _TUNED_PATH)

    # Baseline comparison on the SAME test set at its own 95%-precision
    # threshold (chosen on the same validation slice for a fair head-to-head).
    baseline_eval = None
    if _BASELINE_PATH.exists():
        base_model = joblib.load(_BASELINE_PATH)
        base_imp = SimpleImputer(strategy="median").fit(X_tr)
        _, base_thr, _ = recall_at_precision(
            y_val, base_model.predict_proba(
                _align(pd.DataFrame(base_imp.transform(X_val),
                                    columns=X.columns), base_model)
            )[:, 1]
        )
        baseline_eval = _evaluate(
            base_model, base_imp, base_thr,
            _align(X_test, base_model), y_test, cost_fn, cost_fp
        )

    payload = {
        "search_mode": mode,
        "smoke": smoke,
        "sample_n": sample_n,
        "n_combos_evaluated": len(results),
        "best_params": best,
        "best_cv_recall_at_p95": best_cv,
        "tuned_test": tuned_eval,
        "baseline_test": baseline_eval,
        "all_results": results,
    }
    _TUNE_JSON.write_text(json.dumps(payload, indent=2))

    try:
        run = start_run("tuned-xgboost-best", best)
        mlflow.set_tag("phase", "tuned-final")
        log_results(run, {k: v for k, v in tuned_eval.items()
                          if isinstance(v, (int, float))},
                    tuned_model, list(X.columns))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow final logging failed: %s", exc)

    _print_comparison(baseline_eval, tuned_eval, best)


def _print_comparison(base: dict | None, tuned: dict, best: dict) -> None:
    metrics = ["auc_roc", "auc_pr", "precision", "recall", "f1",
               "false_negative_rate", "false_positive_rate",
               "threshold", "daily_loss_gbp"]
    print("\n" + "=" * 72)
    print(f"BASELINE vs TUNED  (operating point: recall @ "
          f"{TARGET_PRECISION:.0%} precision; same test set)")
    print("=" * 72)
    print(f"Best params: {best}")
    print(f"\n{'Metric':<22}{'Baseline':>14}{'Tuned':>14}{'Delta':>14}")
    print("-" * 64)
    for m in metrics:
        t = tuned[m]
        if base is None:
            print(f"{m:<22}{'n/a':>14}{t:>14.4f}{'':>14}")
            continue
        b = base[m]
        d = t - b
        print(f"{m:<22}{b:>14.4f}{t:>14.4f}{d:>+14.4f}")

    if base is not None:
        saving = base["daily_loss_gbp"] - tuned["daily_loss_gbp"]
        print("\n" + "-" * 64)
        print("DAILY FINANCIAL IMPACT (same assumptions as baseline: "
              f"{DAILY_TXNS:,} txns/day, FN=£{COST_FN_DEFAULT:.0f}, "
              f"FP=£{COST_FP_DEFAULT:.0f})")
        print(f"  Baseline daily loss: £{base['daily_loss_gbp']:,.2f}")
        print(f"  Tuned daily loss:    £{tuned['daily_loss_gbp']:,.2f}")
        verb = "REDUCTION" if saving >= 0 else "INCREASE"
        print(f"  Daily {verb}:       £{abs(saving):,.2f}  "
              f"(~£{abs(saving)*365:,.0f}/year)")
    print("=" * 72)
    print(f"Tuned model -> {_TUNED_PATH}")
    print(f"Results     -> {_TUNE_JSON}")


if __name__ == "__main__":
    main()
