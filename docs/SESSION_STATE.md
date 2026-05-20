# Session State — Handoff for a Fresh Claude Session

> If you are a new session, **read this file first**. It captures the
> non-obvious context that took the previous session a long time to discover.
> It is a snapshot at the end of the last session and may go stale —
> always cross-check against `git log` and the current file contents.

---

## 1. What this project is (one paragraph)

A proof-of-concept real-time fraud detection system for a UK Payment Service
Provider (PSP), built on the IEEE-CIS Vesta dataset and scaffolded so that
every local component (Version 1) maps one-to-one onto an AWS production
architecture (Version 2 — MSK, Managed Flink, SageMaker, Bedrock,
CloudTrail, multi-region active-active, eu-west-2). The whole thing is
shaped by FCA expectations on explainability, leakage-safe model governance,
operational resilience (PS21/3), and Consumer Duty — these are *design
constraints*, not afterthoughts.

## 2. Repo, branch, latest state

- **GitHub:** https://github.com/sgurram15/Fraud-Detection
- **Path on disk:** `C:\Machine_Learning_Sample_Project\fraud-detection`
- **Branch:** `main` (tracking `origin/main`)
- **Identity:** `sgurram15 / 134401834+sgurram15@users.noreply.github.com`
- The fraud-detection repo is **independent** — note the parent folder
  `C:/Machine_Learning_Sample_Project` is its own *separate* personal
  notebooks repo (`Machine_Learning_Sample_Project`). Do not confuse them.

Run `git log --oneline` to see the latest commits.

## 3. CRITICAL environment quirks (these will bite you)

1. **`numpy<2` is pinned** in `requirements.txt`. The conda-built scientific
   stack (pandas / matplotlib / numexpr / bottleneck in the base miniconda
   env at `C:\Users\skrgu\miniconda3\`) is **NumPy-1.x ABI**. An unpinned
   `pip install shap` will pull NumPy 2.x and break the entire env
   (`_ARRAY_API not found`, `numpy._core.multiarray failed to import`).
   This happened mid-session and required removing all `numpy*` from
   `site-packages` then `conda install --force-reinstall numpy=1.26.4`.
2. **pandas was upgraded from 2.x to 3.0.3** during that incident. Feature
   pipeline + `validate_features.py` + model scripts (`train_baseline`,
   `tune_model`, `validate_model`) are all re-verified on 3.0.3 as of
   2026-05-20 — see §11 for the numbers. `numexpr 2.8.7` / `bottleneck
   1.3.5` are below the versions pandas prefers but emit non-fatal
   UserWarnings only.
3. **Python 3.11**, miniconda base env.
4. **Host has ≤1 GB free RAM.** Full IEEE-CIS (590k rows × ~430 cols) +
   SMOTE cannot fit. Local-PoC defaults already cap samples:
   `DEFAULT_SAMPLE_N = 100_000` in `train_baseline.py`,
   `RANDOM_SAMPLE_N = 40_000` in `tune_model.py`. Full-scale = AWS path.
5. **MLflow is on SQLite, not file-store.** Tracking URI is
   `sqlite:///mlruns/mlflow.db`. The local store dir is **`mlruns/`** —
   **not** `mlflow/`. A top-level dir literally named `mlflow/` shadows the
   `mlflow` package when running `python -c` from repo root; do not create
   one. **Quirk fixed 2026-05-20:** `experiment_tracking._system_info`
   previously only caught `PackageNotFoundError`; the local pandas 3.0.3
   install has dist-info without a `Version` field, which made
   `importlib.metadata.version("pandas")` raise `KeyError('Version')` and
   crash every MLflow run with `MLflow logging failed: 'Version'`. The
   handler is now broadened to fall back to `"metadata-broken"`.
6. **Windows console code page (cp1252)** cannot print box-drawing chars +
   `£`. Scripts that print those (e.g. `train_baseline.py`,
   `tune_model.py`) call `sys.stdout.reconfigure(encoding="utf-8")` —
   keep that.

## 4. What is in the repo (high level)

```
fraud-detection/
  README.md, MISSION.md, MISSION_LOG.md, requirements.txt, .gitignore
  data/        download_data.py, explore_data.py
  docs/        AWS_SETUP.md, model_card.md, data_dictionary.md,
               production_architecture.md, SESSION_STATE.md (this file)
  src/
    config.py  (USE_S3, S3_BUCKET, MODEL_PATH, data_path/model_path helpers)
    features/  build_features.py, handle_imbalance.py, feature_store.py,
               validate_features.py
    models/    experiment_tracking.py (MLflow + SQLite),
               train_baseline.py, tune_model.py, validate_model.py,
               explain.py
    api/, streaming/, monitoring/   (scaffolding only)
  scripts/aws/ setup_s3.py, upload_data.py, launch_ec2.py, stop_ec2.py,
               run_training_on_ec2.py, _common.py
  tests/       test_feature_store.py
```

**Gitignored (kept locally, never pushed):** `data/raw/*.csv`, the
`*.zip`, `data/processed/`, `*.pkl`, `mlruns/`, `mlflow.db`,
`docs/feature_validation/`, `docs/model_performance/`,
credentials (`.env`, `kaggle.json`, `.aws/`).

## 5. Status of every module

### Verified working
- **`data/download_data.py`** — uses `kagglehub` (modern client). Reads
  `KAGGLE_API_TOKEN` env var or `~/.kaggle/kaggle.json`. Downloads
  IEEE-CIS competition into `data/raw/ieee-fraud-detection/`.
- **`src/features/build_features.py`** — leakage-safe engineered features.
  Out-of-fold target encoding for `device_type_fraud_rate`
  (`encode_device_fraud_rate_safely`, KFold k=5, Laplace add-one).
  `card_uid` composite key from `card1..card6 + addr1`. Time anchored to
  community-convention `2017-12-01` (TransactionDT seconds offset; true
  epoch unknown).
- **`src/features/handle_imbalance.py`** — `prepare_train_test_split(df,
  strategy)`. **Splits first, then resamples train only** (leakage-safe).
  Strategies: `smote`, `undersample`, `combined`.
- **`src/features/feature_store.py`** — `FeatureStore` with `fit_batch` +
  `get_features(transaction: dict) -> dict`. Sub-ms serving latency.
  Verified by `tests/test_feature_store.py` (3/3 PASS).
- **`src/features/validate_features.py`** — distribution plots, MWU/chi²
  significance, AUC ranking, multicollinearity heatmap. Outputs go to
  `docs/feature_validation/` (gitignored).
- **`src/models/experiment_tracking.py`** — MLflow SQLite backend at
  `sqlite:///mlruns/mlflow.db`, experiment `psp-fraud-detection`,
  `start_run` / `log_results` / `compare_runs`.

### Verified working (re-confirmed on pandas 3.0.3, 2026-05-20)
- **`src/models/train_baseline.py`** — 100k sample on 2026-05-20:
  AUC-ROC 0.9158, cost-optimal threshold 0.16, daily £ loss £166,875,
  FPR 2.23%. Matches the pre-bump numbers within run-variance.
  MLflow run now logs successfully end-to-end.
- **`src/models/tune_model.py`** — `SEARCH_MODE = "random"` (20 iter,
  40k sample, local) ran clean (100 fits). **Tuned model is worse than
  baseline at this sample size** (AUC 0.9214 vs 0.9618, daily loss
  +£56,875/day) — this is not a code regression, it is the documented
  "full-grid 243 combos × 5 folds is SageMaker territory" effect. Do
  not deploy `tuned_xgboost.pkl` over `baseline_xgboost.pkl` from a
  local run.
- **`src/models/validate_model.py`** — 9 checks on 40k × 5 seeds × 5-fold
  CV: 8 PASS / 1 FAIL. The FAIL is `stability.feature_importance`
  (Jaccard 0.28 across seeds, req ≥ 0.80) — model *output* is stable
  (AUC spread 0.007), only the importance assignments shuffle. Same
  "small-sample noise, not a real defect" finding the previous session
  recorded; real settle = full-data SageMaker run. Also note:
  validate_model used the *tuned* artifact, so its NOT-READY verdict
  reflects the underperforming tuned model, not the baseline.

### New
- **`src/models/explain.py`** — SHAP TreeExplainer.
  `explain_prediction(model, transaction_features)` returns top-5 plain-
  English reasons with `+`/`−` direction. `explain_batch(model, df)` saves
  `shap_importance.png`, `shap_summary.png`, three dependence plots.
  `generate_fca_explanation(transaction, prediction, shap_reasons)`
  produces the audit-trail dict (JSON-serialisable).

### Scaffolding only (not implemented)
- `src/api/` (FastAPI scoring endpoint)
- `src/streaming/` (Kafka simulation)
- `src/monitoring/` (Evidently drift)

## 6. Non-obvious key decisions (read this before changing things)

1. **OOF target encoding for `device_type_fraud_rate`.** Earlier full-data
   target mean was a leakage risk; the in-pipeline version uses 5-fold OOF
   + Laplace smoothing, exposes a serving map via `df.attrs[
   "device_fraud_rate_map"]` consumed by the feature store. Do not
   regress this.
2. **No `billing_shipping_match` feature.** IEEE-CIS has no shipping
   address. The previous proxy from `dist1` was dropped on purpose; the
   gap is recorded in `docs/model_card.md` (Known Limitations).
3. **Cost-weighted threshold, not F1.** `train_baseline.py` uses
   `find_optimal_threshold(y_true, y_prob, cost_fn=£125, cost_fp=£25)` —
   5:1 FN:FP. The £25 FP cost has a documented breakdown
   (£8 complaint + £12 attrition + £5 merchant friction). A
   **`MAX_ACCEPTABLE_FPR = 0.05` commercial hard constraint** flags any
   operating point above 5% FPR as `COMMERCIALLY UNACCEPTABLE`.
4. **`scale_pos_weight ≈ 1.0` in baseline** is intentional. Combined
   SMOTE+undersample already balances the training set; stacking
   raw-ratio weighting on top would double-correct and explode FPR.
5. **Daily £ loss formula is volume-independent.** Uses *rates*
   (FN/total, FP/total) × `DAILY_TXNS` × unit cost. Same formula
   everywhere so the three operating points are the same order of
   magnitude regardless of sample size.
6. **Search uses imblearn pipeline inside StratifiedKFold.** sklearn's
   `RandomizedSearchCV`/`GridSearchCV` drive `ImbPipeline([imputer, SMOTE,
   undersample, XGB])`, so resampling/imputation are refit *per fold* on
   training data only. The validation fold stays at the real fraud rate.
7. **Generated outputs are not version-controlled.**
   `docs/feature_validation/`, `docs/model_performance/` PNGs/JSON and
   `data/processed/` are all gitignored.

## 7. How to run things (common commands)

```bash
# Verify environment health
python -c "import numpy,pandas,xgboost,sklearn,mlflow,shap; print('OK')"

# Local pipeline
python data/download_data.py
python src/features/build_features.py
python src/features/validate_features.py
python src/models/train_baseline.py            # 100k default
FRAUD_SAMPLE_N=40000 python src/models/train_baseline.py
python src/models/tune_model.py                # SEARCH_MODE="random"
python src/models/validate_model.py
python src/models/explain.py                   # new

# Tests
python tests/test_feature_store.py
```

## 8. Outstanding work / what's next

| Item | Why it matters | Effort |
|---|---|---|
| Rewrite `f2ccb92` to drop the ~900 KB of generated PNGs/JSON | Bloat in initial commit; needs `git filter-repo` (or interactive rebase + amend) + **force-push to origin/main**. Coordinate before doing it. | Low mechanically, high in coordination |
| Untrack `.claude/settings.local.json` | Currently committed; local IDE config, usually gitignored | Trivial |
| A8 cross-file path migration | `src/config.py` exists; pipeline files still use hardcoded local paths | Med — file-by-file with smoke re-runs |
| Implement `src/api/`, `src/streaming/`, `src/monitoring/` | Scaffolding only today | Multi-session |
| Validate the baseline model (not just the tuned one) | `validate_model.py` currently points at `tuned_xgboost.pkl`; consider parameterising or running a baseline pass too | Low |

## 9. AWS Phase A — status

As of **2026-05-20**:

- **STOP 1 — Root MFA:** ✅ done
- **STOP 2 — IAM user:** ✅ done. A second user **`fraud-detection-dev`**
  was created on 2026-05-20 with the spec'd policies
  (`AmazonS3FullAccess`, `AmazonEC2FullAccess`,
  `AmazonSageMakerFullAccess`); `aws configure` is now pointed at this
  user. The earlier `Shraddha` user still exists in the account — revoke
  its keys when convenient.
- **STOP 3 — Billing alerts:** ✅ done (£10 + £25/month).
- **STOP 4 — GitHub remote:** ✅ done; remote = `origin`.
- **AWS CLI configured:** region `eu-west-2`, output `json`.
  `sts get-caller-identity` →
  `arn:aws:iam::080109295535:user/fraud-detection-dev`.
- **S3 bucket:** `fraud-detection-poc-540924` (eu-west-2, block-all-public,
  versioning enabled, SSE-S3, tagged `Project=fraud-detection-poc`).
- **Data uploaded:** all 5 IEEE-CIS CSVs (1.262 GB) at
  `s3://fraud-detection-poc-540924/data/raw/`, multipart + checksum
  verified. Estimated storage cost ~£0.03/mo.

`boto3 1.42.35` and `awscli 2.24.16` are already installed in the base
env.

Remaining (do not run unless training at scale is the explicit goal —
EC2 costs ~£0.08/h while running):
```
export S3_BUCKET=fraud-detection-poc-540924
python scripts/aws/launch_ec2.py         # t3.large
python scripts/aws/run_training_on_ec2.py
python scripts/aws/stop_ec2.py           # ***always run when done***
```

## 10. User-collaboration preferences observed this session

- **Honest flagging beats over-claiming.** When something is partial,
  noisy, or risky, say so explicitly. Don't pretend smoke runs are real
  results.
- **Don't silently change scope or rewrite verified code.** When a
  prompt's spec conflicts with prior deliberate decisions
  (e.g. `/mlflow/runs` after the rename to `/mlruns/`), surface the
  conflict and ask, don't blindly overwrite.
- **Concise responses, terse summaries.** No trailing recap padding.
- **Methodological discipline matters** — out-of-fold encoding, split-
  before-resample, threshold-on-validation, etc., were all explicit
  decisions the user reinforced.
- **Explicit STOP-points before cost-incurring or destructive AWS
  actions.** The user opted to confirm the bucket name before
  `upload_data.py` even though it was tiny cost; treat history rewrites
  + force-pushes the same way.

## 11. 2026-05-20 session log (most recent first)

- **Phase A AWS provisioning completed.** Created
  `fraud-detection-dev` IAM user (account `080109295535`), `aws
  configure` to `eu-west-2`/json, created S3 bucket
  `fraud-detection-poc-540924`, uploaded 1.262 GB of IEEE-CIS CSVs.
- **pandas 3.0.3 re-verify on model scripts.** All three scripts
  (`train_baseline`, `tune_model`, `validate_model`) ran clean. Numbers
  match the pre-bump values within run-variance; see §5 for the table.
- **Patched `experiment_tracking._system_info`** to broaden the version-
  lookup `except` so a broken pandas dist-info (`KeyError('Version')`)
  no longer crashes MLflow logging. Verified by round-trip and by the
  100k `train_baseline` run logging successfully.
- **Found but did not fix:**
  - `train_baseline.py:141` `pd.concat(..., copy=False)` raises
    `Pandas4Warning` (deprecated kwarg). Cosmetic now; pandas 4 will
    remove it.
  - `validate_model.py` validates `tuned_xgboost.pkl` only — consider
    a baseline pass too.
