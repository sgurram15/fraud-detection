# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Proof-of-concept real-time fraud detection for a UK PSP. Ingests card
transactions, engineers leakage-safe behavioural/velocity features, scores with
XGBoost, and serves the score + a SHAP explanation via FastAPI. It is shaped by
FCA expectations (explainability, no data leakage, documented operating point),
**not** just ML accuracy. PoC only — not for production.

Read `README.md` for the full picture, `docs/model_card.md` before drawing any
conclusions about performance, and `docs/production_architecture.md` for the
local-vs-AWS design.

## Hard rules (do not violate)

1. **Never `git add .`** — always stage specific paths.
2. **Never commit credentials, keys, or passwords.** `kaggle.json`, `.env`, and
   data are gitignored — keep it that way.
3. **AWS region is always `eu-west-2`** (FCA data residency — do not change).
   Tag every AWS resource `Project=fraud-detection-poc`.
4. **EC2 / SageMaker bill by the hour.** Never leave an instance running — run
   `scripts/aws/stop_ec2.py` when done. AWS steps are optional and cost money.
5. **`numpy<2` is pinned** in `requirements.txt`. The scientific stack
   (pandas/matplotlib/numexpr) is NumPy-1.x ABI; do not let any install (e.g.
   an unpinned `shap`) pull NumPy 2.x — it corrupts the environment.
6. If a destructive action is unclear, log it and wait for human confirmation.

## Key project facts

- **Production model:** `src/models/saved/baseline_xgboost.pkl`
  (version `xgboost-baseline-v1`, 18 features). `tuned_xgboost.pkl` also exists
  but baseline is the user-confirmed production choice.
- **Decision threshold: 0.19** (cost-optimal, from `baseline_metrics.json`
  `default_threshold`) — not 0.5.
- **Feature engineering must stay leakage-free.** 18 features; see model card
  for known data limitations (no shipping address, relative `TransactionDT`,
  composite `card_uid`).
- Operating-point cost assumptions (£125 FN / £25 FP) are **illustrative** and
  must be recalibrated with real client data before production.
- Scaffolding-only layers: `src/api/` (in progress), `src/streaming/`,
  `src/monitoring/`.

## Paths & config

Use the helpers in `src/config.py` for all data/model paths — do **not**
hardcode. They branch on `USE_S3` / `S3_BUCKET`:

- `data_path(*parts)` → raw data root
- `processed_path(*parts)` → processed data root
- `model_path(*parts)` → model root

Migrate scripts to these one file at a time **with re-verification** — never a
blind sweep (the pipeline is validated; blind path rewrites are a regression
risk).

## Common commands

```bash
# Pipeline (run in order)
python src/features/build_features.py        # leakage-safe; caches to data/processed/
python src/features/handle_imbalance.py      # SMOTE / undersample / combined
python src/models/train_baseline.py          # FRAUD_SAMPLE_N=all for full dataset
python src/models/tune_model.py              # SEARCH_MODE=random (local) | full_grid (SageMaker)
python src/models/validate_model.py          # pre-production gate

# Serving
uvicorn src.api.main:app --reload

# Tests (standalone, print PASS/FAIL, non-zero exit on failure; pytest also works)
python tests/test_feature_store.py
python tests/test_api.py
```

Environment: Python 3.11, Windows/PowerShell primary. Activate venv with
`.venv\Scripts\Activate.ps1`.

## Mission / progress tracking

Phase progress is tracked in append-only logs — **read these to know current
state, and log every action as you go**:

- `MISSION.md` + `MISSION_LOG.md` — Phase A/B (complete)
- `MISSION_PHASE_C.md` + `MISSION_LOG_C.md` — Phase C (current: API, streaming,
  monitoring, AWS deployment)

Log format: `[timestamp] [task ID] [DONE/FAILED] [description]`. Pause at STOP
points and wait for human confirmation. If a task fails twice, log it and move
on.
