# PSP Real-Time Fraud Detection — Proof of Concept

## 1. What this is

This is a proof-of-concept real-time fraud detection system for a UK Payment
Service Provider (PSP). It ingests card-transaction events, engineers
behavioural and velocity features in a leakage-safe way, scores each
transaction with a gradient-boosted model (XGBoost), and exposes the score and
a SHAP-based explanation through a low-latency serving layer. It is built to
mirror, at small scale, the architecture a regulated PSP would run in
production (the local pieces map one-to-one onto the AWS design in
`docs/production_architecture.md`).

Fraud screening at a PSP is a regulated control, not just an ML model. The
system is shaped by FCA expectations: decisions must be explainable
(Consumer Duty, UK GDPR Art. 22), performance claims must be free of data
leakage and reproducible (model governance), the operating point is a
business decision with a documented commercial constraint (no more than ~5% of
legitimate transactions blocked), and known limitations are recorded rather
than hidden. See `docs/model_card.md` for the FCA-aligned model card and
`docs/data_dictionary.md` for the data.

## Quick start (demo)

Once set up (section 3) with a trained model and feature store in place:

```bash
python scripts/run_demo.py          # full narrated demo (~60s pipeline step)
python scripts/run_demo.py --quick  # short version
```

The demo runs six steps end-to-end with no AWS: system check, baseline-vs-tuned
comparison, five live-scored transactions with SHAP reason codes, the streaming
pipeline at 10 TPS with a live dashboard, a financial-impact projection, and an
ASCII summary of the production architecture.

## 2. Architecture overview

Two versions are described in detail in **`docs/production_architecture.md`**:

- **Version 1 (this repo, local PoC):** simulated stream → in-process feature
  store → XGBoost → FastAPI serving → SHAP → local MLflow (SQLite backend).
- **Version 2 (AWS production):** Amazon MSK → Managed Apache Flink →
  SageMaker (training/registry/endpoints) → Bedrock decision reasoning →
  CloudTrail immutable audit, multi-region active-active, all in `eu-west-2`
  (London) for data residency.

## 3. Local setup

Prerequisites: Python 3.11, git. The IEEE-CIS dataset requires a Kaggle
account/API token.

**Windows (PowerShell):**
```powershell
git clone https://github.com/sgurram15/fraud-detection.git
cd fraud-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (bash):**
```bash
git clone https://github.com/sgurram15/fraud-detection.git
cd fraud-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle credentials: place `kaggle.json` at `~/.kaggle/kaggle.json`
(Windows: `C:\Users\<you>\.kaggle\kaggle.json`). **Never commit it** — it is
gitignored. Accept the competition rules at
https://www.kaggle.com/competitions/ieee-fraud-detection/rules once.

> Note: `numpy<2` is pinned in `requirements.txt`. The scientific stack
> (pandas/matplotlib/numexpr) is NumPy-1.x ABI; allowing NumPy 2.x (e.g. via
> an unpinned `shap` install) corrupts the environment.

## 4. AWS setup

Cloud deployment is **optional** and **incurs cost**. Follow
**`docs/AWS_SETUP.md`** step by step. It contains four human STOP points
(root MFA, IAM user, billing alerts, GitHub remote) that must be completed
before the automated AWS scripts (`scripts/aws/`) can run. All resources are
created in `eu-west-2` and tagged `Project=fraud-detection-poc`.

## 5. Running the pipeline (in order)

```bash
# 1. Download data (Kaggle competition; ~1.3 GB, gitignored)
python data/download_data.py

# 2. Explore / sanity-check
python data/explore_data.py

# 3. Feature engineering (leakage-safe; caches to data/processed/)
python src/features/build_features.py

# 4. Class-imbalance datasets (SMOTE / undersample / combined)
python src/features/handle_imbalance.py

# 5. Train baseline XGBoost (defaults to a 100k sample; FRAUD_SAMPLE_N=all for full)
python src/models/train_baseline.py

# 6. Hyperparameter tuning (SEARCH_MODE="random" local; "full_grid" = SageMaker)
python src/models/tune_model.py

# 7. Pre-production validation gate
python src/models/validate_model.py

# 8. Serving API (FastAPI)
uvicorn src.api.main:app --reload     # (api layer is scaffolding)

# 9. Streaming simulation (Kafka stand-in)
python -m src.streaming.simulate      # (streaming layer is scaffolding)
```

Feature validation plots/report:
`python src/features/validate_features.py`.

## 6. Running the tests

```bash
python tests/test_feature_store.py
```

(Standalone smoke/acceptance tests print PASS/FAIL and exit non-zero on
failure. `pytest` can also be used if installed.)

## 7. Known limitations and model card

The **FCA model card is `docs/model_card.md`** — read it before drawing any
conclusions. Headline limitations: no shipping-address feature (IEEE-CIS has
none), `TransactionDT` is relative not wall-clock, `card_uid` is a composite
approximation (no true account key), local runs use a sample (host RAM can't
hold the full 590k + SMOTE), and several layers (`api/`, `streaming/`) are
scaffolding. Cost assumptions in the operating-point analysis (£125 FN /
£25 FP) are illustrative and must be recalibrated with the client's real
fraud-loss data before production.

## 8. Cost warning

| Step | Cost |
|---|---|
| Local pipeline (sections 5–6) | £0 (your machine) |
| Kaggle data download | £0 |
| `scripts/aws/setup_s3.py` | £0 to create; ~£0.023/GB/month to store |
| `scripts/aws/upload_data.py` | S3 storage of ~1.3 GB ≈ £0.03/month + transfer |
| `scripts/aws/launch_ec2.py` (r5.2xlarge, 60 GB root) | **~£0.48/hour while running** |
| SageMaker training/tuning (Version 2) | per-instance, per-hour — see AWS_SETUP.md |

**EC2 and SageMaker bill by the hour. Always run `scripts/aws/stop_ec2.py`
when done — never leave an instance running overnight.** Billing alerts at
£10 and £25/month are a required setup step (STOP 3).

## 9. Performance

Production model: **XGBoost baseline** (`baseline_xgboost.pkl`,
`xgboost-baseline-v1`), selected over the tuned variant on an identical test
set (`docs/model_performance/model_comparison.json`). Metrics on the full
held-out test set at the deployed cost-optimal operating point (threshold
**0.19**):

| Metric | Value |
|---|---|
| AUC-ROC | 0.923 |
| AUC-PR | 0.647 |
| Recall | 0.678 |
| Precision | 0.426 |
| False-positive rate | 0.033 (within the ≤5% commercial constraint) |

The operating point is a business decision: the 0.19 threshold minimises
expected fraud loss under the illustrative £125-FN / £25-FP cost model while
keeping the false-positive rate well under the ~5% cap. Full numbers and the
threshold analysis are in `docs/model_performance/` and `docs/model_card.md`.

Serving latency is well under the 100ms target (≈30ms end-to-end per request,
measured by `tests/test_api.py`).

## 10. FCA compliance

Fraud screening at a PSP is a regulated control. This system is built to the
FCA expectations that shaped it:

- **Explainability (Consumer Duty, UK GDPR Art. 22):** every decision carries
  SHAP reason codes and a structured `fca_explanation` audit record naming the
  model, threshold, probability, and top contributing features.
- **Immutable audit trail:** every scored transaction is written to an audit
  record (local `data/audit/`; CloudTrail/S3 in production).
- **No data leakage:** features are prior-only (history strictly before the
  transaction); the model was validated on a leakage-free held-out set.
- **Documented operating point:** the threshold and its commercial constraint
  are recorded, not implicit.
- **Model monitoring:** `src/monitoring/` runs Evidently drift detection and a
  performance tracker, with a HEALTHY / WARNING / RETRAINING_REQUIRED verdict.
- **Recorded limitations:** known gaps (no shipping-address feature, relative
  `TransactionDT`, composite `card_uid`, illustrative cost assumptions) are in
  `docs/model_card.md` rather than hidden.
- **Data residency:** all AWS resources are in `eu-west-2` (London).

---

PoC — not for production use. See `docs/model_card.md` and
`docs/production_architecture.md`.
