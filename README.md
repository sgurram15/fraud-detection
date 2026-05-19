# Real-Time Fraud Detection System

A proof of concept (PoC) for a real-time **Payment Service Provider (PSP)** fraud
detection system. The goal is to demonstrate an end-to-end machine learning
pipeline that scores card payment transactions for fraud risk as they stream
through, with the model governance and monitoring controls expected in a
UK-regulated financial services context.

## What this is

This repository is an exploratory PoC, not a production system. It is intended
to validate the approach, surface engineering and modelling trade-offs, and
produce artefacts (including an FCA-aligned model card) that can support a
later production build and any required regulatory engagement.

## Scope

- Ingest a stream of PSP card transactions (Kafka simulation).
- Engineer behavioural and velocity features in real time.
- Score each transaction with a gradient-boosted model (XGBoost) and explain
  predictions with SHAP.
- Serve scores over a low-latency FastAPI endpoint.
- Track experiments and models with MLflow.
- Monitor for data and model drift (Evidently) so degradation is caught early.

## Project structure

```
fraud-detection/
├── data/              Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/         Exploration and experimentation
├── src/
│   ├── features/      Feature engineering pipeline
│   ├── models/        Training and evaluation
│   ├── api/           FastAPI serving layer
│   ├── streaming/     Kafka simulation
│   └── monitoring/    Model drift detection
├── tests/             Unit tests
├── docs/              Architecture and FCA model card
├── requirements.txt
└── README.md
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Datasets are not committed. Use the Kaggle CLI (configured via `kaggle.json`)
to download a card fraud dataset into `data/raw/`.

## Regulatory context

As a UK PSP fraud control, the model is expected to align with FCA
expectations around governance, explainability, and ongoing monitoring. See
`docs/` for the architecture overview and the FCA model card capturing intended
use, data, performance, limitations, and monitoring approach.

## Status

Proof of concept — under active development. Not for production use.
