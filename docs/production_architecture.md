# Production Architecture — PSP Real-Time Fraud Detection

This document describes two versions of the system:

- **Version 1** — the local proof of concept being built in this repository.
- **Version 2** — the AWS production deployment this becomes when operated by
  a UK Payment Service Provider (PSP) regulated by the Financial Conduct
  Authority (FCA).

The intent is to show that the PoC is a faithful, scaled-down model of the
production system: every Version 1 component has a direct Version 2
counterpart, so the modelling and feature logic validated locally transfers
without redesign.

### Regulatory context (why this section exists in a fraud system)

A PSP's fraud control is not just an ML system; it is a regulated control.
The architecture is shaped by: the Payment Services Regulations 2017 (PSRs)
and Strong Customer Authentication; FCA **PS21/3** operational resilience
(important business services, impact tolerances, self-assessment evidence);
SYSC outsourcing and cloud expectations (FG16/5); the Consumer Duty; the
PSR's mandatory APP-scam reimbursement regime; and UK GDPR. These are
referenced per-component below rather than treated as an afterthought.

---

## Version 1 — Local Proof of Concept

```
 transaction CSV ──► [Simulated Kafka] ──► [Feature Store fn] ──► [XGBoost]
                          (src/streaming)     (src/features)      (src/models)
                                                     │                │
                                                     ▼                ▼
                                              [FastAPI endpoint] ──► [SHAP]
                                                  (src/api)        explanation
                                                     │
                                                     ▼
                                          [Local MLflow tracking]
```

### Simulated Kafka stream (`src/streaming`)
A local generator replays historical IEEE-CIS transactions one-by-one to
imitate a live payment event stream, decoupling transaction *arrival* from
transaction *scoring* exactly as a real broker would. **Why it matters for an
FCA-regulated PSP:** fraud screening must happen inline, in the authorisation
path, within tens of milliseconds — a PSP cannot batch-score overnight because
SCA exemption decisions and transaction approval/decline happen in real time.
Proving the streaming contract locally de-risks the latency and ordering
assumptions before they become a regulated, customer-impacting control.

### Feature store Python function (`src/features/feature_store.py`)
A single in-process function maintains per-card rolling state and returns the
engineered feature vector for one transaction in O(1), guaranteeing the
features served online are computed identically to those used in training
(prior-only / out-of-fold semantics). **Why it matters:** online/offline
feature skew is the most common cause of a fraud model that validates well but
fails in production — directly relevant to the FCA's expectation that firms
can evidence that a deployed model performs as tested, and to avoiding
unfair customer outcomes (wrongful declines) under the Consumer Duty.

### XGBoost model (`src/models`)
A gradient-boosted tree classifier produces a calibrated fraud-probability
score per transaction, chosen for strong tabular performance and tractable
explainability. **Why it matters:** a PSP must justify automated decisions
that block or challenge a customer's payment; a tree model with feature
attributions is defensible to the FCA, to internal model governance, and in
APP-scam reimbursement disputes, in a way an opaque deep model is not.

### FastAPI endpoint (`src/api`)
A lightweight HTTP service exposes a synchronous `/score` endpoint returning
the fraud probability, decision, and explanation for a transaction. **Why it
matters:** it defines the real-time integration contract (latency budget,
request/response schema, failure behaviour) that the payment authorisation
flow depends on — a regulated *important business service* under PS21/3 whose
impact tolerance (maximum tolerable disruption) must later be set and tested.

### SHAP explainability
SHAP produces per-transaction feature attributions explaining *why* a given
score was assigned. **Why it matters:** the FCA and UK GDPR (Art. 22,
automated decision-making) expect firms to explain individual automated
outcomes to customers and reviewers; SHAP gives fraud-ops analysts and
complaint handlers a concrete, per-decision rationale rather than "the model
said so", which is essential for fair treatment and audit defensibility.

### Local MLflow tracking
MLflow records experiments, parameters, metrics, and model artefacts, giving a
reproducible lineage from data to trained model. Local development uses SQLite
backend (`mlruns/mlflow.db`). Production uses Amazon SageMaker Experiments
which provides managed experiment tracking, automatic model versioning, and
integration with SageMaker Pipelines for automated retraining. **Why it
matters:** model risk governance requires that any model influencing customer
payment outcomes is versioned, reproducible, and its performance claims
traceable to the exact data and code that produced them — the foundation of
the evidence pack a PSP needs for internal validation and FCA scrutiny.

---

## Version 2 — AWS Production Deployment (UK PSP)

```
        Region: eu-west-2 (London)  ── active ───┐   ┌─── active: eu-west-2 (2nd AZ set)
                                                 ▼   ▼
 card events ─► [Amazon MSK] ─► [Managed Apache Flink] ─► [SageMaker endpoint]
                  (Kafka)        (stream features)          (model + registry)
                                                                  │
                                                                  ▼
                                                       [Amazon Bedrock agent]
                                                       decision reasoning + FCA docs
                                                                  │
                 every action, model version, data access ─────► [AWS CloudTrail]
                                                          immutable audit (PS21/3)
```

### Amazon MSK (managed Kafka)
Amazon Managed Streaming for Apache Kafka is the durable, replicated transport
for the live payment-event stream, retaining ordered events with configurable
retention and replay. **Why it matters for an FCA-regulated PSP:** it provides
the durability and replay needed to reconstruct exactly what the fraud system
saw at decision time — essential for incident investigation, APP-scam dispute
evidence, and demonstrating to the FCA that the firm can reproduce and explain
historical automated decisions.

### Amazon Managed Service for Apache Flink (stream feature engineering)
Managed Flink performs the stateful, low-latency feature computation
(velocity windows, per-card aggregates) on the MSK stream — the production
equivalent of the PoC feature-store function, but horizontally scalable and
fault-tolerant with checkpointed state. **Why it matters:** it makes the
real-time feature pipeline a resilient, recoverable service (checkpoint/restore
after failure) so the fraud control stays within its PS21/3 impact tolerance,
and it enforces the same feature definitions at scale, preserving the
online/offline parity the FCA expects between a tested and a deployed model.

### Amazon SageMaker (training, model registry, real-time endpoints)
SageMaker handles model training, a governed model registry with approval
gates, and autoscaling real-time inference endpoints behind the scoring API.
**Why it matters:** the registry gives auditable model lineage, staged
approval, and one-click rollback — directly supporting model risk governance
and the FCA's expectation that changes to a customer-impacting decision model
are controlled, reversible, and evidenced; managed endpoints provide the
availability and latency guarantees the authorisation path requires.

### Hyperparameter Tuning
Local development uses randomised search over 20 combinations as a
directional guide (SEARCH_MODE=random in tune_model.py).

Production tuning uses SageMaker Automatic Model Tuning (Bayesian
optimisation) which is more efficient than grid search at finding optimal
parameters — typically requiring 50-100 jobs rather than 243 to find a
better optimum. Each job runs on an isolated ml.m5.2xlarge instance.
Tuning jobs are triggered automatically by the retraining pipeline when
model drift is detected by the monitoring layer.

### Amazon Bedrock agent (autonomous decision reasoning + FCA audit documentation)
A Bedrock agent reasons over the model score, SHAP attributions, and customer
context to recommend an action (allow / step-up / block) and to draft the
structured rationale and audit narrative for that decision. **Why it matters:**
under the Consumer Duty and APP-scam reimbursement rules a PSP must
consistently justify why a payment was challenged or stopped; the agent
produces a contemporaneous, human-readable, regulator-ready record for every
borderline decision — turning explainability into documented evidence. It must
operate as decision *support* with human oversight for material outcomes, not
unaccountable automation, consistent with FCA governance expectations.

### AWS CloudTrail (immutable audit logs)
CloudTrail records an immutable, tamper-evident log of every API action: model
deployments, data access, configuration changes, and inference invocations.
**Why it matters:** PS21/3 requires firms to evidence operational resilience —
including testing, scenario analysis, and a maintained self-assessment — and
firms must retain records demonstrating control over a regulated service.
CloudTrail (with log-file integrity validation and locked retention) provides
the immutable evidence chain that the fraud control behaved as governed, who
changed it, and when — defensible in an FCA review or enforcement context.

### Multi-region active-active deployment (impact tolerance compliance)
The stream, feature pipeline, model endpoints, and audit logging run
active-active across multiple AWS regions so that the loss of one region does
not take the fraud control — or payment authorisation — offline. **Why it
matters:** real-time fraud screening is almost certainly an *important
business service* under PS21/3; the PSP must define an impact tolerance (the
maximum tolerable duration/extent of disruption) and remain within it through
severe-but-plausible scenarios. Active-active is the architecture that lets the
firm credibly evidence it can stay within tolerance rather than merely aspire
to.

### Data residency — AWS eu-west-2 (London)
All transaction data, features, model artefacts, and audit logs are confined
to the London region. **Why it matters:** while the FCA does not by itself
mandate UK-only storage, a PSP must maintain effective oversight and control of
outsourced/cloud processing (SYSC 8, FG16/5), meet UK GDPR transfer and
lawful-basis obligations, and satisfy contractual and supervisory expectations
about jurisdiction and law-enforcement access. Pinning data to eu-west-2 keeps
processing within a single, well-understood legal and supervisory perimeter,
simplifying the firm's outsourcing risk assessment and its data-protection
position.

### CloudWatch monitoring + EventBridge retraining
The model is monitored continuously rather than deployed and forgotten. Custom
CloudWatch metrics (transactions processed, fraud rate, false-positive rate,
end-to-end latency, agent decision counts, estimated daily fraud saving) are
published from the pipeline, with alarms on the operationally meaningful
conditions: false-positive rate above the commercial cap, latency breach, a
fraud rate flat-lining at zero (a likely model/feature failure), and abnormal
agent BLOCK volume. A weekly **EventBridge** schedule triggers the monitoring
job (`src/monitoring/run_monitoring.py` in the PoC) — Evidently data-drift
detection plus a performance tracker over confirmed outcomes — which emits a
HEALTHY / WARNING / RETRAINING_REQUIRED verdict and, on RETRAINING_REQUIRED,
kicks off a SageMaker retraining pipeline for human-reviewed promotion. **Why
it matters:** model governance requires that a regulated control's performance
is evidenced over time and that degradation (drift, rising false negatives) is
detected and acted on, not discovered after customer harm.

---

## How Version 1 maps to Version 2

| Concern | Version 1 (PoC) | Version 2 (Production) |
|---|---|---|
| Event transport | Simulated Kafka replay | Amazon MSK |
| Feature engineering | In-process feature-store fn | Managed Apache Flink |
| Model train/serve | XGBoost + local files | SageMaker training/registry/endpoint |
| Explainability | SHAP (local) | SHAP + Bedrock agent narrative |
| Experiment lineage | Local MLflow | SageMaker registry + MLflow |
| Decisioning | API returns score/decision | Bedrock agent (human-overseen) |
| Audit/evidence | Logs / MLflow runs | CloudTrail immutable trail |
| Monitoring/retrain | `run_monitoring.py` (Evidently) | CloudWatch + EventBridge weekly retrain |
| Resilience | Single local process | Multi-region active-active |
| Data location | Local machine | AWS eu-west-2 only |

Because each row is a one-to-one substitution, the feature definitions,
leakage controls (out-of-fold encoding), and model logic proven in the PoC
are exactly what runs in production — the regulated controls are designed in
from Version 1, not retrofitted.

## Cross-cutting FCA considerations (both versions)

- **Online/offline parity** — the same feature definitions train and serve the
  model; divergence is treated as a defect, because it undermines the firm's
  ability to evidence tested performance.
- **No target leakage** — out-of-fold encoding and prior-only windows keep
  offline metrics honest, so performance claims to the FCA and internal
  validation are not optimistic.
- **Explainable, human-overseen decisions** — automated scores are explainable
  per-transaction and material adverse decisions retain human oversight.
- **Reproducibility and rollback** — every model influencing a payment outcome
  is versioned, reproducible, and reversible.
- **Documented limitations** — known gaps (see `docs/model_card.md`) are
  recorded rather than hidden, supporting honest regulatory engagement.
