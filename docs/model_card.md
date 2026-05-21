# Model Card — PSP Fraud Detection (Proof of Concept)

> Status: **proof of concept**, not production. This card is a living document
> and will be expanded as the model is trained and evaluated. It is structured
> to align with FCA expectations on governance, explainability, and ongoing
> monitoring for a UK Payment Service Provider fraud control.

## Model details

- **Purpose:** real-time risk scoring of PSP card transactions to flag likely
  fraud for review / decline.
- **Intended use:** decision support within a PSP fraud-operations workflow.
  Not an autonomous adverse-action system; human review expected for the PoC.
- **Out of scope:** AML/sanctions screening, credit decisioning, any use
  outside card-payment fraud.
- **Data:** IEEE-CIS Fraud Detection (Vesta) — see `docs/data_dictionary.md`.
- **Status:** training/evaluation pending (`src/models/`).

## Data and features

- Feature engineering: `src/features/build_features.py`.
- Class imbalance handling: `src/features/handle_imbalance.py` (test partition
  always kept at the real ~3.5% fraud rate; resampling is train-only).

## Known limitations

These are recorded explicitly for governance and must be revisited before any
production claim:

1. **No billing-vs-shipping match feature.** The user requested a feature
   indicating whether billing and shipping addresses match. IEEE-CIS contains
   **only billing** address fields (`addr1` region, `addr2` country) and **no
   shipping address**, so this signal cannot be computed. Rather than fabricate
   it from a weak proxy (e.g. `dist1`), the feature was **dropped entirely**.
   *Impact:* a recognised fraud signal (address mismatch) is absent. *Action
   for production:* source a dataset / live event stream that carries shipping
   address and reinstate the feature.

2. **Relative, not wall-clock, time.** `TransactionDT` is a seconds offset with
   no disclosed epoch; time-of-day / day-of-week features are anchored to the
   community-convention reference `2017-12-01`. Calendar-derived features are
   *relative*, not literal. *Action:* use true event timestamps in production.

3. **Composite card identity, not an account key.** No customer/account id
   exists; `card_uid` is composed from `card1..card6 + addr1` and only
   approximates "the same card". Velocity/aggregation features inherit this
   approximation. *Action:* use the platform's real account/card token.

4. **Historical device fraud rate — RESOLVED.** `device_type_fraud_rate`
   previously used a smoothed whole-dataset target mean (leakage risk). It now
   uses out-of-fold target encoding with Laplace smoothing. See *Feature
   Engineering Decisions* below.

5. **Categorical raw columns dropped for resampling.** The imbalance pipeline
   keeps numeric columns only (SMOTE requires a numeric, NaN-free matrix);
   raw categoricals (`ProductCD`, `card4/6`, `*_emaildomain`, `M*`,
   `DeviceInfo`, etc.) are not yet encoded. *Action:* add categorical encoding
   in the modelling layer.

## Feature Engineering Decisions

### `device_type_fraud_rate` — out-of-fold target encoding

- **Target encoding.** `device_type_fraud_rate` is a *target-encoded* feature:
  each `DeviceType` is represented by the historical fraud rate of that device
  type. This converts a low-cardinality categorical into a strong numeric
  signal.
- **Out-of-fold encoding prevents data leakage.** Encoding is computed with
  `encode_device_fraud_rate_safely()` in `src/features/build_features.py`. The
  training data is split into **5 folds**; each fold's rows are encoded using
  the device fraud rates calculated **only from the other 4 folds**. A row
  never contributes its own label to its own feature value. Naive full-data
  target encoding would let each row "see" its own outcome, inflating apparent
  performance and producing a model that degrades in production — out-of-fold
  encoding removes that leakage.
- **Laplace (add-one) smoothing.** All rates use `(positives + 1) /
  (total + 2)`, so device types with very few transactions are pulled toward
  a neutral prior instead of taking extreme 0% / 100% rates from noise.
- **Fallback for unseen device types.** A serving map fitted on the full
  training labels is handed to the feature store. At serving time, a device
  type **not seen in training** falls back to the **Laplace-smoothed global
  training fraud rate** (`device_fraud_rate_global`). Using full training data
  for the *serving* map is not leakage: serving transactions are unseen and
  carry no label.
- **Why this matters for production reliability.** Leakage makes offline
  metrics optimistic and the live model silently worse — the most common cause
  of "great in validation, fails in production" fraud models. Out-of-fold
  encoding makes offline estimates honest; Laplace smoothing keeps rare-device
  predictions stable; the explicit global fallback guarantees a defined,
  sensible score for never-before-seen devices instead of a NaN or crash in
  the real-time path. This is also an FCA-relevant control: model performance
  claims must be derived without leakage.

## Operating Point Selection

The baseline model outputs a fraud probability; the decision threshold is a
**business choice**, not a modelling constant. `find_optimal_threshold()` in
`src/models/train_baseline.py` evaluates every threshold and exposes **three
operating points**, all selected on a validation set (never on the test set):

- **Cost-optimal (recommended default).** Minimises expected monetary loss
  using the explicit asymmetric costs: a missed fraud (false negative) costs
  £125 (PSP 50/50 liability on a £250 average APP transaction); a wrongly
  blocked legitimate transaction (false positive) costs £25 — a 5:1 FN:FP
  ratio. This is the default the pipeline applies and the one we recommend.
- **95% Precision (conservative).** The highest-recall threshold that still
  keeps precision ≥ 0.95. Optimises for minimal customer friction (very few
  legitimate transactions blocked) at the cost of catching less fraud.
- **Max Recall, P>0.5 (aggressive).** The highest-recall threshold that keeps
  precision above 0.50. Optimises for catching as much fraud as possible,
  accepting more false positives.

The CTO can select a different operating point based on the firm's specific
tolerance for fraud loss versus customer friction — e.g. shifting toward
aggressive during a fraud attack, or conservative to protect customer
experience. All three, with their test-set recall/precision/expected-loss,
are written to `docs/model_performance/baseline_metrics.json` and visualised
in `docs/model_performance/threshold_analysis.png`.

False positive cost is estimated at £25 per incorrectly blocked transaction,
comprising complaint handling (£8), customer attrition value (£12), and merchant
friction (£5). This estimate should be replaced with the client's actual measured
cost per false positive during production calibration.

A maximum false positive rate of 5% is enforced as a commercial constraint
independent of cost optimisation. Blocking more than 5% of legitimate transactions
is considered commercially unacceptable for a UK PSP regardless of fraud savings,
due to customer attrition risk, merchant relationship damage, and potential FCA
Consumer Duty scrutiny of excessive payment blocking.

Operating points exceeding this threshold are flagged as COMMERCIALLY UNACCEPTABLE
and should not be deployed without explicit board sign-off and FCA notification
consideration.

## Fairness, explainability, monitoring

- Explainability: SHAP attributions planned in `src/models/`; pair opaque
  Vesta `V*`/`id_*` features with interpretable signals for review.
- Monitoring: data/model drift via Evidently (`src/monitoring/`).
- To be completed: performance metrics, threshold/operating point, subgroup
  analysis, validation strategy (time-based split to avoid leakage).

## Model Selection

Production model selected: baseline — tuned does not beat baseline by the required 1pp recall at 95% precision — validated on identical test set, 2026-05-21
