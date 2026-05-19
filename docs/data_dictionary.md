# IEEE-CIS Fraud Detection — Data Dictionary

> **Verified against the downloaded files** at
> `data/raw/ieee-fraud-detection/` on 2026-05-18. Confirmed schema:
> `train_transaction.csv` = 590,540 rows, 394 columns
> (`TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD`,
> `card1`–`card6`, `addr1`, `addr2`, `dist1`, `dist2`,
> `P_emaildomain`, `R_emaildomain`, `C1`–`C14`, `D1`–`D15`, `M1`–`M9`,
> `V1`–`V339`); `train_identity.csv` = 144,233 rows, 41 columns
> (`TransactionID`, `id_01`–`id_38`, `DeviceType`, `DeviceInfo`).
> Identity coverage ≈ **24.4%** of transactions (144,233 / 590,540) — the
> identity-missingness note below is empirically confirmed.

## Context

The data comes from **Vesta Corporation**, a payments / guaranteed-payment
platform. Each row is a card payment transaction. Many columns are deliberately
anonymised: Vesta exposes the *type* of signal (counting, timedelta, match,
engineered) but masks the precise definition for privacy and IP reasons. For a
PSP fraud PoC, treat anonymised families by their **signal semantics**, not by
trying to reverse-engineer individual columns.

The data is split across two files joined on `TransactionID`:

- `train_transaction.csv` / `test_transaction.csv` — the transaction itself.
- `train_identity.csv` / `test_identity.csv` — identity / device / network
  signals. **Not every transaction has an identity row** (left join; expect
  many nulls).

---

## `train_transaction.csv`

| Column | Type | Meaning in a payment-platform context |
|---|---|---|
| `TransactionID` | id | Unique transaction key. Join key to identity file. Not a feature. |
| `isFraud` | **target** | 1 = transaction was reported fraud (chargeback / confirmed). 0 = legitimate. Severely imbalanced (~3.5% positive). |
| `TransactionDT` | numeric | Time **delta in seconds** from a fixed (unknown) reference point — *not* a wall-clock timestamp. Useful for deriving relative time, ordering, and engineered "time since" features. |
| `TransactionAmt` | numeric | Payment amount (USD). Decimal precision sometimes encodes foreign-currency conversion. |
| `ProductCD` | categorical | Product code for the transaction (W, C, R, H, S). A proxy for product line / channel. |
| `card1`–`card3`, `card5` | numeric (anon) | Payment card attributes — issuer bank, card sub-type, country, etc. (masked). Effectively high-cardinality categorical identifiers. |
| `card4` | categorical | Card **network**: visa / mastercard / amex / discover. |
| `card6` | categorical | Card **funding type**: debit / credit (debit vs credit has different fraud profiles). |
| `addr1` | numeric (anon) | Billing **region** (purchaser). |
| `addr2` | numeric (anon) | Billing **country** (purchaser). |
| `dist1`, `dist2` | numeric (anon) | Distance measures (e.g. between billing, shipping, IP, ZIP). Large/anomalous distance is a classic fraud signal. |
| `P_emaildomain` | categorical | **Purchaser** email domain (e.g. gmail.com). Disposable / mismatched domains correlate with fraud. |
| `R_emaildomain` | categorical | **Recipient** email domain. |
| `C1`–`C14` | numeric | **Counting** features — counts of entities associated with the card/account (e.g. how many addresses/phones/devices tied to this card). Definitions masked. High velocity here is strongly fraud-indicative. |
| `D1`–`D15` | numeric | **Timedelta** features — days between events (e.g. days since first/previous transaction on this card). Very short recurrence or brand-new cards skew fraud. |
| `M1`–`M9` | categorical (T/F) | **Match** flags — whether attributes agree (e.g. name on card vs billing, address match). A "no match" is a direct fraud signal. |
| `V1`–`V339` | numeric | **Vesta engineered features** — rich ranking/counting/entity-relationship signals built by Vesta's own fraud system. Highly predictive collectively but opaque individually; heavily correlated and many are sparse. |

---

## `train_identity.csv`

| Column | Type | Meaning in a payment-platform context |
|---|---|---|
| `TransactionID` | id | Join key to the transaction file. |
| `id_01`–`id_11` | numeric (anon) | Continuous identity/behavioural signals — network, timing, ratings collected by Vesta's fraud system and security partners. |
| `id_12`–`id_38` | categorical (anon) | Categorical identity signals — proxy/VPN flags, account/session attributes, device/browser consistency checks. |
| `DeviceType` | categorical | desktop / mobile. |
| `DeviceInfo` | categorical | Device / OS / browser string (high cardinality, dirty). Useful after parsing into OS, browser family, version. |

> Note: in the raw `test_identity.csv` the columns are sometimes named with a
> dash (`id-01`) rather than underscore (`id_01`). Normalise on load.

---

## Columns most likely to be predictive of fraud

Prioritise these in the feature pipeline (`src/features/`):

1. **`C1`–`C14` (counting / velocity)** — account/card velocity is the single
   strongest behavioural fraud signal on this dataset. Engineer ratios and
   rolling windows.
2. **`V1`–`V339` (Vesta engineered)** — collectively the highest-importance
   block in published solutions. Reduce dimensionality (correlation pruning /
   grouped aggregates) rather than dropping.
3. **`D1`–`D15` (timedeltas)** — "card age" and recurrence gaps; brand-new
   cards and rapid re-use are high-risk.
4. **`card1`–`card6`** — card identity + network + debit/credit. Aggregate
   target-rate and frequency encodings on `card1` are very strong.
5. **`addr1`/`addr2` + `dist1`/`dist2`** — geographic mismatch / improbable
   distance between billing, shipping and IP.
6. **`P_emaildomain` / `R_emaildomain`** — disposable, rare, or
   purchaser/recipient-mismatched domains.
7. **`TransactionAmt` + `ProductCD`** — amount distribution differs sharply by
   class and product line; engineer per-product amount z-scores.
8. **`id_*` + `DeviceInfo` / `DeviceType`** — device/network fingerprint
   inconsistency (proxy/VPN, new device, OS/browser anomalies). Strong when an
   identity row is present.
9. **`M1`–`M9` (match flags)** — explicit "attributes don't match" indicators;
   low cardinality, directly interpretable, good for the model card.
10. **`TransactionDT`-derived** — hour-of-day / day-of-week (relative) capture
    fraud's temporal concentration.

### Watch-outs (governance / FCA model card relevance)

- **Leakage risk:** some `D*`/`V*` and `TransactionDT`-derived features can
  encode future or label-correlated information. Validate with time-based
  splits, not random splits.
- **Anonymised features ≠ explainable:** `V*`/`id_*` are powerful but opaque.
  For an FCA-aligned model card, pair them with interpretable signals
  (`M*`, amount, email, distance) and SHAP attributions.
- **Identity coverage bias:** identity columns are missing for a large share of
  transactions; "missingness" itself is predictive — encode it explicitly
  rather than naively imputing.
- **High cardinality:** `card1`, `addr1`, `*_emaildomain`, `DeviceInfo` need
  frequency / target encoding with out-of-fold discipline to avoid overfitting.
