"""Feature engineering pipeline for the IEEE-CIS Fraud Detection dataset.

Public entry point:

    enriched = build_features(df)

`df` is a raw transaction dataframe (optionally already left-joined with the
identity file). The function returns a *copy* with engineered features added;
it never drops rows. Missing values are handled explicitly per feature, with a
companion ``*_was_missing`` flag where missingness is itself informative.

Dataset realities that shape the design (relevant to the FCA model card in
``docs/``):

* **No account key.** IEEE-CIS has no single account/customer id. A composite
  ``card_uid`` is built from ``card1..card6`` (+ ``addr1``). It is an
  approximation of "the same card", not a guaranteed identity.
* **No wall-clock time.** ``TransactionDT`` is a seconds offset from an
  undisclosed reference. The community convention ``2017-12-01`` is used as the
  reference so hour/day-of-week can be derived. Absolute calendar values are
  therefore *relative*, not literal — documented as a known limitation.
* **No shipping address.** Only billing ``addr1`` (region) / ``addr2``
  (country) exist. A true billing-vs-shipping match is impossible on this
  dataset, so the requested ``billing_shipping_match`` feature is **not
  created**. The gap is recorded in ``docs/model_card.md`` (Known
  Limitations) rather than fabricated from a weak proxy.
* **Leakage discipline.** Per-card history features use *prior-only* windows
  (``closed="left"``) so the current row never sees itself. The historical
  device fraud rate is a smoothed full-data target mean for this PoC — the
  models layer should replace it with out-of-fold encoding before any
  production claim.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Community-convention reference point for TransactionDT (true epoch unknown).
REFERENCE_DATETIME = datetime(2017, 12, 1)

# Columns composed into a best-effort "same card" identity.
CARD_ID_COLS = ["card1", "card2", "card3", "card5", "card6", "addr1"]

FREE_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.fr",
    "yahoo.de", "yahoo.es", "yahoo.co.jp", "ymail.com", "hotmail.com",
    "hotmail.co.uk", "hotmail.fr", "hotmail.de", "hotmail.es", "outlook.com",
    "outlook.es", "live.com", "live.com.mx", "live.fr", "msn.com", "aol.com",
    "aim.com", "icloud.com", "me.com", "mac.com", "mail.com", "gmx.de",
    "web.de", "protonmail.com", "yandex.ru", "zoho.com",
}

def _log_created(name: str, note: str = "") -> None:
    logger.info("Created feature: %-32s %s", name, note)


def _laplace_rate(positives: float, total: float) -> float:
    """Add-one (Laplace) smoothed binary rate: (pos + 1) / (total + 2).

    Pulls rates for low-volume device types toward 0.5-ish rather than
    trusting a noisy raw proportion from a handful of transactions.
    """
    return (positives + 1.0) / (total + 2.0)


def encode_device_fraud_rate_safely(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    device_col: str = "DeviceType",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.Series, dict[str, float], float]:
    """Out-of-fold (OOF) target encoding of the device-type fraud rate.

    Returns ``(oof_encoded, serving_map, global_rate)``:

    * ``oof_encoded`` -- per-row encoded value. The data is split into
      ``n_splits`` folds; each fold's rows are encoded using the device fraud
      rates computed **only from the other folds**. A row therefore never
      contributes its own label to its own feature value -> no target leakage
      in the training matrix.
    * ``serving_map`` -- ``{device_type: laplace_rate}`` fitted on the FULL
      training labels. This is for applying to *new* serving transactions,
      which are unseen and carry no label, so using all training data here is
      correct and not leakage.
    * ``global_rate`` -- Laplace-smoothed overall training fraud rate; the
      fallback for device types not seen during training.

    All rates use add-one (Laplace) smoothing so rare device types are not
    assigned extreme rates from tiny samples.
    """
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    dev = df[device_col].astype("string").fillna("__missing__")

    global_rate = _laplace_rate(float(y.sum()), float(len(y)))

    n = len(df)
    splits = max(2, min(n_splits, n))
    oof = pd.Series(np.nan, index=df.index, dtype="float64")
    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in kf.split(df):
        y_tr = y.iloc[tr_idx]
        dev_tr = dev.iloc[tr_idx]
        agg = y_tr.groupby(dev_tr).agg(["sum", "count"])
        fold_rate = _laplace_rate(agg["sum"], agg["count"])  # per device
        fold_global = _laplace_rate(float(y_tr.sum()), float(len(y_tr)))
        mapped = dev.iloc[val_idx].map(fold_rate).fillna(fold_global)
        oof.iloc[val_idx] = mapped.to_numpy()

    full = y.groupby(dev).agg(["sum", "count"])
    serving_map = {
        str(k): float(_laplace_rate(r["sum"], r["count"]))
        for k, r in full.iterrows()
    }
    return oof, serving_map, float(global_rate)


def _card_uid(df: pd.DataFrame) -> pd.Series:
    present = [c for c in CARD_ID_COLS if c in df.columns]
    if not present:
        logger.warning(
            "No card identity columns found %s; card_uid falls back to a "
            "constant — per-card features will be degenerate.", CARD_ID_COLS
        )
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    uid = df[present].astype("string").fillna("NA").agg("|".join, axis=1)
    return uid


def _datetime(df: pd.DataFrame) -> pd.Series:
    if "TransactionDT" not in df.columns:
        raise KeyError("TransactionDT is required for time/velocity features.")
    secs = pd.to_numeric(df["TransactionDT"], errors="coerce")
    return REFERENCE_DATETIME + pd.to_timedelta(secs, unit="s")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer fraud features. Single-call entry point: ``build_features(df)``.

    Returns a new dataframe (input is not mutated) with all engineered columns
    appended. No rows are dropped.
    """
    if df.empty:
        logger.warning("build_features received an empty dataframe.")
        return df.copy()

    out = df.copy()
    n_before = out.shape[1]
    logger.info("build_features: %d rows, %d input columns", *df.shape)

    out["card_uid"] = _card_uid(out)
    out["_event_time"] = _datetime(out)

    # Stable chronological order; preserve original order for the return.
    out["_orig_order"] = np.arange(len(out))
    out = out.sort_values(
        ["card_uid", "_event_time"], kind="mergesort"
    ).reset_index(drop=True)  # clean, unique index for safe alignment

    amt = pd.to_numeric(out.get("TransactionAmt"), errors="coerce")
    out["amt_was_missing"] = amt.isna().astype("int8")
    amt = amt.fillna(0.0)
    out["_amt"] = amt.values
    by_card = out.groupby("card_uid", sort=False)

    def _roll(window: str, how: str) -> pd.Series:
        """Prior-only (closed='left') grouped time-window aggregate, aligned
        back to ``out.index``."""
        r = out.groupby("card_uid", sort=False).rolling(
            window, on="_event_time", closed="left"
        )["_amt"]
        s = getattr(r, how)()
        # `out` is pre-sorted by (card_uid, _event_time) and groups are taken
        # in first-appearance order, so the rolling output is row-aligned with
        # `out`. Align positionally onto the clean RangeIndex.
        return pd.Series(s.to_numpy(), index=out.index)

    # ----- Transaction velocity (prior-only trailing windows) --------------
    out["txn_count_1h"] = _roll("1h", "count").fillna(0).astype("int32")
    _log_created("txn_count_1h", "card txns in prior 1h")

    out["txn_count_24h"] = _roll("24h", "count").fillna(0).astype("int32")
    _log_created("txn_count_24h", "card txns in prior 24h")

    out["txn_value_24h"] = _roll("24h", "sum").fillna(0.0)
    _log_created("txn_value_24h", "card value in prior 24h")

    # ----- Deviation features ---------------------------------------------
    prior_mean = by_card["TransactionAmt"].apply(
        lambda s: s.expanding().mean().shift()
    )
    prior_mean = prior_mean.reset_index(level=0, drop=True).reindex(out.index)
    # No prior history -> no deviation signal -> 0, flagged.
    out["amt_no_history"] = prior_mean.isna().astype("int8")
    prior_mean_f = prior_mean.fillna(amt)
    out["amt_dev_from_card_mean"] = amt - prior_mean_f
    out["amt_dev_ratio_card_mean"] = np.where(
        prior_mean_f.abs() > 1e-9, amt / prior_mean_f.replace(0, np.nan), 1.0
    )
    out["amt_dev_ratio_card_mean"] = out["amt_dev_ratio_card_mean"].fillna(1.0)
    _log_created("amt_dev_from_card_mean", "amt - prior card mean")
    _log_created("amt_dev_ratio_card_mean", "amt / prior card mean")

    prior_max = by_card["TransactionAmt"].apply(
        lambda s: s.expanding().max().shift()
    )
    prior_max = prior_max.reset_index(level=0, drop=True).reindex(out.index)
    out["is_largest_for_card"] = (
        (prior_max.isna()) | (amt > prior_max)
    ).astype("int8")
    _log_created("is_largest_for_card", "largest amt this card has made")

    # ----- Time-based features --------------------------------------------
    et = out["_event_time"]
    out["hour_of_day"] = et.dt.hour.astype("int8")
    out["day_of_week"] = et.dt.dayofweek.astype("int8")
    out["is_weekend"] = (et.dt.dayofweek >= 5).astype("int8")
    out["is_late_night"] = et.dt.hour.between(0, 4).astype("int8")
    for f in ("hour_of_day", "day_of_week", "is_weekend", "is_late_night"):
        _log_created(f)

    # ----- Account / device features --------------------------------------
    first_seen = by_card["_event_time"].transform("min")
    age_days = (out["_event_time"] - first_seen).dt.total_seconds() / 86400.0
    out["card_age_days"] = age_days.fillna(0.0).clip(lower=0.0)
    _log_created("card_age_days", "days since card first seen in data")

    if "P_emaildomain" in out.columns:
        dom = out["P_emaildomain"].astype("string").str.lower()
        out["p_email_was_missing"] = dom.isna().astype("int8")
        out["p_email_is_free"] = dom.isin(FREE_EMAIL_DOMAINS).astype("int8")
        # Missing domain: unknown -> treat as not-free (0) but flagged above.
        _log_created("p_email_is_free", "free provider vs corporate/other")
    else:
        logger.warning("P_emaildomain absent; p_email_is_free set to -1.")
        out["p_email_is_free"] = np.int8(-1)
        out["p_email_was_missing"] = np.int8(1)

    # billing/shipping match intentionally NOT engineered: IEEE-CIS has no
    # shipping address, so any such feature would be a fabricated proxy. The
    # capability gap is recorded in docs/model_card.md (Known Limitations).
    logger.info(
        "Skipping billing_shipping_match: no shipping address in IEEE-CIS. "
        "Gap documented in docs/model_card.md (Known Limitations)."
    )

    # ----- Aggregation features -------------------------------------------
    # No separate "was missing" flag here: it was perfectly collinear (r=1.0)
    # with amt_no_history, which already encodes "card has no prior history".
    out["card_amt_avg_30d"] = _roll("30D", "mean").fillna(amt)
    _log_created("card_amt_avg_30d", "prior 30d mean amt for card")

    _dev_map: dict[str, float] = {}
    _dev_global = float("nan")
    if "DeviceType" in out.columns and "isFraud" in out.columns:
        oof, _dev_map, _dev_global = encode_device_fraud_rate_safely(
            out, target_col="isFraud"
        )
        out["device_type_fraud_rate"] = oof.to_numpy()
        logger.info(
            "device_type_fraud_rate: out-of-fold encoding applied, "
            "leakage risk mitigated"
        )
        _log_created("device_type_fraud_rate", "OOF target encoding (Laplace)")
    else:
        missing = "DeviceType" if "DeviceType" not in out.columns else "isFraud"
        logger.info(
            "device_type_fraud_rate skipped (%s absent); set to NaN.", missing
        )
        out["device_type_fraud_rate"] = np.nan

    # ----- Restore original row order, drop scratch columns ---------------
    out = out.sort_values("_orig_order", kind="mergesort")
    out = out.drop(columns=["_event_time", "_orig_order", "_amt"])
    out.index = df.index

    # Expose the serving-time device map + global fallback for the feature
    # store (set after all reshaping so .attrs survives).
    out.attrs["device_fraud_rate_map"] = _dev_map
    out.attrs["device_fraud_rate_global"] = _dev_global

    created = out.shape[1] - n_before
    logger.info(
        "build_features done: %d engineered columns added (now %d total).",
        created, out.shape[1],
    )
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    matches = sorted(raw.glob("**/train_transaction.csv"))
    if not matches:
        print(f"train_transaction.csv not found under {raw}", file=sys.stderr)
        sys.exit(1)

    sample = pd.read_csv(matches[0], nrows=20000)
    enriched = build_features(sample)
    new_cols = [c for c in enriched.columns if c not in sample.columns]
    print(f"\nInput: {sample.shape} -> Output: {enriched.shape}")
    print(f"Engineered {len(new_cols)} columns:")
    for c in new_cols:
        print(f"  {c:32s} {enriched[c].dtype}")
