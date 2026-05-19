"""Minimal feature store for the PSP fraud PoC.

Sits (conceptually) between Kafka and the model. It solves the core
online/offline skew problem: history-dependent features (velocity, running
mean, card age, 30d average) are trivial to compute over a batch but must be
served in O(1) for a single live transaction.

Two modes:

* **Training mode** -- ``fit_batch(df)``: runs the offline pipeline
  (``build_features``) to materialise the engineered table in a structured
  format (Parquet), and replays the batch chronologically to build the
  per-card online state + the device-type fraud-rate map. State is persisted
  so serving can warm-start from real history.

* **Serving mode** -- ``get_features(transaction: dict) -> dict``: computes
  the same features for one incoming transaction from the maintained state in
  constant time, logs the computation latency, then folds the transaction
  into the state so the next event sees it.

Feature definitions are *prior-only* (they reflect history strictly BEFORE the
current transaction), matching ``build_features``'s ``closed="left"`` /
expanding-``.shift()`` semantics so offline-trained models see the same
distribution online.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Allow running directly (`python src/features/feature_store.py`): ensure the
# project root is importable so `src.*` resolves without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_features import (
    CARD_ID_COLS,
    FREE_EMAIL_DOMAINS,
    REFERENCE_DATETIME,
    build_features,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_STORE_DIR = _ROOT / "data" / "processed" / "feature_store"
_FEATURES_PATH = _STORE_DIR / "training_features.parquet"
_STATE_PATH = _STORE_DIR / "online_state.pkl"

_H1 = 3600
_D1 = 86400
_D30 = 30 * 86400
TARGET = "isFraud"


class _CardState:
    """Compact rolling state for one card_uid. Memory is bounded: the event
    deque only retains the last 30 days."""

    __slots__ = ("events", "n_total", "sum_total", "max_total", "first_dt")

    def __init__(self) -> None:
        self.events: deque[tuple[float, float]] = deque()  # (dt_sec, amount)
        self.n_total = 0
        self.sum_total = 0.0
        self.max_total = float("-inf")
        self.first_dt: float | None = None

    def prune(self, now: float) -> None:
        cutoff = now - _D30
        ev = self.events
        while ev and ev[0][0] < cutoff:
            ev.popleft()

    def update(self, dt: float, amt: float) -> None:
        self.events.append((dt, amt))
        self.n_total += 1
        self.sum_total += amt
        self.max_total = max(self.max_total, amt)
        if self.first_dt is None:
            self.first_dt = dt
        self.prune(dt)


def _card_uid_from_dict(txn: dict) -> str:
    parts = [str(txn.get(c)) if txn.get(c) is not None else "NA"
             for c in CARD_ID_COLS]
    return "|".join(parts) if parts else "UNKNOWN"


class FeatureStore:
    def __init__(self) -> None:
        self._state: dict[str, _CardState] = defaultdict(_CardState)
        self._device_rates: dict[str, float] = {}
        self._global_prior: float = 0.0
        self._max_dt: float = 0.0
        self._last_latency_ms: float | None = None

    # ---------------- Training mode ----------------
    def fit_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process historical batch data: materialise engineered features and
        build the online serving state. Returns the engineered DataFrame."""
        logger.info("Training mode: engineering features for %d rows", len(df))
        enriched = build_features(df)

        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            enriched.to_parquet(_FEATURES_PATH, index=False)
            logger.info("Stored engineered features -> %s", _FEATURES_PATH)
        except Exception as exc:  # pyarrow missing -> CSV fallback
            csv = _FEATURES_PATH.with_suffix(".csv")
            enriched.to_csv(csv, index=False)
            logger.warning("Parquet unavailable (%s); wrote %s", exc, csv)

        # Consume the PRECOMPUTED out-of-fold device map from training
        # (build_features attaches it via .attrs). The feature store no longer
        # recomputes its own rate -- single, leakage-safe source of truth.
        self._device_rates = dict(
            enriched.attrs.get("device_fraud_rate_map", {})
        )
        self._global_prior = float(
            enriched.attrs.get("device_fraud_rate_global", float("nan"))
        )
        if self._device_rates:
            logger.info(
                "Loaded OOF device fraud-rate map (%d device types, "
                "global fallback=%.5f)",
                len(self._device_rates), self._global_prior,
            )
        else:
            logger.warning("No DeviceType/label; device rates unavailable.")

        # Replay chronologically to build per-card rolling state.
        cols = [c for c in (["TransactionDT", "TransactionAmt"] + CARD_ID_COLS)
                if c in df.columns]
        rep = df[cols].sort_values("TransactionDT", kind="mergesort")
        for row in rep.itertuples(index=False):
            d = dict(zip(cols, row))
            dt = float(d.get("TransactionDT") or 0.0)
            amt = float(d.get("TransactionAmt") or 0.0)
            self._state[_card_uid_from_dict(d)].update(dt, amt)
            self._max_dt = max(self._max_dt, dt)

        logger.info("Built online state for %d cards", len(self._state))
        return enriched

    def save(self) -> None:
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_STATE_PATH, "wb") as fh:
            pickle.dump(
                {
                    "state": dict(self._state),
                    "device_rates": self._device_rates,
                    "global_prior": self._global_prior,
                    "max_dt": self._max_dt,
                },
                fh,
            )
        logger.info("Persisted online state -> %s", _STATE_PATH)

    def load(self) -> "FeatureStore":
        if not _STATE_PATH.exists():
            raise FileNotFoundError(
                f"{_STATE_PATH} not found. Run fit_batch()+save() first."
            )
        with open(_STATE_PATH, "rb") as fh:
            blob = pickle.load(fh)
        self._state = defaultdict(_CardState, blob["state"])
        self._device_rates = blob["device_rates"]
        self._global_prior = blob["global_prior"]
        self._max_dt = blob["max_dt"]
        logger.info("Loaded state for %d cards", len(self._state))
        return self

    # ---------------- Serving mode ----------------
    def get_features(self, transaction: dict) -> dict:
        """Compute features for ONE live transaction in O(1). Logs latency."""
        t0 = time.perf_counter()

        amt_raw = transaction.get("TransactionAmt")
        amt_missing = amt_raw is None
        amt = float(amt_raw) if not amt_missing else 0.0

        dt = transaction.get("TransactionDT")
        dt = float(dt) if dt is not None else self._max_dt + 1.0

        uid = _card_uid_from_dict(transaction)
        st = self._state.get(uid)

        # ---- velocity / aggregation from prior events (state pre-update) ----
        if st is not None and st.events:
            st.prune(dt)
            ev = st.events
            cnt_1h = sum(1 for e in ev if e[0] > dt - _H1)
            in_24h = [e[1] for e in ev if e[0] > dt - _D1]
            in_30d = [e[1] for e in ev if e[0] > dt - _D30]
            cnt_24h = len(in_24h)
            val_24h = float(sum(in_24h))
            avg_30d = float(sum(in_30d) / len(in_30d)) if in_30d else amt
        else:
            cnt_1h = cnt_24h = 0
            val_24h = 0.0
            avg_30d = amt

        if st is not None and st.n_total > 0:
            prior_mean = st.sum_total / st.n_total
            dev = amt - prior_mean
            ratio = amt / prior_mean if abs(prior_mean) > 1e-9 else 1.0
            is_largest = int(amt > st.max_total)
            no_history = 0
            card_age_days = max(0.0, (dt - (st.first_dt or dt)) / 86400.0)
        else:
            dev, ratio, is_largest, no_history, card_age_days = (
                0.0, 1.0, 1, 1, 0.0
            )

        # ---- time-based ----
        when = REFERENCE_DATETIME + timedelta(seconds=dt)
        # ---- email / device ----
        dom = transaction.get("P_emaildomain")
        p_email_missing = int(dom is None)
        p_email_is_free = (
            int(str(dom).lower() in FREE_EMAIL_DOMAINS) if dom is not None
            else -1
        )
        dev_type = transaction.get("DeviceType")
        device_rate = self._device_rates.get(
            str(dev_type), self._global_prior
        ) if self._device_rates else float("nan")

        features = {
            "card_uid": uid,
            "amt_was_missing": int(amt_missing),
            "txn_count_1h": cnt_1h,
            "txn_count_24h": cnt_24h,
            "txn_value_24h": val_24h,
            "amt_no_history": no_history,
            "amt_dev_from_card_mean": dev,
            "amt_dev_ratio_card_mean": ratio,
            "is_largest_for_card": is_largest,
            "hour_of_day": when.hour,
            "day_of_week": when.weekday(),
            "is_weekend": int(when.weekday() >= 5),
            "is_late_night": int(0 <= when.hour <= 4),
            "card_age_days": card_age_days,
            "p_email_was_missing": p_email_missing,
            "p_email_is_free": p_email_is_free,
            "card_amt_avg_30d": avg_30d,
            "device_type_fraud_rate": device_rate,
        }

        # Fold this transaction into the state (now visible to the next event).
        self._state[uid].update(dt, amt)
        self._max_dt = max(self._max_dt, dt)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "get_features: card=%s latency=%.3f ms", uid, self._last_latency_ms
        )
        return features


# Module-level singleton + convenience function with the requested signature.
_DEFAULT_STORE: FeatureStore | None = None


def get_features(transaction: dict) -> dict:
    """Serving entry point. Lazily loads the persisted store on first call."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = FeatureStore().load()
    return _DEFAULT_STORE.get_features(transaction)


if __name__ == "__main__":
    import glob

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    txn_csv = glob.glob(
        str(_ROOT / "data" / "raw" / "**" / "train_transaction.csv"),
        recursive=True,
    )
    if not txn_csv:
        raise SystemExit("Raw data not found; run data/download_data.py.")

    sample = pd.read_csv(txn_csv[0], nrows=20000)
    store = FeatureStore()
    store.fit_batch(sample)
    store.save()

    # Simulate live serving on a few held-out raw transactions.
    serving = FeatureStore().load()
    demo = pd.read_csv(txn_csv[0], skiprows=range(1, 20001), nrows=5)
    demo.columns = sample.columns
    print("\n--- Serving simulation ---")
    for rec in demo.to_dict(orient="records"):
        feats = serving.get_features(rec)
        print(
            f"amt={rec['TransactionAmt']:>8} "
            f"v1h={feats['txn_count_1h']} v24h={feats['txn_count_24h']} "
            f"dev={feats['amt_dev_from_card_mean']:.2f} "
            f"age={feats['card_age_days']:.1f}d "
            f"devrate={feats['device_type_fraud_rate']}"
        )
