"""Class-imbalance handling for the IEEE-CIS fraud feature set.

Fraud is ~3.5% of transactions, so naive training is dominated by the
legitimate class. This module:

  * loads the feature-engineered dataset (building it from raw if needed),
  * reports the fraud / legitimate ratio,
  * applies three balancing strategies and saves each to data/processed/,
  * exposes ``prepare_train_test_split(df, strategy)`` for modelling.

LEAKAGE POLICY (important):
  Resampling is a *training-time* operation. ``prepare_train_test_split``
  therefore splits FIRST and resamples the TRAIN partition ONLY. The test
  partition keeps the real, untouched class distribution so evaluation
  metrics are honest. The standalone saved datasets (run as __main__) apply
  resampling to the whole feature set for class-distribution inspection /
  experimentation only -- they must NOT be re-split for evaluation. Use
  ``prepare_train_test_split`` for any modelling claim.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "isFraud"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import USE_S3, data_path, processed_path

# A8: data/processed + data/raw roots come from src/config. This module reads
# raw CSVs via Path.glob and persists parquet locally, so USE_S3=true is
# refused at the local-FS entry points (see load_features / __main__);
# S3 read/write is a separate follow-up (needs s3fs/boto3).
_RAW_DIR = _ROOT / data_path()
_PROC_DIR = _ROOT / processed_path()
_FEATURES_PATH = _PROC_DIR / "features.parquet"

# Columns that must never be model inputs (ids / target / leakage-prone keys).
_NON_FEATURE_COLS = {
    TARGET, "TransactionID", "TransactionDT", "card_uid",
}

STRATEGIES = ("smote", "undersample", "combined")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:  # pyarrow/fastparquet missing -> CSV fallback
        logger.warning("Parquet save failed (%s); falling back to CSV.", exc)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_features() -> pd.DataFrame:
    """Load the feature-engineered dataset.

    Prefers a cached data/processed/features.parquet. If absent, builds
    features from the raw IEEE-CIS files (transaction + identity) via
    ``build_features`` and caches the result for reuse.
    """
    if USE_S3:
        raise NotImplementedError(
            "handle_imbalance.load_features() is local-FS only (Path.glob + "
            "parquet cache); run with USE_S3=false. S3 read/write is a "
            "separate follow-up (needs s3fs/boto3)."
        )
    if _FEATURES_PATH.exists():
        logger.info("Loading cached features from %s", _FEATURES_PATH)
        return _read_table(_FEATURES_PATH)
    csv_cache = _FEATURES_PATH.with_suffix(".csv")
    if csv_cache.exists():
        logger.info("Loading cached features from %s", csv_cache)
        return _read_table(csv_cache)

    logger.info("No cached features; building from raw IEEE-CIS data ...")
    txn = sorted(_RAW_DIR.glob("**/train_transaction.csv"))
    if not txn:
        raise FileNotFoundError(
            f"train_transaction.csv not found under {_RAW_DIR}. "
            "Run `python data/download_data.py` first."
        )
    df = pd.read_csv(txn[0])
    ident = sorted(_RAW_DIR.glob("**/train_identity.csv"))
    if ident:
        df = df.merge(pd.read_csv(ident[0]), on="TransactionID", how="left")

    # Imported lazily so this module is usable without the features module
    # on the path in every context.
    from src.features.build_features import build_features

    df = build_features(df)
    saved = _save_table(df, _FEATURES_PATH)
    logger.info("Cached engineered features to %s", saved)
    return df


# --------------------------------------------------------------------------- #
# Matrix preparation
# --------------------------------------------------------------------------- #
def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric feature matrix + target.

    SMOTE needs a fully numeric, NaN-free matrix. For this PoC we keep numeric
    columns only (object/categorical raw columns such as ProductCD, card4/6,
    *_emaildomain, M*, DeviceInfo are dropped here -- encoding them belongs in
    the modelling layer). Imputation is applied by the caller so that
    train/test fitting stays leak-free.
    """
    if TARGET not in df.columns:
        raise KeyError(f"Target '{TARGET}' not in dataframe.")
    y = pd.to_numeric(df[TARGET], errors="coerce").astype("int8")

    feat = df.drop(columns=[c for c in _NON_FEATURE_COLS if c in df.columns])
    feat = feat.select_dtypes(include=[np.number])
    if feat.shape[1] == 0:
        raise ValueError("No numeric feature columns available.")
    return feat, y


def _ratio_str(y: pd.Series | np.ndarray) -> str:
    y = pd.Series(np.asarray(y))
    counts = y.value_counts().sort_index()
    legit = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    total = legit + fraud
    pos = f"{fraud / total:.3%}" if total else "n/a"
    per = f"1:{legit / fraud:.1f}" if fraud else "1:inf"
    return (
        f"legit={legit:,}  fraud={fraud:,}  "
        f"fraud_pct={pos}  ratio(fraud:legit)={per}"
    )


# --------------------------------------------------------------------------- #
# Resamplers
# --------------------------------------------------------------------------- #
def _make_resampler(strategy: str):
    """Return an imblearn resampler (or pipeline) for the named strategy.

    - smote:       oversample minority to parity (1:1).
    - undersample: randomly drop majority to parity (1:1).
    - combined:    SMOTE minority up to 50% of majority, then undersample
                   majority to parity. This is the recommended recipe (per the
                   original SMOTE paper / imbalanced-learn guidance): synthetic
                   oversampling without exploding to full parity, cleaned up
                   with mild undersampling.
    """
    strategy = strategy.lower()
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if strategy == "combined":
        from imblearn.pipeline import Pipeline

        return Pipeline([
            ("smote", SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),
            ("under", RandomUnderSampler(
                sampling_strategy=1.0, random_state=RANDOM_STATE)),
        ])
    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}."
    )


def _resample(strategy: str, X, y):
    sampler = _make_resampler(strategy)
    return sampler.fit_resample(X, y)


# --------------------------------------------------------------------------- #
# Public modelling API
# --------------------------------------------------------------------------- #
def prepare_train_test_split(
    df: pd.DataFrame,
    strategy: str = "combined",
    test_size: float = 0.20,
):
    """Split 80/20 then apply ``strategy`` to the TRAIN partition only.

    Order of operations (leak-free):
      1. Build numeric X / y.
      2. Stratified train/test split (test stays at the real fraud rate).
      3. Median-impute: imputer fitted on TRAIN, applied to both partitions.
      4. Resample TRAIN only ('none' skips resampling).

    Returns ``X_train, X_test, y_train, y_test`` (X as DataFrames, y as
    Series). The test set is never resampled or fitted on, so there is no
    information leakage between splits.
    """
    X, y = _xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns, index=X_train.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns, index=X_test.index,
    )

    if strategy and strategy.lower() != "none":
        Xtr, ytr = _resample(strategy, X_train, y_train)
        X_train = pd.DataFrame(Xtr, columns=X.columns)
        y_train = pd.Series(ytr, name=TARGET)

    logger.info("Split strategy=%s", strategy)
    logger.info("  TRAIN  %s", _ratio_str(y_train))
    logger.info("  TEST   %s (untouched -- real distribution)",
                _ratio_str(y_test))
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# CLI: build the three balanced datasets
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    df = load_features()
    X, y = _xy(df)

    logger.info("Current distribution: %s", _ratio_str(y))

    # SMOTE needs NaN-free input; impute on the full set for the standalone
    # inspection artifacts only (the modelling path uses the leak-free split).
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
    )

    _PROC_DIR.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        logger.info("Applying strategy: %s ...", strategy)
        Xr, yr = _resample(strategy, X_imp, y)
        logger.info("  -> %s", _ratio_str(yr))

        balanced = pd.DataFrame(Xr, columns=X.columns)
        balanced[TARGET] = np.asarray(yr)
        out = _PROC_DIR / f"balanced_{strategy}.parquet"
        saved = _save_table(balanced, out)
        logger.info("  saved %s rows=%d", saved.name, len(balanced))

    logger.warning(
        "Saved datasets apply resampling to the FULL feature set for "
        "inspection only. For modelling use prepare_train_test_split() so "
        "the test split keeps the real fraud rate (no leakage)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
