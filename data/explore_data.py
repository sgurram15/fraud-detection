"""Quick exploratory summary of the IEEE-CIS Fraud Detection dataset.

Reads the training data from data/raw (train_transaction.csv, optionally
merged with train_identity.csv on TransactionID) and prints:
  - dataset shape
  - fraud vs legitimate ratio
  - missing values per column
  - transaction-amount statistics
  - a sample of 5 fraudulent transactions

Run `python data/download_data.py` first to populate data/raw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET = "isFraud"
AMOUNT = "TransactionAmt"


def _find(name: str) -> Path | None:
    """Locate a CSV directly in data/raw or in a subfolder (kagglehub puts
    competition files under data/raw/ieee-fraud-detection/)."""
    direct = RAW_DIR / name
    if direct.exists():
        return direct
    matches = sorted(RAW_DIR.glob(f"**/{name}"))
    return matches[0] if matches else None


def _load() -> pd.DataFrame:
    txn_path = _find("train_transaction.csv")
    if txn_path is None:
        raise FileNotFoundError(
            f"train_transaction.csv not found under {RAW_DIR}. "
            "Run `python data/download_data.py` first."
        )

    df = pd.read_csv(txn_path)

    id_path = _find("train_identity.csv")
    if id_path is not None:
        identity = pd.read_csv(id_path)
        df = df.merge(identity, on="TransactionID", how="left")
        print(f"Merged train_identity.csv ({identity.shape[1]} id columns).")

    return df


def explore(df: pd.DataFrame) -> None:
    print("\n=== Shape ===")
    print(f"{df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print("\n=== Fraud vs legitimate ===")
    counts = df[TARGET].value_counts().sort_index()
    legit = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    total = legit + fraud
    print(f"Legitimate (0): {legit:,} ({legit / total:.4%})")
    print(f"Fraud      (1): {fraud:,} ({fraud / total:.4%})")
    if fraud:
        print(f"Imbalance ratio: 1 fraud per {legit / fraud:.1f} legitimate")

    print("\n=== Missing values per column (columns with >0 missing) ===")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"missing": missing, "pct": pct})
        with pd.option_context(
            "display.max_rows", None, "display.width", 100
        ):
            print(report)
        print(
            f"\n{len(missing)} of {df.shape[1]} columns have missing values."
        )

    print("\n=== Transaction amount statistics ===")
    print(df[AMOUNT].describe())
    print(f"median: {df[AMOUNT].median():.2f}")
    print("\nBy class (mean / median):")
    print(df.groupby(TARGET)[AMOUNT].agg(["mean", "median", "max"]))

    print("\n=== Sample of 5 fraudulent transactions ===")
    fraud_rows = df[df[TARGET] == 1]
    if fraud_rows.empty:
        print("No fraudulent transactions found.")
    else:
        cols = [
            c
            for c in ["TransactionID", "TransactionDT", AMOUNT,
                      "ProductCD", "card4", "card6", TARGET]
            if c in df.columns
        ]
        sample = fraud_rows.sample(n=min(5, len(fraud_rows)), random_state=42)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    try:
        explore(_load())
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
