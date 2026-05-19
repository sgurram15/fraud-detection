"""Download the IEEE-CIS Fraud Detection dataset from Kaggle into data/raw.

Uses `kagglehub`, the modern Kaggle client. IEEE-CIS Fraud Detection is a
Kaggle *competition*, so this uses `competition_download`.

Authentication (kagglehub picks the first that works):
  1. KAGGLE_API_TOKEN env var (single token) -- what this project uses.
  2. API token file at ~/.kaggle/access_token
  3. Legacy kaggle.json at ~/.kaggle/kaggle.json
  4. kagglehub.login() interactive prompt

You must also accept the competition rules once on the website, or Kaggle
returns a 403:
    https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path(__file__).resolve().parent / "raw"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    import kagglehub

    print(f"Downloading '{COMPETITION}' via kagglehub ...")
    src = Path(kagglehub.competition_download(COMPETITION))

    # kagglehub extracts into its cache and returns that directory. Flatten
    # all CSVs directly into data/raw (no nested competition subfolder) so the
    # rest of the project can rely on a stable, predictable location.
    print(f"Copying CSVs from {src} to {RAW_DIR} ...")
    csvs = sorted(src.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {src}.")
    for item in csvs:
        shutil.copy2(item, RAW_DIR / item.name)

    print("Done. Files in data/raw:")
    for f in sorted(RAW_DIR.iterdir()):
        if f.name != ".gitkeep":
            print(f"  {f.name}")


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nChecklist:\n"
            "  1. pip install kagglehub\n"
            "  2. KAGGLE_API_TOKEN env var set (or ~/.kaggle credentials)\n"
            f"  3. Competition rules accepted at "
            f"https://www.kaggle.com/competitions/{COMPETITION}/rules",
            file=sys.stderr,
        )
        sys.exit(1)
