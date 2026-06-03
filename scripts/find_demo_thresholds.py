"""C_DEMO_FIX — find 18-feature (API-surface) combinations that score > 0.70.

Why this exists: the production model expects the full 417-column IEEE-CIS
vector, but POST /score supplies only the ~18 engineered features and NaN-fills
the ~400 raw C/V/D/id/M columns the model also leans on. The score surface over
the API inputs is therefore NOT intuitively monotonic — the obvious
"£5000, new account, 3am" profile lands in REVIEW, not HOLD.

So we DON'T try to pass raw IEEE columns through the API. Instead we score
candidate transactions through the SAME path the demo uses (the in-process
FastAPI app, a fresh card_id per call for deterministic feature-store state)
and search the 18-feature space for combinations that reliably land in the
HOLD band (fraud_probability > 0.70). The winning combinations are then wired
into scripts/run_demo.py so the demo shows real HOLD decisions.

Run: python scripts/find_demo_thresholds.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

# Quieten the chatty per-request loggers before importing the app, so the
# threshold report is readable.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("src.features.feature_store", "fraud_api", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)
_CID = iter(range(1, 10 ** 9))

HOLD_BAND = 0.70  # decision band: prob > 0.70 -> HOLD

# Neutral baseline (a calm, low-risk transaction).
_BASE = {
    "amount": 100.0, "device_type": "mobile", "hour_of_day": 12,
    "day_of_week": 2, "destination_account_age_days": 100,
    "tx_velocity_1h": 0, "tx_velocity_24h": 0, "amt_deviation": 1.0,
    "is_late_night": False, "is_weekend": False, "card_age_days": 300,
}


def score(**over) -> tuple[float, str]:
    """Score one transaction through POST /score with a fresh card_id."""
    n = next(_CID)
    txn = {**_BASE, "transaction_id": f"FIND-{n}", "card_id": f"FINDCARD-{n}"}
    txn.update(over)
    b = client.post("/score", json=txn).json()
    return float(b["fraud_probability"]), b["decision"]


def sweep() -> None:
    """One-factor-at-a-time sweep from the baseline to see what moves score."""
    base_p, base_d = score()
    print(f"\nBaseline score: {base_p:.4f} ({base_d})")
    print("\nOne-factor sweeps (feature -> value: prob [decision]):")
    grids = {
        "amount": [50, 250, 1000, 1500, 2500, 5000],
        "card_age_days": [0, 30, 90, 200, 600],
        "tx_velocity_1h": [0, 2, 5, 9],
        "tx_velocity_24h": [0, 5, 12, 20],
        "amt_deviation": [0.5, 1.0, 3.0, 10.0, 20.0],
        "hour_of_day": [3, 11, 14, 21],
        "device_type": ["mobile", "desktop"],
        "destination_account_age_days": [1, 7, 100, 365],
    }
    for feat, values in grids.items():
        cells = []
        for v in values:
            p, d = score(**{feat: v})
            mark = "*" if p > HOLD_BAND else " "
            cells.append(f"{v}={p:.3f}[{d[0]}]{mark}")
        print(f"  {feat:<28} " + "  ".join(cells))


def search(max_hits: int = 12) -> list[dict]:
    """Grid-search the most influential features for HOLD combinations."""
    space = {
        "amount": [800, 1500, 2500],
        "card_age_days": [60, 200, 400],
        "tx_velocity_1h": [1, 2, 4],
        "tx_velocity_24h": [3, 5, 8],
        "amt_deviation": [0.5, 1.0, 2.0],
        "hour_of_day": [3, 11, 21],
        "device_type": ["mobile", "desktop"],
    }
    keys = list(space)
    hits: list[dict] = []
    for combo in itertools.product(*space.values()):
        over = dict(zip(keys, combo))
        p, d = score(**over)
        if p > HOLD_BAND:
            hits.append({**over, "prob": round(p, 4)})
    hits.sort(key=lambda h: h["prob"], reverse=True)
    print(f"\nGrid search: {len(hits)} HOLD combinations found "
          f"(prob > {HOLD_BAND}). Top {min(max_hits, len(hits))}:")
    for h in hits[:max_hits]:
        feats = {k: v for k, v in h.items() if k != "prob"}
        print(f"  prob={h['prob']:.4f}  {feats}")
    return hits


def verify_candidates() -> None:
    """Score the specific profiles wired into the demo (post-fix)."""
    print("\nDemo candidate verification:")
    # Final profiles wired into scripts/run_demo.py (DEMO-2, DEMO-3).
    candidates = {
        "DEMO-2 suspicious":    dict(amount=1500.0, device_type="mobile",
                                     hour_of_day=3, is_late_night=True,
                                     tx_velocity_1h=2, tx_velocity_24h=8,
                                     amt_deviation=2.0, card_age_days=60),
        "DEMO-3 high-risk":     dict(amount=2500.0, device_type="mobile",
                                     hour_of_day=3, is_late_night=True,
                                     tx_velocity_1h=4, tx_velocity_24h=8,
                                     amt_deviation=2.0, card_age_days=60),
    }
    for name, over in candidates.items():
        p, d = score(**over)
        print(f"  {name:<24} prob={p:.4f} -> {d}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    print("=" * 64)
    print("DEMO THRESHOLD FINDER — HOLD combos on the /score API surface")
    print("=" * 64)
    sweep()
    search()
    verify_candidates()
    return 0


if __name__ == "__main__":
    sys.exit(main())
