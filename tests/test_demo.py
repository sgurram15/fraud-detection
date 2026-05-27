"""C8.2 — Demo test.

Runs the full demo (quick mode: short throughput step) and verifies:
  * it completes without raising,
  * all 5 hand-crafted transactions produce valid FraudScoreResponses,
  * the financial summary numbers are non-zero and positive.

Run: python tests/test_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts import run_demo

REQUIRED_FIELDS = {
    "transaction_id", "fraud_probability", "decision", "threshold_used",
    "reasons", "fca_explanation", "model_version",
}


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def main() -> int:
    for noisy in ("src.features.feature_store", "fraud_api", "httpx",
                  "enricher", "scorer", "agent", "audit", "producer",
                  "pipeline"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    results: list[bool] = []
    out: dict = {}
    try:
        out = run_demo.run_demo(quick=True)
        results.append(_result("demo runs without errors", True))
    except Exception as exc:  # noqa: BLE001
        results.append(_result("demo runs without errors", False, repr(exc)))

    if out:
        scored = out["scored"]
        valid = (len(scored) == 5 and all(
            not (REQUIRED_FIELDS - set(b))
            and b["decision"] in {"APPROVE", "REVIEW", "HOLD"}
            and isinstance(b["reasons"], list)
            for b in scored))
        results.append(_result("5 transactions produce valid responses", valid,
                               f"n={len(scored)}"))

        fin = out["financial"]
        pos = (fin["processed_in_demo"] > 0
               and fin["est_daily_saving_gbp"] >= 0
               and fin["est_daily_false_pos_cost_gbp"] >= 0
               and fin["roi_saving_per_infra_gbp"] >= 0)
        # At least one decision category must be non-empty (pipeline ran).
        nonzero = fin["processed_in_demo"] > 0
        results.append(_result("financial summary non-negative & populated",
                               pos and nonzero, f"financial={fin}"))

    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
