"""C8.1 — End-to-end demo for a PSP CTO.

A narrated walk-through of the whole POC, runnable on a laptop with no AWS:

  1. System check          model loaded, API healthy
  2. Baseline comparison   baseline vs tuned at the cost-optimal operating point
  3. Live scoring demo     5 hand-crafted transactions, scored + explained
  4. Pipeline throughput   the streaming pipeline at 10 TPS with a live dashboard
  5. Financial impact      fraud caught, daily saving at scale, ROI
  6. Architecture summary  one-page ASCII of the production AWS design

Run:  python scripts/run_demo.py
      python scripts/run_demo.py --quick     (short throughput step, for tests)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_PERF_DIR = _ROOT / "docs" / "model_performance"
_COMPARISON = _PERF_DIR / "model_comparison.json"
_BASELINE_METRICS = _PERF_DIR / "baseline_metrics.json"

# Illustrative cost assumptions (docs/model_card.md — recalibrate for prod).
_FN_COST_GBP = 125.0
_FP_COST_GBP = 25.0
_DAILY_TXNS = 100_000
# Rough local-equivalent infra cost/day for ROI illustration (the AWS POC
# footprint: endpoint + MSK + monitoring ~ £12/day — see docs cost tables).
_DAILY_INFRA_GBP = 12.0


def _hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def _client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _txn(tid: str, **over) -> dict:
    base = {
        "transaction_id": tid, "card_id": f"CARD-{tid}", "amount": 100.0,
        "device_type": "mobile", "hour_of_day": 12, "day_of_week": 2,
        "destination_account_age_days": 100, "tx_velocity_1h": 0,
        "tx_velocity_24h": 0, "amt_deviation": 1.0, "is_late_night": False,
        "is_weekend": False, "card_age_days": 300,
    }
    base.update(over)
    return base


# Five hand-crafted transactions spanning the risk spectrum (mission C8.1).
_DEMO_TXNS = [
    ("1. Clean transaction",
     _txn("DEMO-1", amount=42.50, device_type="desktop", hour_of_day=14,
          tx_velocity_1h=0, tx_velocity_24h=2, amt_deviation=1.0,
          card_age_days=600)),
    ("2. Suspicious (new account, 10x average, late night)",
     _txn("DEMO-2", amount=1000.0, device_type="mobile", hour_of_day=2,
          tx_velocity_1h=3, tx_velocity_24h=6, amt_deviation=10.0,
          card_age_days=2, is_late_night=True)),
    ("3. Definite fraud (new account, 20x average, high velocity, 3am)",
     _txn("DEMO-3", amount=5000.0, device_type="mobile", hour_of_day=3,
          tx_velocity_1h=9, tx_velocity_24h=18, amt_deviation=20.0,
          card_age_days=1, is_late_night=True)),
    ("4. Edge case (high amount, long-standing card, trusted device)",
     _txn("DEMO-4", amount=3000.0, device_type="desktop", hour_of_day=15,
          tx_velocity_1h=1, tx_velocity_24h=3, amt_deviation=2.0,
          card_age_days=900)),
    ("5. Borderline (medium risk, step-up appropriate)",
     _txn("DEMO-5", amount=600.0, device_type="mobile", hour_of_day=21,
          tx_velocity_1h=2, tx_velocity_24h=5, amt_deviation=3.0,
          card_age_days=90)),
]


def step_system_check(client) -> dict:
    _hr("STEP 1 — SYSTEM CHECK")
    health = client.get("/health").json()
    print(f"  model_loaded: {health['model_loaded']}")
    print(f"  model_version: {health['model_version']}")
    print(f"  uptime_seconds: {health['uptime_seconds']}")
    print(f"\nSYSTEM READY — Model: {health['model_version']}")
    return health


def step_baseline_comparison() -> dict:
    _hr("STEP 2 — BASELINE vs TUNED (cost-optimal operating point)")
    if not _COMPARISON.exists():
        print(f"  (model_comparison.json not found at {_COMPARISON}; skipped)")
        return {}
    data = json.loads(_COMPARISON.read_text(encoding="utf-8"))
    pts = data.get("operating_points", {}).get("cost_optimal", {})
    b, t = pts.get("baseline", {}), pts.get("tuned", {})
    rows = [("threshold", "threshold"), ("recall", "recall"),
            ("precision", "precision"), ("auc_roc", "auc_roc"),
            ("false_positive_rate", "FPR"), ("daily_loss_gbp", "daily £ loss")]
    print(f"  {'metric':<16}{'baseline':>14}{'tuned':>14}")
    print(f"  {'-'*44}")
    for key, label in rows:
        bv, tv = b.get(key), t.get(key)
        fb = f"{bv:,.4f}" if isinstance(bv, float) and bv < 1 else f"{bv:,}"
        ft = f"{tv:,.4f}" if isinstance(tv, float) and tv < 1 else f"{tv:,}"
        print(f"  {label:<16}{fb:>14}{ft:>14}")
    print("\n  Production choice: baseline (see docs/model_card.md).")
    return data


def step_live_scoring(client) -> list[dict]:
    _hr("STEP 3 — LIVE SCORING (5 transactions)")
    results = []
    for label, txn in _DEMO_TXNS:
        body = client.post("/score", json=txn).json()
        results.append(body)
        print(f"\n{label}")
        print(f"  £{txn['amount']:,.2f}  {txn['device_type']}  "
              f"{txn['hour_of_day']:02d}:00  card_age={txn['card_age_days']}d  "
              f"velocity_1h={txn['tx_velocity_1h']}")
        print(f"  -> {body['decision']}  (p={body['fraud_probability']:.4f}, "
              f"threshold={body['threshold_used']})")
        for r in body["reasons"][:3]:
            print(f"     • {r}")
    return results


def step_throughput(quick: bool) -> dict:
    _hr("STEP 4 — PIPELINE THROUGHPUT")
    from src.streaming import run_pipeline
    tps = 10
    seconds = 3 if quick else 60
    limit = max(10, tps * seconds)
    print(f"  Running the streaming pipeline: ~{seconds}s at {tps} TPS "
          f"({limit} transactions)\n")
    report = asyncio.run(run_pipeline.run_pipeline(
        limit=limit, tps=tps, dashboard=not quick))
    return report


def step_financial(report: dict, comparison: dict) -> dict:
    _hr("STEP 5 — FINANCIAL IMPACT")
    # Live-demo evidence (throughput/latency) comes from the pipeline run; the
    # £ projection is based on the model's DOCUMENTED full-test performance at
    # the DEPLOYED operating point (baseline_metrics.json, threshold 0.19) —
    # not the small demo sample (far too noisy to extrapolate to 100k/day).
    decisions = report.get("decisions", {})
    processed = sum(decisions.values())
    avg_latency = report.get("avg_latency_ms", 0.0)
    print(f"  Live demo: {processed:,} transactions, "
          f"{report.get('decisions', {})}, avg {avg_latency:.0f}ms e2e")

    fn_cost, fp_cost, daily_txns = _FN_COST_GBP, _FP_COST_GBP, _DAILY_TXNS
    if _BASELINE_METRICS.exists():
        m = json.loads(_BASELINE_METRICS.read_text(encoding="utf-8"))["metrics"]
        recall = m.get("recall", 0.0)
        fpr = m.get("false_positive_rate", 0.0)
        total = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        fraud_rate = (m["tp"] + m["fn"]) / total if total else 0.0
        threshold = m.get("threshold", 0.19)
    else:  # fall back to the model-selection comparison
        base = (comparison.get("operating_points", {})
                .get("cost_optimal", {}).get("baseline", {}))
        recall = base.get("recall", 0.0)
        fpr = base.get("false_positive_rate", 0.0)
        fraud_rate = comparison.get("common_test_set", {}).get(
            "test_fraud_rate", 0.0256)
        threshold = base.get("threshold", 0.19)

    daily_fraud = daily_txns * fraud_rate
    caught = recall * daily_fraud
    saving = caught * fn_cost
    false_pos = fpr * (daily_txns * (1 - fraud_rate))
    fp_cost_total = false_pos * fp_cost
    net = saving - fp_cost_total
    roi = saving / _DAILY_INFRA_GBP if _DAILY_INFRA_GBP else 0.0

    summary = {
        "processed_in_demo": processed,
        "avg_latency_ms": round(avg_latency, 1),
        "basis": f"deployed baseline operating point (threshold {threshold})",
        "model_recall": recall,
        "model_fpr": fpr,
        "est_daily_fraud_caught": round(caught),
        "est_daily_saving_gbp": round(saving),
        "est_daily_false_pos_cost_gbp": round(fp_cost_total),
        "est_daily_net_benefit_gbp": round(net),
        "daily_infra_gbp": _DAILY_INFRA_GBP,
        "roi_saving_per_infra_gbp": round(roi),
    }
    print(f"\n  Projection at {daily_txns:,} txns/day "
          f"(fraud rate {fraud_rate:.2%}, model recall {recall:.1%}, "
          f"FPR {fpr:.2%} @ threshold {threshold}):")
    print(f"    Fraud caught/day:        {summary['est_daily_fraud_caught']:,} "
          f"of ~{round(daily_fraud):,}")
    print(f"    Daily fraud saving:      "
          f"£{summary['est_daily_saving_gbp']:,}")
    print(f"    Daily false-pos cost:    "
          f"£{summary['est_daily_false_pos_cost_gbp']:,}")
    print(f"    Daily net benefit:       "
          f"£{summary['est_daily_net_benefit_gbp']:,}")
    print(f"    Infra cost/day:          £{_DAILY_INFRA_GBP:,.0f} "
          f"(~£{_DAILY_INFRA_GBP * 30:,.0f}/month)")
    print(f"    ROI (saving / infra):    "
          f"{summary['roi_saving_per_infra_gbp']:,}x")
    print("\n  (Costs illustrative — recalibrate with client data per "
          "docs/model_card.md.)")
    return summary


def step_architecture() -> None:
    _hr("STEP 6 — PRODUCTION ARCHITECTURE (AWS, eu-west-2)")
    print(r"""
   Card events
       |
       v
  [ Amazon MSK ] --inbound--> [ Feature enrichment ] --enriched-->
       |                                                    |
       |                                                    v
       |                                       [ SageMaker endpoint ]
       |                                          XGBoost + SHAP
       |                                                    |
       |                 APPROVE / REVIEW / HOLD <----------+
       |                                |
       |                          (HOLD) v
       |                       [ Bedrock agent ] -- reasoning + SAR draft
       |                                |
       +-----------------> [ audit-log ] --> CloudTrail immutable audit
                                              CloudWatch metrics + alarms
                                              EventBridge weekly retrain trigger

   All in eu-west-2 (FCA data residency). See docs/production_architecture.md.
""")


def run_demo(quick: bool = False) -> dict:
    client = _client()
    health = step_system_check(client)
    comparison = step_baseline_comparison()
    scored = step_live_scoring(client)
    report = step_throughput(quick)
    financial = step_financial(report, comparison)
    step_architecture()
    _hr("DEMO COMPLETE")
    print("POC validated end-to-end.")
    return {"health": health, "comparison": comparison, "scored": scored,
            "throughput": report, "financial": financial}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="short throughput step (for tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("src.features.feature_store", "fraud_api", "httpx",
                  "enricher", "scorer", "agent", "audit", "producer",
                  "pipeline"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    run_demo(quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
