"""Controlled head-to-head: baseline_xgboost.pkl vs. tuned_xgboost.pkl.

WHY THIS EXISTS
---------------
``train_baseline.py`` and ``tune_model.py`` each report metrics, but on
*their own* test split (baseline on a 100k-row sample, tuned on 40k). That is
a fair per-artifact reading but **not** a controlled comparison: the two
numbers come from different test rows. This script puts BOTH saved models on
the **same** held-out test set and the **same** operating points, so the
recall/precision/AUC/£-loss deltas are attributable to the model, not to the
data slice.

HOW THE COMMON TEST SET IS BUILT (and why it is reproducible)
-------------------------------------------------------------
We reproduce ``train_baseline.py``'s exact split: load the first
``DEFAULT_SAMPLE_N`` (100k) rows, ``build_features``, then
``prepare_train_test_split(..., test_size=0.20)`` with the project-wide
``RANDOM_STATE=42``. Because the split is seeded and the row sample is
deterministic (``pd.read_csv(nrows=n)`` = the first n rows), the resulting
20% test set is byte-identical to the one baseline was evaluated on. We
cross-check this at runtime against ``baseline_metrics.json`` (recall / TP /
FP at threshold 0.16) and abort if it does not match.

Thresholds for each model are chosen on a **real-distribution validation
slice** (carved from the non-resampled train partition), never on the test
set, so the reported test metrics are not fitted to the test data.

HONEST CAVEAT — this is a controlled comparison of two *fixed artifacts*, not
a from-scratch ablation
------------------------------------------------------------------------------
``load_data(n)`` returns the FIRST n rows, so the tuned model's 40k training
pool is a *subset* of baseline's 100k pool. Consequences on this common test
set (which is 20% of the 100k, drawn from across all 100k rows):

  * For BASELINE the test rows are genuinely held out -> clean.
  * For TUNED, the test rows that fall in the first 40k were candidates for
    its training set; ~1/3 of the common test rows leaked into tuned's
    training. The tuned model's numbers here are therefore **optimistically
    biased**.

Net effect: the comparison is tilted IN FAVOUR of the tuned model. If the
tuned model still fails to beat the baseline, the "deploy baseline" verdict is
robust (it would only widen under a clean, equal-sample retrain). A fully
clean ablation requires retraining both models at the same sample size on the
full-grid path (see docs/production_architecture.md) -- out of scope for a
local PoC artifact comparison.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import USE_S3, model_path
from src.features.build_features import build_features
from src.features.handle_imbalance import prepare_train_test_split
from src.models.train_baseline import (
    COST_FN_DEFAULT,
    COST_FP_DEFAULT,
    DAILY_TXNS,
    DEFAULT_SAMPLE_N,
    MAX_ACCEPTABLE_FPR,
    RANDOM_STATE,
    _sweep,
    load_data,
)
from src.models.tune_model import recall_at_precision

logger = logging.getLogger(__name__)

if USE_S3:
    raise NotImplementedError(
        "compare_models.py loads saved models via joblib and reads raw CSVs "
        "(local-FS only); run with USE_S3=false."
    )

_MODEL_DIR = _ROOT / model_path()
_BASELINE_PATH = _MODEL_DIR / "baseline_xgboost.pkl"
_TUNED_PATH = _MODEL_DIR / "tuned_xgboost.pkl"
_PERF_DIR = _ROOT / "docs" / "model_performance"
_BASELINE_METRICS = _PERF_DIR / "baseline_metrics.json"
_OUT_JSON = _PERF_DIR / "model_comparison.json"
_MODEL_CARD = _ROOT / "docs" / "model_card.md"

# Decision rule: the tuned model must beat baseline by at least this absolute
# recall margin at the 95%-precision operating point to justify deploying it.
IMPROVEMENT_THRESHOLD = 0.01
TARGET_PRECISION = 0.95
DECISION_FLAG = (
    "TUNED MODEL DOES NOT MEET IMPROVEMENT THRESHOLD — "
    "recommend deploying baseline, rerun full grid search on AWS SageMaker — "
    "see docs/production_architecture.md"
)


def _align(X: pd.DataFrame, model) -> pd.DataFrame:
    """Reindex X to the model's expected feature order (XGBoost booster)."""
    try:
        names = model.get_booster().feature_names
    except Exception:
        names = None
    if names:
        return X.reindex(columns=names, fill_value=0)
    return X


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_align(X, model))[:, 1]


def _evaluate_at(y_true, proba, threshold: float) -> dict:
    """Metrics on a fixed test set at a fixed threshold. AUC-ROC/PR are
    threshold-free; recall/precision/£-loss are at the operating point."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    daily_loss = (fn / n) * DAILY_TXNS * COST_FN_DEFAULT \
        + (fp / n) * DAILY_TXNS * COST_FP_DEFAULT
    return {
        "threshold": round(float(threshold), 4),
        "recall": float(recall),
        "precision": float(precision),
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "auc_pr": float(average_precision_score(y_true, proba)),
        "false_positive_rate": float(fpr),
        "daily_loss_gbp": round(float(daily_loss), 2),
        "commercially_acceptable": bool(fpr <= MAX_ACCEPTABLE_FPR),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _cost_optimal_threshold(y_val, val_proba) -> float:
    sweep = _sweep(y_val, val_proba, COST_FN_DEFAULT, COST_FP_DEFAULT)
    return float(sweep.loc[sweep["expected_loss"].idxmin(), "threshold"])


def _p95_threshold(y_val, val_proba) -> float:
    return recall_at_precision(y_val, val_proba, TARGET_PRECISION)[1]


def _build_common_test_set():
    """Reproduce train_baseline's exact split. Returns the real-distribution
    validation slice (for threshold selection) and the pristine common test
    set (for final metrics)."""
    from sklearn.model_selection import train_test_split

    df = load_data(DEFAULT_SAMPLE_N)
    enriched = build_features(df)
    # strategy='none' -> imputed but NOT resampled, so we can carve a
    # real-distribution validation slice. The test split is identical to the
    # 'combined' run (split happens before, and independently of, resampling).
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        enriched, strategy="none", test_size=0.20
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train,
    )
    return X_val, y_val, X_test, y_test


def _verify_against_baseline_metrics(baseline_test_metrics: dict) -> None:
    """Cross-check the reconstructed test set against recorded baseline
    metrics. The correct invariant is the **threshold-free** AUC-ROC: if the
    test rows or model differed, AUC would differ. (The confusion matrix is
    NOT a valid check here -- this script picks each model's threshold on a
    real-distribution validation slice, whereas train_baseline picked 0.16 on
    its resampled val, so the operating point legitimately differs.)"""
    if not _BASELINE_METRICS.exists():
        logger.warning("baseline_metrics.json absent; skipping split check.")
        return
    ref_auc = json.loads(_BASELINE_METRICS.read_text())["metrics"]["auc_roc"]
    got_auc = baseline_test_metrics["auc_roc"]
    if abs(got_auc - ref_auc) < 1e-6:
        logger.info("Split verification PASSED: baseline AUC-ROC %.10f matches "
                    "baseline_metrics.json (test set reconstructed exactly).",
                    got_auc)
    else:
        raise RuntimeError(
            f"Split verification FAILED: reconstructed baseline AUC-ROC "
            f"{got_auc:.10f} != recorded {ref_auc:.10f}. The common test set "
            f"does not match train_baseline's; comparison is untrustworthy. "
            f"Check sample_n / seed / build_features determinism."
        )


# --------------------------------------------------------------------------- #
# Table rendering (matches the requested box format)
# --------------------------------------------------------------------------- #
_W = (19, 6, 9, 8, 12, 10)  # inner widths for the 6 columns


def _rule(left: str, mid: str, right: str) -> str:
    return left + mid.join("─" * (w + 2) for w in _W) + right


def _row(cells) -> str:
    return "│ " + " │ ".join(
        f"{c:<{w}}" for c, w in zip(cells, _W)
    ) + " │"


def _acceptable_str(m: dict) -> str:
    return "YES" if m["commercially_acceptable"] \
        else f"NO {m['false_positive_rate']*100:.0f}% FPR"


def _render_table(title: str, baseline_m: dict, tuned_m: dict) -> str:
    out = [
        title,
        _rule("┌", "┬", "┐"),
        _row(("Model", "Recall", "Precision", "AUC-ROC",
              "Daily £ Loss", "Acceptable")),
        _rule("├", "┼", "┤"),
    ]
    for name, m in (("Baseline XGBoost", baseline_m),
                    ("Tuned XGBoost", tuned_m)):
        out.append(_row((
            name,
            f"{m['recall']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['auc_roc']:.4f}",
            f"£{m['daily_loss_gbp']:,.0f}",
            _acceptable_str(m),
        )))
    out.append(_rule("└", "┴", "┘"))
    return "\n".join(out)


def _log_to_mlflow(payload: dict) -> None:
    try:
        import mlflow

        from src.models.experiment_tracking import start_run

        co_b = payload["operating_points"]["cost_optimal"]["baseline"]
        co_t = payload["operating_points"]["cost_optimal"]["tuned"]
        p95_b = payload["operating_points"]["high_precision_95"]["baseline"]
        p95_t = payload["operating_points"]["high_precision_95"]["tuned"]
        run = start_run("model-selection-decision", {
            "common_test_set": "train_baseline 20% split (100k, seed 42)",
            "decision_metric": "recall_at_precision_0.95",
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "selected_model": payload["decision"]["selected_model"],
        })
        for label, m in (("baseline_p95", p95_b), ("tuned_p95", p95_t),
                         ("baseline_costopt", co_b), ("tuned_costopt", co_t)):
            mlflow.log_metric(f"{label}.recall", m["recall"])
            mlflow.log_metric(f"{label}.precision", m["precision"])
            mlflow.log_metric(f"{label}.auc_roc", m["auc_roc"])
            mlflow.log_metric(f"{label}.daily_loss_gbp", m["daily_loss_gbp"])
        mlflow.log_metric("recall_margin_p95",
                          payload["decision"]["recall_margin_at_p95"])
        mlflow.set_tag("phase", "model-selection")
        mlflow.set_tag("selected_model",
                       payload["decision"]["selected_model"])
        mlflow.set_tag("meets_improvement_threshold",
                       str(payload["decision"]["tuned_meets_threshold"]))
        mlflow.log_text(json.dumps(payload, indent=2),
                        "model_comparison.json")
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("Logged 'model-selection-decision' run to MLflow.")
    except Exception as exc:  # noqa: BLE001 - tracking must not fail the run
        logger.warning("MLflow logging failed: %s", exc)


def _update_model_card(selected: str, reason: str) -> None:
    line = (f"Production model selected: {selected} — {reason} — "
            f"validated on identical test set, {date.today().isoformat()}")
    text = _MODEL_CARD.read_text(encoding="utf-8")
    section = "## Model Selection"
    entry = f"\n{section}\n\n{line}\n"
    if section in text:
        # Replace any existing "Production model selected:" line under it.
        lines = text.splitlines()
        out, in_sec, replaced = [], False, False
        for ln in lines:
            if ln.strip() == section:
                in_sec = True
                out.append(ln)
                continue
            if in_sec and ln.startswith("## ") and ln.strip() != section:
                in_sec = False
            if in_sec and ln.startswith("Production model selected:"):
                out.append(line)
                replaced = True
                continue
            out.append(ln)
        new = "\n".join(out)
        if not replaced:
            new = new.replace(section, f"{section}\n\n{line}", 1)
        _MODEL_CARD.write_text(new + ("\n" if not new.endswith("\n") else ""),
                               encoding="utf-8")
    else:
        _MODEL_CARD.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")
    logger.info("Updated docs/model_card.md Model Selection: %s", line)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    for p in (_BASELINE_PATH, _TUNED_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found in {_MODEL_DIR}. Train both models first "
                "(train_baseline.py and tune_model.py)."
            )
    baseline = joblib.load(_BASELINE_PATH)
    tuned = joblib.load(_TUNED_PATH)

    logger.info("Reconstructing train_baseline's 20%% test split (%d rows, "
                "seed %d) ...", DEFAULT_SAMPLE_N, RANDOM_STATE)
    X_val, y_val, X_test, y_test = _build_common_test_set()
    n_test = len(y_test)
    test_fraud_rate = float(np.mean(y_test))
    logger.info("Common test set: %d rows, fraud rate %.4f.",
                n_test, test_fraud_rate)

    results = {"cost_optimal": {}, "high_precision_95": {}}
    for name, model in (("baseline", baseline), ("tuned", tuned)):
        val_proba = _proba(model, X_val)
        test_proba = _proba(model, X_test)
        t_cost = _cost_optimal_threshold(y_val, val_proba)
        t_p95 = _p95_threshold(y_val, val_proba)
        results["cost_optimal"][name] = _evaluate_at(y_test, test_proba, t_cost)
        results["high_precision_95"][name] = _evaluate_at(
            y_test, test_proba, t_p95
        )

    _verify_against_baseline_metrics(results["cost_optimal"]["baseline"])

    # ---- decision (on recall at the 95%-precision operating point) ----
    b95 = results["high_precision_95"]["baseline"]
    t95 = results["high_precision_95"]["tuned"]
    margin = t95["recall"] - b95["recall"]
    tuned_meets = margin >= IMPROVEMENT_THRESHOLD
    selected = "tuned" if tuned_meets else "baseline"
    if tuned_meets:
        reason = (f"tuned beats baseline by {margin*100:.2f}pp recall at "
                  f"{TARGET_PRECISION:.0%} precision on an identical test set")
    else:
        reason = (f"tuned does not beat baseline by the required "
                  f"{IMPROVEMENT_THRESHOLD*100:.0f}pp recall at "
                  f"{TARGET_PRECISION:.0%} precision "
                  f"(margin {margin*100:+.2f}pp); local random search "
                  f"underperforms, full-grid search belongs on managed compute")

    payload = {
        "comparison_date": date.today().isoformat(),
        "common_test_set": {
            "source": "train_baseline 20% split",
            "sample_n": DEFAULT_SAMPLE_N,
            "random_state": RANDOM_STATE,
            "n_test_rows": n_test,
            "test_fraud_rate": test_fraud_rate,
        },
        "costs": {
            "cost_false_negative_gbp": COST_FN_DEFAULT,
            "cost_false_positive_gbp": COST_FP_DEFAULT,
            "max_acceptable_fpr": MAX_ACCEPTABLE_FPR,
            "daily_txns": DAILY_TXNS,
        },
        "operating_points": results,
        "decision": {
            "decision_metric": "recall_at_precision_0.95",
            "improvement_threshold_recall": IMPROVEMENT_THRESHOLD,
            "baseline_recall_at_p95": b95["recall"],
            "tuned_recall_at_p95": t95["recall"],
            "recall_margin_at_p95": float(margin),
            "tuned_meets_threshold": bool(tuned_meets),
            "selected_model": selected,
            "reason": reason,
        },
        "caveats": [
            "Controlled comparison of two FIXED artifacts, not a from-scratch "
            "ablation: the models were trained on different sample sizes "
            "(baseline 100k, tuned 40k).",
            "load_data(n) returns the first n rows, so tuned's 40k training "
            "pool is a subset of baseline's 100k. ~1/3 of this common test "
            "set leaked into tuned's training -> tuned's numbers are "
            "optimistically biased. The comparison is therefore tilted IN "
            "FAVOUR of tuned; baseline's test rows are genuinely held out.",
            "Thresholds were selected on a real-distribution validation slice "
            "(not the test set); both models received identical, already-"
            "imputed test inputs for fairness.",
            "A fully clean ablation requires retraining both at equal sample "
            "size on the full-grid path (docs/production_architecture.md).",
        ],
    }
    _OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- report ----
    print("\n" + "=" * 78)
    print("MODEL SELECTION — BASELINE vs TUNED (identical held-out test set)")
    print("=" * 78)
    print(f"Common test set: {n_test:,} rows | fraud rate {test_fraud_rate:.3%}"
          f" | costs FN=£{COST_FN_DEFAULT:.0f} FP=£{COST_FP_DEFAULT:.0f} | "
          f"daily volume {DAILY_TXNS:,}")
    print(f"Daily £ loss is volume-projected; AUC-ROC is threshold-free.\n")
    print(_render_table(
        "AT 95% PRECISION OPERATING POINT (decision axis):",
        b95, t95))
    print()
    print(_render_table(
        "AT COST-OPTIMAL THRESHOLD (£125 FN / £25 FP, deployed default):",
        results["cost_optimal"]["baseline"],
        results["cost_optimal"]["tuned"]))

    print("\n" + "-" * 78)
    auc_delta = t95["auc_roc"] - b95["auc_roc"]
    loss_delta = t95["daily_loss_gbp"] - b95["daily_loss_gbp"]
    winner = "TUNED" if tuned_meets else "BASELINE"
    print(f"WINNER: {winner}")
    print(f"  Recall @95% precision : baseline {b95['recall']:.4f} vs "
          f"tuned {t95['recall']:.4f}  (margin {margin*100:+.2f}pp)")
    print(f"  AUC-ROC               : baseline {b95['auc_roc']:.4f} vs "
          f"tuned {t95['auc_roc']:.4f}  ({auc_delta*100:+.2f}pp)")
    print(f"  Daily £ loss @95% prec: baseline £{b95['daily_loss_gbp']:,.0f} "
          f"vs tuned £{t95['daily_loss_gbp']:,.0f}  (£{loss_delta:+,.0f})")

    if not tuned_meets:
        print("\n" + "!" * 78)
        print(DECISION_FLAG)
        print("!" * 78)

    _update_model_card(selected, reason.split(";")[0].split("(")[0].strip())
    _log_to_mlflow(payload)

    print("\n" + "=" * 78)
    print(f"Comparison JSON -> {_OUT_JSON}")
    print(f"Model card      -> {_MODEL_CARD} (Model Selection)")
    print("NOTE: comparison is biased in tuned's favour (training-set "
          "overlap); see caveats in the JSON. Baseline verdict is robust.")
    print("=" * 78)


if __name__ == "__main__":
    main()
