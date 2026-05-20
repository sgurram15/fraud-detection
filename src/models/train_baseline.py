"""Baseline XGBoost training for PSP fraud detection.

Pipeline (reuses the validated project components):
  raw transaction+identity  -> merge on TransactionID
                            -> build_features (engineered, leakage-safe)
                            -> prepare_train_test_split(strategy="combined")
                               (stratified 80/20, SMOTE+undersample on TRAIN
                               only, median-imputed)
                            -> XGBoost baseline + early stopping
                            -> evaluate on the pristine 20% test set
                            -> persist model, metrics, plots; log to MLflow

Methodological choices (documented because they affect honesty of metrics):

* **Early-stopping validation.** Early stopping needs a held-out signal. We
  carve a validation slice out of the (resampled) training set so the 20%
  test set stays completely untouched -- final reported metrics are therefore
  not optimised against. The validation slice contains synthetic SMOTE rows;
  that only influences *when* to stop, not the honesty of the test metrics.
* **Decision threshold chosen on validation, not test.** Threshold-free
  metrics (AUC-ROC, AUC-PR) need no threshold. For F1/precision/recall/
  confusion/FNR/FPR we pick the F1-optimal threshold on the validation set
  and apply it to test, so the operating point is not fitted to the test set.
* **scale_pos_weight vs. resampling.** The combined strategy already balances
  the training classes ~1:1, so scale_pos_weight is computed from the *actual
  training set the model sees* (~1.0) rather than the raw ~28:1 ratio.
  Stacking raw-ratio weighting on top of balanced data would double-correct
  and explode false positives. The raw imbalance is logged for reference.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import USE_S3, data_path, model_path
from src.features.build_features import build_features
from src.features.handle_imbalance import prepare_train_test_split
from src.models.experiment_tracking import log_results, start_run

logger = logging.getLogger(__name__)

# A8: data/model roots come from src/config so USE_S3 / S3_BUCKET flip
# everything centrally. load_data() and joblib.dump below still rely on
# local-FS semantics (Path.glob, write_bytes), so USE_S3=true is refused
# here -- S3 read/write wiring is a separate follow-up (needs s3fs/boto3).
if USE_S3:
    raise NotImplementedError(
        "train_baseline.py does not yet handle USE_S3=true: load_data and "
        "joblib persistence are local-FS only. Run with USE_S3=false."
    )
_RAW_DIR = _ROOT / data_path()
_MODEL_DIR = _ROOT / model_path()
_PERF_DIR = _ROOT / "docs" / "model_performance"
_MODEL_PATH = _MODEL_DIR / "baseline_xgboost.pkl"
_METRICS_PATH = _PERF_DIR / "baseline_metrics.json"
_CM_PLOT = _PERF_DIR / "baseline_confusion_matrix.png"
_PR_PLOT = _PERF_DIR / "baseline_pr_curve.png"
_THRESH_PLOT = _PERF_DIR / "threshold_analysis.png"

# Cost asymmetry defaults. FN = missing one fraud (PSP 50/50 split on a £250
# avg txn = £125).
COST_FN_DEFAULT = 125.0

# FP = wrongly blocking one legitimate transaction. A realistic UK-PSP cost,
# NOT a token friction value:
#   £8  average complaint-handling cost (contact-centre / ops time)
#   £12 customer-attrition value (fraction of annual customer value lost per
#       bad block; some customers churn after a wrongful decline)
#   £5  merchant-relationship friction (acquirer/merchant goodwill cost)
#   ---
#   £25 total. Ratio 5:1 FN:FP. Replace with the client's measured cost per
#   false positive during production calibration (see docs/model_card.md).
COST_FP_DEFAULT = 25.00

# Commercial hard constraint, independent of cost optimisation: blocking more
# than this fraction of legitimate transactions is unacceptable for a UK PSP
# (customer attrition, merchant damage, FCA Consumer Duty scrutiny) regardless
# of fraud savings. Operating points above it are flagged COMMERCIALLY
# UNACCEPTABLE.
MAX_ACCEPTABLE_FPR = 0.05

RANDOM_STATE = 42

# Local-PoC default training sample. Full 590k + SMOTE exceeds modest-host RAM;
# this is statistically representative for a baseline. Full-scale runs belong
# on the AWS/SageMaker path (docs/production_architecture.md). Override via
# the FRAUD_SAMPLE_N env var (=all for the full dataset).
DEFAULT_SAMPLE_N = 100_000

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
}

# Daily transaction volume used to project sample FN/FP *rates* into a true
# daily £ loss (volume-independent: rate * DAILY_TXNS * unit_cost).
DAILY_TXNS = 100_000


def _read_csv_lean(path: Path, nrows: int | None) -> pd.DataFrame:
    """Memory-efficient CSV read.

    IEEE-CIS is mostly float64 columns; loading it whole as float64 needs
    ~1.5 GiB *per block* and OOMs on modest hosts. We read in chunks and
    immediately downcast floats to float32 and ints to the smallest int,
    roughly halving peak memory. float32 is fine for XGBoost (hist).
    """
    if nrows is not None:
        return pd.read_csv(path, nrows=nrows)
    chunks = []
    for ch in pd.read_csv(path, chunksize=50_000):
        for c in ch.select_dtypes(include=["float64"]).columns:
            ch[c] = ch[c].astype("float32")
        for c in ch.select_dtypes(include=["int64"]).columns:
            if c == "TransactionID":
                continue  # keep join key int64 (merge factorizer needs it)
            ch[c] = pd.to_numeric(ch[c], downcast="integer")
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True, copy=False)
    if "TransactionID" in df.columns:
        df["TransactionID"] = df["TransactionID"].astype("int64")
    return df


def load_data(sample_n: int | None = None) -> pd.DataFrame:
    txn = sorted(_RAW_DIR.glob("**/train_transaction.csv"))
    if not txn:
        raise FileNotFoundError(
            f"train_transaction.csv not found under {_RAW_DIR}. "
            "Run data/download_data.py first."
        )
    df = _read_csv_lean(txn[0], sample_n)
    ident = sorted(_RAW_DIR.glob("**/train_identity.csv"))
    if ident:
        # Identity file is small (~26 MB); always read it lean+whole.
        df = df.merge(_read_csv_lean(ident[0], None),
                      on="TransactionID", how="left")
        logger.info("Merged identity (%d cols).", df.shape[1])
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("Loaded %d rows x %d cols (%.0f MB in memory).",
                df.shape[0], df.shape[1], mem_mb)
    return df


def _sweep(y_true, y_prob, cost_fn: float, cost_fp: float) -> pd.DataFrame:
    """Per-threshold (0.01..0.99) metrics.

    Daily £ loss is computed from *rates* (volume-independent) projected onto
    DAILY_TXNS so it is consistent regardless of evaluation-sample size:
      daily_loss = fn_rate*DAILY_TXNS*cost_fn + fp_rate*DAILY_TXNS*cost_fp
    where fn_rate = FN/total and fp_rate = FP/total over all samples.
    ``fpr`` (FP / actual-negatives = fraction of legitimate txns blocked) is
    used for the commercial MAX_ACCEPTABLE_FPR constraint.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = int(y_true.sum())
    neg = len(y_true) - pos
    total = len(y_true)
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pred = y_prob >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = pos - tp
        tn = neg - fp
        recall = tp / pos if pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        fn_rate = fn / total if total else 0.0
        fp_rate = fp / total if total else 0.0
        daily_loss = (fn_rate * DAILY_TXNS * cost_fn
                      + fp_rate * DAILY_TXNS * cost_fp)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0  # legit blocked fraction
        acceptable = fpr <= MAX_ACCEPTABLE_FPR
        rows.append((float(t), recall, precision, f1, float(daily_loss),
                     fn, fp, float(fpr), bool(acceptable)))
    return pd.DataFrame(
        rows,
        columns=["threshold", "recall", "precision", "f1",
                 "expected_loss", "fn", "fp", "fpr", "acceptable"],
    )


def _row_to_option(r) -> dict:
    return {
        "threshold": round(float(r["threshold"]), 2),
        "recall": float(r["recall"]),
        "precision": float(r["precision"]),
        "f1": float(r["f1"]),
        "expected_loss_gbp": round(float(r["expected_loss"]), 2),
        "false_positive_rate": float(r["fpr"]),
        "commercially_acceptable": bool(r["acceptable"]),
    }


def _plot_threshold_analysis(sweep: pd.DataFrame, options: dict) -> None:
    fig, ax_l = plt.subplots(figsize=(9, 5))
    ax_l.plot(sweep["threshold"], sweep["recall"], color="#4c72b0",
              label="Recall")
    ax_l.plot(sweep["threshold"], sweep["precision"], color="#55a868",
              label="Precision")
    ax_l.set_xlabel("Decision threshold")
    ax_l.set_ylabel("Recall / Precision")
    ax_l.set_ylim(0, 1.02)

    ax_r = ax_l.twinx()
    ax_r.plot(sweep["threshold"], sweep["expected_loss"], color="#c44e52",
              ls="--", label="Expected £ loss")
    ax_r.set_ylabel("Expected daily £ loss")

    co = options["cost_optimal"]["threshold"]
    ax_l.axvline(co, color="black", lw=1.5)
    ax_l.annotate(
        f"cost-optimal\nthr={co:.2f}", xy=(co, 0.5),
        xytext=(co + 0.05, 0.55),
        arrowprops=dict(arrowstyle="->"), fontsize=9,
    )
    for key, colr in (("high_precision_95", "#55a868"),
                      ("max_recall_p50", "#4c72b0")):
        t = options[key]["threshold"]
        ax_l.axvline(t, color=colr, ls=":", lw=1)

    lines = ax_l.get_lines()[:2] + ax_r.get_lines()
    ax_l.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    txt = (
        f"Cost-optimal  thr={co:.2f}  "
        f"R={options['cost_optimal']['recall']:.2f} "
        f"P={options['cost_optimal']['precision']:.2f}\n"
        f"95% Precision thr={options['high_precision_95']['threshold']:.2f}  "
        f"R={options['high_precision_95']['recall']:.2f} "
        f"P={options['high_precision_95']['precision']:.2f}\n"
        f"Max Recall    thr={options['max_recall_p50']['threshold']:.2f}  "
        f"R={options['max_recall_p50']['recall']:.2f} "
        f"P={options['max_recall_p50']['precision']:.2f}"
    )
    ax_l.text(0.02, 0.02, txt, transform=ax_l.transAxes, fontsize=8,
              va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax_l.set_title("Threshold analysis — cost-weighted operating points")
    fig.tight_layout()
    fig.savefig(_THRESH_PLOT, dpi=110)
    plt.close(fig)


def find_optimal_threshold(
    y_true,
    y_prob,
    cost_false_negative: float = COST_FN_DEFAULT,
    cost_false_positive: float = COST_FP_DEFAULT,
) -> dict:
    """Cost-weighted threshold selection.

    Sweeps thresholds 0.01..0.99 and returns three operating points so the CTO
    can choose by business priority:

    * ``cost_optimal``     -- minimises expected £ loss
      (FN*cost_fn + FP*cost_fp). **Recommended default.**
    * ``high_precision_95``-- highest recall while precision >= 0.95
      (conservative; minimise customer friction).
    * ``max_recall_p50``   -- highest recall while precision > 0.50
      (aggressive fraud catching).

    Costs are explicit and asymmetric (default 50:1 FN:FP). Also saves the
    threshold-analysis plot. Returns thresholds + their metrics.
    """
    sweep = _sweep(y_true, y_prob, cost_false_negative, cost_false_positive)

    cost_optimal = sweep.loc[sweep["expected_loss"].idxmin()]

    hp = sweep[sweep["precision"] >= 0.95]
    hp_row = (hp.loc[hp["recall"].idxmax()] if not hp.empty
              else sweep.loc[sweep["precision"].idxmax()])

    mr = sweep[sweep["precision"] > 0.50]
    mr_row = (mr.loc[mr["recall"].idxmax()] if not mr.empty
              else sweep.loc[sweep["recall"].idxmax()])

    options = {
        "cost_optimal": _row_to_option(cost_optimal),
        "high_precision_95": _row_to_option(hp_row),
        "max_recall_p50": _row_to_option(mr_row),
        "costs": {
            "cost_false_negative_gbp": cost_false_negative,
            "cost_false_positive_gbp": cost_false_positive,
            "ratio_fn_to_fp": round(
                cost_false_negative / cost_false_positive, 1
            ),
        },
        "fallbacks": {
            "high_precision_95_unreachable": bool(hp.empty),
            "max_recall_p50_unreachable": bool(mr.empty),
        },
    }
    _plot_threshold_analysis(sweep, options)
    return options


def _evaluate(y_true, proba, threshold) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "auc_pr": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _plot_confusion(m: dict) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["legit", "fraud"],
                yticklabels=["legit", "fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Baseline confusion matrix (thr={m['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(_CM_PLOT, dpi=110)
    plt.close(fig)


def _plot_pr(y_true, proba, ap: float) -> None:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, color="#c44e52", label=f"AUC-PR = {ap:.4f}")
    base = float(np.mean(y_true))
    ax.axhline(base, ls="--", color="grey",
               label=f"baseline (prevalence={base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Baseline precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_PR_PLOT, dpi=110)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    # Box-drawing chars + £ crash on Windows' default cp1252 stdout.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    for d in (_MODEL_DIR, _PERF_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # The full IEEE-CIS set (590k x ~430, +SMOTE) needs several GB and OOMs on
    # modest hosts, so the local PoC defaults to a representative sample.
    # Override: FRAUD_SAMPLE_N=<n> for a different size, or FRAUD_SAMPLE_N=all
    # (needs a large host / the AWS SageMaker path -- see
    # docs/production_architecture.md).
    env = os.environ.get("FRAUD_SAMPLE_N")
    if env is None:
        sample_n = DEFAULT_SAMPLE_N
    elif env.strip().lower() in ("all", "0", "full", "none"):
        sample_n = None
    else:
        sample_n = int(env)
    logger.info(
        "Sample size: %s",
        "FULL dataset" if sample_n is None else f"{sample_n:,} rows "
        f"(local PoC default; set FRAUD_SAMPLE_N=all for full)",
    )

    df = load_data(sample_n)
    raw_pos = int(pd.to_numeric(df["isFraud"]).sum())
    raw_neg = len(df) - raw_pos
    raw_ratio = raw_neg / max(raw_pos, 1)

    enriched = build_features(df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        enriched, strategy="combined", test_size=0.20
    )

    # Carve an early-stopping validation slice from the resampled train set.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train,
    )

    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    scale_pos_weight = neg / max(pos, 1)  # ~1.0 after combined resampling

    params = {
        **XGB_PARAMS,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "strategy": "combined (SMOTE+undersample)",
        "raw_imbalance_ratio_neg_per_pos": round(raw_ratio, 2),
        "n_features": X_train.shape[1],
        "n_train_rows": len(X_tr),
        "n_test_rows": len(X_test),
    }

    logger.info("Training XGBoost baseline (%d trees) ...",
                XGB_PARAMS["n_estimators"])
    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    logger.info("Best iteration: %s", model.best_iteration)

    val_proba = model.predict_proba(X_val)[:, 1]
    # Operating points are selected on VALIDATION (not test) so the chosen
    # threshold is never fitted to the reported test metrics.
    options = find_optimal_threshold(y_val, val_proba)
    default_threshold = options["cost_optimal"]["threshold"]

    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate(y_test, test_proba, default_threshold)
    fraud_rate = float(np.mean(y_test))

    # Evaluate all three operating points on the pristine test set. Daily £
    # loss uses volume-independent rates projected onto DAILY_TXNS so it is
    # the same order of magnitude across points and sample sizes.
    cost_fn = options["costs"]["cost_false_negative_gbp"]
    cost_fp = options["costs"]["cost_false_positive_gbp"]
    n_test = len(y_test)
    operating_points = {}
    for key in ("cost_optimal", "high_precision_95", "max_recall_p50"):
        t = options[key]["threshold"]
        m = _evaluate(y_test, test_proba, t)
        daily_loss = (
            (m["fn"] / n_test) * DAILY_TXNS * cost_fn
            + (m["fp"] / n_test) * DAILY_TXNS * cost_fp
        )
        operating_points[key] = {
            "threshold": t,
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "expected_loss_gbp": round(daily_loss, 2),
            "false_positive_rate": m["false_positive_rate"],
            "commercially_acceptable": bool(
                m["false_positive_rate"] <= MAX_ACCEPTABLE_FPR
            ),
            "fn": m["fn"], "fp": m["fp"],
        }

    # ---- persist artefacts ----
    joblib.dump(model, _MODEL_PATH)
    payload = {
        "params": params,
        "selected_operating_point": "cost_optimal",
        "default_threshold": default_threshold,
        "metrics": metrics,
        "operating_points_test": operating_points,
        "threshold_options_validation": options,
        "test_fraud_rate": fraud_rate,
        "max_acceptable_fpr": MAX_ACCEPTABLE_FPR,
    }
    _METRICS_PATH.write_text(json.dumps(payload, indent=2))
    _plot_confusion(metrics)
    _plot_pr(y_test, test_proba, metrics["auc_pr"])

    # ---- MLflow ----
    try:
        run = start_run("baseline-xgboost", params)
        log_results(run, metrics, model, list(X_train.columns))
        logger.info("Logged run to MLflow.")
    except Exception as exc:  # noqa: BLE001 - tracking must not fail training
        logger.warning("MLflow logging failed: %s", exc)

    _print_summary(metrics, fraud_rate, operating_points)


def _op_table(ops: dict) -> str:
    def line(left: str, mid: str, right: str) -> str:
        return (left + "─" * 21 + mid + "─" * 8 + mid + "─" * 11 + mid
                + "─" * 14 + mid + "─" * 13 + right)

    rows = [
        ("Cost-optimal", ops["cost_optimal"]),
        ("95% Precision", ops["high_precision_95"]),
        ("Max Recall (P>0.5)", ops["max_recall_p50"]),
    ]
    out = [
        line("┌", "┬", "┐"),
        f"│ {'Operating Point':<19} │ {'Recall':<6} │ "
        f"{'Precision':<9} │ {'Daily £ Loss':<12} │ {'Acceptable':<11} │",
        line("├", "┼", "┤"),
    ]
    for name, o in rows:
        loss = f"£{o['expected_loss_gbp']:,.0f}"
        if o["commercially_acceptable"]:
            acc = "YES"
        else:
            acc = f"NO {o['false_positive_rate']*100:.0f}% FPR"
        out.append(
            f"│ {name:<19} │ {o['recall']:<6.2f} │ "
            f"{o['precision']:<9.2f} │ {loss:<12} │ {acc:<11} │"
        )
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def _print_summary(metrics: dict, fraud_rate: float,
                   operating_points: dict) -> None:
    order = [
        "auc_roc", "auc_pr", "f1", "precision", "recall",
        "false_negative_rate", "false_positive_rate", "threshold",
    ]
    print("\n" + "=" * 60)
    print("BASELINE XGBOOST - TEST SET PERFORMANCE")
    print("=" * 60)
    for k in order:
        print(f"  {k:24s} {metrics[k]:.4f}")
    print(f"\n  Confusion matrix (thr={metrics['threshold']:.3f}):")
    print(f"    TN={metrics['tn']:,}  FP={metrics['fp']:,}")
    print(f"    FN={metrics['fn']:,}  TP={metrics['tp']:,}")

    print("\n" + "-" * 60)
    print(f"  Fraud caught (recall):              "
          f"{metrics['recall']*100:.2f}%")
    print(f"  Fraud missed (FN rate):             "
          f"{metrics['false_negative_rate']*100:.2f}%")
    print(f"  Legit wrongly blocked (FP rate):    "
          f"{metrics['false_positive_rate']*100:.2f}%")

    print("\n  OPERATING POINTS (test set; thresholds selected on "
          "validation)")
    print(f"  Daily £ loss projected to {DAILY_TXNS:,} txns/day | "
          f"costs: FN=£{COST_FN_DEFAULT:.0f}, FP=£{COST_FP_DEFAULT:.0f} | "
          f"test fraud rate {fraud_rate:.3%}")
    print(_op_table(operating_points))
    co = operating_points["cost_optimal"]
    if co["commercially_acceptable"]:
        print(f"  Recommended default: Cost-optimal (thr="
              f"{co['threshold']:.2f}). CTO may select another point by "
              "business priority.")
    else:
        print(f"  *** WARNING: cost-optimal point (thr={co['threshold']:.2f}, "
              f"FPR {co['false_positive_rate']*100:.1f}%) is COMMERCIALLY "
              f"UNACCEPTABLE (> {MAX_ACCEPTABLE_FPR*100:.0f}% FPR).")
        print("  *** Do NOT deploy without board sign-off + FCA notification "
              "consideration (see docs/model_card.md). Prefer an acceptable "
              "operating point.")
    print("=" * 60)
    print(f"Model  -> {_MODEL_PATH}")
    print(f"Metrics-> {_METRICS_PATH}")
    print(f"Plots  -> {_CM_PLOT.name}, {_PR_PLOT.name}, "
          f"{_THRESH_PLOT.name}")


if __name__ == "__main__":
    main()
