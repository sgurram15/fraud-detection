"""Three-model benchmark: trained XGBoost vs. TabPFN zero-shot vs. majority floor.

Surfaces three CTO-relevant facts in a single table:
  1. The problem is genuinely hard (majority class is useless).
  2. Even a no-training model gets partway there (TabPFN in-context learning).
  3. Our trained model meaningfully outperforms both.

Test set: the exact 20% slice ``train_baseline.py`` produces from
``(sample_n, RANDOM_STATE=42, strategy='combined')`` so numbers are
directly comparable to the deployed-baseline metrics. Override the sample
size with ``FRAUD_SAMPLE_N`` (default = ``train_baseline.DEFAULT_SAMPLE_N``).

TabPFN constraints (real, surfaced honestly):
  * Feature cap (~500 for v2): we pick the top-K features by trained-XGBoost
    importance so TabPFN sees the columns that actually matter.
  * Training-context cap (10k for v2): we sample a stratified context of
    ``TABPFN_CONTEXT_SIZE`` rows from the resampled training set.
  * "Training: None" means *no gradient training* -- TabPFN does see the
    labelled context in a single in-context-learning pass. We label this
    honestly in the table.

Threshold policy: AUC-ROC is threshold-free; recall/precision are reported at
the F1-optimal threshold per model on the test set. This is the most generous
single-point reading for each model and is documented as such; for the
deployed cost-weighted threshold see ``baseline_metrics.json``.

Outputs:
  * Comparison table printed to stdout (UTF-8 box-drawing chars).
  * JSON at ``docs/model_performance/benchmark_comparison.json``.

Run:
    python src/models/benchmark_baseline.py
    FRAUD_SAMPLE_N=20000 python src/models/benchmark_baseline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import USE_S3, model_path
from src.features.build_features import build_features
from src.features.handle_imbalance import prepare_train_test_split
from src.models.train_baseline import DEFAULT_SAMPLE_N, RANDOM_STATE, load_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)

if USE_S3:
    raise NotImplementedError(
        "benchmark_baseline.py: USE_S3=true not yet supported "
        "(load_data is local-FS only)."
    )

_MODEL_PKL = _ROOT / model_path() / "baseline_xgboost.pkl"
_OUT_JSON = _ROOT / "docs" / "model_performance" / "benchmark_comparison.json"

# TabPFN safe-zone settings. v2 caps are ~500 features and 10k context rows;
# we stay well below for memory + speed on this CPU-only host.
TABPFN_TOP_K_FEATURES = 50
TABPFN_CONTEXT_SIZE = 3000


def _best_threshold_by_f1(y_true, y_prob) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5, 0.0
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx])


def _metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1 else 0.5
    )
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = (
        float(precision_score(y_true, y_pred, zero_division=0))
        if int(y_pred.sum()) > 0 else float("nan")
    )
    return {
        "auc_roc": auc,
        "recall": rec,
        "precision": prec,
        "threshold": float(threshold),
    }


def _cost_optimal_footnote() -> str | None:
    """Surface XGBoost's *deployed* operating point so the F1-optimal table
    row isn't read as the only option. Read from baseline_metrics.json."""
    metrics_json = _ROOT / "docs" / "model_performance" / "baseline_metrics.json"
    if not metrics_json.exists():
        return None
    try:
        co = json.loads(metrics_json.read_text())["operating_points_test"][
            "cost_optimal"
        ]
    except (KeyError, ValueError):
        return None
    return (
        "  Note: XGBoost row is at the F1-optimal threshold. The deployed "
        f"cost-optimal\n  threshold ({co['threshold']:.2f}) trades precision "
        f"for recall: recall {co['recall']:.2f}, "
        f"precision {co['precision']:.2f} "
        f"(£{co['expected_loss_gbp']:,.0f}/day)."
    )


def _resolve_sample_n() -> int | None:
    env = os.environ.get("FRAUD_SAMPLE_N")
    if env is None:
        return DEFAULT_SAMPLE_N
    if env.strip().lower() in ("all", "0", "full", "none"):
        return None
    return int(env)


def _xgboost_row(xgb, X_test, y_test, train_n: int) -> dict:
    prob = xgb.predict_proba(X_test)[:, 1]
    thr, _ = _best_threshold_by_f1(y_test, prob)
    out = _metrics(y_test, prob, thr)
    out["training"] = f"{train_n:,}-row train"
    return out


def _dummy_row(X_train, y_train, X_test, y_test) -> dict:
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    prob = dummy.predict_proba(X_test)[:, 1]
    out = _metrics(y_test, prob, 0.5)
    out["training"] = "None"
    return out


def _tabpfn_skipped(reason: str, label: str = "(skipped)") -> dict:
    logger.warning("TabPFN row skipped: %s", reason)
    return {
        "auc_roc": float("nan"),
        "recall": float("nan"),
        "precision": float("nan"),
        "threshold": float("nan"),
        "training": label,
    }


def _tabpfn_row(xgb, X_train, y_train, X_test, y_test) -> dict:
    # TabPFN v8 gates pretrained weights behind a one-time Prior Labs license.
    # Disable the interactive browser flow so a missing token fails cleanly
    # instead of hanging on stdin when run headless / in the background.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")

    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return _tabpfn_skipped(
            "tabpfn not installed (pip install tabpfn)", "(not installed)"
        )

    if not (os.environ.get("TABPFN_TOKEN") or os.environ.get("HF_TOKEN")):
        cached = Path.home() / "AppData" / "Roaming" / "tabpfn"
        if not cached.exists() or not any(cached.glob("*.ckpt")):
            return _tabpfn_skipped(
                "no TABPFN_TOKEN and weights not cached; "
                "register at ux.priorlabs.ai, set TABPFN_TOKEN, re-run",
                "(needs token)",
            )

    imp = getattr(xgb, "feature_importances_", None)
    if imp is None:
        raise RuntimeError("Trained XGBoost has no feature_importances_")
    k = min(TABPFN_TOP_K_FEATURES, X_train.shape[1])
    top_idx = np.argsort(imp)[-k:][::-1]
    top_feats = X_train.columns[top_idx].tolist()
    logger.info("TabPFN: top-%d features by XGB importance", k)

    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = (
        y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    )
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    ctx_total = min(TABPFN_CONTEXT_SIZE, len(y_arr))
    ctx_pos = min(len(pos_idx), ctx_total // 2)
    ctx_neg = min(len(neg_idx), ctx_total - ctx_pos)
    sel = np.concatenate([
        rng.choice(pos_idx, ctx_pos, replace=False),
        rng.choice(neg_idx, ctx_neg, replace=False),
    ])
    rng.shuffle(sel)

    X_ctx = X_train.iloc[sel][top_feats].astype("float32").to_numpy()
    y_ctx = y_arr[sel].astype("int64")
    X_te = X_test[top_feats].astype("float32").to_numpy()

    logger.info(
        "TabPFN: context=%d (pos=%d/neg=%d), predicting on %d rows ...",
        len(X_ctx), ctx_pos, ctx_neg, len(X_te),
    )
    clf = TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=True,
        memory_saving_mode=True,
        random_state=RANDOM_STATE,
    )
    try:
        clf.fit(X_ctx, y_ctx)
        prob = clf.predict_proba(X_te)[:, 1]
    except Exception as exc:  # noqa: BLE001 - auth/license/download failures
        return _tabpfn_skipped(
            f"weight load failed: {type(exc).__name__}", "(load failed)"
        )
    thr, _ = _best_threshold_by_f1(y_test, prob)
    out = _metrics(y_test, prob, thr)
    out["training"] = (
        f"None (in-context: {len(X_ctx):,} rows / {k} feats)"
    )
    return out


def _fmt(value, nan="N/A") -> str:
    return nan if value is None or np.isnan(value) else f"{value:.2f}"


def _print_table(
    test_n: int, rows: list[tuple[str, dict]], footnote: str | None = None
) -> None:
    print()
    print("=" * 70)
    print(f"BASELINE COMPARISON  (test rows = {test_n:,}; "
          f"thr = F1-optimal per model)")
    print("=" * 70)
    print("┌──────────────────────┬────────┬───────────┬─────────┬──────────────────┐")
    print("│ Model                │ Recall │ Precision │ AUC-ROC │ Training         │")
    print("├──────────────────────┼────────┼───────────┼─────────┼──────────────────┤")
    for name, m in rows:
        rec = _fmt(m["recall"])
        prec = _fmt(m["precision"])
        auc = _fmt(m["auc_roc"])
        training = m["training"][:16]
        print(f"│ {name:<20} │ {rec:>6} │ {prec:>9} │ {auc:>7} │ {training:<16} │")
    print("└──────────────────────┴────────┴───────────┴─────────┴──────────────────┘")
    if footnote:
        print(footnote)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if not _MODEL_PKL.exists():
        sys.exit(
            f"Missing {_MODEL_PKL}. Train it first: "
            "python src/models/train_baseline.py"
        )

    sample_n = _resolve_sample_n()
    logger.info(
        "Reproducing train_baseline test set "
        "(sample_n=%s, RANDOM_STATE=%d) ...",
        "FULL" if sample_n is None else f"{sample_n:,}",
        RANDOM_STATE,
    )
    df = load_data(sample_n)
    enriched = build_features(df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        enriched, strategy="combined", test_size=0.20
    )
    logger.info(
        "Train rows=%d  test rows=%d  test fraud rate=%.3f",
        len(y_train), len(y_test), float(y_test.mean()),
    )

    logger.info("Loading XGBoost from %s", _MODEL_PKL)
    xgb = joblib.load(_MODEL_PKL)

    rows = [
        ("Majority class", _dummy_row(X_train, y_train, X_test, y_test)),
        ("TabPFN (zero-shot)",
         _tabpfn_row(xgb, X_train, y_train, X_test, y_test)),
        ("XGBoost (ours)", _xgboost_row(xgb, X_test, y_test, len(y_train))),
    ]
    _print_table(len(y_test), rows, footnote=_cost_optimal_footnote())

    payload = {
        "sample_n": sample_n,
        "random_state": RANDOM_STATE,
        "test_rows": int(len(y_test)),
        "test_fraud_rate": float(y_test.mean()),
        "tabpfn": {
            "top_k_features": TABPFN_TOP_K_FEATURES,
            "context_size": TABPFN_CONTEXT_SIZE,
        },
        "results": {name: m for name, m in rows},
    }
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON -> {_OUT_JSON}")


if __name__ == "__main__":
    main()
