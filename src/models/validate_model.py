"""Pre-production model validation gate.

Runs performance, fairness, stability, leakage and serving checks against the
saved model and prints a single PRODUCTION READY / NOT READY verdict with
specific failure reasons. Full report -> docs/model_performance/
validation_report.json.

Notes / honest design choices:
* Recall and FPR are threshold-dependent. The validation operating point is
  chosen on a held-out validation slice as "max recall subject to FPR <= the
  FPR limit", then applied to the untouched test set -- so the reported
  numbers are not fitted to the test set.
* Stability and leakage checks RETRAIN the model (5 seeds + 5-fold CV) using
  the same combined SMOTE+undersample + median-impute pipeline as the rest of
  the project, fit per-split (leakage-safe).
* This is multi-train heavy; on a modest host use FRAUD_SAMPLE_N to bound it
  (default below). VALIDATE_SMOKE=1 forces a tiny fast configuration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.features.build_features import build_features
from src.features.handle_imbalance import _xy
from src.models.train_baseline import DEFAULT_SAMPLE_N, RANDOM_STATE, load_data

logger = logging.getLogger(__name__)

_MODEL_DIR = _ROOT / "src" / "models" / "saved"
_TUNED_PATH = _MODEL_DIR / "tuned_xgboost.pkl"
_BASELINE_PATH = _MODEL_DIR / "baseline_xgboost.pkl"
_PERF_DIR = _ROOT / "docs" / "model_performance"
_REPORT = _PERF_DIR / "validation_report.json"
_TUNING_JSON = _PERF_DIR / "tuning_results.json"

# ---- Acceptance thresholds ----
AUC_MIN = 0.85
RECALL_MIN = 0.75
FPR_MAX = 0.10
STABILITY_TOL = 0.02          # max-min spread allowed across seeds
LEAKAGE_REL_TOL = 0.05        # |cv-test|/cv must be within this
SERVING_MS_MAX = 100.0
FAIRNESS_RECALL_DROP = 0.10   # a subgroup may not be >10pp below overall
FAIRNESS_MIN_GROUP = 50       # ignore tiny subgroups
N_SEEDS = 5
N_SPLITS = 5
TOPK_FI = 15                  # feature-importance overlap window


def _params() -> dict:
    """Tuned best params if available, else sane baseline params."""
    if _TUNING_JSON.exists():
        try:
            bp = json.loads(_TUNING_JSON.read_text())["best_params"]
            return {**bp}
        except Exception:  # noqa: BLE001
            pass
    return {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8}


def _make_model(seed: int, params: dict) -> XGBClassifier:
    return XGBClassifier(
        **params, n_estimators=300, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=seed,
    )


def _resampler(seed: int) -> Pipeline:
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=seed)),
        ("under", RandomUnderSampler(sampling_strategy=1.0,
                                     random_state=seed)),
    ])


def _train(seed: int, params: dict, X_tr, y_tr):
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X_tr)
    Xr, yr = _resampler(seed).fit_resample(Xi, y_tr)
    m = _make_model(seed, params)
    m.fit(Xr, yr)
    return m, imp


def _rates(y_true, y_pred) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(recall), float(fpr)


def _pick_threshold(y_val, proba_val) -> float:
    """Max recall subject to FPR <= FPR_MAX, evaluated on validation."""
    best_t, best_r = 0.5, -1.0
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        r, f = _rates(y_val, (proba_val >= t).astype(int))
        if f <= FPR_MAX and r > best_r:
            best_r, best_t = r, float(t)
    return best_t


def _check(name: str, passed: bool, detail: str) -> dict:
    logger.info("[%s] %s -- %s", "PASS" if passed else "FAIL", name, detail)
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    _PERF_DIR.mkdir(parents=True, exist_ok=True)

    smoke = os.environ.get("VALIDATE_SMOKE", "").strip() in ("1", "true")
    env = os.environ.get("FRAUD_SAMPLE_N")
    if env and env.strip().lower() in ("all", "0", "full", "none"):
        sample_n = None
    elif env:
        sample_n = int(env)
    else:
        sample_n = 6000 if smoke else min(DEFAULT_SAMPLE_N, 40000)
    seeds = [42, 7] if smoke else [42, 7, 13, 21, 99][:N_SEEDS]
    n_splits = 3 if smoke else N_SPLITS

    logger.info("Validation: sample=%s, seeds=%s, cv=%d",
                "ALL" if sample_n is None else f"{sample_n:,}",
                seeds, n_splits)

    df = load_data(sample_n)
    enriched = build_features(df)
    X, y = _xy(enriched)
    # Subgroup labels aligned to X rows (kept for the fairness check).
    grp_device = (enriched["DeviceType"] if "DeviceType" in enriched
                  else pd.Series(["__na__"] * len(X), index=X.index))
    amt = (pd.to_numeric(enriched["TransactionAmt"], errors="coerce")
           if "TransactionAmt" in enriched
           else pd.Series(0.0, index=X.index))
    del df, enriched

    idx = np.arange(len(X))
    tr_i, te_i = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    trv_i, val_i = train_test_split(
        tr_i, test_size=0.15, random_state=RANDOM_STATE, stratify=y.iloc[tr_i]
    )
    X_tr, y_tr = X.iloc[trv_i], y.iloc[trv_i]
    X_val, y_val = X.iloc[val_i], y.iloc[val_i]
    X_te, y_te = X.iloc[te_i], y.iloc[te_i]

    params = _params()
    results: list[dict] = []

    # Production artefact under test (tuned preferred, else baseline, else
    # a freshly trained seed-42 model so validation can still run).
    if _TUNED_PATH.exists():
        prod_model = joblib.load(_TUNED_PATH)
        prod_imp = SimpleImputer(strategy="median").fit(X_tr)
        model_src = _TUNED_PATH.name
    elif _BASELINE_PATH.exists():
        prod_model = joblib.load(_BASELINE_PATH)
        prod_imp = SimpleImputer(strategy="median").fit(X_tr)
        model_src = _BASELINE_PATH.name
    else:
        prod_model, prod_imp = _train(RANDOM_STATE, params, X_tr, y_tr)
        model_src = "freshly-trained (no saved artefact found)"
    logger.info("Model under test: %s", model_src)

    def _proba(model, imp, Xdf):
        cols = getattr(model.get_booster(), "feature_names", None)
        Xa = Xdf.reindex(columns=cols, fill_value=0) if cols else Xdf
        return model.predict_proba(imp.transform(Xa))[:, 1]

    # ---------------- Performance ----------------
    thr = _pick_threshold(y_val, _proba(prod_model, prod_imp, X_val))
    p_te = _proba(prod_model, prod_imp, X_te)
    auc = float(roc_auc_score(y_te, p_te))
    pred_te = (p_te >= thr).astype(int)
    recall, fpr = _rates(y_te, pred_te)

    results.append(_check(
        "performance.auc_roc", auc > AUC_MIN,
        f"AUC-ROC={auc:.4f} (req > {AUC_MIN})"))
    results.append(_check(
        "performance.recall", recall > RECALL_MIN,
        f"recall={recall:.4f} @thr={thr:.2f} (req > {RECALL_MIN})"))
    results.append(_check(
        "performance.false_positive_rate", fpr < FPR_MAX,
        f"FPR={fpr:.4f} @thr={thr:.2f} (req < {FPR_MAX})"))

    # ---------------- Fairness ----------------
    dev_te = grp_device.iloc[te_i]
    fair_fail = []
    for label, mask in (
        [(f"device={d}", (dev_te == d).to_numpy())
         for d in dev_te.dropna().unique()]
        + [(f"amt_band_{i+1}",
            (pd.qcut(amt.iloc[te_i], 4, labels=False,
                     duplicates="drop") == i).to_numpy())
           for i in range(4)]
    ):
        if mask.sum() < FAIRNESS_MIN_GROUP:
            continue
        g_recall, _ = _rates(y_te[mask], pred_te[mask])
        if g_recall < recall - FAIRNESS_RECALL_DROP:
            fair_fail.append(f"{label}: recall {g_recall:.2f} "
                             f"(overall {recall:.2f})")
    results.append(_check(
        "fairness.subgroup_recall", not fair_fail,
        "no subgroup >10pp below overall" if not fair_fail
        else "; ".join(fair_fail)))

    # ---------------- Stability ----------------
    seed_auc, seed_rec, seed_fpr, fi_ranks = [], [], [], []
    for s in seeds:
        m, imp = _train(s, params, X_tr, y_tr)
        ps = _proba(m, imp, X_te)
        seed_auc.append(float(roc_auc_score(y_te, ps)))
        r, f = _rates(y_te, (ps >= thr).astype(int))
        seed_rec.append(r)
        seed_fpr.append(f)
        fi = pd.Series(m.feature_importances_, index=X.columns)
        fi_ranks.append(set(fi.sort_values(ascending=False)
                            .head(TOPK_FI).index))
    spreads = {
        "auc": max(seed_auc) - min(seed_auc),
        "recall": max(seed_rec) - min(seed_rec),
        "fpr": max(seed_fpr) - min(seed_fpr),
    }
    stable = all(v < STABILITY_TOL for v in spreads.values())
    results.append(_check(
        "stability.metric_spread", stable,
        ", ".join(f"{k} spread={v:.4f}" for k, v in spreads.items())
        + f" (req < {STABILITY_TOL})"))
    common = set.intersection(*fi_ranks)
    jacc = len(common) / len(set.union(*fi_ranks))
    results.append(_check(
        "stability.feature_importance", jacc >= 0.80,
        f"top-{TOPK_FI} overlap (Jaccard)={jacc:.2f} across "
        f"{len(seeds)} seeds (req >= 0.80)"))

    # ---------------- Leakage ----------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
    cv_aucs = []
    Xtr_all, ytr_all = X.iloc[tr_i], y.iloc[tr_i]
    for cv_tr, cv_va in skf.split(Xtr_all, ytr_all):
        m, imp = _train(RANDOM_STATE, params,
                        Xtr_all.iloc[cv_tr], ytr_all.iloc[cv_tr])
        pv = _proba(m, imp, Xtr_all.iloc[cv_va])
        cv_aucs.append(float(roc_auc_score(ytr_all.iloc[cv_va], pv)))
    cv_auc = float(np.mean(cv_aucs))
    rel_gap = abs(cv_auc - auc) / cv_auc if cv_auc else 1.0
    results.append(_check(
        "leakage.cv_vs_test", rel_gap < LEAKAGE_REL_TOL,
        f"CV AUC={cv_auc:.4f} vs test AUC={auc:.4f} "
        f"(rel gap {rel_gap:.2%}, req < {LEAKAGE_REL_TOL:.0%})"))

    # ---------------- Serving ----------------
    n_serve = min(1000, len(X_te))
    Xserve = X_te.iloc[:n_serve]
    cols = getattr(prod_model.get_booster(), "feature_names", None)
    Xserve_a = (Xserve.reindex(columns=cols, fill_value=0)
                if cols else Xserve)
    Xserve_i = prod_imp.transform(Xserve_a)
    t0 = time.perf_counter()
    for r in range(n_serve):
        prod_model.predict_proba(Xserve_i[r:r + 1])
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n_serve
    results.append(_check(
        "serving.latency", avg_ms < SERVING_MS_MAX,
        f"avg {avg_ms:.3f} ms/txn over {n_serve} (req < "
        f"{SERVING_MS_MAX:.0f} ms)"))

    serve_ok, serve_detail = True, "missing-value & unseen-device rows scored"
    try:
        row = Xserve_a.iloc[[0]].copy()
        row.iloc[0, :5] = np.nan  # missing values
        prod_model.predict_proba(prod_imp.transform(row))
        row2 = Xserve_a.iloc[[0]].copy()
        if "device_type_fraud_rate" in row2.columns:
            row2["device_type_fraud_rate"] = 0.999999  # unseen device proxy
        prod_model.predict_proba(prod_imp.transform(row2))
    except Exception as exc:  # noqa: BLE001
        serve_ok = False
        serve_detail = f"raised: {exc!r}"
    results.append(_check("serving.robustness", serve_ok, serve_detail))

    # ---------------- Verdict ----------------
    failures = [r for r in results if not r["passed"]]
    ready = not failures
    report = {
        "model_under_test": model_src,
        "sample_n": sample_n,
        "operating_threshold": thr,
        "thresholds": {
            "auc_min": AUC_MIN, "recall_min": RECALL_MIN,
            "fpr_max": FPR_MAX, "stability_tol": STABILITY_TOL,
            "leakage_rel_tol": LEAKAGE_REL_TOL,
            "serving_ms_max": SERVING_MS_MAX,
        },
        "test_metrics": {"auc_roc": auc, "recall": recall, "fpr": fpr},
        "checks": results,
        "verdict": "PRODUCTION READY" if ready else "NOT READY",
        "failed_checks": [r["check"] for r in failures],
    }
    _REPORT.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 64)
    print(f"MODEL VALIDATION VERDICT: {report['verdict']}")
    print("=" * 64)
    for r in results:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: "
              f"{r['detail']}")
    if failures:
        print("\nNOT READY -- failing checks:")
        for r in failures:
            print(f"  - {r['check']}: {r['detail']}")
    else:
        print("\nAll checks passed.")
    print(f"\nReport -> {_REPORT}")


if __name__ == "__main__":
    main()
