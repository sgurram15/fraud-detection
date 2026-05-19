"""Validate the engineered fraud features.

For every feature added by ``build_features`` (i.e. every column not present in
the raw IEEE-CIS files), this script produces:

  * a distribution plot comparing fraud vs legitimate transactions,
  * a statistical-significance test that the two distributions differ,
  * the feature's correlation with the ``isFraud`` label,

then prints a predictiveness ranking and flags multicollinear feature pairs.
Plots are written to ``docs/feature_validation/``.

Test choice:
  * continuous features  -> Mann-Whitney U (non-parametric; no normality
    assumption) + rank-biserial effect size.
  * binary/low-cardinality -> chi-square test of independence + Cramer's V.
With ~590k rows almost any difference is "statistically significant", so the
ranking is driven by **effect size / univariate ROC AUC**, not p-values alone.
Missing values are dropped per-feature for each test (never row-wise on the
whole frame) and the dropped count is reported.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.features.handle_imbalance import load_features

logger = logging.getLogger(__name__)

TARGET = "isFraud"
_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw"
_PLOT_DIR = _ROOT / "docs" / "feature_validation"

ALPHA = 0.05
MULTICOLLINEAR_HI = 0.90   # hard flag
MULTICOLLINEAR_WARN = 0.80  # softer warning
_CATEGORICAL_MAX_UNIQUE = 10

# Engineered but target-derived -> inflated predictiveness, leakage risk.
_LEAKAGE_PRONE = {"device_type_fraud_rate"}


def _raw_columns() -> set[str]:
    cols: set[str] = set()
    for name in ("train_transaction.csv", "train_identity.csv"):
        hits = sorted(_RAW_DIR.glob(f"**/{name}"))
        if hits:
            cols |= set(pd.read_csv(hits[0], nrows=0).columns)
    return cols


def _engineered_numeric(df: pd.DataFrame) -> list[str]:
    raw = _raw_columns()
    feats = []
    for c in df.columns:
        if c in raw or c == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
        else:
            logger.info("Skipping non-numeric engineered column: %s", c)
    return feats


def _is_categorical(s: pd.Series) -> bool:
    return s.nunique(dropna=True) <= _CATEGORICAL_MAX_UNIQUE


def _plot(df: pd.DataFrame, feat: str, y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    s = df[feat]
    legit = s[y == 0].dropna()
    fraud = s[y == 1].dropna()

    if _is_categorical(s):
        ct = (
            pd.crosstab(s, y, normalize="columns")
            .rename(columns={0: "legit", 1: "fraud"})
        )
        ct.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"])
        ax.set_ylabel("within-class proportion")
        ax.set_xlabel(feat)
    else:
        lo, hi = np.nanpercentile(s.astype(float), [1, 99])
        if not np.isfinite([lo, hi]).all() or lo == hi:
            lo, hi = float(np.nanmin(s)), float(np.nanmax(s)) + 1e-9
        bins = np.linspace(lo, hi, 40)
        ax.hist(legit.clip(lo, hi), bins=bins, density=True, alpha=0.5,
                label="legit", color="#4c72b0")
        ax.hist(fraud.clip(lo, hi), bins=bins, density=True, alpha=0.5,
                label="fraud", color="#c44e52")
        ax.set_ylabel("density")
        ax.set_xlabel(f"{feat} (clipped 1-99 pct)")
        ax.legend()

    ax.set_title(f"{feat}: fraud vs legitimate")
    fig.tight_layout()
    fig.savefig(_PLOT_DIR / f"{feat}.png", dpi=110)
    plt.close(fig)


def _test_and_score(df: pd.DataFrame, feat: str, y: pd.Series) -> dict:
    s = df[feat]
    mask = s.notna()
    dropped = int((~mask).sum())
    sv, yv = s[mask], y[mask]
    legit = sv[yv == 0]
    fraud = sv[yv == 1]

    result = {
        "feature": feat,
        "n_used": int(mask.sum()),
        "n_missing": dropped,
        "kind": "categorical" if _is_categorical(s) else "continuous",
    }

    # Correlation with the label (point-biserial == Pearson vs 0/1 target).
    if sv.nunique() > 1:
        result["corr_with_fraud"] = float(np.corrcoef(sv, yv)[0, 1])
    else:
        result["corr_with_fraud"] = 0.0

    # Univariate ROC AUC, direction-agnostic (strong predictiveness measure).
    try:
        auc = roc_auc_score(yv, sv)
        result["auc"] = float(max(auc, 1 - auc))
    except ValueError:
        result["auc"] = float("nan")

    # Significance test.
    if result["kind"] == "categorical":
        ct = pd.crosstab(sv, yv)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            n = ct.to_numpy().sum()
            k = min(ct.shape) - 1
            result["test"] = "chi2"
            result["p_value"] = float(p)
            result["effect_size"] = float(np.sqrt(chi2 / (n * k))) if k else 0.0
        else:
            result.update(test="chi2", p_value=1.0, effect_size=0.0)
    else:
        if len(legit) and len(fraud):
            u, p = stats.mannwhitneyu(fraud, legit, alternative="two-sided")
            rbc = 1.0 - (2.0 * u) / (len(fraud) * len(legit))
            result["test"] = "mannwhitneyu"
            result["p_value"] = float(p)
            result["effect_size"] = float(abs(rbc))  # |rank-biserial|
        else:
            result.update(test="mannwhitneyu", p_value=1.0, effect_size=0.0)

    result["significant"] = bool(result["p_value"] < ALPHA)
    return result


def _multicollinearity(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    corr = df[feats].corr().abs()
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = corr.iloc[i, j]
            if r >= MULTICOLLINEAR_WARN:
                pairs.append((feats[i], feats[j], float(r)))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[feats].corr(), cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Engineered feature correlation matrix")
    fig.tight_layout()
    fig.savefig(_PLOT_DIR / "_correlation_matrix.png", dpi=110)
    plt.close(fig)

    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_corr"]) \
        .sort_values("abs_corr", ascending=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if TARGET not in df.columns:
        raise KeyError(f"'{TARGET}' not in feature set; cannot validate.")
    y = pd.to_numeric(df[TARGET], errors="coerce").astype("int8")

    feats = _engineered_numeric(df)
    logger.info("Validating %d engineered numeric features.", len(feats))

    rows = []
    for feat in feats:
        _plot(df, feat, y)
        rows.append(_test_and_score(df, feat, y))
        logger.info("  validated %s", feat)

    report = pd.DataFrame(rows)
    report["abs_corr"] = report["corr_with_fraud"].abs()
    # Rank by univariate AUC (fallback to effect size), most predictive first.
    report = report.sort_values(
        ["auc", "effect_size"], ascending=False
    ).reset_index(drop=True)

    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 78)
    print("FEATURE PREDICTIVENESS RANKING (most predictive first)")
    print("=" * 78)
    cols = ["feature", "auc", "abs_corr", "corr_with_fraud", "effect_size",
            "test", "p_value", "significant", "n_missing"]
    print(report[cols].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    leaky = [f for f in report["feature"] if f in _LEAKAGE_PRONE]
    if leaky:
        print("\n[!] LEAKAGE WARNING: target-derived feature(s) inflate the "
              f"ranking: {leaky}")
        print("    Recompute with out-of-fold encoding before trusting these.")

    not_sig = report.loc[~report["significant"], "feature"].tolist()
    if not_sig:
        print(f"\n[i] Not statistically significant at alpha={ALPHA}: "
              f"{not_sig}")

    mc = _multicollinearity(df.assign(**{TARGET: y}), feats)
    print("\n" + "=" * 78)
    print(f"MULTICOLLINEARITY (|corr| >= {MULTICOLLINEAR_WARN})")
    print("=" * 78)
    if mc.empty:
        print("None. No engineered feature pair exceeds the threshold.")
    else:
        for _, r in mc.iterrows():
            tag = "DROP-CANDIDATE" if r.abs_corr >= MULTICOLLINEAR_HI else "watch"
            print(f"  [{tag:14s}] {r.feature_a} <-> {r.feature_b} "
                  f"= {r.abs_corr:.3f}")
        print(f"\n  >= {MULTICOLLINEAR_HI} pairs are strong drop candidates "
              "(keep one per pair).")

    print(f"\nPlots written to {_PLOT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
