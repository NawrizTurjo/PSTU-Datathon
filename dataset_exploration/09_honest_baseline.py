"""
Item 9: Honest baseline to anchor everything else. HistGradientBoosting
(sklearn's LightGBM-alike; the real solution should use actual LightGBM/
XGBoost/CatBoost on Kaggle, none of which are available in this local env).
5-fold stratified OOF. Applies the cleanup this exploration has established:
  - drop the 44 exact-constant/duplicate columns from 04
  - treat -999999 (feat_109) and 9999999999 (23 cols) as missing (NaN) --
    HistGradientBoosting has native missing-value support, no imputation needed
  - categorical columns ordinal-encoded (fit on train only; unseen levels in
    test would map past the training range, which HGB treats as just another
    split point -- acceptable for this anchor baseline, not the final model)
Reports OOF ROC-AUC plus binary-F1 / macro-F1 threshold sweep so this ties
directly into 07's metric-behaviour findings.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
OUT = "dataset_exploration/09_honest_baseline_report.txt"
SWEEP_OUT = "dataset_exploration/09_threshold_sweep.csv"

CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
TARGET = "TARGET"
SENTINEL2_COLS = ["feat_11", "feat_21", "feat_26", "feat_30", "feat_31", "feat_36", "feat_74",
                   "feat_77", "feat_96", "feat_124", "feat_135", "feat_144", "feat_149",
                   "feat_158", "feat_171", "feat_196", "feat_204", "feat_226", "feat_301",
                   "feat_315", "feat_330", "feat_336", "feat_340"]

train = pd.read_csv(TRAIN)
y = train[TARGET].values
feat_cols = [c for c in train.columns if c != TARGET]

# --- drop exact-constant/duplicate columns (recomputed inline, see 04 for detail) ---
numeric_cols_all = [c for c in feat_cols if c not in CAT_COLS]
const_both = [c for c in numeric_cols_all if train[c].nunique(dropna=False) == 1]
dup_search_cols = [c for c in numeric_cols_all if c not in const_both]
col_hashes = {}
for c in dup_search_cols:
    h = pd.util.hash_pandas_object(train[c], index=False).sum()
    col_hashes.setdefault(h, []).append(c)
redundant = set()
seen = set()
for h, cols in col_hashes.items():
    if len(cols) < 2:
        continue
    remaining = list(cols)
    while remaining:
        base = remaining.pop(0)
        if base in seen:
            continue
        group, still = [base], []
        for c in remaining:
            if train[base].equals(train[c]):
                group.append(c)
                seen.add(c)
            else:
                still.append(c)
        remaining = still
        if len(group) > 1:
            keep, *drop = sorted(group)
            redundant.update(drop)
droppable = set(const_both) | redundant

use_cols = [c for c in feat_cols if c not in droppable]

X = train[use_cols].copy()

# --- sentinel -> NaN ---
if "feat_109" in X.columns:
    X["feat_109"] = X["feat_109"].replace(-999999, np.nan)
for c in SENTINEL2_COLS:
    if c in X.columns:
        X[c] = X[c].replace(9999999999, np.nan)

# --- categorical encoding: ordinal, fit on train only ---
use_cat_cols = [c for c in CAT_COLS if c in X.columns]
for c in use_cat_cols:
    X[c] = pd.factorize(X[c])[0].astype(float)
    X[c] = X[c].replace(-1, np.nan)  # factorize gives -1 for NaN input, none expected here

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                      l2_regularization=1.0, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y))
fold_aucs = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    clf.fit(X.iloc[tr_idx], y[tr_idx])
    p = clf.predict_proba(X.iloc[va_idx])[:, 1]
    oof_proba[va_idx] = p
    fold_aucs.append(roc_auc_score(y[va_idx], p))

oof_auc = roc_auc_score(y, oof_proba)

rows = []
for t in np.arange(0.02, 0.99, 0.01):
    pred = (oof_proba >= t).astype(int)
    bf1 = f1_score(y, pred, average="binary", zero_division=0)
    mf1 = f1_score(y, pred, average="macro", zero_division=0)
    rows.append({"threshold": round(t, 2), "binary_f1": bf1, "macro_f1": mf1,
                 "n_predicted_positive": int(pred.sum())})
results = pd.DataFrame(rows)
results.to_csv(SWEEP_OUT, index=False)
best_binary = results.loc[results["binary_f1"].idxmax()]
best_macro = results.loc[results["macro_f1"].idxmax()]
at_half = results.iloc[(results["threshold"] - 0.5).abs().idxmin()]

# feature importance proxy: permutation not run here (slow); use built-in
# training-set gain via a fresh full-data fit's feature usage is not exposed
# by HGB directly, so report top single-feature AUCs again for reference is
# skipped -- see 08 for that. Instead report top predictors via a quick
# correlation of OOF residual-free proba with each feature (cheap proxy).

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Honest baseline: HistGradientBoostingClassifier, 5-fold stratified OOF ===\n")
    f.write("(local stand-in for LightGBM/XGBoost/CatBoost, unavailable in this env;\n"
            " swap in real GBDT libraries on Kaggle for the actual solution)\n\n")
    f.write(f"columns used: {len(use_cols)} / {len(feat_cols)} "
            f"(dropped {len(droppable)} exact-constant/duplicate columns)\n")
    f.write(f"sentinels treated as NaN: feat_109 (-999999), {len(SENTINEL2_COLS)} cols "
            f"(9999999999)\n\n")

    f.write(f"Per-fold ROC-AUC: {[round(a, 4) for a in fold_aucs]}\n")
    f.write(f"Mean fold AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}\n")
    f.write(f"OOF ROC-AUC (all folds pooled): {oof_auc:.4f}\n\n")

    f.write("=== Threshold sweep summary (full sweep -> 09_threshold_sweep.csv) ===\n")
    f.write(f"Best threshold for BINARY F1: t={best_binary['threshold']:.2f}  "
            f"binary_f1={best_binary['binary_f1']:.4f}  macro_f1={best_binary['macro_f1']:.4f}\n")
    f.write(f"Best threshold for MACRO F1:  t={best_macro['threshold']:.2f}  "
            f"binary_f1={best_macro['binary_f1']:.4f}  macro_f1={best_macro['macro_f1']:.4f}\n")
    f.write(f"Naive t=0.5:                  "
            f"binary_f1={at_half['binary_f1']:.4f}  macro_f1={at_half['macro_f1']:.4f}\n\n")

    f.write("=== This anchors Stage 2/3 ===\n")
    f.write(f"A cleaned-up out-of-the-box GBDT already reaches OOF AUC {oof_auc:.4f} and "
            f"tuned binary_f1 {best_binary['binary_f1']:.4f} / macro_f1 {best_macro['macro_f1']:.4f}.\n"
            "Real LightGBM/XGBoost/CatBoost with proper hyperparameter tuning, categorical\n"
            "target-encoding, and row-aggregate features (per the Santander-lineage playbook)\n"
            "should beat this. Treat these numbers as a floor, not a ceiling.\n")

print("wrote", OUT)
print(f"oof_auc={oof_auc:.4f} best_binary_f1={best_binary['binary_f1']:.4f} "
      f"best_macro_f1={best_macro['macro_f1']:.4f}")
