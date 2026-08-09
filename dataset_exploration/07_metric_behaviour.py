"""
Item 7: Metric behaviour. The competition page contradicts itself: "F1 Score"
in the evaluation section vs "Macro F1" in the submission section. These
diverge hugely at ~3.96% positive rate. Compute the degenerate floors (all
zeros / all ones) for both interpretations to set up the diagnostic LB probe,
then sweep decision thresholds on a quick OOF baseline to show how the
optimal cut-point differs between binary F1 and macro F1.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
OUT = "dataset_exploration/07_metric_behaviour_report.txt"
SWEEP_OUT = "dataset_exploration/07_threshold_sweep.csv"

CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
TARGET = "TARGET"

train = pd.read_csv(TRAIN)
y = train[TARGET].values
n = len(y)
pos_rate = y.mean()

# --- degenerate floors ---
all_zero = np.zeros(n, dtype=int)
all_one = np.ones(n, dtype=int)

binary_f1_zero = f1_score(y, all_zero, average="binary", zero_division=0)
binary_f1_one = f1_score(y, all_one, average="binary", zero_division=0)
macro_f1_zero = f1_score(y, all_zero, average="macro", zero_division=0)
macro_f1_one = f1_score(y, all_one, average="macro", zero_division=0)

# --- quick OOF baseline for threshold sweep (fast RF; the honest baseline
#     with full CV rigor lives in 09_honest_baseline.py) ---
feat_cols = [c for c in train.columns if c != TARGET]
X = train[feat_cols].copy()
for c in CAT_COLS:
    X[c] = pd.factorize(X[c])[0]

clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                              class_weight="balanced", n_jobs=-1, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
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

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"positive rate: {pos_rate:.6f} ({int(y.sum())}/{n})\n\n")

    f.write("=== Degenerate floors: binary F1 vs macro F1 ===\n")
    f.write(f"{'submission':14s}{'binary_f1':>12s}{'macro_f1':>12s}\n")
    f.write(f"{'all zeros':14s}{binary_f1_zero:12.4f}{macro_f1_zero:12.4f}\n")
    f.write(f"{'all ones':14s}{binary_f1_one:12.4f}{macro_f1_one:12.4f}\n\n")

    f.write("=== Diagnostic probe recommendation ===\n")
    f.write(f"Submit all-zeros as an early LB probe.\n")
    f.write(f"  LB ~= {macro_f1_zero:.4f} -> grader is using MACRO F1.\n")
    f.write(f"  LB ~= {binary_f1_zero:.4f} -> grader is using BINARY F1.\n")
    f.write("This single submission resolves the contradiction in the competition page and\n"
            "determines the entire thresholding strategy downstream.\n\n")

    f.write("=== Quick OOF baseline (RandomForest, 5-fold, NOT the final model) ===\n")
    f.write(f"OOF ROC-AUC (threshold independent): {oof_auc:.4f}\n\n")
    f.write("=== Threshold sweep (0.02-0.98, step 0.01) -> 07_threshold_sweep.csv ===\n")
    f.write(f"Best threshold for BINARY F1: t={best_binary['threshold']:.2f}  "
            f"binary_f1={best_binary['binary_f1']:.4f}  macro_f1={best_binary['macro_f1']:.4f}\n")
    f.write(f"Best threshold for MACRO F1:  t={best_macro['threshold']:.2f}  "
            f"binary_f1={best_macro['binary_f1']:.4f}  macro_f1={best_macro['macro_f1']:.4f}\n")
    f.write(f"Naive t=0.5:                  "
            f"binary_f1={at_half['binary_f1']:.4f}  macro_f1={at_half['macro_f1']:.4f}\n\n")
    f.write(f"Binary-F1 gain from tuning threshold vs naive 0.5: "
            f"{best_binary['binary_f1'] - at_half['binary_f1']:+.4f}\n")
    f.write(f"Macro-F1 gain from tuning threshold vs naive 0.5:  "
            f"{best_macro['macro_f1'] - at_half['macro_f1']:+.4f}\n\n")

    f.write("=== Recommendation ===\n")
    f.write(
        "Run the diagnostic probe FIRST. Whichever metric it confirms, tune the decision\n"
        "threshold on out-of-fold predictions to maximize that exact metric (binary_f1 or\n"
        "macro_f1 from this sweep), not accuracy or AUC. Because the grader applies a fixed\n"
        "0.5 cut to whatever is submitted, encode the chosen operating point as hard 0/1\n"
        "labels (or rank-shift probabilities so the 0.5 cut lands at the tuned threshold).\n"
        "Re-tune the threshold every time the model or feature set changes -- the optimum\n"
        "is not stable across model versions.\n"
    )

print("wrote", OUT)
print(f"binary_f1(zero)={binary_f1_zero:.4f} macro_f1(zero)={macro_f1_zero:.4f} "
      f"binary_f1(one)={binary_f1_one:.4f} macro_f1(one)={macro_f1_one:.4f}")
