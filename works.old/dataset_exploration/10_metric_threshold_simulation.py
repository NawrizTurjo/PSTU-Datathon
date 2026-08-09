"""
Item 6: The leaderboard metric is a weighted composite of 6 threshold-dependent
+ threshold-free scores. Fit a quick baseline model (RandomForest, out-of-fold
via CV), then sweep decision thresholds to see how the composite score responds
and find the threshold that would have been optimal - almost certainly not 0.5
given the 5% positive rate.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                              balanced_accuracy_score, confusion_matrix)

TRAIN = "dataset_exploration/converted_train.csv"
TARGET = "Your_Target_Column"

train = pd.read_csv(TRAIN).drop(columns=["id"])
X = train.drop(columns=[TARGET])
y = train[TARGET].values

clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                              class_weight="balanced", n_jobs=-1, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

roc_auc = roc_auc_score(y, oof_proba)  # threshold-independent


def composite_score(y_true, y_pred, y_proba):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2
    total = 0.30 * f1 + 0.25 * auc + 0.15 * prec + 0.15 * rec + 0.10 * bal_acc + 0.05 * spec
    return dict(f1=f1, roc_auc=auc, precision=prec, recall=rec,
                specificity=spec, balanced_accuracy=bal_acc, composite=total)


rows = []
for t in np.arange(0.02, 0.91, 0.02):
    y_pred = (oof_proba >= t).astype(int)
    m = composite_score(y, y_pred, oof_proba)
    m["threshold"] = round(t, 2)
    rows.append(m)

results = pd.DataFrame(rows)[["threshold", "f1", "precision", "recall", "specificity",
                               "balanced_accuracy", "roc_auc", "composite"]]
results.to_csv("dataset_exploration/threshold_sweep.csv", index=False)

best = results.loc[results["composite"].idxmax()]
at_half = results.iloc[(results["threshold"] - 0.5).abs().idxmin()]

with open("dataset_exploration/threshold_simulation_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Baseline model: RandomForest, 5-fold out-of-fold predictions ===\n")
    f.write("(quick baseline for threshold-behavior exploration only, NOT a final model -\n"
            " no feature cleanup / sentinel handling / duplicate-row dedup applied yet)\n\n")
    f.write(f"Out-of-fold ROC-AUC (threshold independent): {roc_auc:.4f}\n\n")
    f.write("=== Threshold sweep (0.02 to 0.90 step 0.02) ===\n")
    f.write(results.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n\n")
    f.write(f"=== Best threshold by composite score ===\n{best.to_string()}\n\n")
    f.write(f"=== Naive threshold=0.5 (for comparison) ===\n{at_half.to_string()}\n\n")
    f.write(f"Composite score gain from tuning threshold vs naive 0.5: "
            f"{best['composite'] - at_half['composite']:+.4f}\n\n")
    f.write("=== Recommendation ===\n")
    f.write("Do not submit argmax/0.5-threshold binary predictions blindly. Pick the decision\n"
            "threshold that maximizes this exact weighted composite on out-of-fold predictions\n"
            "(reuse composite_score() above during model selection), and re-tune it per model/\n"
            "feature-set change since the optimum shifts with model calibration. Keep predicted\n"
            "PROBABILITIES uncalibrated/raw for Target_Probability (drives ROC-AUC, 0.25 weight)\n"
            "but use the tuned threshold only for the separate Target_Binary column.\n")

print("wrote dataset_exploration/threshold_simulation_report.txt")
print("wrote dataset_exploration/threshold_sweep.csv")
print(f"best threshold={best['threshold']} composite={best['composite']:.4f} "
      f"vs t=0.5 composite={at_half['composite']:.4f}")
