"""
Item 8: Leak diagnostics -- the checks that proved the old Santander-lineage
dataset was leak-free/leak-compromised last time. Four probes:
  (a) row-index leak: does raw row order predict TARGET?
  (b) best single-feature AUC: any one "magic feature" separating the target?
  (c) capacity test: unregularized deep tree -- if train AUC -> 1.0 while a
      held-out val AUC also approaches 1.0, there is a near-deterministic
      rule (leak). If val AUC craters relative to train, there's a real,
      unavoidable ceiling (overfitting, not leak).
  (d) duplicate-row label consistency: already established as N/A in
      06_duplicate_rows.py (zero duplicate feature-rows in train), restated
      here for completeness of the leak-diagnostic checklist.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
OUT = "dataset_exploration/08_leak_diagnostics_report.txt"

CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
TARGET = "TARGET"

train = pd.read_csv(TRAIN)
y = train[TARGET].values
n = len(y)
feat_cols = [c for c in train.columns if c != TARGET]
numeric_cols = [c for c in feat_cols if c not in CAT_COLS]

with open(OUT, "w", encoding="utf-8") as f:
    # --- (a) row-index leak ---
    f.write("=== (a) Row-index leak ===\n")
    idx = np.arange(n)
    idx_auc = roc_auc_score(y, idx)
    idx_auc = max(idx_auc, 1 - idx_auc)
    f.write(f"AUC of raw row-index predicting TARGET: {idx_auc:.4f} "
            f"(0.5 = no leak, direction-agnostic)\n")
    n_blocks = 10
    block_size = n // n_blocks
    f.write(f"Positive rate per {n_blocks} equal row-index blocks:\n")
    for b in range(n_blocks):
        lo = b * block_size
        hi = n if b == n_blocks - 1 else (b + 1) * block_size
        rate = y[lo:hi].mean()
        f.write(f"  rows [{lo:6d}:{hi:6d}): pos_rate={rate:.4f}\n")
    f.write("\n")

    # --- (b) best single-feature AUC ---
    f.write("=== (b) Best single-feature AUC (numeric columns, direction-agnostic) ===\n")
    aucs = []
    for c in numeric_cols:
        v = train[c].values
        try:
            a = roc_auc_score(y, v)
        except ValueError:
            continue
        a = max(a, 1 - a)
        aucs.append((c, a))
    aucs.sort(key=lambda r: -r[1])
    f.write("Top 15 single-feature AUCs:\n")
    for c, a in aucs[:15]:
        f.write(f"  {c:12s} AUC={a:.4f}\n")
    f.write(f"\nHighest single-feature AUC: {aucs[0][1]:.4f} ({aucs[0][0]}).\n"
            f"Interpretation: a 'magic feature' would show AUC > ~0.85-0.90 alone.\n"
            f"{'A magic feature IS present.' if aucs[0][1] > 0.85 else 'No single-feature magic bullet found.'}\n\n")

    # --- (c) capacity test ---
    f.write("=== (c) Capacity test: unregularized DecisionTree, train vs held-out AUC ===\n")
    X = train[feat_cols].copy()
    for c in CAT_COLS:
        X[c] = pd.factorize(X[c])[0]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    tree = DecisionTreeClassifier(random_state=42)  # no depth/leaf regularization
    tree.fit(X_tr, y_tr)
    train_auc = roc_auc_score(y_tr, tree.predict_proba(X_tr)[:, 1])
    val_auc = roc_auc_score(y_val, tree.predict_proba(X_val)[:, 1])
    f.write(f"train AUC (fit set, unregularized -> near-perfect memorization expected): {train_auc:.4f}\n")
    f.write(f"held-out val AUC: {val_auc:.4f}\n")
    gap = train_auc - val_auc
    f.write(f"gap: {gap:.4f}\n")
    if val_auc > 0.95:
        f.write("Val AUC also near 1.0 -> suspicious, suggests a near-deterministic rule (possible leak).\n")
    else:
        f.write("Val AUC drops well below train AUC -> normal overfitting, no deterministic rule found.\n"
                "This confirms a real generalization ceiling exists (unlike a leaked dataset).\n")
    f.write("\n")

    # --- (d) duplicate-group label consistency ---
    f.write("=== (d) Duplicate-row label consistency ===\n")
    f.write("See 06_duplicate_rows_report.txt: 0 duplicate feature-rows found in train, so\n"
            "there are zero duplicate-feature groups to check for label conflicts. This\n"
            "diagnostic is N/A for this dataset (unlike the old competition, which had\n"
            "confirmed label-conflicting duplicate rows).\n")

print("wrote", OUT)
print(f"idx_auc={idx_auc:.4f} best_feat_auc={aucs[0][1]:.4f} ({aucs[0][0]}) "
      f"train_auc={train_auc:.4f} val_auc={val_auc:.4f}")
