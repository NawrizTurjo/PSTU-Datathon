"""
Item 5: Adversarial validation. Train a classifier to distinguish train rows
from test rows. AUC near 0.5 => train/test look like an iid split (CV should
transfer to LB). AUC well above 0.5 => real covariate shift; flags which
features drive the separability so they can be down-weighted or dropped.
Categorical columns are label-encoded (fit on the train+test union of levels
-- fine for this diagnostic since we are not building a submission model).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/05_adversarial_validation_report.txt"

CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
TARGET = "TARGET"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST).drop(columns=["id"])
feat_cols = [c for c in train.columns if c != TARGET]

X = pd.concat([train[feat_cols], test[feat_cols]], ignore_index=True)
y = np.array([0] * len(train) + [1] * len(test))  # 1 = is_test

for c in CAT_COLS:
    X[c] = pd.factorize(X[c])[0]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Adversarial validation: classifier predicting is_test ===\n")
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    f.write(f"5-fold CV ROC-AUC (train vs test discrimination): {scores.mean():.4f} +/- {scores.std():.4f}\n")
    f.write("Interpretation: ~0.50 = train/test indistinguishable (safe, iid split).\n"
            "  Notably >0.5 (e.g. >0.55-0.60) = real covariate shift exists; CV may\n"
            "  overstate leaderboard performance, consider adversarial-weighting or\n"
            "  dropping the most shift-driving features.\n\n")

    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    f.write("Top 20 features driving train/test separability (importance in is_test classifier):\n")
    f.write(imp.head(20).to_string(float_format=lambda v: f"{v:.4f}") + "\n\n")

    f.write("=== Out-of-range features: test values outside train's [min,max] (numeric only) ===\n")
    numeric_cols = [c for c in feat_cols if c not in CAT_COLS]
    train_num = train[numeric_cols]
    test_num = pd.read_csv(TEST)[numeric_cols]
    oor_rows = []
    for c in numeric_cols:
        tr_min, tr_max = train_num[c].min(), train_num[c].max()
        below = (test_num[c] < tr_min).sum()
        above = (test_num[c] > tr_max).sum()
        if below + above > 0:
            oor_rows.append((c, tr_min, tr_max, below, above, test_num[c].min(), test_num[c].max()))
    oor_rows.sort(key=lambda r: -(r[3] + r[4]))
    f.write(f"{'column':12s}{'train_min':>16s}{'train_max':>18s}{'n_below':>9s}{'n_above':>9s}"
            f"{'test_min':>16s}{'test_max':>18s}\n")
    for r in oor_rows[:30]:
        f.write(f"{r[0]:12s}{r[1]:16.2f}{r[2]:18.2f}{r[3]:9d}{r[4]:9d}{r[5]:16.2f}{r[6]:18.2f}\n")
    f.write(f"\nTotal numeric columns with out-of-range test values: {len(oor_rows)} / {len(numeric_cols)}\n")

print("wrote", OUT)
print(f"adversarial AUC mean={scores.mean():.4f}")
