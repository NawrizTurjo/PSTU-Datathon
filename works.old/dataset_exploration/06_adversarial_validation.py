"""
Item 2: Adversarial validation - train a classifier to distinguish train rows
from test rows. If it can (AUC well above 0.5), train and test come from
different distributions and CV scores may not reflect leaderboard performance.
Also flags per-feature train/test range mismatches and boolean-flag rate shifts.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

TRAIN = "dataset_exploration/converted_train.csv"
TEST = "dataset_exploration/converted_test.csv"
SCHEMA = "dataset_exploration/column_groups.csv"
TARGET = "Your_Target_Column"

train = pd.read_csv(TRAIN).drop(columns=["id"])
test = pd.read_csv(TEST).drop(columns=["id"])
feat_cols = [c for c in train.columns if c != TARGET]

X = pd.concat([train[feat_cols], test[feat_cols]], ignore_index=True)
y = np.array([0] * len(train) + [1] * len(test))  # 1 = is_test

with open("dataset_exploration/adversarial_validation_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Adversarial validation: classifier predicting is_test ===\n")
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    f.write(f"5-fold CV ROC-AUC (train vs test discrimination): {scores.mean():.4f} +/- {scores.std():.4f}\n")
    f.write("Interpretation: ~0.50 = train/test indistinguishable (safe, iid split).\n"
            "  Notably >0.5 (e.g. >0.55-0.60) = real covariate shift exists; CV may\n"
            "  overstate leaderboard performance, consider adversarial-weighting or\n"
            "  dropping the most shift-driving features.\n\n")

    # feature importance from a full-data fit -> which features drive train/test separability
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    f.write("Top 20 features driving train/test separability (importance in is_test classifier):\n")
    f.write(imp.head(20).to_string() + "\n\n")

    # --- range mismatch scan: values in test outside train's observed [min,max] ---
    f.write("=== Out-of-range features: test values outside train's [min,max] ===\n")
    schema = pd.read_csv(SCHEMA)
    bool_cols = set(schema.loc[schema.group == "boolean_text_flag", "column"])
    oor_rows = []
    for c in feat_cols:
        if c in bool_cols:
            continue
        tr_min, tr_max = train[c].min(), train[c].max()
        below = (test[c] < tr_min).sum()
        above = (test[c] > tr_max).sum()
        if below + above > 0:
            oor_rows.append((c, tr_min, tr_max, below, above,
                              test[c].min(), test[c].max()))
    oor_rows.sort(key=lambda r: -(r[3] + r[4]))
    f.write(f"{'column':45s}{'train_min':>12s}{'train_max':>14s}{'n_below':>9s}{'n_above':>9s}"
            f"{'test_min':>12s}{'test_max':>14s}\n")
    for r in oor_rows[:30]:
        f.write(f"{r[0]:45s}{r[1]:12.3f}{r[2]:14.3f}{r[3]:9d}{r[4]:9d}{r[5]:12.3f}{r[6]:14.3f}\n")
    f.write(f"\nTotal numeric columns with out-of-range test values: {len(oor_rows)} / "
            f"{len(feat_cols) - len(bool_cols)}\n\n")

    # --- boolean flag rate shift ---
    f.write("=== Boolean flag positive-rate shift (train vs test) ===\n")
    shift_rows = []
    for c in bool_cols:
        tr_rate = train[c].mean()
        te_rate = test[c].mean()
        shift_rows.append((c, tr_rate, te_rate, abs(tr_rate - te_rate)))
    shift_rows.sort(key=lambda r: -r[3])
    f.write(f"{'column':45s}{'train_rate':>12s}{'test_rate':>12s}{'abs_diff':>10s}\n")
    for r in shift_rows[:20]:
        f.write(f"{r[0]:45s}{r[1]:12.4f}{r[2]:12.4f}{r[3]:10.4f}\n")

print("wrote dataset_exploration/adversarial_validation_report.txt")
