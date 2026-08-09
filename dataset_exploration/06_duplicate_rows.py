"""
Item 6: Duplicate rows and label conflicts. CLAUDE.md first-pass claims zero
duplicate rows and zero label conflicts (unlike the old Santander dataset).
Verify: exact full-row duplicates within train, duplicate rows across
train/test on feature columns only (id/target excluded), and whether any
duplicate-feature group in train carries conflicting TARGET values.
"""
import pandas as pd

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/06_duplicate_rows_report.txt"

TARGET = "TARGET"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
feat_cols = [c for c in train.columns if c != TARGET]
test_feat_cols = [c for c in test.columns if c != "id"]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Exact full-row duplicates (all feature columns identical) ===\n")
    dup_mask_train = train[feat_cols].duplicated(keep=False)
    n_dup_train_rows = int(dup_mask_train.sum())
    f.write(f"train: {n_dup_train_rows} rows involved in a full-feature-row duplicate "
            f"({train[feat_cols].duplicated(keep='first').sum()} would be dropped by keep='first')\n")

    dup_mask_test = test[test_feat_cols].duplicated(keep=False)
    n_dup_test_rows = int(dup_mask_test.sum())
    f.write(f"test:  {n_dup_test_rows} rows involved in a full-feature-row duplicate\n\n")

    f.write("=== Label conflicts within duplicate-feature-row groups (train only) ===\n")
    if n_dup_train_rows > 0:
        dup_rows = train[dup_mask_train]
        grp = dup_rows.groupby(feat_cols)[TARGET].nunique()
        n_conflicting_groups = int((grp > 1).sum())
        f.write(f"duplicate-feature groups in train: {dup_rows.groupby(feat_cols).ngroups}\n")
        f.write(f"groups with conflicting TARGET values: {n_conflicting_groups}\n")
        if n_conflicting_groups > 0:
            f.write("!!! LABEL NOISE PRESENT: identical feature rows with different TARGET.\n")
        else:
            f.write("All duplicate-feature groups agree on TARGET -- no label conflicts.\n")
    else:
        f.write("No duplicate feature-rows in train -> no label conflicts possible.\n")

    f.write("\n=== Train rows whose features exactly match a test row ===\n")
    train_keyed = train[feat_cols].copy()
    test_keyed = test[test_feat_cols].copy()
    train_keyed["_src"] = "train"
    test_keyed["_src"] = "test"
    combined = pd.concat([train_keyed, test_keyed], ignore_index=True)
    cross_dup_mask = combined.duplicated(subset=feat_cols, keep=False)
    cross_dups = combined[cross_dup_mask]
    n_cross = cross_dups.groupby(feat_cols)["_src"].apply(lambda s: set(s) == {"train", "test"}).sum()
    f.write(f"feature-groups appearing in BOTH train and test: {int(n_cross)}\n")

print("wrote", OUT)
print(f"train_dup_rows={n_dup_train_rows} test_dup_rows={n_dup_test_rows}")
