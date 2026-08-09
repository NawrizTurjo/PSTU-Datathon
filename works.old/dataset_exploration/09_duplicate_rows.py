"""
Item 5: exact duplicate rows within train (with conflicting labels = label
noise risk), and exact feature-row overlap between train and test (possible
leakage / lookup shortcut).
"""
import pandas as pd

TRAIN = "dataset_exploration/converted_train.csv"
TEST = "dataset_exploration/converted_test.csv"
TARGET = "Your_Target_Column"

train = pd.read_csv(TRAIN).drop(columns=["id"])
test = pd.read_csv(TEST).drop(columns=["id"])
feat_cols = [c for c in train.columns if c != TARGET]

with open("dataset_exploration/duplicate_rows_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Exact duplicate rows within train (feature columns only) ===\n")
    dup_mask = train.duplicated(subset=feat_cols, keep=False)
    n_dup_rows = dup_mask.sum()
    f.write(f"Rows involved in an exact feature-duplicate group: {n_dup_rows} "
            f"({n_dup_rows/len(train):.3%} of train)\n")

    if n_dup_rows > 0:
        group_id = train.groupby(feat_cols, sort=False).ngroup()
        gdf = pd.DataFrame({"group": group_id, TARGET: train[TARGET]})
        dup_groups = gdf.groupby("group")[TARGET].agg(["count", "nunique", "mean"])
        dup_groups = dup_groups[dup_groups["count"] > 1]
        conflicting = dup_groups[dup_groups["nunique"] > 1]
        f.write(f"Duplicate groups: {len(dup_groups)}\n")
        f.write(f"Largest duplicate group size: {dup_groups['count'].max()}\n")
        f.write(f"Duplicate groups with CONFLICTING target labels: {len(conflicting)}\n")
        if len(conflicting) > 0:
            f.write(f"Rows affected by label conflict: {conflicting['count'].sum()} "
                    f"({conflicting['count'].sum()/len(train):.3%} of train)\n")
            f.write("Sample conflicting groups (group_id, count, target_mean):\n")
            f.write(conflicting.sort_values('count', ascending=False)
                    .head(10)[["count", "mean"]].to_string() + "\n")
        else:
            f.write("No label conflicts - every duplicate feature-row group agrees on target.\n")

        # is this dominated by one all/mostly-zero mega-row (the very sparse default profile)?
        biggest_group_id = dup_groups["count"].idxmax()
        n_zero_feats = (train.loc[group_id == biggest_group_id, feat_cols].iloc[0] == 0).sum()
        f.write(f"\nLargest duplicate group: {dup_groups['count'].max()} rows, "
                f"{n_zero_feats}/{len(feat_cols)} of its feature values are exactly 0\n"
                f"(context: dataset is heavily zero-inflated, see script 08 - a large chunk of\n"
                f"duplicate rows are likely the 'nothing happened this period' default profile\n"
                f"colliding by chance rather than true duplicate station logs).\n")
    else:
        f.write("No exact duplicate rows found in train.\n")

    f.write("\n=== Train-test exact feature overlap ===\n")
    train_keyed = train[feat_cols].copy()
    test_keyed = test[feat_cols].copy()
    train_keyed["_src"] = "train"
    test_keyed["_src"] = "test"
    combined = pd.concat([train_keyed, test_keyed], ignore_index=True)
    dup_mask2 = combined.duplicated(subset=feat_cols, keep=False)
    overlap = combined[dup_mask2]
    cross_overlap_rows = overlap[overlap["_src"] == "test"]
    f.write(f"Test rows with an identical feature-row also present in train: "
            f"{len(cross_overlap_rows)} / {len(test)} ({len(cross_overlap_rows)/len(test):.3%})\n")
    if len(cross_overlap_rows) > 0:
        f.write("If this is non-trivial, matched train targets could be used as a direct\n"
                "lookup/leak for those specific test rows - worth exploiting if the overlap\n"
                "rate is meaningfully above what all-zero-heavy rows would produce by chance.\n")
    else:
        f.write("No train-test exact feature-row overlap found.\n")

print("wrote dataset_exploration/duplicate_rows_report.txt")
