"""
Item 1: Santander-derived datasets often carry constant (zero-variance) columns
and exact duplicate columns. Detect both across train+test combined (a column
must be constant/duplicate in BOTH to be safely dropped).
"""
import pandas as pd

TRAIN = "dataset_exploration/converted_train.csv"
TEST = "dataset_exploration/converted_test.csv"
TARGET = "Your_Target_Column"

train = pd.read_csv(TRAIN).drop(columns=["id"])
test = pd.read_csv(TEST).drop(columns=["id"])

feat_cols = [c for c in train.columns if c != TARGET]

with open("dataset_exploration/zero_variance_and_duplicates_report.txt", "w", encoding="utf-8") as f:
    # --- zero variance ---
    const_train = [c for c in feat_cols if train[c].nunique(dropna=False) <= 1]
    const_test = [c for c in feat_cols if test[c].nunique(dropna=False) <= 1]
    const_both = sorted(set(const_train) & set(const_test))

    f.write(f"=== Zero-variance (constant) columns ===\n")
    f.write(f"Constant in train: {len(const_train)}\n")
    f.write(f"Constant in test:  {len(const_test)}\n")
    f.write(f"Constant in BOTH (safe to drop): {len(const_both)}\n")
    for c in const_both:
        f.write(f"  - {c} (value={train[c].iloc[0]})\n")
    only_train = sorted(set(const_train) - set(const_test))
    if only_train:
        f.write(f"\nConstant in train only ({len(only_train)}) - KEEP, test has variance:\n")
        for c in only_train:
            f.write(f"  - {c}\n")

    # --- duplicate columns (identical values row-for-row) ---
    f.write(f"\n=== Duplicate columns (exact value match across all rows, train) ===\n")
    # hash each column's values to group candidates cheaply, then verify equality
    remaining = [c for c in feat_cols if c not in const_both]  # skip already-constant
    hashes = {}
    for c in remaining:
        h = pd.util.hash_pandas_object(train[c], index=False).sum()
        hashes.setdefault(h, []).append(c)

    dup_groups = []
    for h, cols in hashes.items():
        if len(cols) < 2:
            continue
        # verify exact equality within hash bucket (avoid hash collisions)
        base = cols[0]
        group = [base]
        for c in cols[1:]:
            if train[c].equals(train[base]):
                group.append(c)
        if len(group) > 1:
            dup_groups.append(group)

    f.write(f"Duplicate column groups found: {len(dup_groups)}\n")
    total_dupe_cols = 0
    for group in dup_groups:
        f.write(f"  IDENTICAL: {group}\n")
        total_dupe_cols += len(group) - 1
    f.write(f"\nTotal redundant columns to drop (keep 1 per group): {total_dupe_cols}\n")

    f.write(f"\n=== Summary ===\n")
    f.write(f"Original feature count: {len(feat_cols)}\n")
    f.write(f"After dropping constants ({len(const_both)}) and dupes ({total_dupe_cols}): "
            f"{len(feat_cols) - len(const_both) - total_dupe_cols}\n")

    f.write(f"\n=== Santander naming notes (for context, not a real un-obfuscation) ===\n")
    f.write("This dataset's obfuscated numeric columns (num_var*, num_op_var*, num_aport*,\n"
            "num_compra*, num_trasp*, num_venta*, num_reemb*, num_sal*, num_ent*, num_med*)\n"
            "match Santander Customer Satisfaction column names exactly. In the original\n"
            "Santander competition var3 (here: base_number_of_dependent_farmers) carried the\n"
            "-999999 sentinel confirmed in script 04, and num_var38 was widely reverse-engineered\n"
            "by the Kaggle community as a customer's total balance/relationship value (here would\n"
            "map to some financial/engagement magnitude, likely num_var38-equivalent if present).\n"
            "There is no public, verified full field-by-field mapping beyond var3 and num_var38 -\n"
            "treat any other 'meaning' assignment as speculative and not worth relying on for\n"
            "feature engineering; the renamed domain names (sensor_*, cost_*, count_*) in THIS\n"
            "dataset are the actual synthetic semantics assigned by the competition, independent\n"
            "of Santander's original meaning.\n")

print("wrote dataset_exploration/zero_variance_and_duplicates_report.txt")
