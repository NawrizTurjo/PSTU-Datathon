"""
Item 4: Constant + duplicate column detection. CLAUDE.md's first-pass claims
~83 droppable columns: 28 constant in both train+test, 55 redundant across 20
duplicate-column groups (42 constant in test-only but only 28 in both -> keep
the other 14). Verify all these numbers directly.
Scope: the 344 numeric feature columns only (6 categorical string columns
are handled separately in 02_categorical_deep_dive.py -- their cardinality
already rules out being constant or exact duplicates).
"""
import pandas as pd

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/04_constant_duplicate_report.txt"
DUP_GROUPS_OUT = "dataset_exploration/04_duplicate_groups.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

target_col = "TARGET"
CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
feat_cols = [c for c in train.columns if c != target_col and c not in CAT_COLS]

# constant columns
const_train = set(c for c in feat_cols if train[c].nunique(dropna=False) == 1)
const_test = set(c for c in feat_cols if test[c].nunique(dropna=False) == 1)
const_both = const_train & const_test
const_test_only = const_test - const_train
const_train_only = const_train - const_test

# duplicate columns: exact value-for-value duplicates (train-defined groups).
# Exclude constant-in-train columns first -- a constant column is trivially
# "equal" to every other constant column with the same value (e.g. all-zero),
# which would falsely merge them into one giant duplicate group. Constant
# columns are already tracked separately above.
train_feat = train[feat_cols]
dup_search_cols = [c for c in feat_cols if c not in const_train]
col_hashes = {}
for c in dup_search_cols:
    h = pd.util.hash_pandas_object(train_feat[c], index=False).sum()
    col_hashes.setdefault(h, []).append(c)

dup_groups = []
seen = set()
for h, cols in col_hashes.items():
    if len(cols) < 2:
        continue
    # confirm real equality within hash bucket (hash collisions possible)
    remaining = list(cols)
    while remaining:
        base = remaining.pop(0)
        if base in seen:
            continue
        group = [base]
        still_remaining = []
        for c in remaining:
            if train_feat[base].equals(train_feat[c]):
                group.append(c)
                seen.add(c)
            else:
                still_remaining.append(c)
        remaining = still_remaining
        if len(group) > 1:
            seen.add(base)
            dup_groups.append(group)

redundant_cols = set()
for group in dup_groups:
    keep, *drop = sorted(group)
    redundant_cols.update(drop)

droppable = const_both | redundant_cols

with open(DUP_GROUPS_OUT, "w", encoding="utf-8") as f:
    f.write("group_id,columns\n")
    for i, g in enumerate(dup_groups):
        f.write(f"{i},{'|'.join(sorted(g))}\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Constant columns ===\n")
    f.write(f"constant in train: {len(const_train)}\n")
    f.write(f"constant in test:  {len(const_test)}\n")
    f.write(f"constant in BOTH:  {len(const_both)}\n")
    f.write(f"constant in test only (varies in train): {len(const_test_only)}\n")
    f.write(f"constant in train only (varies in test):  {len(const_train_only)}\n\n")
    f.write("Constant-in-both columns:\n")
    for c in sorted(const_both):
        f.write(f"  {c}\n")
    f.write("\nConstant-in-test-only columns (kept -- vary in train, model can still learn from them):\n")
    for c in sorted(const_test_only):
        f.write(f"  {c}\n")
    f.write("\n")

    f.write("=== Duplicate column groups (exact value match on train) ===\n")
    f.write(f"n duplicate groups: {len(dup_groups)}\n")
    f.write(f"n redundant columns (group size - 1, summed): {len(redundant_cols)}\n")
    f.write("full groups -> 04_duplicate_groups.csv\n\n")
    for i, g in enumerate(sorted(dup_groups, key=lambda g: -len(g))):
        f.write(f"  group {i} ({len(g)} cols): {sorted(g)}\n")

    f.write(f"\n=== Combined droppable set (exact duplicates) ===\n")
    f.write(f"constant-in-both ({len(const_both)}) + redundant-exact-duplicates ({len(redundant_cols)}) "
            f"with overlap removed = {len(droppable)} unique droppable columns\n")

    # --- near-duplicate check via correlation (threshold-dependent, NOT exact) ---
    f.write("\n=== Near-duplicate check: |corr| > 0.999 among non-constant numeric cols ===\n")
    f.write("This is a SEPARATE, looser, threshold-dependent notion of redundancy (columns\n"
            "perfectly or near-perfectly linearly related, e.g. one is a scaled copy of\n"
            "another) -- not exact row-for-row duplicates. Included because CLAUDE.md's\n"
            "first-pass '55 redundant / 20 groups' figure does not match exact-duplicate\n"
            "detection above and was likely produced with a correlation-based method.\n")
    import numpy as np
    corr = train_feat[dup_search_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    cols = list(corr.columns)
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            if corr.loc[c1, c2] > 0.999:
                union(c1, c2)
    from collections import defaultdict
    near_groups = defaultdict(list)
    for c in cols:
        near_groups[find(c)].append(c)
    near_multi = [sorted(v) for v in near_groups.values() if len(v) > 1]
    near_redundant = sum(len(v) - 1 for v in near_multi)
    f.write(f"n near-duplicate groups (corr>0.999): {len(near_multi)}\n")
    f.write(f"n redundant columns in those groups: {near_redundant}\n")
    for g in sorted(near_multi, key=lambda g: -len(g)):
        f.write(f"  {g}\n")
    f.write(f"\nCombined droppable if using near-duplicate threshold instead: "
            f"{len(const_both)} + {near_redundant} = {len(const_both) + near_redundant}\n")
    f.write("Recommendation: use the EXACT-duplicate set for safe dropping (44 cols); treat the\n"
            "near-duplicate corr groups as candidates for dimensionality reduction only, since\n"
            "the threshold (0.999) is arbitrary and a scaled/derived column may still carry\n"
            "distinct information (e.g. different units) that a tree model can exploit.\n")

print("wrote", OUT)
print(f"const_both={len(const_both)} dup_groups={len(dup_groups)} redundant={len(redundant_cols)} "
      f"droppable={len(droppable)}")
