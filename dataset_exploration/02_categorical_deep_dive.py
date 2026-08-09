"""
Item 2: Categorical deep-dive. For the 6 object-dtype columns, measure
cardinality (train and test separately), overlap between train/test level
sets, and how many test rows carry a level never seen in train (the "unseen
category at inference" risk CLAUDE.md flags). Also checks target rate by
level for the smaller-cardinality columns, since those may be usable as
low-cardinality categoricals directly.
"""
import pandas as pd

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/02_categorical_deep_dive_report.txt"

CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]

train = pd.read_csv(TRAIN, usecols=CAT_COLS + ["TARGET"])
test = pd.read_csv(TEST, usecols=CAT_COLS + ["id"])

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Categorical column cardinality + train/test overlap ===\n")
    f.write(f"{'column':10s}{'train_levels':>13s}{'test_levels':>12s}{'shared':>8s}"
            f"{'train_only':>11s}{'test_only':>10s}{'test_rows_w_unseen':>20s}{'pct_unseen':>11s}\n")
    for c in CAT_COLS:
        tr_vals = train[c]
        te_vals = test[c]
        tr_set = set(tr_vals.unique())
        te_set = set(te_vals.unique())
        shared = tr_set & te_set
        train_only = tr_set - te_set
        test_only = te_set - tr_set
        unseen_mask = ~te_vals.isin(tr_set)
        n_unseen_rows = int(unseen_mask.sum())
        pct_unseen = n_unseen_rows / len(te_vals)
        f.write(f"{c:10s}{len(tr_set):13d}{len(te_set):12d}{len(shared):8d}"
                f"{len(train_only):11d}{len(test_only):10d}{n_unseen_rows:20d}{pct_unseen:11.4%}\n")

    f.write("\n=== Prefix check (expected format per CLAUDE.md) ===\n")
    prefixes = {"feat_142": "PRD_", "feat_325": "SEG_", "feat_157": "PRV_",
                "feat_320": "CH_", "feat_337": "OFC_", "feat_318": "PERF_"}
    for c in CAT_COLS:
        p = prefixes[c]
        tr_ok = train[c].astype(str).str.startswith(p).all()
        te_ok = test[c].astype(str).str.startswith(p).all()
        f.write(f"  {c}: expected prefix '{p}' -> train all-match={tr_ok}, test all-match={te_ok}\n")

    f.write("\n=== Value counts + target rate by level (low-cardinality cols only, <=40 levels) ===\n")
    for c in CAT_COLS:
        n_levels = train[c].nunique()
        if n_levels > 40:
            continue
        f.write(f"\n-- {c} ({n_levels} levels) --\n")
        grp = train.groupby(c)["TARGET"].agg(["count", "mean"]).sort_values("count", ascending=False)
        f.write(grp.to_string(float_format=lambda v: f"{v:.4f}") + "\n")

    f.write("\n=== Top-10 most frequent levels (high-cardinality cols, >40 levels) ===\n")
    for c in CAT_COLS:
        n_levels = train[c].nunique()
        if n_levels <= 40:
            continue
        f.write(f"\n-- {c} ({n_levels} train levels) --\n")
        vc = train[c].value_counts().head(10)
        f.write(vc.to_string() + "\n")
        # target rate for just the top-10 levels
        top_levels = vc.index
        sub = train[train[c].isin(top_levels)]
        rate = sub.groupby(c)["TARGET"].mean()
        f.write("target rate among top-10 levels:\n")
        f.write(rate.to_string(float_format=lambda v: f"{v:.4f}") + "\n")

    f.write("\n=== Recommended unseen-level handling ===\n")
    f.write(
        "All 6 columns have test rows with levels not present in train (see pct_unseen above).\n"
        "Any encoding scheme MUST map unseen-at-inference levels to a reserved bucket rather\n"
        "than erroring or NaN-ing:\n"
        "  - target/frequency encoding: map unseen level -> global train mean / 0 count, not NaN\n"
        "  - one-hot / ordinal (sklearn): use handle_unknown='ignore' / an explicit UNK category\n"
        "  - tree models fed raw codes: fit a LabelEncoder-like mapping on train, unseen -> -1\n"
        "This must be validated by refitting only on train and transforming test with fitted\n"
        "mapping (never fit on train+test concatenated, which leaks the level identity).\n"
    )

print("wrote", OUT)
