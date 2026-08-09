"""
Load compact converted_train.csv (small now, decoded bools included).
Profile every column, hunt the "ghost" missing-value marker mentioned
in overview.md (an anomalous number outside physical sensor range),
check target imbalance, and rank features by correlation with target.
"""
import pandas as pd
import numpy as np

TRAIN = "dataset_exploration/converted_train.csv"
TARGET = "Your_Target_Column"

pd.set_option("display.width", 140)

df = pd.read_csv(TRAIN)
df = df.drop(columns=["id"])

with open("dataset_exploration/profile_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Shape: {df.shape}\n\n")

    # --- target imbalance ---
    vc = df[TARGET].value_counts()
    f.write("=== Target balance ===\n")
    f.write(str(vc) + "\n")
    f.write(f"Positive rate: {vc.get(1,0)/len(df):.4%}\n\n")

    # --- describe numeric ---
    desc = df.describe().T
    desc.to_csv("dataset_exploration/numeric_describe.csv")
    f.write("Full describe() saved to numeric_describe.csv\n\n")

    # --- ghost value hunt ---
    # Heuristic: for each column, look at max value vs 99.5th pct and min vs 0.5th pct.
    # A ghost marker shows up as a single far-outlier value shared by many rows,
    # way outside the rest of the column's realistic spread.
    f.write("=== Ghost-value hunt (candidates: value far outside typical range,"
            " appearing on many rows) ===\n")
    candidates = []
    for col in df.columns:
        if col == TARGET:
            continue
        s = df[col]
        if s.nunique() <= 2:  # skip decoded bools / near-constant
            continue
        p1, p50, p99 = s.quantile([0.01, 0.5, 0.99])
        vmax, vmin = s.max(), s.min()
        spread = max(p99 - p1, 1e-9)
        # how many "extreme" (top magnitude) rows share the exact same value
        top_val_counts = s.value_counts().head(3)
        for val, cnt in top_val_counts.items():
            if cnt < 5:
                continue
            dist_from_median = abs(val - p50)
            if dist_from_median > 20 * spread and abs(val) > 50:
                candidates.append((col, val, cnt, p50, p1, p99, vmin, vmax))

    if candidates:
        f.write(f"{'column':45s} {'ghost_val':>15s} {'count':>7s} {'median':>10s} "
                f"{'p1':>10s} {'p99':>10s} {'min':>10s} {'max':>10s}\n")
        for col, val, cnt, p50, p1, p99, vmin, vmax in sorted(candidates, key=lambda x: -x[2]):
            f.write(f"{col:45s} {val:15.4f} {cnt:7d} {p50:10.4f} {p1:10.4f} {p99:10.4f} "
                    f"{vmin:10.4f} {vmax:10.4f}\n")
    else:
        f.write("No obvious single-value outlier candidates found by this heuristic.\n")

    # cross-column: does the SAME numeric value recur as an outlier across many
    # different columns? (that'd be strong evidence of one shared ghost marker)
    from collections import Counter
    val_counter = Counter(round(v, 2) for (_, v, *_ ) in candidates)
    f.write("\nMost common candidate ghost values across columns (value: n_columns):\n")
    for val, n in val_counter.most_common(10):
        f.write(f"  {val}: {n} columns\n")

    # --- correlation with target ---
    f.write("\n=== Top |correlation| with target (Pearson, decoded bools included) ===\n")
    corrs = df.corr(numeric_only=True)[TARGET].drop(TARGET).dropna()
    top = corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(30)
    f.write(top.to_string() + "\n")
    top.to_csv("dataset_exploration/top_target_correlations.csv")

print("done -> dataset_exploration/profile_report.txt")
