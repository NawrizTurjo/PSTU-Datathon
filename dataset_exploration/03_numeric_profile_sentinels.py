"""
Item 3: Numeric profiling. Describe all 344 numeric feature columns, hunt for
sentinel values (-999999 fingerprint in feat_109, and whatever is producing
feat_169's reported ~-1.11e8 min), and measure zero-inflation ratio per
column (share of rows == 0), since CLAUDE.md claims 252/350 numeric cols are
>=90% zero.
"""
import pandas as pd
import numpy as np

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/03_numeric_profile_report.txt"
DESCRIBE_OUT = "dataset_exploration/03_numeric_describe.csv"
ZERO_OUT = "dataset_exploration/03_zero_inflation_ratios.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

target_col = "TARGET"
id_col = "id"
cat_cols = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]
feat_cols = [c for c in train.columns if c not in (target_col,)]
numeric_cols = [c for c in feat_cols if c not in cat_cols]

desc = train[numeric_cols].describe().T
desc.to_csv(DESCRIBE_OUT)

# zero-inflation ratio (train)
zero_ratio = (train[numeric_cols] == 0).mean().sort_values(ascending=False)
zero_ratio.name = "zero_ratio"
zero_ratio.to_frame().to_csv(ZERO_OUT)

n_ge90 = int((zero_ratio >= 0.90).sum())

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Numeric columns: describe() -> 03_numeric_describe.csv ===\n")
    f.write(f"n numeric columns: {len(numeric_cols)}\n\n")

    f.write("=== Sentinel hunt: feat_109 (-999999 fingerprint) ===\n")
    c = "feat_109"
    n_sentinel_train = int((train[c] == -999999).sum())
    n_sentinel_test = int((test[c] == -999999).sum())
    f.write(f"train: min={train[c].min()}, count(==-999999)={n_sentinel_train} "
            f"({n_sentinel_train/len(train):.4%})\n")
    f.write(f"test:  min={test[c].min()}, count(==-999999)={n_sentinel_test} "
            f"({n_sentinel_test/len(test):.4%})\n")
    # check if -999999 appears anywhere else
    other_sentinel_cols = []
    for oc in numeric_cols:
        if oc == c:
            continue
        n = int((train[oc] == -999999).sum())
        if n > 0:
            other_sentinel_cols.append((oc, n))
    f.write(f"other numeric columns containing exact value -999999: {len(other_sentinel_cols)}\n")
    for oc, n in other_sentinel_cols:
        f.write(f"  {oc}: {n} rows\n")
    f.write("\n")

    f.write("=== Sentinel hunt: feat_169 (reported min ~-1.11e8) ===\n")
    c = "feat_169"
    f.write(f"train: min={train[c].min()}, max={train[c].max()}\n")
    f.write(f"test:  min={test[c].min()}, max={test[c].max()}\n")
    vc = train[c].value_counts().head(10)
    f.write("most frequent values (train):\n")
    f.write(vc.to_string() + "\n")
    extreme_mask = train[c] < train[c].quantile(0.001)
    f.write(f"rows below 0.1th percentile: {int(extreme_mask.sum())}\n")
    f.write(f"value at min: {train[c].min()}, 2nd smallest distinct: "
            f"{sorted(train[c].unique())[1] if train[c].nunique() > 1 else 'n/a'}\n\n")

    f.write("=== General extreme-value scan: columns with |min| or |max| > 1e6 ===\n")
    extreme_cols = []
    for oc in numeric_cols:
        mn, mx = train[oc].min(), train[oc].max()
        if abs(mn) > 1e6 or abs(mx) > 1e6:
            extreme_cols.append((oc, mn, mx))
    f.write(f"{'column':12s}{'min':>18s}{'max':>18s}\n")
    for oc, mn, mx in sorted(extreme_cols, key=lambda r: r[0]):
        f.write(f"{oc:12s}{mn:18.2f}{mx:18.2f}\n")
    f.write(f"total: {len(extreme_cols)} columns\n\n")

    f.write("=== NEW FINDING (not in CLAUDE.md): 9999999999 sentinel across 23 columns ===\n")
    f.write("Many columns cap at exactly 9999999999 (1e10-1). This is the classic Santander\n"
            "'delta_imp_*' sentinel fingerprint. Same treat-as-missing logic as -999999 in\n"
            "feat_109 should apply here.\n")
    sentinel2_cols = []
    for oc in numeric_cols:
        n_tr = int((train[oc] == 9999999999).sum())
        n_te = int((test[oc] == 9999999999).sum())
        if n_tr > 0 or n_te > 0:
            sentinel2_cols.append((oc, n_tr, n_te))
    f.write(f"columns affected: {len(sentinel2_cols)}\n")
    f.write(f"{'column':12s}{'n_train':>10s}{'n_test':>10s}\n")
    for oc, n_tr, n_te in sentinel2_cols:
        f.write(f"{oc:12s}{n_tr:10d}{n_te:10d}\n")
    f.write("\n")

    f.write("=== Zero-inflation ===\n")
    f.write(f"numeric columns with zero_ratio >= 0.90: {n_ge90} / {len(numeric_cols)}\n")
    f.write(f"full ranked list -> 03_zero_inflation_ratios.csv\n")
    f.write("Top 15 most zero-inflated:\n")
    f.write(zero_ratio.head(15).to_string(float_format=lambda v: f"{v:.4f}") + "\n\n")
    f.write("Bottom 15 (least zero-inflated):\n")
    f.write(zero_ratio.tail(15).to_string(float_format=lambda v: f"{v:.4f}") + "\n")

print("wrote", OUT)
print(f"zero_ratio>=0.90 columns: {n_ge90}/{len(numeric_cols)}")
