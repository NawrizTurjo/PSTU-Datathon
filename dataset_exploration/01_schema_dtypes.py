"""
Item 1: Schema / dtype classification. Load train + test fully (small enough,
~235MB in RAM), split columns into numeric (int/float) vs categorical
(object/string), record shape, missing-value counts, and target balance.
Writes a plain-text report. No mutation of source files.
"""
import pandas as pd

TRAIN = "pstu-data-thon-2026-vol-1/train.csv"
TEST = "pstu-data-thon-2026-vol-1/test.csv"
OUT = "dataset_exploration/01_schema_dtypes_report.txt"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

target_col = "TARGET"
id_col = "id"

assert target_col in train.columns, "TARGET missing from train"
assert target_col not in test.columns, "TARGET unexpectedly present in test"
assert id_col in test.columns, "id missing from test"
assert id_col not in train.columns, "id unexpectedly present in train"
assert list(test.columns)[-1] == id_col, "id is not the last column of test"

feat_cols = [c for c in train.columns if c != target_col]
test_feat_cols = [c for c in test.columns if c != id_col]
assert feat_cols == test_feat_cols, "train/test feature columns differ or are out of order"

dtypes = train[feat_cols].dtypes
numeric_cols = [c for c in feat_cols if dtypes[c] != object]
cat_cols = [c for c in feat_cols if dtypes[c] == object]
int_cols = [c for c in feat_cols if dtypes[c] == "int64"]
float_cols = [c for c in feat_cols if dtypes[c] == "float64"]

train_missing = train.isna().sum().sum()
test_missing = test.isna().sum().sum()

target_counts = train[target_col].value_counts().sort_index()
pos_rate = train[target_col].mean()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("=== Shape ===\n")
    f.write(f"train.csv: {train.shape[0]} rows x {train.shape[1]} cols\n")
    f.write(f"test.csv:  {test.shape[0]} rows x {test.shape[1]} cols\n")
    f.write(f"feature columns (excl target/id): {len(feat_cols)}\n\n")

    f.write("=== Column order check ===\n")
    f.write("train/test feature columns identical and same order: PASS\n")
    f.write(f"id is last column of test.csv: PASS\n\n")

    f.write("=== Dtypes (train, feature cols only) ===\n")
    f.write(f"int64:   {len(int_cols)}\n")
    f.write(f"float64: {len(float_cols)}\n")
    f.write(f"object:  {len(cat_cols)}\n\n")

    f.write("=== Object (categorical string) columns ===\n")
    for c in cat_cols:
        f.write(f"  {c}\n")
    f.write("\n")

    f.write("=== Missing values ===\n")
    f.write(f"train total NaN cells: {train_missing}\n")
    f.write(f"test total NaN cells:  {test_missing}\n\n")

    f.write("=== Target balance ===\n")
    f.write(f"{target_counts.to_string()}\n")
    f.write(f"positive rate: {pos_rate:.6f} ({int(train[target_col].sum())} / {len(train)})\n\n")

    f.write("=== Sample dtypes per column (first 15 feature cols) ===\n")
    f.write(dtypes.head(15).to_string() + "\n")

print("wrote", OUT)
print(f"numeric={len(numeric_cols)} categorical={len(cat_cols)} pos_rate={pos_rate:.6f}")
