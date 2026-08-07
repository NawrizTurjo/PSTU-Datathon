"""
Item 4: Scan every numeric column for common Santander-style secondary
sentinel candidates (-1, 99, 999, 9999, 99999, 999999, 9999999999, and their
negatives), and profile the exact zero-inflation ratio per column.
"""
import pandas as pd

TRAIN = "dataset_exploration/converted_train.csv"
SCHEMA = "dataset_exploration/column_groups.csv"
TARGET = "Your_Target_Column"

train = pd.read_csv(TRAIN).drop(columns=["id"])
schema = pd.read_csv(SCHEMA)
bool_cols = set(schema.loc[schema.group == "boolean_text_flag", "column"])
numeric_cols = [c for c in train.columns if c != TARGET and c not in bool_cols]

CANDIDATES = [-1, 99, 999, 9999, 99999, 999999, 9999999999,
              -99, -999, -9999, -99999, -999999]

with open("dataset_exploration/secondary_sentinels_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Secondary sentinel scan (exact match on common missing-value codes) ===\n")
    f.write(f"Candidates checked: {CANDIDATES}\n\n")
    any_hit = False
    for val in CANDIDATES:
        hit_cols = []
        for c in numeric_cols:
            cnt = (train[c] == val).sum()
            if cnt > 0:
                hit_cols.append((c, cnt))
        if hit_cols:
            any_hit = True
            f.write(f"value {val}:\n")
            for c, cnt in sorted(hit_cols, key=lambda x: -x[1]):
                f.write(f"    {c}: {cnt} rows ({cnt/len(train):.3%})\n")
    if not any_hit:
        f.write("No hits for any candidate other than the already-confirmed -999999 in\n"
                "base_number_of_dependent_farmers (see script 04). No secondary sentinel\n"
                "value found by this exact-match scan.\n")

    f.write("\n=== Zero-inflation ratio per numeric column ===\n")
    zero_ratios = []
    for c in numeric_cols:
        r = (train[c] == 0).sum() / len(train)
        zero_ratios.append((c, r))
    zero_ratios.sort(key=lambda x: -x[1])
    zr = pd.DataFrame(zero_ratios, columns=["column", "zero_ratio"])
    zr.to_csv("dataset_exploration/zero_inflation_ratios.csv", index=False)

    n_90plus = (zr.zero_ratio >= 0.90).sum()
    n_99plus = (zr.zero_ratio >= 0.99).sum()
    f.write(f"Columns with >=90% zeros: {n_90plus} / {len(numeric_cols)}\n")
    f.write(f"Columns with >=99% zeros: {n_99plus} / {len(numeric_cols)}\n")
    f.write(f"Full ranked list saved to zero_inflation_ratios.csv\n\n")
    f.write("20 sparsest columns:\n")
    f.write(zr.head(20).to_string(index=False) + "\n\n")
    f.write("20 least sparse (most populated) numeric columns:\n")
    f.write(zr.tail(20).to_string(index=False) + "\n")

    f.write("\n=== Recommendation ===\n")
    f.write("Most operational_count/financial/sensor columns are extremely zero-inflated\n"
            "(this mirrors the original Santander data's sparsity). For tree models this is\n"
            "largely free (native sparse-aware splitting). For linear/distance-based models,\n"
            "consider: (a) a binary 'is_nonzero' indicator per sparse column in addition to\n"
            "the raw value, (b) log1p-transform on the nonzero-heavy-tailed ones flagged in\n"
            "script 04, (c) dropping near-constant (>99.5% zero) columns if they add no signal\n"
            "after checking correlation with target.\n")

print("wrote dataset_exploration/secondary_sentinels_report.txt")
print("wrote dataset_exploration/zero_inflation_ratios.csv")
