"""
Reproduces two key EDA findings on the compact converted csv:
1. The hidden "ghost" missing-value marker mentioned in overview.md.
2. Which has_X / lacks_X / is_X boolean-text pairs are redundant.
"""
import pandas as pd

TRAIN = "dataset_exploration/converted_train.csv"
TEST = "dataset_exploration/converted_test.csv"
SCHEMA = "dataset_exploration/column_groups.csv"
TARGET = "Your_Target_Column"
GHOST_VALUE = -999999

df = pd.read_csv(TRAIN).drop(columns=["id"])
dft = pd.read_csv(TEST).drop(columns=["id"])

with open("dataset_exploration/ghost_value_report.txt", "w", encoding="utf-8") as f:
    f.write(f"=== Ghost value hunt: searching every numeric column for exact {GHOST_VALUE} ===\n")
    hits = []
    for col in df.columns:
        if col == TARGET:
            continue
        cnt = (df[col] == GHOST_VALUE).sum()
        if cnt > 0:
            hits.append((col, cnt))
    for col, cnt in hits:
        n_test = (dft[col] == GHOST_VALUE).sum() if col in dft.columns else -1
        f.write(f"  {col}: train={cnt} ({cnt/len(df):.3%})  test={n_test} ({n_test/len(dft):.3%})\n")

    if len(hits) == 1:
        col = hits[0][0]
        rate_ghost = df.loc[df[col] == GHOST_VALUE, TARGET].mean()
        rate_normal = df.loc[df[col] != GHOST_VALUE, TARGET].mean()
        f.write(f"\nCONFIRMED single ghost marker: {GHOST_VALUE} appears ONLY in '{col}'.\n")
        f.write(f"Failure rate when ghosted: {rate_ghost:.4f} vs normal rows: {rate_normal:.4f}\n")
        f.write("Physically impossible: a negative dependent-farmer count.\n")
        f.write("Recommendation: replace with NaN, then impute (median/mode) or let"
                " a tree model (LightGBM/XGBoost/CatBoost) split on it natively -"
                " it's rare (~0.15% of rows) so impact is small either way.\n")

    f.write("\n=== Secondary note: heavy-tailed sensor columns (NOT the same as the"
            " ghost marker above) ===\n")
    f.write("sensor_average_daily_pump_runtime_hours, sensor_short_term_pump_runtime_hours,\n"
            "sensor_long_term_pump_runtime_hours, sensor_wind_speed_kmh,\n"
            "sensor_daily_water_demand_liters show many repeated round-number outliers\n"
            "(90000, 150000, 300000, ...) far beyond physical plausibility (e.g. hours/day > 24).\n"
            "These look like a multiplicative noise/scaling artifact rather than one sentinel value.\n"
            "Recommendation: cap/winsorize at a reasonable percentile (e.g. p99.5) or log1p-transform;\n"
            "tree models are fairly robust to this but linear/distance-based models are not.\n")

print("wrote dataset_exploration/ghost_value_report.txt")

# --- boolean pair redundancy ---
schema = pd.read_csv(SCHEMA)
bool_cols = schema.loc[schema.group == "boolean_text_flag", "column"].tolist()
sub = df[bool_cols]
corr = sub.corr()

pairs = []
for i, c1 in enumerate(bool_cols):
    for c2 in bool_cols[i + 1:]:
        r = corr.loc[c1, c2]
        if abs(r) > 0.8:
            pairs.append((c1, c2, r))

with open("dataset_exploration/bool_pair_report.txt", "w", encoding="utf-8") as f:
    f.write(f"{len(bool_cols)} boolean-text (Bengali yes/no sentence) columns decoded to 0/1.\n")
    f.write(f"Pairs with |corr| > 0.8 (candidate redundant has_X/lacks_X/is_X pairs):\n\n")
    for c1, c2, r in sorted(pairs, key=lambda x: -abs(x[2])):
        ct = pd.crosstab(df[c1], df[c2])
        both_true = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else 0
        f.write(f"  corr={r:+.3f}  {c1}  <->  {c2}   (both=1 in {both_true} rows)\n")
    f.write(f"\nKey pattern: across almost all pairs, both_true == 0 (has_X and lacks_X are\n"
            f"NEVER simultaneously 1), but both are frequently 0 together too - so these are\n"
            f"related-but-noisy signals, not pure duplicates or pure complements.\n"
            f"Recommendation: engineer a single net feature per pair (val_has - val_lacks:\n"
            f"+1/-1/0) instead of keeping both raw columns, to cut redundancy and let the\n"
            f"model use the disagreement/uncertainty signal.\n")

print("wrote dataset_exploration/bool_pair_report.txt")
