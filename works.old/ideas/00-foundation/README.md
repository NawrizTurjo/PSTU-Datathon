# 00 — Foundation

Shared setup every other idea assumes. Get this right once and reuse it.

**Also read [metric-decomposition.md](metric-decomposition.md)** — it explains what the
scoring formula rewards and is the single most useful document in `ideas/`.

## Data facts (from `dataset_exploration/`)

| | |
|---|---|
| Train | 48,128 rows × 286 features + target |
| Test | 12,032 rows × 286 features |
| Positive rate | 5.00% (2,406 of 48,128) |
| Ghost/sentinel | `-999999` in `base_number_of_dependent_farmers` only (66 train / 23 test rows) |
| Droppable columns | 12 (6 constant, 6 exact duplicates) |
| Train/test shift | none — adversarial AUC 0.4985 |
| Station grouping | none recoverable — use `StratifiedKFold`, not `GroupKFold` |
| Sparsity | 143 of 223 numeric columns are ≥90% zero |
| Label noise | 3.3% of rows in duplicate groups with conflicting labels |

## Input files

Use the **converted** files, not the raw ones. `dataset/train.csv` is 910 MB because 63
boolean columns were stored as full Bengali sentences; `dataset_exploration/02_convert_to_numeric.py`
decodes them to 0/1 and gets it down to 63 MB with zero undecodable values.

```
dataset_exploration/converted_train.csv   # 63 MB, includes `id` + target
dataset_exploration/converted_test.csv    # 16 MB, includes `id`
```

On Kaggle, either upload these as a dataset or run the conversion script once in the
notebook (it streams the raw CSV and takes ~10 seconds).

## Preprocessing pipeline

Applied identically to train and test. All steps are justified by measured EDA findings.

```python
import pandas as pd, numpy as np

TARGET = "Your_Target_Column"

# 6 columns constant in BOTH train and test
CONSTANT_COLS = [
    "has_no_medium_term_fund_balance", "has_no_reimbursement_delta",
    "has_zero_grid_power_balance", "has_zero_medium_term_avg_balance",
    "has_zero_solar_efficiency_balance", "has_zero_water_tank_balance",
]

# keep the first of each identical pair, drop the second
DUPLICATE_COLS = [
    "is_pump_draw_dry",                      # ≡ has_dust_accumulation_on_panels
    "count_battery_failures",                # ≡ count_pump_motor_faults
    "trend_maintenance_claim_count_1y3",     # ≡ trend_maintenance_cost_increase_1y3
    "trend_repair_claim_count_1y3",          # ≡ trend_repair_cost_increase_1y3
    "trend_expense_transaction_count_1y3",   # ≡ trend_outgoing_expense_increase_1y3
    "trend_internal_in_count_1y3",           # ≡ trend_internal_transfer_in_1y3
]

def preprocess(df):
    df = df.drop(columns=CONSTANT_COLS + DUPLICATE_COLS, errors="ignore")
    # the confirmed sentinel — a negative farmer count is physically impossible
    df["base_number_of_dependent_farmers"] = (
        df["base_number_of_dependent_farmers"].replace(-999999, np.nan))
    return df
```

Notes:

- **Do not impute the sentinel** if you're using LightGBM / XGBoost / CatBoost /
  HistGradientBoosting — they all handle `NaN` natively and will learn a split for it.
  Only impute (median) for models that can't, e.g. linear or distance-based.
- **Winsorizing the heavy-tailed sensor columns is optional.** Columns like
  `sensor_average_daily_pump_runtime_hours` and `sensor_wind_speed_kmh` have physically
  implausible round-number tails (90000, 150000, 300000…). Trees are invariant to
  monotone transforms, so this only matters for linear/NN members of an ensemble.
  Don't bother for the GBDT path.
- Dropping the 12 columns is safe but worth roughly nothing on its own — it's for
  cleanliness and speed, not score.

## CV protocol

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **`StratifiedKFold`, not `GroupKFold`.** There is no recoverable station id — the
  apparent `base_*` groups are coincidental collisions on a dominant fill value
  (`base_distance_from_coastal_river_km = 0.509…` in 19.5% of rows). Within-group target
  rates scatter around the global 5% rate rather than clustering. Details in
  `dataset_exploration/pseudo_station_report.txt`.
- **No adversarial reweighting needed.** Train and test are iid (adversarial AUC 0.4985).
  Your CV score should track the leaderboard closely — which also means a large CV/LB gap
  is a signal that *you* introduced leakage, not that the split is unfair.
- **Use the same seed and fold assignment everywhere** so OOF predictions from different
  models can be stacked and compared. Consider repeating with 2–3 seeds and averaging when
  comparing close candidates: with only 2,406 positives, fold noise on the composite is
  roughly ±0.003–0.005, so differences smaller than that are not real.

## Submission format — traps

The spec is strictly enforced. Three columns, exact names, exact order:

```
id,Target_Binary,Target_Probability
0,0,0.0123
1,1,0.8745
```

- **`id` is 0-indexed row order of `test.csv`.** There is no id column in the raw file —
  row $i$ gets id $i$. Never shuffle test rows.
- **Exactly 12,032 rows** plus header.
- **`Target_Probability` must be strictly inside (0.0, 1.0)** and contain no `NaN` or
  `inf`. Some models output exact 0.0 or 1.0 — always clip:
  ```python
  proba = np.clip(proba, 1e-6, 1 - 1e-6)
  ```
- `Target_Binary` must be integer `0`/`1`, not boolean or float.

A validation function is worth writing once:

```python
def validate_submission(sub, n_test=12032):
    assert list(sub.columns) == ["id", "Target_Binary", "Target_Probability"]
    assert len(sub) == n_test
    assert sub["id"].tolist() == list(range(n_test))
    assert sub["Target_Binary"].isin([0, 1]).all()
    assert sub["Target_Probability"].between(0, 1, inclusive="neither").all()
    assert np.isfinite(sub["Target_Probability"]).all()
    return True
```

An "Evaluation Error" scores nothing at all — this check is cheaper than a wasted submission.

## Kaggle resources

The dataset is small. Budget accordingly:

| Task | Hardware | Time |
|---|---|---|
| Load + preprocess | CPU | <10 s |
| LightGBM 5-fold CV | CPU | 1–3 min |
| CatBoost 5-fold CV | CPU | 3–8 min |
| Full ensemble + threshold search | CPU | 15–30 min |
| TabNet / FT-Transformer 5-fold | GPU | 20–40 min |

**Don't use the GPU for tree models.** At 48k × 286, GPU LightGBM is typically slower
than CPU because of launch overhead. Kaggle's CPU notebooks have no weekly quota, unlike
the ~30 GPU hours/week — so keep iterating on CPU and save GPU quota for the one
neural experiment in [../04-ensemble-diversity/](../04-ensemble-diversity/), if you get to it.

Memory is not a concern: peak usage for the full pipeline is well under 4 GB against
Kaggle's ~30 GB.
