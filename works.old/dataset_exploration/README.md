# Dataset Exploration — findings & plan

> ## ⛔ FINDINGS INVALIDATED — DATASET WITHDRAWN (2026-08-08)
>
> The organizers announced the released dataset **contained leaks** and will re-upload it.
> **Everything below describes the withdrawn data.**
>
> The 10 scripts are generic — they *detect* schema, sentinels, constants and duplicates
> rather than hardcoding them — so re-run them as-is on the new data and read the
> regenerated reports fresh. Do not carry any figure below forward.
>
> See [`../CLAUDE.md`](../CLAUDE.md) for what survives.

Original `dataset/train.csv` is **910 MB** for only 48,128 rows (`dataset/test.csv` is
227 MB / 12,032 rows). Not row count that's huge — it's column bloat: ~63 of the
286 feature columns are **boolean flags encoded as full Bengali sentences**
(e.g. `"না, এই স্টেশনটির... রেকর্ড নেই..."`) instead of `0`/`1`. Everything else
is numeric.

Run scripts in order (each is idempotent, re-run anytime):

1. `01_explore_schema.py` → `column_groups.csv`, `schema_summary.txt`
   Classifies all 287 columns into groups: `base_attribute`, `financial`,
   `sensor_reading`, `operational_count`, `trend_pct`, `obfuscated_numeric`
   (Santander-style `num_var*`/`num_op_var*` names — this dataset's numeric
   skeleton is a renamed/rescaled Santander Customer Satisfaction feature set),
   `boolean_text_flag` (63 cols), `target`.

2. `02_convert_to_numeric.py` → `converted_train.csv` (63 MB), `converted_test.csv` (16 MB)
   Streams the raw CSVs once, decodes every boolean-text sentence to `1` (starts
   with `হ্যাঁ` = yes) / `0` (starts with `না` = no), discards the raw text, adds
   an `id` column. **910 MB → 63 MB.** Zero unexpected/undecodable values in
   either file. Use these compact files for all further work — pandas can load
   them fully in RAM (this machine only has ~8 GB total).

3. `03_profile_and_ghost_hunt.py` → `profile_report.txt`, `numeric_describe.csv`,
   `top_target_correlations.csv`
   Full `describe()`, target balance, first-pass outlier scan, target correlations.

4. `04_ghost_value_and_bool_pairs.py` → `ghost_value_report.txt`, `bool_pair_report.txt`
   Confirms the hidden ghost value and checks boolean-pair redundancy (see below).

5. `05_zero_variance_and_duplicate_cols.py` → `zero_variance_and_duplicates_report.txt`
   Constant and exact-duplicate columns (`missing-exploration.md` item 1).

6. `06_adversarial_validation.py` → `adversarial_validation_report.txt`
   Train-vs-test classifier + out-of-range/rate-shift scan (item 2).

7. `07_pseudo_station_clustering.py` → `pseudo_station_report.txt`
   Checks whether `base_*` attributes recover a real station id (item 3).

8. `08_secondary_sentinels_and_sparsity.py` → `secondary_sentinels_report.txt`,
   `zero_inflation_ratios.csv`
   Scans for secondary sentinel codes and profiles per-column sparsity (item 4).

9. `09_duplicate_rows.py` → `duplicate_rows_report.txt`
   Exact duplicate/conflicting-label rows in train, train-test row overlap (item 5).

10. `10_metric_threshold_simulation.py` → `threshold_simulation_report.txt`,
    `threshold_sweep.csv`
    Baseline model + composite-metric threshold sweep (item 6).

## Key findings

### 1. Target is heavily imbalanced
`Your_Target_Column`: **45,722 normal (0) vs 2,406 failure (1) — 5.0% positive rate.**
Matches the overview's warning: plain accuracy is useless here. Use
`class_weight`/`scale_pos_weight`, stratified CV splits, and tune the decision
threshold for F1 rather than defaulting to 0.5 — the leaderboard formula weighs
F1 (0.30) and ROC-AUC (0.25) highest.

### 2. The hidden "ghost" missing-value marker: **`-999999`**
Found in exactly one column: **`base_number_of_dependent_farmers`**
(train: 66 rows / 0.14%, test: 23 rows / 0.19%). A negative farmer count is
physically impossible — exactly the "highly anomalous value outside physical
range" the overview describes. No other column contains this literal value.
Failure rate on ghosted rows (3.0%) vs normal rows (5.0%) — not hugely
different, so simple imputation (median/mode) or leaving it for a tree model
to split on natively should be fine; this is a low-impact column (rare, single
feature).

**Separately** (not the same phenomenon — don't conflate the two): several
`sensor_*` columns (`sensor_average_daily_pump_runtime_hours`,
`sensor_short_term_pump_runtime_hours`, `sensor_long_term_pump_runtime_hours`,
`sensor_wind_speed_kmh`, `sensor_daily_water_demand_liters`) have long right
tails with many repeated round-number values (90000, 150000, 300000, ...) that
are physically implausible (e.g. "daily" runtime hours > 24). This looks like
multiplicative scaling noise from how the columns were synthesized, not one
sentinel value. Recommend winsorizing at ~p99.5 or `log1p`-transforming these
before feeding to non-tree models; tree models handle it fine as-is.

### 3. Boolean-text columns are noisy-related pairs, not clean duplicates
Pairs like `has_X` / `lacks_X` or `is_X_active` / `is_X_inactive` are **never
both `1`** on the same row, but are **frequently both `0`** — i.e. related but
not strict complements or duplicates (see `bool_pair_report.txt` for the full
list, e.g. `has_battery_backup_system` vs `lacks_battery_backup_system`,
corr = -0.93). A few pairs *are* perfect complements (corr = -1.0):
`has_solar_charge_controller`/`lacks_solar_charge_controller`,
`is_community_trained_for_maintenance`/`is_community_untrained_for_maintenance`,
`is_pipe_corroded_by_salt`/`is_pipe_free_from_salt_corrosion`,
`has_alternative_water_source`/`lacks_alternative_water_source`.
Suggestion: engineer a net feature per pair (`has - lacks` ∈ {-1, 0, 1}) to cut
redundancy and preserve the "neither answered" signal, rather than keeping
both raw 0/1 columns for all 63 flags.

### 4. Strongest single-feature correlations with target (see `top_target_correlations.csv`)
Top few (all still weak individually, |r| < 0.17 — this will need real feature
interactions, not one strong predictor):
`count_months_since_tank_cleaning` (-0.16), `is_pump_motor_overheating` (-0.16),
`count_water_level_readings` (-0.15), `has_primary_solar_inverter` (-0.15),
`count_water_tanks_connected` (-0.15), `count_solar_panel_cleanings` (-0.15),
`base_station_installation_age_years` (+0.12).

### 5. Zero-variance & duplicate columns (12 droppable)
6 columns are constant (same value on every row) in **both** train and test —
all `has_zero_*_balance`/`has_no_*_balance` flags, e.g. `has_zero_grid_power_balance`.
6 more are **exact duplicates** of another column (drop one from each pair):
`has_dust_accumulation_on_panels`≡`is_pump_draw_dry`,
`count_pump_motor_faults`≡`count_battery_failures`,
`trend_maintenance_cost_increase_1y3`≡`trend_maintenance_claim_count_1y3`,
`trend_repair_cost_increase_1y3`≡`trend_repair_claim_count_1y3`,
`trend_outgoing_expense_increase_1y3`≡`trend_expense_transaction_count_1y3`,
`trend_internal_transfer_in_1y3`≡`trend_internal_in_count_1y3`.
→ 286 features can be safely trimmed to 274 with zero information loss.
(On the Santander-naming question: beyond the confirmed `var3`→`-999999` sentinel,
there's no reliable public field-by-field mapping worth chasing — the competition's
own `sensor_*`/`cost_*`/`count_*` names are the real semantics to use.)

### 6. Adversarial validation: train and test are effectively iid
5-fold train-vs-test classifier ROC-AUC = **0.4985** (chance level). No meaningful
covariate shift. A handful of numeric columns have test values slightly outside
train's observed min/max (33/223 columns, usually just 1-2 rows spilling past the
boundary) and boolean flag rates differ by <0.5pp — none of this rises to a real
distribution shift. **Plain `StratifiedKFold` CV should track the leaderboard
reliably**; no adversarial reweighting needed.

### 7. No recoverable station id — don't use GroupKFold
Grouping by the 5 `base_*` attributes looks tempting (19% of train rows collide
into a group of size >1, biggest group has 1265 rows) but this is a **false
signal**: `base_distance_from_coastal_river_km` has one dominant fill value
(`0.509...`, 19.5% of all rows) and `base_solar_panel_tilt_angle_degrees` only
has 5 distinct values, so unrelated rows collide by coincidence. Within-group
target rates scatter close to the global 5% rate with high variance (std 0.113)
instead of clustering per "station" — confirming these aren't real repeated
station logs. Row order also shows no chronological/sequential structure per
group. **Use plain `StratifiedKFold`, not `GroupKFold`.**

### 8. No secondary sentinel beyond `-999999`; heavy zero-inflation
Scanned every numeric column for common missing-value codes (`-1, 99, 999,
9999, 99999, 999999, 9999999999` and negatives). Only `99` shows up, and only
in obfuscated `num_var*/num_op_var*` columns at <0.15% of rows each — these are
plausible real small counts (columns range into the hundreds), not confident
sentinel evidence, unlike the unambiguous `-999999` case. Separately: **143/223
numeric columns (64%) are ≥90% zero, 87/223 (39%) are ≥99% zero** — full ranked
list in `zero_inflation_ratios.csv`. This mirrors the original Santander data's
sparsity and is expected, not a data-quality problem.

### 9. Real label noise: duplicate rows with conflicting targets
**7.35% of train rows (3,539)** are exact feature-duplicates of another row
(626 groups; the biggest group is 430 rows, mostly the all-zero "nothing
happened" default profile). Of those groups, **105 have conflicting target
labels** — same features, different outcome — affecting **3.3% of train rows**.
This is real label noise (or reflects unmodeled information the features don't
capture). Note precisely what it does and doesn't limit: it **caps precision/F1**,
but **not AUC** — predicting every row as its own duplicate-group mean gives an
oracle AUC of 0.9993, so ranking headroom remains. Separately, **~7% of test rows
(841)** have an exact feature-match in train.

> ⚠️ **Correction:** an earlier version of this file suggested exploiting that
> train-test overlap as a direct lookup/blend. That was written before it was
> measured, and **measurement refutes it** — on the rows the lookup covers, the
> model scores AUC 0.8295 vs the lookup's 0.7210, and every blend weight lowers the
> composite (full override costs −0.026). See `ideas/06-dead-ends/` for the numbers.
> The matched rows collide because they're the sparse "nothing happened" default
> profile, not because they're the same station.

### 10. Composite-score threshold tuning is worth real points
Baseline RandomForest (5-fold OOF, no cleanup applied yet) gets **ROC-AUC
0.811**. Sweeping the decision threshold against the actual competition formula
(`0.30·F1 + 0.25·AUC + 0.15·Precision + 0.15·Recall + 0.10·BalAcc + 0.05·Spec`):
best threshold ≈ **0.60** → composite **0.5167**, vs naive 0.5 → composite
**0.4989** — **+0.018 just from threshold choice**, on an unpolished baseline.
Full curve in `threshold_sweep.csv`; reuse `composite_score()` from
`10_metric_threshold_simulation.py` during model selection instead of relying
on standard `predict()` argmax.

## Suggested modeling plan
- Load `converted_train.csv` / `converted_test.csv` directly (small, fast) instead
  of the raw text CSVs.
- Drop the 12 zero-variance/duplicate columns identified in finding 5.
- Replace `-999999` in `base_number_of_dependent_farmers` with NaN; let
  LightGBM/XGBoost/CatBoost handle NaN natively, or median-impute for other
  model types.
- Optionally winsorize the handful of extreme-tailed `sensor_*` columns noted above.
- Gradient-boosted trees (LightGBM/XGBoost/CatBoost) are a strong default given:
  mixed numeric + binary features, no need for scaling, native NaN handling,
  robust to the remaining outliers.
- Handle imbalance via `scale_pos_weight` (≈ 45722/2406 ≈ 19) or focal loss;
  use plain `StratifiedKFold` for CV — adversarial validation (finding 6) confirms
  train/test are iid, and there's no real station id to `GroupKFold` on (finding 7).
- Be aware ~3.3% of train rows carry conflicting labels on identical features
  (finding 9) — this caps achievable precision/F1 (but not AUC), so don't over-tune
  to it. Do **not** try to exploit the train-test row overlap as a lookup; it was
  measured and it hurts (see `ideas/06-dead-ends/`).
- Tune the classification threshold on out-of-fold predictions against the
  *exact* weighted composite formula (see `10_metric_threshold_simulation.py`'s
  `composite_score()`), not a plain F1 or accuracy proxy — naive 0.5 leaves
  measurable points on the table (finding 10).
- Submission needs `id` (0-indexed row order, matches `test.csv` row order —
  there is no id column in the raw files), `Target_Binary`, `Target_Probability`.
