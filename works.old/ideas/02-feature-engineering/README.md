# 02 — Feature engineering

The most likely source of a real edge after the threshold and model work, and the place
where the dataset's Santander lineage is genuinely exploitable.

**Validate each block independently.** Add one block, re-run 5-fold OOF AUC, keep it only
if it clears fold noise (±0.003–0.005). Adding everything at once on 2,406 positives is a
reliable way to overfit and never find out which part helped.

## Block A — Santander-style row aggregates (try first)

This dataset's numeric skeleton is the Santander Customer Satisfaction feature set. The
single most reproducible trick on that data was **row-wise aggregation across the sparse
columns** — it captures "how much activity does this record show at all", which no
individual column expresses.

Given 143 of 223 numeric columns are ≥90% zero, this should be informative here too.

```python
num_cols = [c for c in df.columns if c not in bool_cols + [TARGET, "id"]]

df["agg_n_zeros"]    = (df[num_cols] == 0).sum(axis=1)
df["agg_n_nonzero"]  = (df[num_cols] != 0).sum(axis=1)
df["agg_sum"]        = df[num_cols].sum(axis=1)
df["agg_mean"]       = df[num_cols].mean(axis=1)
df["agg_std"]        = df[num_cols].std(axis=1)
df["agg_max"]        = df[num_cols].max(axis=1)
df["agg_skew"]       = df[num_cols].skew(axis=1)
df["agg_n_negative"] = (df[num_cols] < 0).sum(axis=1)
```

**`agg_n_zeros` is the highest-prior feature in this entire document.** On Santander it
was consistently among the strongest single engineered features.

Then repeat per semantic group — the groups are already labelled in
`dataset_exploration/column_groups.csv`:

```python
for grp in ["sensor_reading", "operational_count", "financial",
            "trend_pct", "obfuscated_numeric"]:
    cols = schema.loc[schema.group == grp, "column"]
    cols = [c for c in cols if c in df.columns]
    df[f"{grp}_n_nonzero"] = (df[cols] != 0).sum(axis=1)
    df[f"{grp}_sum"]       = df[cols].sum(axis=1)
    df[f"{grp}_mean"]      = df[cols].mean(axis=1)
    df[f"{grp}_max"]       = df[cols].max(axis=1)
```

Cost: ~20 new features. Cheap, fast, and the block most likely to pay.

## Block B — boolean net-flags

63 boolean columns collapse into far fewer real signals. EDA measured that `has_X` and
`lacks_X` pairs are **never both 1**, but are **frequently both 0** — so they aren't
strict complements, and "neither" is itself a state worth encoding.

```python
PAIRS = [
    ("has_primary_solar_inverter",        "lacks_primary_solar_inverter"),
    ("has_battery_backup_system",         "lacks_battery_backup_system"),
    ("is_salinity_sensor_active",         "is_salinity_sensor_inactive"),
    ("is_submersible_pump_operational",   "is_submersible_pump_non_operational"),
    ("has_solar_panel_cleaning_schedule", "lacks_solar_panel_cleaning_schedule"),
    ("has_remote_monitoring_system",      "lacks_remote_monitoring_system"),
    ("is_groundwater_level_stable",       "is_groundwater_level_fluctuating"),
    ("is_pump_motor_cool",                "is_pump_motor_overheating"),
    ("has_auto_voltage_regulator",        "lacks_auto_voltage_regulator"),
    ("has_flood_submersion_history",      "has_no_flood_submersion_history"),
    ("is_local_technician_available",     "is_local_technician_unavailable"),
    ("has_alternative_water_source",      "lacks_alternative_water_source"),
]
for pos, neg in PAIRS:
    df[f"net_{pos}"] = df[pos] - df[neg]          # +1 / 0 / -1
    df[f"unk_{pos}"] = ((df[pos] == 0) & (df[neg] == 0)).astype(int)
```

The `unk_` indicator — "the log answered neither way" — may be the more interesting half:
missing-ness in maintenance records often correlates with neglected stations.

Also add a global count of risk flags:

```python
RISK_FLAGS = ["is_pump_motor_overheating", "is_pipe_corroded_by_salt",
              "has_dust_accumulation_on_panels", "has_flood_submersion_history",
              "is_local_technician_unavailable", "lacks_battery_backup_system",
              "lacks_auto_voltage_regulator", "has_constant_charge_controller_issue"]
df["risk_flag_count"] = df[[c for c in RISK_FLAGS if c in df.columns]].sum(axis=1)
```

Note: four pairs are perfect complements (correlation exactly −1.0) — for those, the
`net_` feature is fully redundant with either raw column, so drop one of the originals
rather than adding a third copy.

## Block C — domain ratios and stress indices

Physically motivated interactions. Trees can approximate these, but ratios and products
are exactly what axis-aligned splits struggle to represent, so explicit construction can help.

```python
eps = 1e-6

# degradation per unit of age — an old station with many events is different
# from a new station with the same count
age = df["base_station_installation_age_years"] + eps
df["dry_runs_per_year"]     = df["count_dry_run_events"] / age
df["surges_per_year"]       = df["count_voltage_surge_events"] / age
df["repairs_per_year"]      = df["count_major_repairs_total"] / age
df["maint_visits_per_year"] = df["count_maintenance_visits_total"] / age

# maintenance debt: time since service vs how often it's normally serviced
df["cleaning_debt"] = (df["count_months_since_panel_cleaning"] /
                       (df["count_solar_panel_cleanings"] + eps))
df["maint_debt"]    = (df["count_months_since_last_maintenance"] /
                       (df["count_maintenance_visits_total"] + eps))

# financial fragility: is anyone actually funding this station's upkeep?
df["repair_cost_per_farmer"] = (df["cost_total_repair_bdt"] /
                                (df["base_number_of_dependent_farmers"].abs() + eps))
df["grant_to_repair_ratio"]  = (df["cost_govt_grant_bdt"] /
                                (df["cost_total_repair_bdt"] + eps))
df["total_funding"] = (df["cost_govt_grant_bdt"] + df["cost_community_contribution_bdt"] +
                       df["cost_ngo_funding_3m_bdt"])
df["funding_deficit"] = df["cost_total_maintenance_bdt"] - df["total_funding"]

# environmental stress interactions
df["salinity_x_age"]     = df["sensor_water_salinity_ppm"] * age
df["vibration_x_age"]    = df["sensor_motor_vibration_level_mm_s"] * age
df["heat_stress"]        = (df["sensor_panel_surface_temperature_celsius"] *
                            df["sensor_ambient_temperature_celsius"])
```

Caveat: many of these source columns are ≥90% zero, so a lot of these ratios will be
0 or degenerate for most rows. Check `agg_n_nonzero` on the inputs before investing —
a ratio between two mostly-zero columns is mostly noise.

## Block D — recency / trend deltas

The dataset has parallel `_last_month`, `_2_months_ago`, `_3_months_ago`,
`_last_3_months` columns for several sensors. Differences between them encode direction
of travel, which the raw levels don't.

```python
for base in ["water_tank", "temp", "demand", "humidity", "vibration", "irradiance",
             "short_runtime", "long_runtime"]:
    last, prev = f"sensor_avg_{base}_last_month", f"sensor_avg_{base}_2_months_ago"
    if last in df.columns and prev in df.columns:
        df[f"delta_{base}"] = df[last] - df[prev]
        df[f"ratio_{base}"] = df[last] / (df[prev] + eps)
```

A station whose vibration is *rising* is a different risk from one with steady high
vibration — this is the closest thing to a temporal signal available, given there's no
true per-station time series (see [../06-dead-ends/](../06-dead-ends/)).

## Block E — sparsity indicators

For the heavily zero-inflated columns, "did this ever happen" can matter more than
"how much".

```python
SPARSE = zero_ratios.query("zero_ratio >= 0.90")["column"]   # from zero_inflation_ratios.csv
for c in SPARSE:
    df[f"nz_{c}"] = (df[c] != 0).astype(int)
```

This adds ~143 columns — likely too many. Prefer applying it only to the sparse columns
that show non-trivial target correlation, or skip in favour of Block A's aggregate counts,
which capture the same idea far more compactly. **Test Block A first; only try E if A helps
and you want to push further.**

## What to expect

Honest estimate: **+0.005 to +0.015 composite** in total, most of it from Block A.
Blocks C and D are plausible but unproven on this data — they're the ones worth
experimenting with if you have time after the basics.

## Anti-overfitting discipline

- Add one block → measure OOF AUC → keep or discard. Never batch.
- With 2,406 positives, treat any gain under 0.003 as noise. Confirm survivors on a
  second CV seed before believing them.
- **No target encoding on the duplicate-row groups.** 7.35% of rows are exact duplicates;
  target-encoding anything that identifies those groups leaks the label directly. This was
  measured to be a trap — see [../06-dead-ends/](../06-dead-ends/).
- Check feature importance after each block. A newly added feature that dominates
  importance is usually leakage, not insight.
