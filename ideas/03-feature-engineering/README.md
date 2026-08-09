# 03 — Feature engineering

~4–8 hours. The highest-variance idea in the folder: it is where the remaining headroom
plausibly lives, and it is also the one most likely to return exactly zero.

## What it is

Three concrete feature families, each motivated by a specific measured property of this data:

1. **Row-wise aggregates** over the 252 near-all-zero numeric columns.
2. **Encodings for the three high-cardinality categoricals** (2,333 / 1,710 / 627 levels).
3. **Sentinel and missingness indicators** for the 24 sentinel-bearing columns.

## Why it should work (measured evidence)

### Row-wise aggregates — the sparsity is the signal

**252 of 344 numeric columns are ≥90% zero** (measured, `03_zero_inflation_ratios.csv`). In a
sparse, zero-inflated tabular problem, *which* columns are nonzero for a row, and *how many*,
is frequently more informative than any individual column's value — and a tree model cannot
construct "count of nonzero across 252 columns" on its own, because that requires 252
simultaneous splits. You have to hand it the feature.

Corroborating: the best single-feature AUC is only **0.6986** (`feat_175`) — no individual
column carries the signal, so it is distributed across many, which is exactly the situation
aggregates are designed for.

This dataset's structural fingerprint matches the public Santander Customer Satisfaction data
(row count, exact positive count, `-999999` sentinel), where row-aggregate features are a
well-documented, effective pattern. **This is a structural analogy only — no external data may
be used, and none is needed here.**

### Categorical encoding — cardinality is measured and awkward

| Column | Levels | Rows per level (avg) | Positives per level (avg) |
|---|---|---|---|
| `feat_142` | 2,333 | ~33 | **~1.3** |
| `feat_325` | 1,710 | ~44 | **~1.8** |
| `feat_157` | 627 | ~121 | **~4.8** |

At ~1.3 positives per level, naive target encoding on `feat_142` is nearly pure noise and will
leak badly without out-of-fold computation and heavy smoothing. Measured target rates for the
top-10 levels of `feat_142` range 0.0140–0.0578 around a 0.0396 base — a spread entirely
consistent with sampling noise at those counts. **Frequency/count encoding is the safer first
move; target encoding is the higher-risk, higher-reward follow-up.**

`feat_320` (119), `feat_337` (39) and `feat_318` (12) are low enough cardinality to one-hot or
leave as native categoricals — measured target rates there are similarly flat, so expect little.

### Sentinel indicators — cheap and measured

`feat_109 == -999999` affects 116 train / 89 test rows; `9999999999` spans **23 columns**. Once
[00](../00-foundation/) converts these to `NaN`, the *fact that a value was sentinel* is
destroyed unless you capture it. It is a one-line feature per column.

## Concrete steps

### 1. Row-wise aggregates (do this first — cheapest, best-motivated)

Computed per row, over the numeric columns only, **after** the 44-column drop and sentinel→NaN
conversion:

```python
num = X[numeric_cols]
X["agg_n_zero"]      = (num == 0).sum(axis=1)
X["agg_n_nonzero"]   = (num != 0).sum(axis=1)
X["agg_n_nan"]       = num.isna().sum(axis=1)          # = sentinel count for that row
X["agg_sum"]         = num.sum(axis=1)
X["agg_mean_nonzero"]= num.replace(0, np.nan).mean(axis=1)
X["agg_std"]         = num.std(axis=1)
X["agg_max"]         = num.max(axis=1)
X["agg_n_negative"]  = (num < 0).sum(axis=1)
```

**Every one of these is per-row with no fitted state** — which makes them completely safe for
the [06](../06-inference-notebook/) hidden-test run. That property is worth as much as the
score.

Refinement if the plain version helps: compute the same aggregates *within column blocks*
grouped by zero-inflation ratio (e.g. the ≥99% zero group separately from the 90–99% group),
since mixing a mostly-dense column into a sum dominated by sparse ones dilutes the signal.

### 2. Frequency / count encoding for categoricals

```python
freq_map = train[c].value_counts(normalize=True).to_dict()   # FIT ON TRAIN ONLY
X[f"{c}_freq"] = X[c].map(freq_map).fillna(0.0)              # unseen -> 0.0, never NaN
```

Unseen levels legitimately have frequency 0 in train — a meaningful, honest value, and it
doubles as an implicit "this is a rare/new category" flag. Save `freq_map` to the artifacts.

### 3. Out-of-fold target encoding (higher risk)

Only with **out-of-fold computation and smoothing**, or it leaks and your OOF score becomes a
fiction:

```python
prior = y_train.mean()                       # 0.039569
enc = (level_sum + prior * m) / (level_count + m)      # m ≈ 50–200, tune it
```

Compute the encoding using only the training folds within each CV split, apply to the held-out
fold. For the test/hidden set, use a map fitted on all training data. **Use a large smoothing
constant** — at ~1.3 positives per level for `feat_142`, `m=100` means the encoding stays near
the prior for all but the most common levels, which is the correct behaviour.

Honestly: expect this to help on `feat_157` (627 levels, ~4.8 positives each), be marginal on
`feat_325`, and be noise on `feat_142`. Consider applying it only to `feat_157`/`feat_320`.
CatBoost's ordered target statistics ([02](../02-gbdt-core/)) do this correctly by construction
and may be the better route to the same signal.

### 4. Sentinel indicator flags

```python
X["sent_neg999999_feat_109"] = (raw["feat_109"] == -999999).astype(int)
X["sent_big_count"] = sum((raw[c] == 9999999999) for c in SENTINEL_BIG_COLS)
```

Two features, five minutes. `agg_n_nan` from step 1 partly subsumes these — check whether they
add anything beyond it before keeping all of them.

### 5. Measure each family separately

Four runs on the pinned folds: baseline, +aggregates, +encodings, +indicators, then the
combination. Against the measured **±0.0060** fold noise, a family that moves OOF AUC by
+0.002 has done nothing — drop it rather than carrying it into the ensemble.

## Kaggle cost

Cheap. Row-aggregates over 76,020 × ~300 are seconds in vectorized pandas/numpy. Out-of-fold
target encoding adds a pass per fold — still minutes. CPU only. The real cost is the retraining
needed to evaluate each family (~2–3 min per 5-fold LightGBM run), so budget by number of
experiments, not by feature-computation time.

## Honest expected gain

| Family | Expected | Confidence |
|---|---|---|
| Row-wise aggregates | **+0.002 to +0.006 OOF AUC** | medium |
| Frequency encoding | +0.000 to +0.003 AUC | medium |
| OOF target encoding | −0.003 to +0.004 AUC (**can hurt via leakage/noise**) | **low** |
| Sentinel indicators | +0.000 to +0.002 AUC | low |

Combined realistic: **+0.003 to +0.010 OOF AUC**, i.e. binary-F1 ~+0.005 to ~+0.020.
**This family could plausibly deliver nothing at all** — the measured evidence says the signal
is diffuse (best single feature 0.6986) and a GBDT already recovers most diffuse signal on its
own. The aggregates are the part most likely to pay; everything else is speculative.

## When to abandon

- **Row-aggregates:** if +0.002 AUC or less after one run, keep only `agg_n_nonzero` and
  `agg_n_nan` (nearly free) and stop.
- **Target encoding:** abandon the moment OOF AUC jumps implausibly (>+0.02) — that is leakage,
  not signal, and it will reverse on the leaderboard. Also abandon if it fails to beat plain
  frequency encoding, which is simpler and safer for
  [06](../06-inference-notebook/).
- **Whole idea:** if the combination of all families is under +0.003 AUC after ~3 hours, stop
  and put the remaining time into [05](../05-ensemble-diversity/) and the presentation. A
  feature that does not clear fold noise is not a feature.
- **Any feature requiring fitted state that cannot be pickled and reloaded** — abandon
  immediately regardless of gain. It endangers 40% of the grade to win 0.005 F1.
