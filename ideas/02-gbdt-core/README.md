# 02 — GBDT core

~3–5 hours. Start after [00](../00-foundation/), [01](../01-threshold-engine/) and the
[06](../06-inference-notebook/) skeleton are done and a valid submission exists.

## What it is

Replace the sklearn `HistGradientBoostingClassifier` stand-in with a real gradient-boosting
library (LightGBM primary, XGBoost and CatBoost as alternates), tune it properly for a sparse
high-imbalance tabular problem, and settle the class-imbalance handling question with a
measurement rather than a default.

## Why it should work (measured evidence)

- The baseline is **deliberately untuned**: OOF AUC **0.8868** (per-fold ±0.0060) from
  `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
  l2_regularization=1.0)` with no search at all. There is ordinary tuning headroom above it.
- HistGBM is a close cousin of LightGBM but consistently a little weaker on sparse,
  high-cardinality tabular data, and it lacks native categorical handling. **252 of 344 numeric
  columns are ≥90% zero** and three categorical columns have 627–2,333 levels — precisely the
  regime where LightGBM's exclusive-feature-bundling and native categorical splits earn their
  keep, and where CatBoost's ordered target statistics are designed to help.
- The capacity test ([`dead-ends`](../dead-ends/)) measured train AUC 1.0 vs held-out 0.5916 on
  an unregularized tree — this problem **overfits readily**, so the tuning that matters is
  regularization (leaf count, `min_child_samples`, subsampling, L1/L2), not capacity.

## Concrete steps

### 1. Straight swap first, measure it

LightGBM with sensible sparse-data defaults on the identical pinned folds from
[00](../00-foundation/):

```python
params = dict(
    objective="binary", metric="auc",
    learning_rate=0.03, num_leaves=31, min_child_samples=100,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.1, lambda_l2=1.0,
    n_estimators=3000, seed=SEED, deterministic=True, n_jobs=-1,
)
```

with `early_stopping(200)` on each fold's validation split. Record OOF AUC + both F1s. **This
single number tells you whether the rest of this idea is worth doing.**

### 2. Feed the categoricals natively

Pass the 6 categorical columns as `categorical_feature=CAT_COLS` with pandas `category` dtype
rather than as ordinal integers. Ordinal codes impose a meaningless order on `PRD_*`/`SEG_*`
labels; LightGBM's categorical splits do not. Keep the unseen-level fallback from
[00](../00-foundation/) — a category absent from the fitted dtype must land in a defined bucket,
not `NaN`-by-accident.

Measure this as a separate A/B against step 1. On high-cardinality columns native handling can
also *overfit* — `feat_142` has 2,333 levels over 76,020 rows, i.e. ~33 rows per level and ~1.3
positives per level. If it hurts, use frequency encoding instead and leave target encoding to
[03](../03-feature-engineering/).

### 3. Settle class imbalance empirically — three arms, one measurement

At a 3.96% positive rate the reflex is `scale_pos_weight` or `is_unbalance`. Do not assume;
measure all three on the same folds:

| Arm | Setting |
|---|---|
| A | no reweighting (plain `objective="binary"`) |
| B | `scale_pos_weight = 73012/3008 ≈ 24.3` |
| C | `is_unbalance=True` |

Judge on **F1 at the tuned threshold from [01](../01-threshold-engine/)**, not on AUC and not on
F1-at-0.5. Reweighting mostly rescales the probability distribution, which the threshold engine
then undoes — so it frequently changes AUC by ~nothing while changing the *optimal threshold*
enormously. It is common for arm A to win once the threshold is tuned. Whichever wins, note that
arms B and C shift the optimal threshold far from arm A's, so never reuse a threshold across
arms.

### 4. Tune, but modestly

Random/Optuna search, ~30–60 trials, 5-fold, over:

```
num_leaves        [15, 63]        min_child_samples [20, 300]
feature_fraction  [0.5, 0.95]     bagging_fraction  [0.6, 0.95]
lambda_l1         [0, 5]          lambda_l2         [0, 10]
learning_rate     0.02–0.05 fixed low, let early stopping choose n_estimators
```

**Compare against the ±0.0060 measured fold noise.** A trial that "wins" by 0.002 AUC has won
nothing. Take the best *region* of hyperparameters, not the single best trial, and prefer the
more regularized end of any tie.

### 5. Alternates, only if time allows

- **XGBoost** (`hist` tree method) — mostly a diversity source for
  [05](../05-ensemble-diversity/) rather than a winner on its own.
- **CatBoost** — genuinely worth a shot here specifically because of the three high-cardinality
  categoricals; its ordered target statistics handle exactly the `feat_142`-shaped problem that
  naive target encoding leaks on. Slower to train; budget accordingly.

Keep all three only if they will actually be blended. A single well-tuned LightGBM plus a
correct threshold beats a rushed three-model blend.

## Kaggle cost

CPU only — **do not enable the GPU.** At 76,020 × ~306 columns, CPU LightGBM trains a 5-fold
model in a couple of minutes; GPU adds transfer overhead and burns the ~30 h/week cap for no
gain. A 40-trial Optuna search over 5 folds is roughly 1–2 CPU-hours; Kaggle CPU notebooks have
no weekly quota.

CatBoost on 2,333-level categoricals is the one genuinely slow item here — budget 20–40 minutes
per 5-fold run.

## Honest expected gain

| Step | Expected | Confidence |
|---|---|---|
| HistGBM → tuned LightGBM | **+0.002 to +0.008 OOF AUC** | medium-high |
| …translated to F1 | **+0.005 to +0.020 binary-F1** | medium |
| Native categorical handling | −0.002 to +0.004 AUC (**may hurt**) | low |
| Imbalance arm selection | 0 to +0.010 F1, mostly by *not* picking the wrong arm | medium |

Total realistic: **binary-F1 ~0.384 → ~0.395–0.405**, macro-F1 ~0.679 → ~0.685–0.695. This is a
real but modest gain — an eighth of what the threshold mechanic in
[01](../01-threshold-engine/) is worth. Budget accordingly.

## When to abandon

- If the straight LightGBM swap (step 1) does not beat OOF AUC 0.8868 by at least **+0.003**,
  stop tuning and go to [03](../03-feature-engineering/) — the model is not the bottleneck, the
  features are.
- If 40 search trials produce nothing outside the ±0.0060 noise band, take the default params
  and move on. That is a real result, not a failure: it means you are at the data's ceiling and
  further tuning is fitting fold noise.
- Drop CatBoost immediately if a single 5-fold run exceeds ~45 minutes; the ensemble diversity
  is not worth a day of the four you have.
