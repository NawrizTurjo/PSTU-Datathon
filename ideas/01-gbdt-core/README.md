# 01 — GBDT core

The workhorse. On 48k × 286 mixed numeric/binary tabular data with 5% positives,
gradient-boosted trees are the right default and very likely the final answer.

## Measured starting point

| Model | OOF AUC | Best composite |
|---|---|---|
| RandomForest (`max_depth=8`, balanced) | 0.8107 | 0.5167 |
| HistGradientBoosting (400 iters, lr 0.05) | **0.8189** | **0.5269** |

`HistGradientBoostingClassifier` is sklearn's LightGBM-equivalent and needs no extra
install — it's a good first move if you want a result in five minutes. A properly tuned
LightGBM/CatBoost ensemble should land around **0.83–0.84 AUC**; treat anything above
0.85 with suspicion and check for leakage.

## Why AUC improvements pay twice

Raising AUC lifts the 0.25-weighted AUC term *and* raises the entire
precision/recall frontier that the threshold search operates on. At the measured optimum
precision is only 0.183 — it is the binding constraint, and the only way to relax it is
better ranking. This is why model quality still matters even though threshold tuning is
the cheaper win.

## Recommended models

Train all three; they disagree usefully and blend well (see
[../04-ensemble-diversity/](../04-ensemble-diversity/)).

### LightGBM (primary)

```python
params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,       # important: 2,406 positives, keep leaves populated
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.1,
    lambda_l2=1.0,
    n_estimators=3000,          # with early stopping on the fold's validation AUC
    n_jobs=-1,
    verbosity=-1,
)
```

Use `early_stopping_rounds=200` on fold AUC. Do **not** early-stop on the composite —
it's threshold-dependent and noisy; optimize ranking during training and handle the
threshold afterwards.

### CatBoost (best single model on noisy tabular data, often)

```python
params = dict(
    loss_function="Logloss", eval_metric="AUC",
    learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
    iterations=3000, od_type="Iter", od_wait=200,
    random_seed=42, verbose=200,
)
```

CatBoost's ordered boosting is genuinely helpful given the label noise here
(see [../05-label-noise/](../05-label-noise/)) — it resists memorizing conflicting rows.
Slower than LightGBM but usually worth including.

### XGBoost (third opinion)

```python
params = dict(
    objective="binary:logistic", eval_metric="auc",
    learning_rate=0.03, max_depth=5, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.7,
    reg_lambda=2.0, n_estimators=3000, n_jobs=-1, tree_method="hist",
)
```

`tree_method="hist"` on CPU. Don't bother with `device="cuda"` at this data size.

## Class imbalance — test, don't assume

Three options, and the right one is an empirical question:

1. **Nothing.** Train on raw labels, fix everything at the threshold stage.
2. **`scale_pos_weight ≈ 19`** (= 45722/2406) or `class_weight="balanced"`.
3. **Focal loss** (custom objective).

Important subtlety: **weighting mainly moves where probabilities sit, not how rows are
ranked.** Since AUC is rank-based and the threshold is tuned afterwards, aggressive
weighting often buys less than expected — it can even hurt by distorting the loss surface.
The measured baselines above used balanced weighting, but option 1 plus a tuned threshold
is a genuinely competitive alternative and worth a direct A/B on OOF AUC.

Rule: **compare weighting schemes on OOF AUC** (weighting-invariant to the threshold
choice), then tune the threshold separately for whichever wins.

## Missing values

Leave the `-999999 → NaN` sentinel as `NaN`. All four libraries handle it natively and
will learn a dedicated split direction. Only 66 train rows are affected, so this is a
correctness detail rather than a scoring lever.

## Hyperparameter tuning

With 5-fold CV taking 1–3 minutes, you can afford ~50–100 Optuna trials on CPU inside a
single Kaggle session.

```python
import optuna
def objective(trial):
    p = dict(
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        feature_fraction=trial.suggest_float("ff", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bf", 0.5, 1.0),
        lambda_l2=trial.suggest_float("l2", 1e-3, 10.0, log=True),
    )
    return cv_oof_auc(p)          # optimize AUC, not the composite
optuna.create_study(direction="maximize").optimize(objective, n_trials=80)
```

**Tune on OOF AUC**, not the composite. The composite is threshold-dependent and its
noise would make the search chase fold artifacts. Ranking quality is the thing tuning can
actually improve; the threshold is handled separately and deterministically.

Given ±0.003–0.005 fold noise on the composite, don't over-tune — the difference between
a decent parameter set and a heavily optimized one on 48k rows is usually smaller than the
gain from [../02-feature-engineering/](../02-feature-engineering/).

## Regularization matters more than usual here

2,406 positives across 286 features, plus 3.3% of rows carrying conflicting labels, makes
overfitting easy. Bias toward:

- higher `min_child_samples` / `min_child_weight` (40+, not the default 20)
- `feature_fraction` around 0.6–0.8 — with 143 near-constant sparse columns, forcing
  subsampling helps trees find the informative ones
- meaningful `lambda_l2`
- more trees at a lower learning rate rather than fewer deep ones

## Checklist

- [ ] Preprocessing from [../00-foundation/](../00-foundation/) applied identically to train/test
- [ ] Same `StratifiedKFold(5, shuffle=True, random_state=42)` as every other experiment
- [ ] OOF predictions saved to disk for later stacking and threshold work
- [ ] Weighting scheme A/B-tested on OOF AUC
- [ ] Threshold re-tuned per [../03-threshold-engine/](../03-threshold-engine/) after tuning
- [ ] Sanity: composite comfortably above the 0.4388 all-ones floor
