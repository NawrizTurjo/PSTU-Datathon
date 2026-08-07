# 05 — Label-noise handling

**Speculative.** Expected gain 0 to +0.005. Try only after ideas 00–04. Included because
the noise is real and measured, but no approach here has been validated on this dataset.

## The finding

- **7.35%** of train rows (3,539) are exact feature-duplicates of at least one other row.
- **105 duplicate groups**, covering **3.30% of train rows (1,590)**, have
  **conflicting target labels** — byte-identical features, different outcome.
- The largest duplicate group has 430 rows and a target rate of 0.033; another has 25 rows
  at 0.280. Several sit at 0.5.

These rows are genuinely unpredictable from the features. Either the labels carry noise,
or the outcome depends on information not present in the 286 columns.

## What it does and doesn't cap

This is worth stating precisely, because the intuition is misleading:

- **It does not meaningfully cap AUC.** Measured: predicting every row as its own
  duplicate-group mean gives an oracle AUC of **0.9993**. The conflicting groups are too
  small a slice to bound ranking performance. So AUC ~0.83 is nowhere near a noise ceiling —
  there's real headroom.
- **It does cap precision and $F_1$.** Within a group with target rate 0.28, no model can
  do better than predicting 0.28 for all of them. Any threshold either labels all of them
  positive (72% false alarms) or all negative (missing 28% of the failures). Since
  precision is already the binding constraint at the optimum (measured 0.183), these rows
  are a direct drag on the 0.30-weighted $F_1$ term.

Practical implication: **don't interpret a stalled $F_1$ as a modelling failure.** Part of
it is structural.

## Approaches worth testing

### A. Sample weighting (safest)

Downweight rows in conflicting groups so the model doesn't waste capacity fitting
contradictions.

```python
gid = train.groupby(feat_cols, sort=False).ngroup()
stats = pd.DataFrame({"g": gid, "y": y}).groupby("g")["y"].agg(["count", "mean"])
# purity: 1.0 for a group that agrees, 0.5 for a group split down the middle
purity = stats["mean"].apply(lambda m: max(m, 1 - m))
w = purity.reindex(gid).values          # in [0.5, 1.0]
model.fit(X, y, sample_weight=w)
```

Low risk, easy to A/B. Compare on OOF AUC.

### B. Soft labels / group-mean targets

Replace hard labels with the group's empirical rate and train a regressor on the
probability, or use a soft-label-capable objective. Theoretically the "correct" thing to
do, but changes the training objective enough that it needs careful validation.

**Leakage warning:** the group mean must be computed from the training fold only. Computing
it across the full training set and using it as a target or feature leaks labels directly
into validation — you'll see a large CV gain that vanishes on the leaderboard. See
[../06-dead-ends/](../06-dead-ends/), where a related approach was measured and failed.

### C. Deduplication

Collapse each duplicate group to a single row, weighted by group size, labelled with the
majority (or the group mean under approach B).

Reduces train from 48,128 to ~44,600 rows. Speeds things up and prevents large groups from
dominating the loss. Risk: the 430-row group genuinely represents 430 real stations — its
frequency may be legitimate signal about how common that configuration is, and collapsing
it discards that.

### D. Just use CatBoost

CatBoost's ordered boosting is specifically designed to resist target leakage and
overfitting to noisy labels. It may handle this for free with no special treatment —
which is part of why it's recommended in [../01-gbdt-core/](../01-gbdt-core/).

**This is the highest effort-to-reward option: it costs nothing extra.** Try it before
building any custom noise handling.

## Realistic expectation

None of A–C is likely to move the composite much. The conflicting rows are only 3.3% of
train, and gradient boosting with sensible regularization already handles moderate label
noise reasonably well. The main value of this document is **diagnostic**: knowing that
part of your $F_1$ gap is structural stops you from burning hours trying to close it with
more feature engineering.

If you test one thing here, make it **A** (sample weighting) — it's a two-line change with
a clean A/B.
