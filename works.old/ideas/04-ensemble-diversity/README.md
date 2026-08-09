# 04 — Ensemble & diversity

Reliable but modest. Expect **+0.003 to +0.008 composite**. Do this after the model and
feature work, as the final squeeze.

## Rank-average for the probability column

The probability column feeds **only** AUC, and AUC depends only on ordering. So blend
**ranks, not probabilities** — this makes the blend immune to members being calibrated
differently, which matters because CatBoost, LightGBM and a neural net will produce
probabilities on quite different scales.

```python
from scipy.stats import rankdata

def rank_blend(pred_list, weights=None):
    weights = weights or [1.0] * len(pred_list)
    stacked = sum(w * rankdata(p) / len(p) for w, p in zip(weights, pred_list))
    return stacked / sum(weights)      # in (0, 1], clip before submitting
```

Then clip into the open interval before writing the submission:
`np.clip(blend, 1e-6, 1 - 1e-6)`.

Choose weights by maximizing **OOF AUC** (not the composite — the AUC term is the only
thing this column affects). A coarse grid or `scipy.optimize.minimize` over the simplex is
plenty; with 3–5 members there's no need for anything fancier.

## Sources of diversity, in order of value

1. **Different algorithms.** LightGBM / CatBoost / XGBoost genuinely disagree — CatBoost's
   ordered boosting in particular behaves differently on the noisy duplicate rows.
   This is the highest-value diversity axis.
2. **Different seeds.** Cheapest possible gain. 5 seeds of the same LightGBM config,
   rank-averaged, typically adds +0.002–0.004 AUC for zero thought. Always do this.
3. **Different feature subsets.** One model on raw features, one on raw + Block A
   aggregates, one on a reduced set. Decorrelates errors.
4. **Different depths.** Shallow (`depth=4`) and deep (`depth=8`) models make different
   mistakes.
5. **RandomForest / ExtraTrees.** Weaker individually (0.8107 measured) but sufficiently
   different to earn a small blend weight.

## Stacking

With clean OOF predictions from the shared fold assignment, a level-2 model is
straightforward:

```python
# meta-features: OOF probabilities from each base model
meta_X = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_rf])
meta   = LogisticRegression(C=1.0, max_iter=1000)
```

Keep the meta-model **simple** — logistic regression or a depth-2 LightGBM. With 2,406
positives, a complex stacker overfits the OOF predictions and the gain evaporates on the
leaderboard. In practice a well-weighted rank-average usually matches stacking here at a
fraction of the risk; try stacking only if the blend has plateaued.

**Critical:** every base model must use the identical `StratifiedKFold(5, shuffle=True,
random_state=42)` split, or the stacker trains on contaminated meta-features.

## Neural tabular models — honest assessment

The prompt suggests TabNet / FT-Transformer. Realistic expectations for this dataset:

- **48,128 rows × 286 features with 2,406 positives is small for a neural tabular model.**
  The regime where FT-Transformer and friends beat well-tuned GBDTs generally starts in
  the hundreds of thousands of rows, with more signal per feature than we have here
  (best single-feature correlation is |r| < 0.17).
- Published benchmarks on this data scale consistently favour gradient-boosted trees.
  Expect a standalone neural model to land **below** LightGBM, probably 0.79–0.81 AUC
  against LightGBM's ~0.83.
- **Its value is as a blend member, not a contender.** A model that scores slightly worse
  but errs differently can still add +0.002–0.004 AUC at a 10–20% blend weight. That's
  the only reason to build one.

Resource reality: such a model would use **well under 2 GB of your 16 GB VRAM**, and cost
20–40 minutes of a limited GPU quota per 5-fold run. Weigh that against the fact that
every idea in folders 01–03 is CPU-only and unlimited.

**Recommendation: skip unless the GBDT path is exhausted.** If you do build one, a plain
3-layer MLP with embeddings for the binary columns, batch-norm and dropout is a better
time investment than TabNet — simpler, faster, and usually just as useful as a diversity
member.

Non-negotiable if you go this route: standardize the numeric columns and winsorize the
heavy-tailed sensor columns first (`sensor_wind_speed_kmh` reaches 3,008,077 — this will
destroy a neural net's training dynamics while leaving trees entirely unaffected).

## Blend the binary column separately

Remember the two levers are independent:

1. Build the best-ranked probability vector you can → that's `Target_Probability`.
2. Run the threshold/top-$k$ search from [../03-threshold-engine/](../03-threshold-engine/)
   **on the blended OOF predictions** → that's `Target_Binary`.

Do not average the individual models' binary predictions by voting. Blend the scores
first, then threshold once. Voting discards the ranking information the threshold
search needs.

## Checklist

- [ ] All base models share the exact same fold split
- [ ] OOF predictions saved per model
- [ ] Seed-averaging applied (cheapest win)
- [ ] Blend weights chosen on OOF AUC
- [ ] Rank-averaged, not probability-averaged
- [ ] Threshold re-tuned on the *final blended* OOF, not on any single model
- [ ] Final probabilities clipped to $(0, 1)$
