# 01 — Threshold engine (the operating point)

**Highest value-per-hour in this folder.** ~1–2 hours. Do it before any modelling work.

## What it is

Two separable things, and conflating them is the classic way to lose this competition:

1. **The submission mechanic** — the grader applies a fixed 0.5 cut to whatever you upload. So
   you upload **hard 0/1 labels** at your chosen operating point (or rank-shift probabilities so
   0.5 lands exactly there). This is not optimization, it is not optional, and it is worth more
   than every other idea combined.
2. **Choosing the operating point** — an exhaustive O(n log n) sweep over every possible cut on
   the OOF predictions, scoring binary-F1 and macro-F1 at each, then picking a *robust* point on
   the resulting curve rather than the raw argmax.

## Why it should work (measured evidence)

### The mechanic — a measured +0.207

Same model, same predictions, two ways of submitting them:

| Submitted as | Binary F1 | Macro F1 |
|---|---|---|
| Raw probabilities (grader's 0.5 cut applies) | **0.1777** | **0.5791** |
| Hard labels at tuned cut (t≈0.19) | **0.3841** | **0.6786** |
| | **+0.2064** | **+0.0995** |

Because the positive rate is 3.96%, a calibrated model almost never emits P > 0.5, so a 0.5 cut
predicts far too few positives and recall collapses. This is a property of the class balance,
not of the model — it will not fix itself with a better model.

### The selection — the peak is flat, so don't chase the argmax

Measured sweep on the baseline OOF (`../dataset_exploration/09_threshold_sweep.csv`):

| t | binary F1 | macro F1 | n predicted positive |
|---|---|---|---|
| 0.15 | 0.3806 | 0.6742 | 4,507 |
| 0.16 | 0.3823 | 0.6758 | 4,191 |
| 0.17 | 0.3815 | 0.6761 | 3,881 |
| 0.18 | 0.3818 | 0.6769 | 3,602 |
| **0.19** | **0.3841** | **0.6786** | **3,366** |
| 0.20 | 0.3821 | 0.6780 | 3,143 |
| 0.21 | 0.3798 | 0.6772 | 2,953 |
| 0.24 | 0.3670 | 0.6718 | 2,430 |

Everything in **t ∈ [0.15, 0.21] is within 0.005 binary-F1 of the peak** — that is well inside
the measured ±0.0060 per-fold noise. The argmax at 0.19 is *not* meaningfully better than 0.16
or 0.20; it just won a coin flip on this particular OOF sample. Picking the argmax overfits the
fold noise; picking the **centre of the plateau** transfers better.

Note the asymmetry: the curve decays gently upward of the peak (0.24 → 0.3670) but the
low-threshold side degrades faster in macro-F1 terms. A slight bias toward the *upper* half of
the plateau is cheap insurance.

### The predicted-positive-rate sanity check

At the optimum the model flags 3,366 / 76,020 = **4.43%** of rows, against a true positive rate
of 3.96%. F1 rewards slightly over-predicting the positive class. That gives a concrete
cross-check for any submission: **expect ≈ 0.0443 × 60,654 ≈ 2,700 positives in
`submission.csv`.** If a submission has 300 positives or 20,000, something upstream broke —
catch it before uploading, not on the leaderboard.

## Concrete steps

1. **Port the O(n) cut-point optimizer** from `works.old/solution/pstu_kaggle_solution.py`
   (`cutoff_curve()`) — sort predictions once, take cumulative sums of the label vector, and you
   get the full confusion matrix at *every* distinct threshold in one pass. It was verified
   against brute force to 1e-9. **Swap its scoring line from the old composite metric to F1** —
   both binary and macro.

2. **Sweep on OOF predictions only.** Never tune the threshold on data the model trained on;
   in-fold probabilities are systematically over-confident and would push the cut far too high.

3. **Pick the plateau centre, not the argmax:**

   ```python
   peak = curve["binary_f1"].max()
   plateau = curve[curve["binary_f1"] >= peak - 0.005]   # 0.005 ≈ measured fold noise
   t_star = plateau["threshold"].median()
   ```

   Report the argmax alongside it so you can see how far apart they are. On the baseline they
   differ by ~0.02 in threshold and ~0.002 in F1 — i.e. nothing, which is exactly the point.

4. **Check per-fold threshold stability.** Compute the optimal threshold *within each of the 5
   folds separately*. If those five thresholds are spread across [0.12, 0.30], the operating
   point is unstable and you should widen the plateau tolerance. If they cluster tightly, you
   can trust a narrower choice. This measurement costs nothing and tells you how much to trust
   the number.

5. **Emit hard labels and validate:**

   ```python
   preds = (test_proba >= t_star).astype(int)
   assert 1500 < preds.sum() < 4500, f"implausible positive count: {preds.sum()}"
   ```

6. **Re-tune after every model change.** The optimum moves with calibration — the RF baseline
   peaked at t≈0.75, the HistGBM baseline at t≈0.19, for near-identical F1. The *threshold value
   is meaningless across models*; only the resulting predicted-positive rate is comparable.
   Track that rate, not the raw threshold.

7. **If the LB probe says macro-F1**, re-run selection against `macro_f1`. On the baseline both
   metrics peaked at the same t, but that is a coincidence of this model's calibration and is
   not guaranteed to survive.

## Kaggle cost

Negligible — seconds. The sweep is a sort plus a cumulative sum over 76,020 rows.

## Honest expected gain

- **The mechanic: +0.207 binary-F1 / +0.099 macro-F1** vs naively uploading probabilities.
  Measured, high confidence. This is a catastrophe-avoidance number — the baseline in the index
  table already includes it, so treat it as "the cost of getting this wrong" rather than as
  headroom to be gained.
- **Plateau-centred selection over argmax: +0.000 to +0.010 on the *test* metric**, medium
  confidence. It cannot help OOF by construction (the argmax is by definition the OOF maximum);
  the whole claim is that it degrades less on unseen data. Given the measured 0.5742 adversarial
  shift, the OOF argmax is more likely than usual to be a fold-noise artifact.
- **Per-fold stability check: no direct gain**, but it tells you whether to trust the operating
  point at all — worth the 10 minutes.

## When to abandon

Don't. The whole idea is 1–2 hours and the mechanic alone is decisive. The only genuinely
optional part is step 4 (per-fold stability); skip it if time-pressed and just use the plateau
median with a 0.005 tolerance.

## Interaction with other ideas

- Every model change in [02](../02-gbdt-core/), [03](../03-feature-engineering/) and
  [05](../05-ensemble-diversity/) **invalidates the current threshold.** Re-run this after each.
- Rank-averaged ensemble scores ([05](../05-ensemble-diversity/)) are not probabilities at all,
  so the threshold there is purely a quantile choice — pick the cut that yields the target
  predicted-positive rate (~4.4%) rather than any fixed number.
