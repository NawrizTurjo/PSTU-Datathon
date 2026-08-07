# 03 — Threshold engine

**Priority: do this first, before model tuning.** Measured gain: **+0.018 composite**,
for maybe an hour of work. That is more than the entire gain from upgrading
RandomForest to a gradient-boosted model (+0.010).

## The problem

Five of the six sub-metrics ($F_1$, precision, recall, specificity, balanced accuracy)
depend on where you cut probabilities into 0/1. Only AUC doesn't. Default `predict()`
cuts at 0.5, which is close to worthless at a 5% base rate.

Measured on a RandomForest baseline:

| threshold | $F_1$ | $P$ | $R$ | $S$ | composite |
|---|---|---|---|---|---|
| 0.50 (naive) | 0.2162 | 0.1285 | 0.6820 | 0.7566 | **0.4989** |
| 0.60 (tuned) | 0.2843 | 0.1899 | 0.5648 | 0.8732 | **0.5167** |

**+0.0178 for changing one number.**

## The optimum moves with the model

Do not hardcode 0.60. Measured optima:

| Model | OOF AUC | Optimal threshold | Composite |
|---|---|---|---|
| RandomForest | 0.8107 | **0.60** | 0.5167 |
| HistGradientBoosting | 0.8189 | **0.53** | 0.5269 |

The threshold tracks each model's calibration. **Re-tune it every time you change the
model, the features, or the class weighting.**

## Method 1 — threshold sweep (start here)

```python
import numpy as np

def best_threshold(y_true, oof_proba, lo=0.01, hi=0.99, step=0.005):
    best = (None, -1)
    for t in np.arange(lo, hi, step):
        score, _ = composite_score(y_true, (oof_proba >= t).astype(int), oof_proba)
        if score > best[1]:
            best = (t, score)
    return best  # (threshold, composite)
```

Run it on **out-of-fold** predictions, never on in-fold predictions — in-fold
probabilities are overconfident and will pick a threshold that doesn't transfer.

## Method 2 — search over $k$ instead of $t$ (more robust)

Rather than searching a probability cutoff, search the **number of rows you label
positive**, then take the top-$k$ by probability. For a fixed score vector these are
mathematically equivalent (cutting at the $k$-th highest score gives exactly $k$
positives), but $k$ has two practical advantages:

- **It is comparable across models and folds** regardless of how each is calibrated. A
  threshold of 0.53 means something different for every model; "label the top 9% as
  positive" means the same thing for all of them.
- **It avoids probability cliffs.** The RandomForest sweep had a discontinuity at
  $t=0.76$ where the composite collapsed from 0.508 to 0.433, because many rows shared
  near-identical probabilities and crossed the cutoff together. Searching over $k$ steps
  through them one at a time and cannot land inside such a cliff.

```python
def best_k(y_true, oof_proba, k_grid=None):
    n = len(y_true)
    k_grid = k_grid or range(int(0.01*n), int(0.40*n), max(1, n // 2000))
    order = np.argsort(-oof_proba)
    best = (None, -1)
    for k in k_grid:
        pred = np.zeros(n, dtype=int)
        pred[order[:k]] = 1
        score, _ = composite_score(y_true, pred, oof_proba)
        if score > best[1]:
            best = (k, score)
    return best  # (k, composite)
```

Convert $k$ to a test-set rule by **positive rate**, not by absolute count:
label the top $k/n_{\text{train}}$ fraction of test rows positive. At the HGB optimum
this was ~17% of rows — well above the 5% base rate, which is exactly what the recall
tilt in the metric predicts (see
[../00-foundation/metric-decomposition.md](../00-foundation/metric-decomposition.md)).

## Guarding against threshold overfitting

With 2,406 positives, the threshold itself can overfit the OOF predictions. Two checks:

**Per-fold stability.** Compute the optimal threshold within each fold separately. If the
five values scatter widely (say 0.45–0.70), the optimum is noise-driven — prefer the
median, or a value in the middle of a flat region rather than a sharp peak.

```python
per_fold = []
for tr_i, va_i in cv.split(X, y):
    t, _ = best_threshold(y[va_i], oof[va_i])
    per_fold.append(t)
print(np.median(per_fold), np.std(per_fold))
```

**Prefer plateaus over peaks.** Plot composite vs threshold. The RandomForest curve is
fairly flat between 0.58 and 0.66 (composite 0.514–0.517) — anywhere in that band is
safe. A threshold that wins by 0.001 over a narrow spike is not a real optimum. Pick the
centre of the widest near-optimal plateau.

**Nested selection (if you want to be thorough).** Choose the threshold on inner folds and
evaluate it on the outer fold, so the reported composite doesn't include the threshold's
own selection advantage. This gives an honest estimate of what the leaderboard will show;
it doesn't produce a better threshold.

## Calibration — worth knowing, mostly not worth doing

Isotonic regression, Platt scaling and temperature scaling are all **monotone**
transformations. A monotone transform cannot change the ordering of predictions, therefore:

- It **cannot change AUC** (0.25 of the score is untouched).
- It **cannot change the achievable composite** at the optimal threshold, since the
  optimal top-$k$ set is identical.

Calibration is therefore not a score improvement. It's useful only to make a *fixed*
threshold transfer more reliably between CV and test, or to make thresholds comparable
across ensemble members. If you're already searching over $k$ (Method 2), calibration
buys you nothing — skip it.

(Contrast: **class weighting during training** changes the fitted model itself, so it
*can* change AUC and is worth testing empirically. See [../01-gbdt-core/](../01-gbdt-core/).)

## Optional: Nelder-Mead / Bayesian search

The prompt suggests these. For a **one-dimensional bounded** search, a dense grid is
simpler, exhaustive, and takes milliseconds — there is no reason to use a
derivative-free optimizer here. They become relevant only if you extend the search to
multiple dimensions simultaneously, e.g. jointly tuning blend weights and the threshold.

## Checklist

- [ ] `composite_score()` implemented and unit-checked against the official formula
- [ ] Threshold chosen on OOF predictions only
- [ ] Search over $k$ / positive-rate, not a raw probability cutoff
- [ ] Per-fold threshold spread inspected; plateau preferred over peak
- [ ] Threshold re-tuned after *every* model or feature change
- [ ] Probability column submitted raw (clipped to $(0,1)$), independent of the threshold
