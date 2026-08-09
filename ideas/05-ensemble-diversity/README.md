# 05 — Ensemble & diversity

~2–4 hours. Reliable, small, and boring — which is exactly what you want on day 3. Every step
here trades compute for variance reduction, and none of it requires a new idea to work.

## What it is

Three escalating levels of averaging:

1. **Seed averaging** — same model, several seeds, average the probabilities.
2. **Rank blending** — different model families, averaged in rank space rather than
   probability space.
3. **Stacking** — a meta-model over OOF predictions. Listed for completeness; probably not
   worth it in four days.

## Why it should work (measured evidence)

- **The measured per-fold AUC spread is ±0.0060** (baseline: 0.8797 / 0.8812 / 0.8931 / 0.8857 /
  0.8945). That spread *is* the variance seed-averaging removes. It is also larger than the
  entire expected gain from [02](../02-gbdt-core/) — so reducing it is competitive with
  improving the model.
- **The threshold sits on a knife-edge of a flat plateau** (see
  [01](../01-threshold-engine/)): binary-F1 varies by under 0.005 across t ∈ [0.15, 0.21]. A
  smoother, lower-variance probability distribution makes the chosen operating point more likely
  to transfer — averaging helps F1 *twice*, once through ranking and once through stability.
- **3,008 positives** is a small positive set. Single-model predictions on the marginal cases
  near the cut are noisy almost by definition, and those marginal cases are precisely what
  determines F1.
- **No leak, no magic feature, genuine generalization ceiling** (measured, see
  [dead-ends](../dead-ends/)). When there is no clever win available, variance reduction is what
  is left.

## Concrete steps

### 1. Seed averaging (do this — near-free, most reliable step in the folder)

```python
SEEDS = [42, 1337, 2026, 7, 99]
oof = np.zeros(len(train)); test_p = np.zeros(len(test))
for s in SEEDS:
    params["seed"] = s
    o, t = run_cv(params)          # same pinned folds every time
    oof += o / len(SEEDS); test_p += t / len(SEEDS)
```

Keep the **fold split pinned** across seeds and vary only the model seed. Varying folds too
conflates two sources of variance and makes the comparison against a single-seed run
uninterpretable.

Then **re-run the threshold engine on the averaged OOF** — averaged probabilities are less
extreme, so the optimal threshold moves. Track the predicted-positive rate (~4.4%) rather than
the raw threshold value across configurations.

### 2. Rank blending across model families

Only worth doing if [02](../02-gbdt-core/) produced two or three genuinely different models
(LightGBM / XGBoost / CatBoost).

```python
from scipy.stats import rankdata
blend = sum(rankdata(p) / len(p) for p in model_probas) / len(model_probas)
```

Rank space, not probability space — the three libraries calibrate differently, and averaging raw
probabilities lets the most confident model dominate for no good reason. Since the grade depends
only on the binary decision, calibration is irrelevant and only the induced ordering matters,
which makes rank averaging the natural choice.

Weight by OOF AUC if the models differ materially, but **only if the gap exceeds the ±0.0060
noise band** — otherwise use equal weights. Fitted blend weights on 5 folds overfit readily.

**Inference-notebook warning:** `rankdata` over the test file is computed *within that file*. On
the hidden test (different size, different composition) the transform is well-defined and
deterministic per-file, which is acceptable — but you must verify that the resulting
predicted-positive rate is still ~4.4%, because a rank-space threshold is a **quantile**, not a
probability. Choose the cut as "top q% by blended rank," and it transfers cleanly. See
[06](../06-inference-notebook/).

### 3. Stacking (probably skip)

A logistic regression or shallow LightGBM over the OOF prediction columns. Requires nested CV to
evaluate honestly, or the meta-model's score is optimistic. With 3,008 positives and four days,
the added complexity and the extra artifacts to save and reload in the inference notebook are
poorly matched to an expected gain of a few thousandths. **Recommend against unless everything
else is finished and verified.**

## Kaggle cost

Linear in the number of models. A 5-fold LightGBM run is ~2–3 CPU-minutes, so 5 seeds ≈ 15
minutes; three families × 5 seeds ≈ 45–60 minutes. CPU only, no quota concerns.

The real cost is **artifact size and inference-notebook complexity** — 15 models to pickle,
reload, and predict with, all of which must be deterministic. If artifacts get unwieldy,
prefer fewer seeds over dropping the determinism checks.

## Honest expected gain

| Step | Expected | Confidence |
|---|---|---|
| Seed averaging (5 seeds) | **+0.001 to +0.003 OOF AUC**; F1 +0.003 to +0.008 | **medium-high** |
| Rank blending 2–3 families | +0.002 to +0.005 AUC; F1 +0.003 to +0.010 | medium |
| Stacking | +0.000 to +0.003 AUC | low |

Combined realistic: **binary-F1 +0.003 to +0.010.** Small, but among the most *dependable*
numbers in this folder — averaging essentially always helps a little and essentially never
hurts. The secondary benefit (threshold stability) is not captured in these figures and may
matter more on the hidden test than the headline AUC gain does.

## When to abandon

- **Seed averaging:** don't. It is 15 minutes and reliably non-negative. The only reason to cut
  it is inference-notebook artifact bloat — in which case drop to 3 seeds, or refit a single
  model on all data using the averaged configuration.
- **Rank blending:** abandon if the second and third families are more than ~0.01 OOF AUC behind
  LightGBM. Blending in a materially weaker model dilutes rather than diversifies.
- **Stacking:** abandon on sight unless it is 13 Aug morning and everything else is finished,
  verified, and submitted.
- **All of it:** if the inference notebook cannot reproduce the ensemble bit-exactly, cut the
  ensemble, not the reproducibility. A +0.005 F1 gain is worth a fraction of the 40% hidden-test
  component it would put at risk.
