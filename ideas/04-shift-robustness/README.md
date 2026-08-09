# 04 — Shift robustness

**Status: step 1 is done. Step 2 (feature audit) is now automated, and folded in with idea
02's imbalance-weighting A/B under one shift-aware evaluation methodology, in
`solution/pstu_train.py` sections 5 and 7.** Steps 3 (rank-transform) and 4 (per-row adversarial
sample weighting) below remain undone — candidates if the automated search's gain proves
insufficient. What follows describes the reasoning; the implemented parts run automatically on
every training run, no manual A/B required.

A real Kaggle submission from `solution/pstu_train.ipynb` (real LightGBM, tuned threshold)
scored public LB **0.1849 binary-F1** against an OOF binary-F1 of **0.3954** (AUC 0.8920) —
roughly half. Submission positive rate was 2,597/60,654 = 0.0428 vs its own OOF rate 0.0494
(ratio 0.87), matching the ~0.86 rate-shrinkage pattern already measured locally with the
HistGBM stand-in. That rate shift is real but too small to explain a ~2× F1 drop on its own —
most of the gap is genuine generalization loss, exactly what this idea addresses. Full run
archived at `../../results/run-1/`.

**What changed since this was written:** rather than running the feature/imbalance A-B-C arms
below by hand and eyeballing OOF, `pstu_train.py` now scores every candidate config on a
**shift holdout** (train rows an adversarial classifier says look most like test) and picks the
winner automatically. Ordinary CV can't make this choice — every fold looks equally "trainy," so
nothing in it would ever penalize a config for failing to generalize to *shifted* data
specifically. See `solution/KAGGLE_INSTRUCTIONS.md`'s "What changed from run-1" section for the
mechanism, and note the same honesty caveat applies here: **this cannot guarantee a specific
leaderboard score** — it optimizes against the measured failure mode, which is the most that's
achievable without observing the hidden test directly.

**Run-2 (2026-08-09, real LightGBM, full output at `../../results/run-2/`): the search ran and
picked `drop_top5_shift` / no reweighting** (shift_holdout_f1 0.4127, beating `keep_all`'s
0.4044 and `drop_top1_feat182`'s 0.4040 — a real but modest margin among the 6 arms, see
`arm_search.csv`). OOF AUC 0.8928 and OOF binary-F1 0.3957 are both within the ±0.006 fold-noise
band of run-1's 0.8920 / 0.3954 — the search doesn't move the optimistic number, which is
expected; it was never trying to. **Public LB for run-2 has not been recorded yet** — that
submission is the actual test of whether this helped, and isn't in yet.

**⚠️ Audit finding, worth reading before trusting the shift-holdout number for anything else:**
in every arm of `arm_search.csv`, `shift_holdout_f1` is *higher* than `quick_full_f1`, and the
final model's shift-holdout F1 (0.4068) is likewise higher than its full OOF F1 (0.3957) — the
opposite of what "this should track real LB more closely" implied, since run-1's real LB
(0.1849) sits far *below* full OOF. The likely cause: the shift holdout's positive rate (0.0472)
runs about 1.19× the overall rate (0.0396), and a higher base rate mechanically inflates F1 at a
fixed threshold regardless of whether predictions are actually more reliable there. **Treat
`shift_holdout_f1` as valid only for its narrow original purpose — ranking candidate arms
against each other, which is all section 7 actually uses it for — not as an absolute estimate of
LB, and not as proof this run will close the OOF→LB gap.** That claim stands or falls on run-2's
own LB score once submitted, compared against run-1's 0.1849. Full detail in `CLAUDE.md`'s
"Run-2" section.

## What it is

Respond to the measured train↔test covariate shift: quantify how much CV is overstating
leaderboard performance, then decide whether to (a) do nothing but monitor, (b) down-weight or
clip the shift-driving features, or (c) weight training rows by their test-likeness.

Given the scoring split — **40% private + 40% hidden vs 10% public** — a model that scores
slightly worse on OOF but degrades less on unseen data is straightforwardly the better
submission.

## Why it should work (measured evidence)

- **Adversarial validation AUC = 0.5742 ± 0.0027** (`05_adversarial_validation_report.txt`).
  Not catastrophic, but well clear of the ~0.50 an iid split would give, and the ±0.0027 band
  means it is unambiguously real, not noise.
- **`feat_182` alone drives it** — importance 0.1706 in the adversarial classifier, **~2.4× the
  next feature** (`feat_44`, 0.0702). A single column carrying that much train/test
  discriminative power is a red flag: it is the most likely candidate to behave differently on
  the hidden slice.
- **93 of 344 numeric columns have test values outside their train `[min, max]`.** The worst
  (`feat_116`, `feat_44`, `feat_334`, `feat_306`, `feat_97`) are also the top adversarial
  features — the shift is concentrated in a small, identifiable set, which is the good case:
  it can be targeted.
- **Corroborating suspicion:** the baseline OOF AUC of **0.8868** is high for a problem of this
  shape. On the structurally-similar public Santander dataset, reported leaderboard AUCs topped
  out considerably lower (**~0.83–0.84 from memory — treat as an unverified prior, not a
  measurement**). If that prior is even roughly right, OOF 0.887 will not transfer intact, and
  the gap is exactly what this idea is about. **Do not act on this number; act on the
  CV-vs-LB gap you actually observe.**

## Concrete steps

### 1. Measure the CV↔LB gap first — ✅ done, 2026-08-09

| Observation | Interpretation | Action |
|---|---|---|
| LB ≈ OOF (within ~0.02) | shift is benign | do nothing, skip to [05](../05-ensemble-diversity/) |
| **LB well below OOF** ← **this happened** | CV is optimistic | **proceed to steps 2–4** |
| LB wildly below OOF (e.g. ~0) | leakage in your own pipeline | audit target encoding first |

Measured: OOF ~0.39–0.40 → LB **0.1849**. Middle row. Not the catastrophic bottom row (rules
out a gross pipeline leak — the model clearly works, just not as well as CV suggests), and far
past the top row's "within ~0.02" tolerance. **Proceed to steps 2–4.**

Caveat still worth keeping in mind: **public LB is only 10% of test data** (~6,065 rows, ~240
positives), with sampling noise on the order of ±0.02–0.03 on a metric this size. A single
submission is not proof of the *exact* gap size — but a ~0.20 drop is roughly 7–10x that noise
band, so the *existence* of a real gap is not in doubt, even if its precise magnitude will move
a bit submission to submission. Don't over-index on 0.1849 specifically; do treat "LB
substantially below OOF" as settled.

### 2. Adversarial feature audit — ✅ automated in `pstu_train.py` sections 5/7

**Arm C (clip to train range) is already the default** — every numeric feature is clipped to
its saved train `[min, max]` before both training and inference (`build_features`'s
`clip_bounds` step). The 0.1849 LB score already includes this mitigation, which means clipping
alone was not sufficient to close the gap — the remaining arms are what's left to try, not
optional extras layered on top of an unclipped baseline.

For the top shift-driving features (`feat_182`, `feat_44`, `feat_116`, `feat_306`, `feat_97`),
check each one's actual predictive value for `TARGET`. `feat_182` is measured at single-feature
AUC **0.6824** — third-highest in the dataset. So it is *both* the biggest shift driver *and* a
genuinely strong predictor. That is the awkward case, and exactly why this shouldn't be decided
by argument: `FEATURE_ARMS` in the training script defines it as an A/B/C exactly as originally
proposed here —

- `keep_all`: baseline (clip active, `feat_182` kept) — what run-1 (LB 0.1849) used.
- `drop_top1_feat182`: drop `feat_182` only, clip still active.
- `drop_top5_shift`: drop the full top-5 shift-driving set, clip still active.

— except now **the judge is the shift holdout from section 5, not plain OOF.** The original
plan here ("confirm the more promising one with an actual LB submission") is exactly the
weakness this automation removes: OOF alone can't distinguish the arms because the gap it needs
to detect is precisely what OOF doesn't see. Scoring on rows the adversarial classifier already
flagged as test-like is the closest available local proxy.

**Note on imbalance weighting:** `pstu_train.py`'s search also includes a `scale_pos_weight`
arm (none vs the measured neg:pos ratio, ~24.3), judged by the same shift-holdout metric. That's
idea `02-gbdt-core`'s step 3 A/B, not this idea's — it's folded in here because the shift holdout
is a better judge for it too than plain OOF, for the same reason.

### 3. Rank-transform the worst offenders — still open, not yet implemented

For the ~5 worst-shifted columns, replace raw values with within-file percentile ranks. This
makes the feature invariant to monotone distributional drift, at the cost of discarding
magnitude information. Not built into `pstu_train.py` — remains a candidate if the feature-drop
arms in step 2 don't close enough of the gap.

**Warning for [06](../06-inference-notebook/):** a rank transform computed *within the test file*
is a different transform than the one fitted on train, and the hidden test is a different size
and composition. Either fit the rank mapping on train and interpolate at inference (safe,
reproducible), or accept file-dependent behaviour (not reproducible — do not do this). Given the
40% at stake, prefer the already-automated clipping + feature-drop search (step 2) over
rank-transforming unless step 2's gain proves insufficient.

### 4. Adversarial sample weighting — still open, not yet implemented

Weight each training row by its probability of looking like a test row, from the adversarial
classifier:

```python
w = p_test / (1 - p_test)          # importance weight
w = np.clip(w, 0.25, 4.0)          # clipping is essential — unclipped weights explode
```

Train the final model with `sample_weight=w`.

Be clear-eyed: at an adversarial AUC of only 0.5742 these weights are close to uniform, so the
effect will be small — and importance weighting at low separability mostly adds variance. **This
is the least likely step here to pay off.** It is included for completeness and should be the
first thing cut for time.

### 5. Prefer the robust choice at every tie

Whenever two configurations are within the measured ±0.0060 fold noise, take the one with fewer
features, more regularization, or less dependence on the shift-driving columns. Free insurance
against the 80% of the grade you cannot see.

## Kaggle cost

The automated version (section 5 + 7) adds real time to every training run: the adversarial
classifier is now out-of-fold (`cross_val_predict`, ~3 RF fits instead of 1), plus 6 quick
reduced-estimator CV configs in the arm search. Estimated +10–15 minutes over the pre-search
pipeline — see `solution/KAGGLE_INSTRUCTIONS.md`'s runtime table. `SKIP_ARM_SEARCH = True`
reverts to the old, fast, unsearched `keep_all`/`none` behavior if that overhead isn't
affordable on a given run. CPU only throughout.

## Honest expected gain

- **On OOF: zero or slightly negative, by design.** If these steps improve OOF, be suspicious.
- **On public/private/hidden LB: +0.02 to +0.06 binary-F1**, **medium confidence** (upgraded
  from the original low-medium — the gap this idea targets is now measured, not hypothetical).
  Closing even a fraction of a confirmed ~0.20 gap is worth more than any tuning pass in
  [02](../02-gbdt-core/) or [03](../03-feature-engineering/); closing all of it is implausible
  in the time remaining, hence the modest range.
- **The step-1 diagnostic is complete** and delivered exactly what it promised: proof the OOF
  numbers ranking every other idea in this folder overstate real performance by roughly 2x.
  Read every other idea's "expected gain" figure with that discount in mind.

Expected-value note: because private + hidden is 80% of the leaderboard component, and the LB
gap is now confirmed rather than assumed, this idea's expected value likely exceeds
[02](../02-gbdt-core/) and [03](../03-feature-engineering/) combined — hence its promotion to
position 4 in [`../README.md`](../README.md)'s priority table.

## When to abandon

- Step 1 is done; its "abandon if LB ≈ OOF" condition did not apply — the gap is real.
- **Step 2 (feature/imbalance arm search) is automatic** — there's no manual "abandon" decision
  during a run, but if `keep_all`/`none` keeps winning across several runs (check
  `arm_search.csv`), that's informative in itself: it means the shift isn't concentrated enough
  in these 5 columns for dropping them to help, and effort is better spent on steps 3–4 or
  elsewhere. Use `SKIP_ARM_SEARCH = True` to save the ~10–15 min overhead once that pattern is
  clear.
- **If a follow-up LB submission doesn't close at least half the gap** between run-1's LB
  (0.1849) and the winning arm's `shift_holdout_f1`, treat that as evidence the shift-holdout
  proxy isn't as predictive of real LB as hoped — worth noting for future runs, not a reason to
  keep tuning against it blindly.
- **Abandon step 4 (sample weighting)** if it changes OOF AUC by more than ±0.01 in either
  direction — at this adversarial AUC that magnitude of change means the weights are
  destabilizing training, not correcting it.
- **Abandon rank-transforms (step 3)** if they cannot be made deterministic for the inference
  notebook. No exceptions; see [06](../06-inference-notebook/).
- Hard stop at 4 hours of *additional* work on steps 3–4 specifically — steps 1–2 are done and
  don't count against this budget. This is insurance and gap-closing, not a from-scratch scoring
  strategy — don't let it crowd out shipping a resubmission.
