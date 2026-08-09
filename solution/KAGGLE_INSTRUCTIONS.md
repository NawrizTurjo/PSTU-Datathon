# How to run this on Kaggle

Two notebooks, run in order. Total wall time ≈ 25–40 minutes on a free CPU kernel (up from the
first version — see "What changed" below).

| File | What it does | Runtime |
|---|---|---|
| `pstu_train.ipynb` | shift-aware config search, trains, picks the operating point, writes `submission.csv` + `artifacts.joblib` | ~20–35 min |
| `pstu_inference.ipynb` | **mandatory deliverable** — loads artifacts, writes `submission.csv` | ~1–2 min |

## What changed from run-1

Run-1 (real LightGBM, first submission) scored OOF binary-F1 ~0.395 but public LB only
**0.1849** — confirming the measured 0.5742 adversarial train/test shift causes real
generalization loss, not just LB noise. Plain cross-validation can't see this: every fold looks
equally "trainy," so nothing rewards a config for surviving the shift specifically.

This version adds a **shift-aware configuration search** (new sections 5 and 7): an adversarial
classifier ranks train rows by how much they resemble test rows; the most test-like 15% become a
standing holdout; 3 feature-drop arms × 2 imbalance arms are each quickly cross-validated and
scored on **that holdout**, not on ordinary OOF; the winner is retrained with the full budget.
Section 9 also prints a **shift-holdout binary-F1** alongside the standard OOF number.

**⚠️ Audit correction (after run-2, 2026-08-09):** an earlier version of this doc said to "watch
[the shift-holdout number], not the full-OOF one, when guessing at LB" — that framing doesn't
hold up. In run-2's `arm_search.csv`, `shift_holdout_f1` came out *higher* than the quick-CV full
F1 for every single arm, and the final model's shift-holdout F1 (0.4068) was likewise higher
than its full OOF F1 (0.3957) — the opposite of run-1's real LB (0.1849), which landed far
*below* full OOF. Likely cause: the holdout's positive rate (0.0472) runs ~1.19× the overall
rate (0.0396), which mechanically inflates F1 at a fixed threshold. **Don't use
`shift_holdout_f1` to predict what LB will read.** Its one legitimate use is picking a winner
*among* the candidate arms (which is what section 7 does with it) — not estimating the score
that winner will get. See `CLAUDE.md`'s "Run-2" section for the full writeup.

**This cannot guarantee a specific leaderboard score.** The hidden test's true distribution isn't
observable locally. What it does is optimize against the actual measured failure mode (the
confirmed OOF→LB gap) instead of against a number already known to overstate LB by roughly 2x.
Whether it actually closes any of that gap is only known once run-2 (or a later run) is
submitted and its LB score compared against run-1's 0.1849.

---

## ⚡ Use CPU, not GPU

**Set Accelerator to `None` for both notebooks.** This is deliberate, not an oversight.

You have 2×T4 or P100 available. Do not use them here:

- At 76,020 rows × ~320 columns, CPU LightGBM trains a fold in **seconds**. GPU histogram
  construction has host↔device transfer overhead that dominates at this data size — the GPU
  build is typically *slower* for this shape of problem, not faster.
- Kaggle **CPU notebooks have no weekly quota**. GPU is capped at ~30 h/week. Spending that cap
  on a job that runs faster without it is a straight loss.
- `device="gpu"` LightGBM uses a different histogram binning path whose floating-point reduction
  order is **not guaranteed reproducible across runs**. The inference notebook asserts
  bit-identical reproduction; a GPU run can fail that assertion for no benefit.
- A second T4 does nothing at all here — nothing in this pipeline is multi-GPU aware.

**The only reason to touch the GPU** would be adding a neural tabular model, which
[`../ideas/dead-ends/`](../ideas/dead-ends/) recommends against for this four-day sprint.

If you want more speed, raise `n_jobs` (already `-1`) or cut `SEEDS` from 3 to 1 — that is a
~3× saving for roughly −0.002 AUC.

---

## Step 1 — Training notebook

1. **New Notebook** → **File → Import Notebook** → upload `pstu_train.ipynb`.
2. **Add Data** → *Competitions* → add **PSTU Data Thon 2026 Vol-1**.
   It mounts at `/kaggle/input/competitions/pstu-data-thon-2026-vol-1/`. The notebook probes that
   path first and falls back to `/kaggle/input/pstu-data-thon-2026-vol-1/`, so either layout works
   without edits.
3. **Settings** → Accelerator **None**, Internet **Off** (nothing here needs it; off is also the
   safer posture for a no-external-data competition).
4. **Run All.**
5. Confirm in the output:
   - `backend: lightgbm` — if it says `histgbm`, LightGBM is missing and you are running the
     local stand-in. Fix before submitting.
   - `droppable columns: 44`
   - `cutoff_curve verified against sklearn on 40 cut points` ← the correctness gate
   - `adversarial train/test AUC` near **0.57–0.60** (Stage-1 measured: 0.5742)
   - the 6-row arm-search table, then a `winning config:` line
   - seed-averaged OOF AUC (expect ~0.88–0.90)
   - `shift-holdout binary_f1` in the final operating-point block — **do NOT use this to predict LB directly** (confounded by higher holdout positive rate 0.0472 vs 0.0396 overall; see audit note above). It is used for relative arm comparison in Section 7.
   - submission positive rate, typically ~0.04–0.05 (varies with which arm won)
6. **Save Version → Save & Run All (Commit).** This persists `/kaggle/working/` outputs.

**Outputs:** `submission.csv`, `artifacts.joblib`, `threshold_curve.csv`, `run_summary.json`,
`arm_search.csv` (all 6 candidate configs and their scores, for inspection).

Submit `submission.csv` from the notebook's Output tab.

---

## Step 2 — Publish the artifacts as a dataset

The inference notebook needs `artifacts.joblib`, and a notebook cannot read another notebook's
working directory directly.

1. Open the committed training version → **Output** tab.
2. **New Dataset** from the output (or download `artifacts.joblib` and upload it manually).
3. Name it **`pstu-artifacts`** — the inference notebook probes
   `/kaggle/input/pstu-artifacts/artifacts.joblib` first.

Any other name still works: the notebook falls back to scanning `/kaggle/input/**` for any
`*.joblib`. Naming it as above just makes the resolution explicit.

---

## Step 3 — Inference notebook (the graded one)

1. Import `pstu_inference.ipynb`.
2. **Add Data** → the competition **and** your `pstu-artifacts` dataset.
3. Accelerator **None**, Internet **Off**.
4. **Run All.**
5. Confirm in the output:
   - `build_features matches training copy: <hash>` ← proves no logic drift
   - unseen-level counts per categorical column (expect ~0.15% or less on the public test)
   - `REPRODUCTION OK: all 60654 predictions match the training run exactly`
6. **Save & Run All (Commit)**, then submit this notebook as the competition's required
   inference notebook.

### If `REPRODUCTION OK` does not appear

That assertion is the whole point of the notebook — **do not submit until it passes.**

| Symptom | Cause | Fix |
|---|---|---|
| `REPRODUCTION FAILED: N predictions differ` | non-determinism, usually GPU or unpinned threads | set Accelerator to None; confirm `deterministic=True` in `LGB_PARAMS` |
| `build_features has DIVERGED` | the two copies of the function are out of sync | copy the function body from `pstu_train.py` verbatim into `pstu_inference.py`, re-run training to refresh the hash |
| `reproduction check skipped` | test file has a different row count | expected on the hidden test; on the public `test.csv` it means you loaded the wrong file |
| `features missing at inference` | artifacts are from an older training run | re-run training, republish the dataset |

---

## Running against the hidden test

The organizers point the notebook at a different test file. One line changes:

```python
TEST_PATH = os.path.join(DATA_DIR, "test.csv")   # <-- repoint here
```

Everything downstream is derived from artifacts, so no other edit is needed. The notebook
already handles the two things that differ on an unseen file:

- **Unseen categorical levels** → mapped to a fallback bucket (frequency `0.0`, code `-1`) and
  the count is printed. Measured basis: `feat_142` already has 55 test-only levels.
- **Values outside the training range** → clipped to the saved train bounds. Measured basis:
  93 of 344 numeric columns already exceed their train range on the public test set.

The reproduction check self-skips on a different row count, which is the expected path.

---

## v1 baseline (superseded) — kept for comparison

This is the pipeline *before* the shift-aware config search (sections 5/7) existed — no arm
search, always `keep_all`/`none`. Real LightGBM run of exactly this version scored public LB
**0.1849** against OOF binary-F1 ~0.395 (see `results/run-1/`). That gap is what motivated the
rebuild; see "What changed from run-1" near the top of this file. Numbers below are local-only
(`HistGradientBoostingClassifier` stand-in) and predate the current pipeline — the current
version's own verified numbers are in the section below this one.

```
droppable columns: 44                       <- matches Stage 1 exactly
cutoff_curve verified against sklearn on 40 cut points (tol 1e-9)
seed 42   : OOF AUC 0.8862  (folds 0.8862 +/- 0.0061)
seed 1337 : OOF AUC 0.8868  (folds 0.8869 +/- 0.0066)
seed 2026 : OOF AUC 0.8863  (folds 0.8863 +/- 0.0059)
seed-averaged OOF AUC: 0.8887

operating point: threshold 0.1691, 3886 OOF positives (rate 0.0511)
  binary_f1 0.3853 | macro_f1 0.6780
  cost_vs_argmax: 0.0008          <- plateau centre gives up almost nothing
  plateau: t in [0.152, 0.213]

raw probabilities @ 0.5 cut : binary_f1 0.1724 | macro_f1 0.5764  (507 positives)
hard labels @ tuned cut     : binary_f1 0.3853 | macro_f1 0.6780  (3886 positives)
gain: binary_f1 +0.2129 | macro_f1 +0.1016

submission: 60654 rows | 2661 positives | rate 0.0439
artifacts.joblib: 5.4 MB, 15 models
```

Inference notebook against the same `test.csv`:

```
build_features matches training copy: bb6fe1831b922114
feat_142:  89 rows with unseen levels (0.1467%) -> fallback bucket
feat_157:  11 rows with unseen levels (0.0181%)
feat_325:  47 rows with unseen levels (0.0775%)
REPRODUCTION OK: all 60654 predictions match the training run exactly.
```

Those unseen-level counts match the Stage-1 EDA measurements exactly, which is a useful
end-to-end check that preprocessing is doing what it claims.

### Two honest caveats

1. **The engineered features and seed averaging did not clearly beat the Stage-1 baseline.**
   OOF AUC went 0.8868 → 0.8887 (**+0.0019**) and binary-F1 0.3841 → 0.3853 (**+0.0012**) —
   both *inside* the ±0.006 per-fold noise band. Treat this as "no measured regression," not as
   a win. The real gain in this pipeline is the operating-point mechanic (+0.2129), which was
   already known from Stage 1. Switching to LightGBM on Kaggle is the next thing likely to move
   the number; per [`../ideas/02-gbdt-core/`](../ideas/02-gbdt-core/) expect +0.002–0.008 AUC.

2. **The test predicted-positive rate runs below the OOF rate** — 0.0439 vs 0.0511, a ratio of
   0.86. The notebook prints this ratio and warns outside [0.5, 2.0]. It is consistent with the
   measured 0.5742 adversarial train/test shift: test scores skew slightly lower, so the fixed
   threshold selects proportionally fewer rows. It is not a bug, but it does mean the effective
   operating point on test is a little more conservative than the one tuned on OOF. If the LB
   score comes in below expectations, [`../ideas/04-shift-robustness/`](../ideas/04-shift-robustness/)
   step 1 is the diagnostic — and selecting the cut by *target positive rate* rather than by
   threshold value is the fix.

---

## Verified real Kaggle run-2 numbers (shift-aware search)

Full output archived at `results/run-2/`. Real LightGBM, 3 seeds × 5 folds, shift-aware config search active across 6 arms.

```
winning config: feature_arm='drop_top5_shift', imbalance_arm='none'
dropped columns: ['feat_182', 'feat_44', 'feat_116', 'feat_306', 'feat_97']
adversarial train/test AUC: 0.5733

arm_search.csv:
  drop_top5_shift   / none             : quick_full_f1 0.3890 | shift_holdout_f1 0.4127 (WINNER)
  drop_top5_shift   / scale_pos_weight : quick_full_f1 0.3848 | shift_holdout_f1 0.4090
  keep_all          / none             : quick_full_f1 0.3913 | shift_holdout_f1 0.4044
  drop_top1_feat182 / none             : quick_full_f1 0.3860 | shift_holdout_f1 0.4040
  drop_top1_feat182 / scale_pos_weight : quick_full_f1 0.3864 | shift_holdout_f1 0.3994
  keep_all          / scale_pos_weight : quick_full_f1 0.3808 | shift_holdout_f1 0.3894

seed 42   : OOF AUC 0.8917
seed 1337 : OOF AUC 0.8907
seed 2026 : OOF AUC 0.8917
seed-averaged OOF AUC: 0.8928

operating point: threshold 0.1830, 3547 OOF positives (rate 0.0467)
  binary_f1 0.3957 | macro_f1 0.6843
  plateau: t in [0.1558, 0.2170]
  cost vs argmax: 0.0026

shift-holdout (11403 rows): binary_f1 0.4068 | macro_f1 0.6873 (positive rate 0.0472)
submission: 60654 rows | 2408 positives | rate 0.0397
```

---

## Configuration knobs

All at the top of `pstu_train.py` / the first code cell of `pstu_train.ipynb`.

| Setting | Default | Notes |
|---|---|---|
| `TARGET_METRIC` | `"binary_f1"` | already confirmed correct by the all-zeros LB probe |
| `SEEDS` | `[42, 1337, 2026]` | 3-seed averaging. `[42]` is ~3× faster for ≈−0.002 AUC. |
| `PLATEAU_TOL` | `0.005` | plateau width for cut selection ≈ measured per-fold noise |
| `USE_NATIVE_CATEGORICAL` | `False` | LightGBM native categorical splits — see below |
| `HOLDOUT_FRAC` | `0.15` | fraction of train rows (most test-like) held out for arm search |
| `SKIP_ARM_SEARCH` | `False` | `True` forces keep_all/no-weight and skips straight to the final CV — faster, but loses the shift-aware config choice |
| `SMOKE_TEST` | `False` | subsamples to 8k rows for a fast pipeline check |
| `N_FOLDS` | `5` | pinned split, shared across seeds and the arm search |

`SCALE_POS_WEIGHT` is no longer a manual knob — section 7 now searches it automatically
(`none` vs the measured neg:pos ratio, ~24.3) alongside the feature-drop arms, scored on the
shift holdout, and sets it for you. `FEATURE_ARMS` (keep_all / drop `feat_182` / drop the top-5
shift-driving columns) is the equivalent list for feature selection — edit that dict directly if
you want to try a different candidate set.

### The metric — resolved: binary F1

The competition page said both "F1 Score" and "Macro F1". **Settled**: `all_zeros_submission.csv`
scored public LB **0.0000000**, matching the measured binary-F1 floor exactly (vs macro-F1's
0.4899) → **the grader uses binary F1.** `TARGET_METRIC = "binary_f1"` is already the default;
no config change needed.

The training notebook still prints both floors every run as a sanity check.

### `USE_NATIVE_CATEGORICAL`

Off by default. Turning it on passes the ordinal code columns to LightGBM as true categoricals
instead of numbers. It can help — `PRD_*`/`SEG_*` labels have no meaningful numeric order — but
`feat_142` has 2,333 levels over 76,020 rows, i.e. **~1.3 positives per level**, which overfits
readily. Treat it as an A/B, not an upgrade: run both, keep the winner, and only believe a
difference larger than the ±0.006 fold noise.

---

## Common Kaggle failures

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train.csv not found` | competition data not attached | Add Data → Competitions → PSTU Data Thon 2026 Vol-1 |
| `backend: histgbm` on Kaggle | LightGBM import failed | `!pip install lightgbm` in the first cell, or accept the ~0.005 AUC loss |
| `artifacts.joblib not found` | artifacts dataset not attached | Step 2 above |
| Notebook times out | arm search + 3 seeds × 5 folds on a slow kernel | set `SKIP_ARM_SEARCH = True` and/or `SEEDS = [42]` |
| Arm search picks `keep_all`/`none` every time | the shift may genuinely not be fixable by dropping these 5 columns | expected outcome sometimes — `feat_182` is a strong predictor, not just a shift driver; check `arm_search.csv` to see how close the arms were |
| `implausible positive rate` assertion | threshold/model mismatch | check the OOF vs submission rate both print near 0.044 |
| Submission rejected | wrong column names or row count | the validator asserts all of this before writing; re-read its message |

---

## What to check before the deadline

- [x] All-zeros probe submitted → binary F1 confirmed (LB 0.0000000), `TARGET_METRIC` already correct
- [x] Run-1 (real LightGBM, pre-arm-search) submitted → LB 0.1849, confirmed OOF/LB gap is real
- [ ] Training notebook shows `backend: lightgbm` and `droppable columns: 44`
- [ ] `cutoff_curve verified against sklearn` appears in the output
- [ ] `adversarial train/test AUC` near 0.57–0.60 and the 6-arm search table both print
- [ ] Note which arm won (`winning config:` line) and its `shift_holdout_f1` (used for selecting the winning arm)
- [ ] `submission.csv` has 60,654 rows and a plausible positive count (varies with winning arm, typically ~2,400–2,900)
- [ ] Artifacts published as a Kaggle dataset
- [ ] Inference notebook prints `REPRODUCTION OK`
- [ ] Inference notebook committed and submitted as the required deliverable
- [ ] Both notebooks run top-to-bottom from a fresh session with no manual cell ordering
- [ ] After submitting: compare LB against run-1's LB (0.1849) to verify whether the shift-aware arm search (`drop_top5_shift`) improved generalization on public LB
