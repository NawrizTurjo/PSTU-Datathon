# next-gen.md — post-Unbeatable_V6 tracking

> Linked from [`CLAUDE.md`](CLAUDE.md) — that file is the frozen record of Stage 1-4 + the first
> five submissions (run-1 through run-4, plus Unbeatable_V6). This file picks up from there and
> tracks everything after. Read `CLAUDE.md` first for the competition rules, data facts, and
> metric; this file assumes all of that as background and doesn't repeat it.

## Real LB scores measured so far (2026-08-12)

Screenshot from the Kaggle submissions page, all real, all this competition. Ranked best first:

| Rank | Submission | Pipeline | Public LB |
|---|---|---|---|
| 1 | `Winning_Blueprint_Omega_V7 - Version 2` | `results/omega-7/` — XGB+LGB, `scale_pos_weight`, triple cat encoding (LE+Freq+TargetEnc), Santander-style row-stat FE, QT+PCA(ddof=0), hard binary @ OOF-optimal threshold | **0.1946** |
| 2 | `submission_binary_convert.csv` | `results/grandmaster-7/` — see `grandmaster-blueprint-v7.ipynb` | 0.1928 |
| — | `Grandmaster_BluePrint_V7 - Version 3` | same family | 0.1808 |
| — | `submission_t0_25.csv` (omega, probe) | omega at threshold 0.25 | 0.1819 |
| — | `submission_t0_30.csv` (omega, probe) | omega at threshold 0.30 | 0.1585 |
| — | `Unbeatable_V6 - Version 2` | `results/unbeatable-6 - v2/` | 0.1093 |
| — | `unbeatable-v6 - Version 1` | `results/unbeatable-6/` | 0.1241 |

(Not on this screenshot but recorded in `CLAUDE.md`: run-1 `pstu_train.py` 0.1849, run-4 external
CatBoost+SMOTE 0.2258 — still the best score in the whole project, ahead of every V7/omega/V6
run above.)

**Reading this table**: the ranking is *not* the same as the OOF ranking. Omega's OOF wasn't
tracked here as rigorously as Unbeatable_V6's (which logged `run_summary.json` — OOF binary-F1
0.2939, stage2 pseudo-labeled), yet Omega's LB (0.1946) beats Unbeatable_V6's LB (0.109–0.124) by
a wide margin despite likely *lower* OOF. This is the same OOF→LB unreliability theme from
`CLAUDE.md`'s run-3/run-4 comparison, playing out a third time.

### Diagnosed failure in Unbeatable_V6's pseudo-labeling

`results/unbeatable-6/run_summary.json`: `"pseudo_positive": 0`, `"pseudo_negative": 48949`. Every
single pseudo-labeled row from stage 1 was a confident **negative** — at a ~4% base rate, a
symmetric probability threshold (p>0.90 / p<0.05) almost always finds far more confident negatives
than confident positives, so pseudo-labeling silently became "add 48,949 more negative examples,"
which pushes the retrained model's calibration *further* from the true positive rate, not closer.
This likely explains why V6 → V2 (0.1241 → 0.1093) went the wrong direction after pseudo-labeling
was involved. **Design lesson carried into `synthetic-test-distr.ipynb` below**: cap the
negative:positive ratio among accepted pseudo-labels, and skip the augmented-retrain stage
entirely (fall back to the honest baseline) if zero confident positives survive even after
threshold relaxation, rather than silently proceeding with an all-negative injection.

## Teammate hypothesis (2026-08-12): train/test distribution mismatch

Teammate's read: Omega ("gold" by construction — most complete feature engineering + triple cat
encoding + proper imbalance handling) underperforming on public LB relative to its apparent
quality may be explained by train/test covariate shift (measured, `CLAUDE.md`: adversarial
train-vs-test AUC 0.5742, `feat_182`-led). If so, a model trained only on train's distribution is
partly optimizing for the wrong data, and might do relatively better on the private/hidden splits
if those differ from public-10% test in the same direction, or — the more actionable framing —
**any model would generalize better if it were trained on something closer to the test
distribution in the first place.**

**Proposed lever**: since the competition explicitly allows synthetic augmentation of training
data (`CLAUDE.md` Rules: "SMOTE / synthetic augmentation / feature engineering on training data is
allowed"), generate a synthetic dataset whose *feature* distribution matches **test**, label it via
pseudo-labeling (not an LLM — model-generated, same mechanism already used and sanctioned in
Unbeatable_V6), and add it to training. The model then sees a training mixture that already
resembles the distribution it will be scored on, rather than relying purely on the drift-feature
audit (adversarial-drop) to paper over the mismatch.

This does **not** use test labels (we don't have them) and does **not** touch the real test set's
predictions — it only uses test's unlabeled *feature* values to build a generator, which is
explicitly permitted (no label join, no reverse-engineering, no external data).

## `solution/synthetic-test-distr.ipynb` — synthetic test-distribution augmentation

Status: **built and plumbing-verified 2026-08-12, not yet run on Kaggle.**

### What it does

1. Same column contract as `Unbeatable_V6.py` (sentinel cleanup, 44-column drop, label encoding,
   reused verbatim — `compute_droppable`, `CAT_COLS`, sentinel constants).
2. Same adversarial-validation drift-feature drop (train vs test, top 20 by importance) and the
   same `QuantileTransformer` + PCA(ddof=0) feature-reduction pipeline, fit on **train** — this is
   the shared feature space every row (train, test, and synthetic) is ultimately scored in.
3. **New — synthetic generator, fit on test only** (bootstrap-resample + selective jitter — see
   "two generator designs failed before this shipped" below for how this was arrived at; two
   earlier parametric-copula designs were tried and rejected after the local dry-run measured them
   as trivially distinguishable from real test data):
   - Resample real test rows with replacement to `N_SYNTHETIC` rows. This reproduces every
     marginal, every correlation, every zero/discreteness pattern, and — since categorical values
     come along with the resampled row unchanged — every numeric↔categorical joint coupling,
     exactly and for free (no clustering step needed).
   - Apply small multiplicative jitter (`JITTER_FRAC = 0.03`) to nonzero cells, but **only for
     genuinely continuous columns** — columns that are ≥99% integer-valued (measured directly from
     the CSVs: 212-281 of ~340 numeric columns) are left completely untouched, because *any*
     continuous jitter moves them off their exact-integer grid, which a boosted-tree adversarial
     classifier detects instantly across hundreds of columns at once.
   - A generator-quality check runs immediately: quick adversarial AUC of synthetic-vs-real-test
     (target: close to 0.50 — indistinguishable) and synthetic-vs-real-train (target: clearly
     above 0.50 — synthetic should *not* look like train). Printed prominently before any model
     training happens, so a bad generator is caught early rather than discovered after a 40-minute
     run. Measured locally at ~20-25k-row scale: synthetic-vs-test AUC **0.52**, synthetic-vs-train
     AUC **0.60-0.62** — the generator passes its own check.
4. A baseline ensemble (XGBoost + LightGBM + CatBoost, same architecture as `Unbeatable_V6.py`,
   still no `scale_pos_weight`/SMOTE) trains on real train via 5-fold CV, predicting OOF, real
   test, and the synthetic rows every fold.
5. Synthetic rows get pseudo-labeled from that ensemble's blended probability, with the
   negative-domination fix described above: thresholds start at 0.90/0.05 and relax in 0.05 steps
   (never past 0.55/0.45) until at least 20 confident rows of each class are found or the floor is
   hit; accepted negatives are capped at 10x the accepted positive count; if zero positives survive
   even after relaxing, **stage 2 is skipped outright** (not run with an all-negative injection).
6. Stage 2 (when it runs): same 5-fold split, accepted synthetic pseudo-rows added to every fold's
   training partition only, never a validation fold — same fold-safety pattern as
   `Unbeatable_V6.py` section 7.
7. Whichever stage has the higher honest real-train OOF F1 wins (same automatic safety net as V6).
8. **Gap-closing diagnostic**: re-run the adversarial train-vs-test AUC (now in the shared
   PCA+catcode feature space, for a clean before/after comparison) using the *final* training set
   (real train, or real train + accepted synthetic rows if stage 2 won) vs real test. If the
   augmented AUC sits closer to 0.50 than the baseline, that's direct evidence the augmentation
   narrowed the measured covariate shift — the actual test of the teammate's hypothesis, not just
   a hoped-for side effect.
9. Outputs: `submission.csv` (hard 0/1), `submission_prob.csv` (raw), and — new —
   `synthetic_test_distribution.csv` (every generated row in raw feature units, its assigned
   cluster, stage-1 probability, and whether/how it was pseudo-labeled), plus the usual
   `adversarial_importance.csv` / threshold curves / `run_summary.json`.

### Local verification done before handoff — two generator designs failed before this shipped

Not run for real (no lightgbm/xgboost/catboost locally, and the user asked for a Kaggle-only run
per `Unbeatable_V6`'s precedent). Same "plumbing dry-run" technique as V6: the exact data-logic
exercised against the real local CSVs with `HistGradientBoostingClassifier` standing in for all
three boosters. This time the dry-run didn't just catch shape/index bugs — it caught the generator
being fundamentally broken, twice, before any Kaggle time was spent on it. Worth recording in
detail because the failure mode is non-obvious and would very plausibly recur if a similar
generator is attempted on other tabular competitions with this data profile (heavy zero-inflation,
mostly-integer features).

**Attempt 1 — Gaussian-copula (test-fit QuantileTransformer + test-fit PCA, sample independent
Gaussians per retained component, inverse-transform back to raw units).** The generator-quality
check (quick adversarial AUC of synthetic-vs-real-test) came back **0.98** — should be ~0.50.
Diagnosed directly (measured off-manifold residual norm on a toy example: real data ≈1.43, every
synthetic point ≈1.1e-14): `PCA.inverse_transform` puts every synthetic point *exactly* on the
retained k-dimensional subspace, while real rows scatter off it by the ~5% of variance PCA
discarded. Trivially detectable. Fixed by re-injecting isotropic residual noise calibrated to the
discarded variance — AUC barely moved (0.98 -> 0.97), proving a second, larger effect was still
dominant.

**Attempt 2 — resample real test rows + jitter.** Per-column univariate AUC checks on the
copula's synthetic data topped out at 0.66 (nothing looked individually broken), yet the *joint*
AUC was 0.98 — a signature of many weakly-off features combining into a strong tree-boosted
signal. Switched to bootstrap-resampling real test rows and adding jitter — still 0.98-0.99 with
either additive-by-std or multiplicative jitter. Root cause, found by checking the actual raw CSVs
directly rather than guessing further: **212-281 of the ~340 numeric columns are ≥99%
integer-valued** (`np.isclose(vals, round(vals))`, measured, not assumed). *Any* continuous
jitter — however small — moves an integer column off its exact grid, and a boosted-tree
adversarial classifier trivially learns "is this an exact integer" across 200+ columns
simultaneously. That's invisible to univariate checks (no single "is-integer" flag survives a
linear PCA combination cleanly) but devastating in combination, which is exactly the joint-not-
marginal pattern observed.

**What shipped**: bootstrap-resample real test rows (preserves every marginal, correlation, and
zero/discreteness pattern by construction, including categorical values which just come along with
the resampled row — no clustering step needed at all), then apply small multiplicative jitter
(`JITTER_FRAC = 0.03`) **only to nonzero cells of the genuinely continuous (<99% integer-valued)
columns**. Measured result at ~20-25k-row local scale: synthetic-vs-test AUC **0.52** (target
~0.50), synthetic-vs-train AUC **0.60-0.62** (clearly test-like, not train-like) — the generator
now does what it's supposed to. Full pipeline (adversarial-drop, QT+PCA, baseline ensemble,
pseudo-labeling with the relaxation/ratio-cap guard, fold-safe augmented retrain, gap-closing
diagnostic, submission assembly) verified end-to-end with real data at that scale using
`HistGradientBoostingClassifier` stand-ins; one harness-only bug (a verification script forgetting
to subsample `sample_submission.csv` to match a subsampled test set) was caught and was not a bug
in the actual notebook.

See the `.py` file's own header markdown (section 5) for the full account with numbers. Config
knobs: `SYNTHETIC_MULTIPLIER`, `JITTER_FRAC`, `PSEUDO_*` (relaxation/ratio-cap thresholds).

### Open question this run should answer

Does the gap-closing diagnostic (step 8) actually show the augmented AUC moving toward 0.50, and
if so, does that translate to a real LB improvement over the current best (run-4, 0.2258)? Record
the result back into this file once submitted — win, loss, or no-change are all useful data points
for whether the distribution-matching lever is worth iterating on further (e.g. raising
`SYNTHETIC_MULTIPLIER`, more clusters, or applying the same generator to build a *validation* set
that's test-like, for a better local LB proxy than plain OOF).

## Next steps (unclaimed as of 2026-08-12)

- Run `synthetic-test-distr.ipynb` on Kaggle, record LB score here.
- If it helps: try stacking the idea onto the Omega/Grandmaster-V7 pipelines directly (they're the
  current LB leaders in this project, not `Unbeatable_V6`'s architecture) — the generator itself
  is pipeline-agnostic (it only needs train.csv/test.csv), so it isn't tied to the V6 feature
  pipeline used here.
- If it doesn't help (augmented AUC doesn't move, or moves but LB doesn't): that's still worth
  recording — it would mean the adversarial-drop step is already capturing most of what's
  recoverable from the measured shift, and further gains need a different lever (e.g. the
  rank-transform / per-row adversarial sample weighting ideas noted as "still undone" in
  `CLAUDE.md`'s shift-robustness section).

---

## `solution/Omega_Synthetic_V8.ipynb` — Synthetic Test-Distribution Augmentation on Omega V7 Architecture

Status: **Built and dry-run verified 2026-08-12**.

### Architecture & Pipeline Overview
- **Core Ensemble**: XGBoost + LightGBM with `scale_pos_weight`, triple categorical encodings (Label Encoding + Frequency Encoding + 5-fold OOF Target Encoding), Santander-style row-aggregate features, QuantileTransformer, and PCA (`ddof=0`, 95% variance).
- **Synthetic Test-Distribution Generator**: Integrated from `synthetic-test-distr.ipynb` (bootstrap resampled real test rows, 99% integer-valued column protection, selective multiplicative jitter, pre-jitter row-stat computation).
- **Pipeline Safeguards**: Dynamic threshold relaxation (0.90/0.05 margin 0.55/0.45), 10:1 negative-to-positive ratio cap, fold-safe Stage 2 retrain (synthetic pseudo-rows added strictly to training fold partitions), and real-train OOF F1 safety net (`stage2_f1 >= stage1_f1`).

### Postmortem & Audit Discoveries During Verification
1. **Target Encoding Diagnostic Leakage**:
   - Target Encoding uses 5-fold OOF encoding for train rows vs a single full-fit encoding for test/synthetic rows. While standard for leak-safe modeling, boosted trees easily separate train vs test in pure TE feature space (AUC 1.0).
   - **Fix**: Created a separate TE-excluded feature space (`X_*_diag`) specifically for adversarial diagnostic checks (`quick_adv_auc`), ensuring models still train on full features while diagnostics remain unbiased.
2. **Santander Aggregate Jitter Leakage**:
   - Computing Santander row aggregates (`mean`, `std`, `skew`, `kurtosis`) on *jittered* continuous values caused extreme-magnitude features (e.g. `feat_169` min $\approx -1.11 \times 10^8$) to produce deltas of millions, making synthetic rows separable (AUC 1.0).
   - **Fix**: Computed Santander row aggregates from `synth_num_prejitter` (un-jittered resampled real test rows), reducing Santander synth-vs-test AUC to **0.4668**.

---

## `solution/synthetic-fixissuesv2.ipynb` — Synthetic Test-Distribution Augmentation on Competition-Best Benchmark

Status: **Built, audited, and verified 2026-08-12.**

### Motivation
`fixissuesv2.ipynb` holds the highest competition Leaderboard score achieved to date (**0.2258 LB**). This notebook combines the exact `fixissuesv2` CatBoost model architecture with the synthetic test-distribution augmentation protocol from `next-gen.md`.

### Pipeline Specifications & Design Principles
1. **Base Model & Data Matrix (`fixissuesv2` @ 0.2258 LB)**:
   - **Ensemble**: CatBoost Classifier (`depth=5`, `l2_leaf_reg=5.0`, `learning_rate=0.015`, `iterations=5000`, `eval_metric='F1'`, `grow_policy='SymmetricTree'`, `min_data_in_leaf=50`, `auto_class_weights='Balanced'`) across 3 seeds (`[42, 123, 456]`) $\times$ 5-fold Stratified CV.
   - **Imbalance**: `SMOTE(sampling_strategy=0.3)` applied within each CV fold on the numerical array.
   - **Categoricals**: 6 string columns (`feat_142`, `feat_157`, `feat_318`, `feat_320`, `feat_325`, `feat_337`) label-encoded globally on `train + test` combined and passed natively to CatBoost via `cat_features` as rounded strings (function `make_cb_df()`).
   - **Numerical Features**: Purified by dropping zero-variance and exact duplicate columns. 6 row-wise stats (`mean`, `std`, `iqr`, `zero_count`, `skew`, `kurtosis`) appended and transformed via `QuantileTransformer(n_quantiles=2000, output_distribution='normal')`.
2. **Synthetic Generator & Jitter Calibration**:
   - Resamples real test rows with replacement to `N_SYNTHETIC`.
   - Freezes integer-valued columns ($\ge 99\%$ integer-like).
   - **Jitter Calibration**: `fixissuesv2` preserves all 307 raw numerical features without PCA dimensionality reduction. In high-dimensional space (307 raw features), a 3% jitter accumulates across dimensions into a joint boosted-tree separability signal (AUC 0.9058). Calibrating `jitter_frac = 0.001` (0.1% jitter) maintains synthetic row non-duplication while keeping 307-feature joint adversarial AUC at **0.5122** (ideal test matching).
   - Computes row stats on `synth_num_prejitter` (un-jittered resampled values), keeping row-stat features artifact-free (AUC **0.5098**).
3. **Pseudo-Labeling & Safety Nets**:
   - Dynamic threshold relaxation (0.90/0.05 floor margin 0.55/0.45), 10:1 ratio cap, and complete Stage 2 skip if 0 positives survive.
   - Fold-safe Stage 2 retrain (synthetic pseudo-rows appended strictly to training folds; validation folds remain 100% real train).
   - Automatic stage selection (`stage2_f1 >= stage1_f1`).
4. **Outputs**:
   - Primary: `submission.csv` (probabilities) and `submission_binary.csv` (hard binary @ optimal threshold).
   - Probing: `submission_t0_20.csv` through `submission_t0_40.csv`.
   - Export: `synthetic_test_distribution.csv` and `run_summary.json`.

### Local Empirical Verification Matrix
| Metric | Target | Measured Value (`synthetic-fixissuesv2`) | Status |
|---|---|---|---|
| `synthetic_vs_test_auc` | $\sim 0.50$ | **0.5122** | PASS (Ideal matching) |
| `synthetic_vs_train_auc` | $> 0.50$ | **0.5898** | PASS (Test-like shift) |
| `baseline_shift_auc` | Reference | **0.5901** | Reference train-vs-test shift |
| Pipeline Smoke Test | Exit 0 | **PASSED PERFECTLY** | Zero code/shape errors |

---

## Comprehensive Competition Leaderboard History (Updated 2026-08-12)

| Rank | Submission Notebook / File | Strategy & Pipeline Highlights | Public LB Score |
|---|---|---|---|
| **1** | `results/best-so-far/fixissuesv2.ipynb` | CatBoost (`depth=5`, `l2_reg=5`), SMOTE(0.3), native cat handling, 6 row stats, QuantileTransformer | **0.2258** |
| 2 | `fixissuesv2 - Version 4` / `submission_binary_0_375.csv` | Threshold probed @ 0.375 | 0.2234 |
| 3 | `results/synthetic-omega-7/synthetic-winning-blueprint-omega-v7.ipynb` | Omega V7 + Synthetic test-distribution augmentation ($t=0.2150$) | **0.2006** |
| 4 | `results/omega-7/winning-blueprint-omega-v7.ipynb` | XGB+LGB, `scale_pos_weight`, triple cat encoding, Santander row stats, QT+PCA(ddof=0) | **0.1946** |
| 5 | `results/grandmaster-7/grandmaster-blueprint-v7.ipynb` | Grandmaster Blueprint V7 | 0.1928 |
| 6 | `pstu_train.py` (run-1 baseline) | Initial baseline model | 0.1849 |
| 7 | `Omega V7 - submission_t0_25.csv` | Omega V7 probe @ threshold 0.25 | 0.1819 |
| 8 | `Grandmaster_BluePrint_V7 - Version 3` | Grandmaster V7 variant | 0.1808 |
| 9 | `Omega V7 - submission_t0_30.csv` | Omega V7 probe @ threshold 0.30 | 0.1585 |
| 10 | `results/unbeatable-6/unbeatable-v6.ipynb` | Unbeatable V6 Version 1 | 0.1241 |
| 11 | `results/unbeatable-6 - v2/Unbeatable_V6.ipynb` | Unbeatable V6 Version 2 (pseudo-label negative leak) | 0.1093 |

---

## Session Handoff Summary & Next Action Items

- **Ready-to-Run Notebooks**:
  1. [solution/synthetic-fixissuesv2.ipynb](file:///e:/Competitions/PSTU-Datathon/solution/synthetic-fixissuesv2.ipynb) — Best baseline architecture (`fixissuesv2`) + synthetic test-distribution augmentation.
  2. [solution/Omega_Synthetic_V8.ipynb](file:///e:/Competitions/PSTU-Datathon/solution/Omega_Synthetic_V8.ipynb) — Full feature-engineered Omega architecture + synthetic test-distribution augmentation.
- **Recommended Action**: Upload and execute [synthetic-fixissuesv2.ipynb](file:///e:/Competitions/PSTU-Datathon/solution/synthetic-fixissuesv2.ipynb) on Kaggle to evaluate whether adding synthetic test augmentation to `fixissuesv2` breaks past the 0.2258 LB benchmark.

