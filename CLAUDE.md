# CLAUDE.md — project context

> ## 🆕 THIS IS A NEW COMPETITION (2026-08-09)
>
> The previous competition's dataset was withdrawn for leaks. The organizers replaced it with
> an **entirely different problem**: financial account-instability prediction, 350 anonymized
> features, **F1 metric**, different submission format.
>
> All prior work moved to `works.old/`. **Nothing in `works.old/` is factually valid for this
> competition** — different domain, different data, different metric. Only the *tooling
> patterns* carry over (see [Reusable from works.old](#reusable-from-worksold)).

## Competition

[PSTU Data Thon 2026 Vol-1](https://www.kaggle.com/competitions/pstu-data-thon-2026-vol-1/)

Predict whether a financial account will be flagged at-risk (`TARGET = 1`).
Binary classification, 350 anonymized features, **F1 score**.

**Timeline (GMT+6):** starts 9 Aug 2026 18:00 · **final submission 13 Aug 2026 18:00** ·
private LB 13 Aug 18:30 · inference notebook due 13 Aug 23:59 · winners 15 Aug.

**Scoring is only 50% leaderboard:**

| Component | Weight | Notes |
|---|---|---|
| Public LB | 10% | live, on **10%** of test data — small, do not overfit to it |
| Private LB | 40% | on 50% of test data |
| **Hidden test** | **40%** | 40% of *completely unseen* data, run via your inference notebook |
| Presentation + code + format | 10% | |

The hidden-test component means **generalization and a reproducible inference notebook matter
as much as leaderboard score.** A notebook that fails to run, or is non-deterministic, forfeits
40%.

---

## Measured facts about the new data

From a first-pass profile (2026-08-09). Re-verify in the full EDA, but these are solid:

| | |
|---|---|
| `train.csv` | **76,020 rows × 351 cols** (350 `feat_*` + `TARGET`). 128 MB, ~235 MB in RAM |
| `test.csv` | **60,654 rows × 351 cols** (350 `feat_*` + **`id` as the LAST column**) |
| `sample_submission.csv` | 60,654 rows, `id,TARGET`; ids match `test.csv` **in the same order** |
| Positive rate | **3.957%** (3,008 of 76,020) |
| Missing values | **none** (no `NaN` in train or test) |
| Feature dtypes | 212 int, 132 float, **6 object (string)** |

### ⚠️ Six features are categorical strings, not numbers

The competition description says "350 numerical features". **It is wrong.** These are
high-cardinality categorical codes:

| Column | Levels (train) | Prefix | Test-only levels |
|---|---|---|---|
| `feat_142` | 2,333 | `PRD_*` | **55** |
| `feat_325` | 1,710 | `SEG_*` | **27** |
| `feat_157` | 627 | `PRV_*` | **8** |
| `feat_320` | 119 | `CH_*` | 0 |
| `feat_337` | 39 | `OFC_*` | 0 |
| `feat_318` | 12 | `PERF_*` | 0 |

Consequences: `pd.read_csv` yields `object` dtype and most models will crash or silently
mis-handle them. **Unseen categories appear at inference** — any encoding must have a fallback
for levels never seen in training, or the hidden-test run will break.

### Measured findings — Stage 1 EDA complete (2026-08-09, see `dataset_exploration/README.md`)

Full EDA now done (`dataset_exploration/01`–`09` + README). These numbers are **measured**,
superseding the first-pass estimates below where they differ:

- **`-999999` sentinel in `feat_109`** (exactly one column) confirmed — 116 train rows / 89
  test rows (~0.15%). Treat as missing.
- `feat_169`'s min ≈ **-1.11e8** is **not** a sentinel — just a genuine heavy-tailed continuum
  of large-magnitude values, no repeated constant. No special handling needed beyond normal
  scaling/robustness.
- **NEW — second sentinel `9999999999`** (1e10−1) spans **23 columns** (`feat_11, feat_21,
  feat_26, feat_30, feat_31, feat_36, feat_74, feat_77, feat_96, feat_124, feat_135, feat_144,
  feat_149, feat_158, feat_171, feat_196, feat_204, feat_226, feat_301, feat_315, feat_330,
  feat_336, feat_340`) — the classic Santander `delta_imp_*` fingerprint. Not caught by the
  first-pass profile. Treat as missing alongside `feat_109`.
- **44 safely-droppable columns**, not 83: 28 constant in *both* train and test (42 constant in
  test only — keep the other 14, they vary in train), plus **16** exact-duplicate columns
  across **16 groups** (all size 2). The first-pass "55 redundant / 20 groups" figure does not
  reproduce under exact row-value matching — a looser `|corr| > 0.999` check gets closer
  (49 redundant / 36 groups, 77 combined) but is threshold-dependent, not exact. Use 44 as the
  safe drop list; treat corr-based near-duplicates as feature-selection candidates only.
- **252 of the 344 *numeric* columns are ≥90% zero** (350 total feature columns minus the 6
  categorical ones = 344; the first pass's "252 of 350" had the wrong denominator).
- **Zero duplicate rows and zero label conflicts** confirmed exactly (train, test, and across
  train/test on features alone).
- **Adversarial validation: train vs test AUC = 0.5742 ± 0.0027** — a real, moderate covariate
  shift (not the ~0.50 "iid" case). `feat_182` dominates the shift (importance ~0.17, ~2.4x the
  next feature); 93/344 numeric columns have out-of-range test values. Watch the CV-vs-LB gap.
- **Leak diagnostics: clean.** Row-index AUC 0.5047 (no leak), best single-feature AUC 0.6986
  (`feat_175`, no magic feature), capacity test shows train AUC 1.0 collapsing to held-out AUC
  0.59 (real ceiling, no deterministic rule) — confirms there is *no* proven noise ceiling this
  time and higher scores may genuinely be reachable, as the first pass suspected.
- **Honest baseline** (`HistGradientBoostingClassifier`, 5-fold OOF, cleaned columns, sentinels
  → NaN): OOF AUC **0.8868** (0.8869 ± 0.0060 per-fold), tuned macro-F1 **0.6786** / binary-F1
  **0.3841** at t≈0.19 (vs macro-F1 0.5791 / binary-F1 0.1777 at naive t=0.5). This is the floor
  Stage 2/3 needs to beat.

### Lineage note — do NOT act on it

76,020 train rows with exactly 3,008 positives (3.957%) and a `-999999` sentinel is the exact
fingerprint of the public **Santander Customer Satisfaction** dataset. Test `id`s run 0–75,817,
matching that competition's test indices.

**External datasets are explicitly PROHIBITED and joining labels is instant disqualification**
(see Rules). Also, Santander's test labels were never public, so there is nothing to join
anyway. Record the lineage only because it predicts *structure* (extreme sparsity, the
sentinel, row-aggregate features working well) — never as a data source.

---

## The metric — ✅ RESOLVED (2026-08-09): it's binary F1

The competition page was **self-contradictory**:

- Evaluation section: "Submissions are evaluated using the **F1 Score**"
- Submission section: "converted to binary predictions using a threshold of 0.5 before
  computing the **Macro F1** score"

**Settled by the diagnostic probe:** `all_zeros_submission.csv` scored public LB **0.0000000**.
That matches the measured binary-F1 floor exactly (0.0000, vs macro-F1's 0.4899) — **the grader
uses binary F1.** `TARGET_METRIC = "binary_f1"` in `solution/pstu_train.py` was already the
default; no code change was needed.

These behave very differently at a 3.96% positive rate:

| Submission | Binary F1 | Macro F1 |
|---|---|---|
| all zeros | **0.0000** ← confirmed live | **0.4899** |
| all ones | 0.0761 | 0.0381 |

Binary F1 is the harsher, less forgiving of the two — it does not reward getting the negative
class right (which is nearly free at 96% of rows), so precision/recall on the rare positive
class is all that counts. Every threshold-selection decision in `ideas/01-threshold-engine/`
and `solution/pstu_train.py` already targets this by default; nothing downstream needs to
change on account of this result.

### The threshold is still yours to control

The grader applies a fixed 0.5 cut to whatever you submit — but **you choose the numbers you
submit**, so threshold optimization is fully available and is still the biggest single lever:

- Simplest: submit **hard `0`/`1`** (the sample submission uses integers).
- Or rank-transform your scores so that your chosen operating point lands exactly at 0.5.

Because the metric is **F1 only** (no AUC term), *only the binary decision matters*. Ranking
quality matters solely through the split it produces. This is a bigger shift than it looks:
the old competition gave 25% of the score to AUC; here, ranking earns nothing on its own.

---

## Submission format

```
id,TARGET
3496,0
17271,0
44259,0
```

- `id` comes from the **`id` column of `test.csv`** (last column). It is **not** `0..n-1` and
  is **not** contiguous — never regenerate it with `range()`.
- `test.csv` row order already matches `sample_submission.csv`, so
  `sub = sample_submission.copy(); sub["TARGET"] = preds` is safe.
- 60,654 rows + header. Two columns only.
- `TARGET` may be a hard `0`/`1` (preferred, see above).

---

## Rules that constrain the solution

- ❌ **No external datasets.** Only the provided files.
- ❌ **No test-set tampering or label reverse-engineering.** Instant disqualification.
- ❌ **Do not generate target values with an LLM.** Explicitly banned.
- ✅ SMOTE / synthetic augmentation / feature engineering **on training data** is allowed.
- ⚠️ Pre-trained models allowed **only with disclosure**.
- ✅ **Inference notebook is mandatory** and must reproduce submitted predictions
  **deterministically** — fix every seed, pin the fold split, and save model artifacts.
  Top 20 must also submit code + presentation.

---

## The workflow to follow (same four stages as last time)

This is the process that worked previously. Repeat it on the new data.

### Stage 1 — Dataset exploration → `dataset_exploration/` — ✅ DONE (2026-08-09)

Complete. 9 numbered, idempotent, self-contained scripts + `dataset_exploration/README.md`
(full findings summary and corrections against this file's first-pass numbers). Results are
folded into "Measured findings" above. Re-run any script any time — none mutate the source CSVs.

1. Schema / dtype classification, categorical vs numeric split
2. Categorical deep-dive: cardinality, train-vs-test level overlap, unseen-level handling
3. Numeric profiling, sentinel hunt (`-999999`, `feat_169`, **and the newly found `9999999999`**), zero-inflation ratios
4. Constant + duplicate column detection (44 exact droppable, not the estimated 83)
5. Adversarial validation (train vs test covariate shift — AUC 0.5742, real shift)
6. Duplicate rows / label conflicts (confirmed zero, as the first pass guessed)
7. Metric behaviour: binary vs macro F1, threshold sweep, degenerate floors
8. Leak diagnostics — row-index leak, best single-feature AUC, capacity test, dup-label
   consistency. All clean; no leak, real generalization ceiling confirmed.
9. Honest baseline (HistGradientBoosting, 5-fold OOF) — OOF AUC 0.8868, anchors Stage 2/3.

### Stage 2 — Idea-generation prompt → `prompt.md` — ✅ DONE (2026-08-09)

Self-contained master prompt, pre-loaded with every Stage 1 measured finding **and the
confirmed negatives** (no leak, no duplicate rows, `feat_169` not a sentinel, the 83-droppable
figure unreproducible) so Opus builds on them instead of re-proposing them.

### Stage 3 — Idea generation → `ideas/` — ✅ DONE (2026-08-09)

Seven priority-ordered direction folders + `dead-ends/`, each with what it is, why it should
work (citing measured evidence), concrete steps, Kaggle cost, honest expected gain with stated
confidence, and an abandon condition. Start at [`ideas/README.md`](ideas/README.md).

**The roadmap's three headline conclusions:**

1. **The operating point outweighs the model.** Submitting the baseline's raw probabilities
   against the grader's fixed 0.5 cut scores binary-F1 **0.1777**; submitting hard labels at the
   tuned cut scores **0.3841** — a **+0.207** swing from a mechanical choice, larger than every
   modelling idea combined. The F1 plateau is flat (t ∈ [0.15, 0.21] all within 0.005 of peak,
   well inside the ±0.0060 fold noise), so pick the **plateau centre, not the OOF argmax**.
   Sanity check every submission against ≈4.4% predicted-positive rate (≈2,700 of 60,654).
2. **Modelling headroom is modest.** Untuned HistGBM already hits OOF AUC 0.8868 with zero
   feature engineering. Real LightGBM + tuning + row-aggregate features is worth maybe
   +0.005–0.015 AUC total. Plan for a few points of F1, not a transformation.
3. **Ship early.** Day 1: foundation → threshold engine → a valid end-to-end submission →
   inference-notebook skeleton. Only then improve the model. The inference notebook is 40% of
   the grade and its most likely failure mode (unseen categorical levels) is a *measured*
   property of this data.

### Stage 4 — Runnable notebook → `solution/` — ✅ DONE (2026-08-09)

| File | Purpose |
|---|---|
| `solution/pstu_train.py` / `.ipynb` | training, operating-point selection, `submission.csv` + `artifacts.joblib` |
| `solution/pstu_inference.py` / `.ipynb` | **mandatory deliverable** — loads artifacts, deterministic, reproduction-asserted |
| `solution/KAGGLE_INSTRUCTIONS.md` | step-by-step Kaggle run guide, config knobs, failure table |

Implements ideas 00/01/02/03/05/06. Smoke-tested locally with `HistGradientBoostingClassifier`
(both the subsampled and the full-test path); `cutoff_curve` is asserted against sklearn's
`f1_score` to 1e-9 at runtime, and the inference notebook asserted **60,654/60,654 predictions
bit-identical** to the training run.

**Verified full local run** (HistGBM stand-in, 3 seeds × 5 folds): droppable columns **44** ✓,
seed-averaged OOF AUC **0.8887**, binary-F1 **0.3853** / macro-F1 **0.6780** at t=0.169,
submission 2,661 positives (rate 0.0439). Inference reproduced it exactly, and its unseen-level
counts (89 / 11 / 47) match the Stage-1 EDA figures precisely.

⚠️ **Honest result: the engineered features + seed averaging did not clearly beat the Stage-1
baseline.** OOF AUC 0.8868 → 0.8887 (+0.0019) and binary-F1 0.3841 → 0.3853 (+0.0012) are both
*inside* the ±0.006 fold-noise band. Read it as "no regression," not a win. The pipeline's real
value is the operating-point mechanic (+0.2129 binary-F1 vs a naive 0.5 cut) plus the
reproducibility guards. Switching to real LightGBM on Kaggle is the next thing likely to move
the score (`ideas/02-gbdt-core/` estimates +0.002–0.008 AUC).

⚠️ **Test predicted-positive rate runs below OOF** — 0.0439 vs 0.0511 (ratio 0.86), consistent
with the measured 0.5742 adversarial shift. The notebook warns on this. If LB underperforms,
select the cut by *target positive rate* rather than by threshold value.

### First real Kaggle submission — ✅ MEASURED (2026-08-09)

`solution/pstu_train.ipynb` run on Kaggle, confirmed `backend: lightgbm` (real LightGBM, not
the local stand-in), 3 seeds × 5 folds, submitted as `run-1-PTSU.csv`:

| | OOF | Public LB |
|---|---|---|
| AUC | 0.8920 | — |
| Binary F1 | **0.3954** (t=0.1736) | **0.1849** |
| Predicted-positive rate | 0.0494 | 0.0428 (2,597 / 60,654), ratio **0.87** |

(Full output archived at `results/run-1/` — notebook, submission csv, `run_summary.json`,
`threshold_curve.csv`.)

The rate-shrinkage ratio (0.87) matches the local HistGBM run's 0.86 almost exactly — that
pattern is now confirmed with the real model, not just the stand-in. It does **not** explain a
~2× binary-F1 drop on its own; most of the OOF→LB gap is genuine generalization loss, consistent
with the measured adversarial train/test AUC (0.5742) but larger in effect than that AUC alone
would suggest.

**Leaderboard context at time of writing:** top public-LB entry **0.2955**, six teams clustered
0.20–0.30. This is real evidence — the `ideas/README.md` "realistic ceiling ~0.42–0.45" figure
(written before any submission existed) was too optimistic and has been **revised down to
~0.28–0.32**. Plan remaining effort around closing a ~0.02–0.06 F1 gap, not chasing a 0.10+ one.

**Consequence: `ideas/04-shift-robustness/` is promoted** from "low-medium confidence, do late"
to "medium confidence, do next" — its step-1 diagnostic (LB well below OOF → proceed to steps
2–4) is exactly the pattern this submission confirmed. Note: **clipping to train range (idea
04's arm C) was already the default** in `pstu_train.py`'s `build_features`, so the 0.1849 score
already includes that mitigation — the remaining levers were the `feat_182`-family feature
audit and, as a last resort, adversarial sample weighting. See below — both are now implemented.

### Pipeline rebuilt: shift-aware configuration search — ✅ DONE (2026-08-09)

`solution/pstu_train.py` restructured (sections renumbered 1–11) to operationalize
`ideas/04-shift-robustness/`'s step 2 (feature audit), folded together with idea
`02-gbdt-core`'s imbalance-weighting A/B, under one shift-aware evaluation methodology —
**automatically**, rather than as a manual A/B a human has to run and interpret. Idea 04's
step 3 (rank-transform) and step 4 (per-row adversarial sample weighting) remain undone,
candidates if this proves insufficient:

- **Section 5 (new):** an adversarial classifier (train vs test, out-of-fold via
  `cross_val_predict`) scores every train row by how much it resembles a test row. The top 15%
  become a standing **shift holdout** — tracked by index, not removed from training.
- **Section 7 (new):** 3 feature arms (`keep_all`, drop `feat_182`, drop the top-5 shift-driving
  columns) × 2 imbalance arms (none, `scale_pos_weight` ≈ neg:pos ratio) — 6 configs, each
  scored with a fast reduced-estimator CV **on the shift holdout**, not on ordinary OOF. Ordinary
  CV cannot make this choice: every fold looks equally "trainy," so it would reward whichever
  config overfits the very shift that's hurting the real score. The winner is retrained with the
  full 3-seed × 5-fold budget.
- **Section 9:** now also reports **shift-holdout binary-F1** — the final model's F1 restricted
  to the same holdout rows, at the same deployed threshold. This should track real LB more
  closely than the standard OOF number; compare it against the next submission's LB score to
  check whether the proxy is actually predictive.

**This does not guarantee a specific leaderboard score** — the hidden test's true distribution
isn't observable locally, so no local number can promise a public/private/hidden LB outcome.
What changed is that the pipeline now optimizes against the actual measured failure mode (the
confirmed run-1 OOF→LB gap) instead of against an OOF metric already known to overstate LB by
roughly 2×.

**A bug was caught and fixed during smoke testing**, worth recording because it's a subtle
pandas trap: `winner = arm_df.iloc[0].to_dict()` silently coerced `scale_pos_weight=None` to
`NaN` (a DataFrame column mixing `None` and floats upcasts to `float64` with `NaN` for the
missing entries), and `if spw:` in `make_lgb_params` treats `NaN` as truthy — so the "no
reweighting" arm would have silently applied a bogus `scale_pos_weight` if it won. Fixed by
re-deriving `winner["spw"]` from `IMBALANCE_ARMS` directly instead of trusting the DataFrame
round-trip. Caught by the smoke test's printed `final imbalance setting:` line before this ever
reached a real Kaggle run — the value to take from this: **always print resolved config values,
not just intended ones, at run time.**

Smoke-tested (subsampled train, full test file, HistGBM backend): pipeline runs end to end,
`build_features` sha256 re-synced with `pstu_inference.py` after a comment-only edit changed the
hash (the guard did exactly what it's for), and the inference notebook reproduced **60,654/60,654
predictions bit-identical** to the training run. Full-scale local numbers: see
`solution/KAGGLE_INSTRUCTIONS.md`'s "verified local baseline" section once populated from the
full run.

### Run-2: second real Kaggle submission — ✅ MEASURED (2026-08-09)

`solution/pstu_train.ipynb` (the shift-aware-search version above) run on Kaggle, real
LightGBM. Full output archived at `results/run-2/` (notebook, submission csv, `run_summary.json`,
`arm_search.csv`, `threshold_curve.csv`).

| | run-1 (pre-search) | run-2 (arm search) |
|---|---|---|
| Winning config | `keep_all` / no reweighting (only option) | **`drop_top5_shift`** / no reweighting |
| OOF AUC | 0.8920 | 0.8928 |
| OOF binary F1 (tuned t) | 0.3954 (t=0.1736) | 0.3957 (t=0.1830) |
| Shift-holdout binary F1 | *(mechanism didn't exist yet)* | 0.4068 |
| Adversarial train/test AUC | — | 0.5733 (Stage-1 measured: 0.5742 — matches) |
| Submission positives | 2,597 (rate 0.0428) | 2,408 (rate 0.0397) |
| Public LB | **0.1849** | **not yet submitted / no score recorded here** |

The arm search's own comparison table (`arm_search.csv`), all 6 candidates:

| feature_arm | imbalance_arm | quick_full_f1 | shift_holdout_f1 |
|---|---|---|---|
| drop_top5_shift | none | 0.3890 | **0.4127** ← winner |
| drop_top5_shift | scale_pos_weight | 0.3848 | 0.4090 |
| keep_all | none | 0.3913 | 0.4044 |
| drop_top1_feat182 | none | 0.3860 | 0.4040 |
| drop_top1_feat182 | scale_pos_weight | 0.3864 | 0.3994 |
| keep_all | scale_pos_weight | 0.3808 | 0.3894 |

OOF AUC and OOF binary-F1 are essentially unchanged from run-1 (both differences are inside the
measured ±0.006 fold-noise band) — the arm search did not find a config that improves the
*optimistic* in-distribution number, which is expected and fine, since that was never its job.

**⚠️ Audit finding — the shift-holdout metric does not behave as originally hoped, and the
documentation's framing needs correcting.** In every row of both the local smoke-test arm search
and this real run's `arm_search.csv`, **`shift_holdout_f1` is *higher* than `quick_full_f1`**
(e.g. winning arm: 0.4127 vs 0.3890), and the final model's shift-holdout F1 (0.4068) is likewise
higher than its full OOF F1 (0.3957). This is the *opposite* of what would be expected if the
holdout genuinely captured "how hard the shift makes generalization" — run-1's real LB (0.1849)
sits far *below* full OOF, so a metric meant to anticipate that gap should also read low, not
high.

The likely explanation: the shift holdout's positive rate is **0.0472, about 1.19× the overall
0.0396** (rows an adversarial classifier finds test-like happen to skew toward the positive
class in this data). At a fixed threshold, a higher base rate mechanically inflates F1 regardless
of whether the underlying predictions are actually more or less reliable — so the holdout metric
is confounded by class balance, not a clean read on generalization difficulty.

**Practical consequence:** treat `shift_holdout_f1` as useful only for its original, narrower
purpose — *relative* ranking between candidate arms trained and evaluated identically (which is
what section 7 actually uses it for, and that usage remains sound) — not as an absolute estimate
of what LB will read, and not as proof the OOF→LB gap has narrowed. **The real test of whether
run-2's arm search helped is its own LB score once submitted**, compared against run-1's 0.1849.
That comparison is not yet available. `solution/KAGGLE_INSTRUCTIONS.md` and
`ideas/04-shift-robustness/README.md`'s claims that this number "should track real LB more
closely" should be read with this caveat until a real LB-vs-shift-holdout data point exists to
confirm or refute it.

Three guards worth knowing about, each targeting a measured failure mode:
- **`build_features` sha256** is stored in the artifacts and re-checked by the inference
  notebook — silent drift between the two copies of the function becomes a loud failure.
- **Unseen categorical levels** map to a fallback bucket (freq `0.0`, code `-1`) and the count
  is printed per column.
- **Clip to saved train range** at inference, against the measured 0.5742 adversarial shift
  (93 of 344 numeric columns already exceed their train range on the public test).

⚠️ **Kaggle data path is `/kaggle/input/competitions/pstu-data-thon-2026-vol-1/`** (with the
`competitions/` segment), not the path this file originally assumed. Both are probed.

**Run on CPU, not GPU** — see `solution/KAGGLE_INSTRUCTIONS.md`. At this data size CPU LightGBM
is faster, CPU notebooks are unmetered, and the GPU histogram path is not reproducibly
deterministic, which the inference notebook's assertion would catch.

### Run-3 & Run-4: external/alternate-pipeline submissions — ✅ MEASURED (2026-08-10)

Two more real Kaggle submissions, archived at `results/run-3/` and `results/run-4/` (each with
its own `README.md` — full detail there, this is the summary). **Neither is built from
`solution/pstu_train.py`** — both are a separately authored pipeline (LightGBM+XGBoost+CatBoost
blend with heavy feature engineering, later a leaner CatBoost-only rewrite). Recorded here
because they are real, measured data points about this competition's data and metric, even
though they don't share code with this project's own solution.

| | run-3 (V1) | run-4 (V2, fixes run-3) |
|---|---|---|
| Models | LightGBM + XGBoost + CatBoost blend, F1-calibrated | CatBoost only, 3-seed ensemble, no calibration |
| Categorical encoding | 5-fold OOF target encoding | native CatBoost handling (label-encoded) |
| Extra features | 22 row-stats + 28 KMeans-cluster feats + PCA(50) | 6 row-stats only, no KMeans, no PCA |
| Imbalance | SMOTE(0.5) | SMOTE(0.3) + `scale_pos_weight=12` |
| CV | 10-fold | 5-fold |
| OOF/CV F1 | 0.3624 (ensemble) / 0.373 (best single model) | 0.28724 (probability-avg ensemble) |
| Test positive rate vs train (3.96%) | — | **7.05%, i.e. 1.78× train** (inflated, not shrunk) |
| **Public LB** | **0.195681511470** | **0.225758329189** ← best LB measured so far, any pipeline |

Run-4 is a direct, documented fix of run-3 (its own notebook contains a "V1 Post-Mortem" table
diagnosing calibration-shift leakage, target-encoding leakage, SMOTE over-aggression, KMeans
train/test peeking, weak regularization, and a "model-diversity trap" from blending 3 differently
calibrated models). Removing all of that dropped OOF from 0.36 to 0.29 but **raised LB from
0.1957 to 0.2258** — the CV→LB relative gap shrank from ~46% to ~21%. This is an independent
confirmation, from a completely different codebase, of this project's own central finding
(`ideas/README.md`, `ideas/04-shift-robustness/`): on this dataset, a lower/honester OOF that
hasn't absorbed leakage-prone tricks transfers to LB far better than a higher one that has.

**Run-4's 0.2258 is now the best real LB score recorded anywhere in this project**, ahead of the
project's own run-1 (0.1849, `solution/pstu_train.py`). Run-2 (`solution/pstu_train.py`'s
shift-aware arm search, OOF 0.3957) has still not been submitted to LB — see the Run-2 section
above. Ceiling context from `ideas/README.md` (top public-LB team 0.2955, six teams clustered
0.20–0.30 as of 2026-08-09) puts run-4 inside that cluster.

See `results/run-3/README.md` and `results/run-4/README.md` for full configs, root-cause tables,
and "to improve on this later" notes (kept in those files, not acted on here, per the scope of
this documentation pass).

#### Original Stage 4 spec (kept for reference)

One end-to-end Kaggle notebook, plus a `# %%` cell-marked `.py` twin for diffing.
Must include: data loading, preprocessing, categorical encoding, CV harness, models,
threshold/operating-point selection, submission builder **with a format validator**.

Write the source as a `# %%` cell-marked `.py`, then convert it with the saved helper:

```bash
python tools/to_ipynb.py solution/<name>.py solution/<name>.ipynb
```

`tools/to_ipynb.py` splits on `# %%` / `# %% [markdown]` markers and emits valid nbformat-4
JSON. Validate after converting — load the JSON and `compile()` the concatenated code cells.

Smoke-test locally before handing it over — patch in `HistGradientBoostingClassifier` for the
libraries not installed locally (see Conventions). The pattern that worked: read the `.py`,
string-replace the model flags off, inject a scikit-learn stand-in, then `exec` it. That
catches integration bugs the notebook would otherwise hit on Kaggle.

**This time, also produce the mandatory inference notebook**: loads saved artifacts, runs on an
arbitrary test file, writes `submission.csv`, fully deterministic.

---

## Reusable from `works.old/`

Patterns only — never the numbers.

- `works.old/dataset_exploration/*.py` — structure of the EDA scripts (schema classifier,
  constant/duplicate detector, adversarial validation, duplicate-row check).
- `works.old/solution/pstu_kaggle_solution.py` — the CV harness, seed-averaging loop,
  rank-blending, submission validator, and especially the **exhaustive O(n) cut-point
  optimizer** (`cutoff_curve()`): sort once, take cumulative sums, and you get the confusion
  matrix at *every* threshold. Verified against brute force to 1e-9. Adapt its scoring line
  from the old composite to F1.
- `works.old/ideas/` — the *format* of the roadmap, and the general modelling playbook
  (row-wise aggregate features for sparse Santander-lineage data, seed averaging,
  rank blending, plateau-centred threshold choice).

**Do not reuse:** the Bengali text decoder (no such columns now), the 6-component composite
metric decomposition (metric is F1 now), any measured number, or the old dead-ends list.

---

## Conventions

- Data lives in `pstu-data-thon-2026-vol-1/` (train.csv, test.csv, sample_submission.csv).
  Add it to `.gitignore` — currently `.gitignore` only contains `dataset/`.
- On Kaggle the data mounts at
  `/kaggle/input/pstu-data-thon-2026-vol-1/`. Auto-detect the path so the notebook also runs
  locally (the old notebook has a working `CANDIDATE_DIRS` pattern).
- Local environment has `pandas`, `numpy`, `scikit-learn`, `scipy` but **not** lightgbm /
  xgboost / catboost. Kaggle has all three. Use `HistGradientBoostingClassifier` as the local
  stand-in when smoke-testing.
- **Don't enable the GPU for tree models** — at this data size CPU is faster, and Kaggle CPU
  notebooks have no weekly quota (GPU is capped ~30 h/week).
- `train.csv` loads fine with plain `pd.read_csv` (~235 MB in RAM) — no streaming needed this
  time, unlike the old 910 MB file.
- **Fix every random seed** and pin the fold split. The inference notebook must be
  bit-reproducible or the 40% hidden-test component is at risk.

---

## Next session — start here

**All four stages are complete.** The pipeline runs end to end and reproduces itself. What
remains is leaderboard work and the graded deliverables.

1. ✅ **Metric probe done.** All-zeros submission scored **0.0000000** → confirmed **binary F1**.
2. ✅ **First real Kaggle submission done.** Real LightGBM, LB **0.1849**, confirmed OOF→LB gap
   (~0.39–0.40 → 0.1849) — see "First real Kaggle submission" above. Ceiling estimate revised to
   ~0.28–0.32 against a live leaderboard (top entry 0.2955).
3. **Do `ideas/04-shift-robustness/` next**, not last — promoted to position 4 in
   [`ideas/README.md`](ideas/README.md)'s priority table because its trigger condition (LB well
   below OOF) is now confirmed rather than hypothetical. Start with the `feat_182`-family A/B
   (step 2) since clipping (arm C) is already active by default and the 0.1849 score already
   reflects it.
4. Then `02-gbdt-core` (tuning) and `03-feature-engineering`, evaluated on top of whatever
   survives step 3 — not before it, since OOF alone can't distinguish a real improvement from
   one that only helps the ~2x-inflated in-sample number.
5. Read [`ideas/dead-ends/`](ideas/dead-ends/) before trying anything new — nine measured
   negatives are ruled out there.
6. Reserve time before the 13 Aug deadline for the **code submission and presentation** —
   with the inference notebook that is 50% of the final mark and none of it is a leaderboard
   activity.
7. ✅ **Two more real submissions recorded (2026-08-10), from an external/alternate pipeline —
   not `solution/pstu_train.py`.** LB **0.2258** (run-4) is now the best LB score measured
   anywhere in this project, ahead of run-1's 0.1849. See "Run-3 & Run-4" above and
   `results/run-3/README.md` / `results/run-4/README.md` for the full audit — the headline is an
   independent confirmation that removing calibration/target-encoding/KMeans leakage traded OOF
   0.36→0.29 for LB 0.196→0.226, the same OOF-vs-LB lesson idea 04 is built around.
