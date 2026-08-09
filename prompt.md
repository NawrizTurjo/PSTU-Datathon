# Master idea-generation prompt — PSTU Data Thon 2026 Vol-1

You are generating a prioritized roadmap of modeling ideas for a Kaggle-style competition. This
prompt is self-contained: everything you need is below. Do not assume access to any other files
or conversation history. Where this prompt states a fact as **measured**, treat it as ground
truth — it was produced by running code against the actual data, not estimated. Where it states
something as a **confirmed negative**, do not propose it — it was tried/checked and does not
apply here. Proposing a measured negative wastes review time; build on top of these findings
instead of re-deriving or contradicting them.

## What you're building

One folder per idea under `ideas/`, plus a priority-ordered `ideas/README.md` index. Each idea
folder must contain, in its own `README.md`:

- **What it is** — one paragraph, concrete enough to implement without further clarification.
- **Why it should work** — cite the specific measured finding below that motivates it. "Because
  it's a common Kaggle trick" is not sufficient justification on its own.
- **Concrete steps** — an implementation-ready outline, not just a description.
- **Kaggle cost** — rough compute/time budget (CPU-only tree models are cheap; anything needing
  GPU-hours or long training should say so explicitly and justify it against the ~30h/week GPU
  cap).
- **Honest expected gain** — a number or range against the baseline below, with your confidence
  in that estimate. Inflated expectations are worse than modest honest ones — this roadmap will
  be used to allocate a handful of days of work before a hard deadline.
- **When to abandon** — a concrete stopping condition (e.g. "if OOF AUC doesn't improve by
  >0.002 after step 3, drop this").

Also produce `ideas/dead-ends/` — a folder for ideas you considered and are explicitly *not*
recommending, with a one-paragraph note on why, so nobody re-investigates them later.

Prioritize by (expected gain / implementation cost), not by novelty. This is a ~4-day sprint
before a hard submission deadline, not a research project.

---

## Competition

**PSTU Data Thon 2026 Vol-1.** Predict whether a financial account will be flagged at-risk
(`TARGET = 1`). Binary classification, 350 anonymized features. Timeline (GMT+6): final
submission 13 Aug 2026 18:00, private LB 18:30, inference notebook due 23:59, winners 15 Aug.
**Today is 2026-08-09** — roughly 4 days remain.

**Scoring is only 50% leaderboard:**

| Component | Weight | Notes |
|---|---|---|
| Public LB | 10% | live, on 10% of test data — small, do not overfit to it |
| Private LB | 40% | on 50% of test data |
| Hidden test | 40% | 40% of *completely unseen* data, run via a mandatory inference notebook |
| Presentation + code + format | 10% | |

**Implication for idea ranking:** generalization and a deterministic, reproducible inference
notebook matter as much as raw leaderboard score. A notebook that fails to run, or is
non-deterministic, forfeits 40% outright. Weight ideas that improve robustness/generalization
at least as highly as ideas that only chase public-LB score — the public LB is only 10% and is
explicitly flagged as too small to trust for model selection.

## The metric — still partially unresolved

The competition page contradicts itself: the evaluation section says "F1 Score", the submission
section says predictions are thresholded at 0.5 and scored as "**Macro F1**". These diverge
sharply at this dataset's ~3.96% positive rate:

| Submission | Binary F1 | Macro F1 |
|---|---|---|
| all zeros | 0.0000 | 0.4899 |
| all ones | 0.0761 | 0.0381 |

**An all-zeros diagnostic-probe submission may or may not have been made by the time you read
this** — do not assume the ambiguity is resolved. Propose ideas that are robust to *either*
metric where possible (i.e. tune the decision threshold against both binary-F1 and macro-F1 and
report both), and flag clearly if an idea's value depends on which metric turns out to be live.

Because scoring is F1-only (no AUC term in the grade), **only the binary decision the model
outputs matters** — ranking quality (AUC) matters solely through the split it produces at the
chosen threshold. Prioritize threshold-selection/calibration ideas accordingly; they are cheap
and were measured to be worth **+0.21 binary-F1 / +0.10 macro-F1** over a naive 0.5 cut on the
baseline model (see below). The grader applies a fixed 0.5 cut to whatever is *submitted*, so
the practical mechanism is: submit hard 0/1 labels at your chosen operating point, or rank-shift
probabilities so 0.5 lands exactly there.

## Data shape (measured, exact)

- `train.csv`: 76,020 rows × 351 cols (350 `feat_*` + `TARGET`).
- `test.csv`: 60,654 rows × 351 cols (350 `feat_*` + `id` as the **last** column). `id` is not
  `0..n-1` and not contiguous.
- Zero missing values in either file.
- Positive rate: 3.9569% (3,008 / 76,020) — a hard class-imbalance problem.
- Dtypes: 212 int64, 132 float64, **6 object (categorical string)** columns among the 350
  `feat_*` columns (344 are numeric).

### The 6 categorical columns (measured cardinality + train/test overlap)

| Column | Train levels | Prefix | Test-only levels | % test rows with unseen level |
|---|---|---|---|---|
| `feat_142` | 2,333 | `PRD_*` | 55 | 0.1467% |
| `feat_325` | 1,710 | `SEG_*` | 27 | 0.0775% |
| `feat_157` | 627 | `PRV_*` | 8 | 0.0181% |
| `feat_320` | 119 | `CH_*` | 0 | 0.0000% |
| `feat_337` | 39 | `OFC_*` | 0 | 0.0000% |
| `feat_318` | 12 | `PERF_*` | 0 | 0.0000% |

Every value in every one of these 6 columns matches its expected prefix 100% of the time — the
categories are structured/clean, not noisy free text. `feat_318` and `feat_337` are low enough
cardinality to treat as ordinary categoricals; `feat_142`/`feat_325`/`feat_157` need
high-cardinality handling (target encoding, hashing, or leave as raw codes for a tree model that
handles categoricals natively). **Any encoding must have an explicit fallback for unseen levels
at inference** (global mean / reserved bucket / `handle_unknown='ignore'`) — the hidden-test
component runs on the mandatory inference notebook, so an unhandled unseen-category error there
is a real risk, not a hypothetical.

### Sentinels (measured)

- `feat_109 == -999999` is a real sentinel: 116 train rows (~0.15%), 89 test rows (~0.15%). No
  other numeric column carries this exact value. Treat as missing.
- **`9999999999`** (1e10−1) is a second, separate sentinel spanning **23 columns**: `feat_11,
  feat_21, feat_26, feat_30, feat_31, feat_36, feat_74, feat_77, feat_96, feat_124, feat_135,
  feat_144, feat_149, feat_158, feat_171, feat_196, feat_204, feat_226, feat_301, feat_315,
  feat_330, feat_336, feat_340`. Treat as missing alongside `feat_109`.
- **Confirmed negative:** `feat_169`'s extreme minimum (~-1.11e8) is *not* a sentinel. It is a
  genuine heavy-tailed continuum of large-magnitude values with no repeated constant at the
  extreme. Do not special-case it as missing; ordinary robust scaling is sufficient.

### Sparsity

252 of the 344 numeric columns are ≥90% zero. This is a heavily sparse, zero-inflated tabular
dataset, structurally similar to the public Santander Customer Satisfaction dataset (see
Lineage note below) — row-wise aggregate features (count of nonzero, sum, mean-of-nonzero,
etc. across feature blocks) are a known-effective pattern for this data shape.

### Redundant columns (measured — corrects an earlier, wrong estimate)

- 28 columns are constant (single unique value) in *both* train and test — safe to drop.
- 14 more are constant in test only (they vary in train) — **keep** these, a model can still
  learn from their train-time variance even though test predictions won't use that signal.
- 16 columns are *exact*, row-for-row duplicates of another column (16 groups, all size 2) —
  safe to drop the redundant half of each pair.
- **Combined safe-drop list: 44 columns.** An earlier estimate of "83 droppable, 55 redundant
  across 20 groups" does **not** reproduce under exact-value duplicate checking and should be
  treated as wrong. A looser check (Pearson `|corr| > 0.999`, i.e. scaled/linear near-copies
  rather than byte-identical values) finds 36 groups / 49 redundant columns (77 combined) —
  closer but still not exactly 83, and threshold-dependent. **Confirmed negative: do not budget
  time re-deriving the 83 figure — it was not reproducible.** If you want a
  dimensionality-reduction idea, propose it against the 36 corr-based groups explicitly as
  "candidates for reduction, not guaranteed-safe drops," since a scaled copy can still carry
  distinct information a tree model exploits.

### Train/test distribution shift (measured)

5-fold adversarial-validation AUC (classifier predicting train-vs-test) = **0.5742 ± 0.0027**.
This is a real, moderate covariate shift — not the ~0.50 you'd see on a clean iid split.
`feat_182` alone drives the most separability (importance ~0.17 in the adversarial classifier,
~2.4x the next-highest feature `feat_44`). 93 of 344 numeric columns have test values outside
their train min/max range, with `feat_116`, `feat_44`, `feat_334` the worst offenders (also
top adversarial-importance features). **Implication:** CV may optimistically overstate LB
performance; ideas that address this (adversarial-validation sample weighting, robustifying or
clipping the worst-shifted features, monitoring CV-vs-LB gap) are worth proposing given the
scoring structure's heavy weight on private/hidden test (80% combined) over public LB (10%).

### Duplicate rows / label conflicts (measured, confirmed negative)

Zero duplicate feature-rows in train, zero in test, zero rows shared between train and test on
features alone. **Confirmed negative: there is no duplicate-row label-noise cleanup to do** —
unlike some public tabular datasets, this one has none. Do not propose deduplication ideas.

### Leak diagnostics (measured, confirmed negative — no leak)

- Row-index AUC (does raw row order predict the target?): 0.5047 — no leak. Positive rate is
  flat (~3.4%–4.5%) across 10 equal row-index blocks.
- Best single-feature AUC across all 344 numeric columns: 0.6986 (`feat_175`). A "magic feature"
  leak would show >0.85–0.90 alone; this is far below that. **Confirmed negative: no single
  feature comes close to solving this on its own.**
- Capacity test: an unregularized `DecisionTreeClassifier` reaches train AUC 1.0000 (expected —
  full memorization) but held-out AUC only 0.5916 — a 0.41 gap. **Confirmed negative: there is
  no deterministic rule to be found; the score has a genuine generalization ceiling.** Do not
  propose "find the hidden formula" style approaches — they were checked for and ruled out.

### Lineage — informational only, do NOT source external data

The row count, exact positive count (3,008/76,020), and `-999999` sentinel match the public
Santander Customer Satisfaction Kaggle dataset's fingerprint, and this dataset structurally
resembles it (extreme sparsity, sentinel values, row-aggregate features likely to help).
**This is recorded only to justify structural modeling choices** (e.g. row-aggregate features,
sparse-data handling). **External datasets are explicitly prohibited by the competition rules
and joining any external labels is instant disqualification.** Santander's true test labels were
never public in the first place, so there is nothing to leak even if this were attempted — but
the point is moot because it's against the rules regardless. Do not propose anything that uses
external data, pretrained embeddings without disclosure, or any join against public Santander
data or IDs.

### Honest baseline (measured — beat this, don't just match it)

`HistGradientBoostingClassifier` (sklearn's GBDT, used only because real LightGBM/XGBoost/
CatBoost aren't installed in the local dev environment — the actual solution should use one of
those on Kaggle), 5-fold stratified OOF, trained on the 306 remaining columns after dropping the
44 safe-drop columns, sentinels replaced with NaN, categoricals ordinal-encoded on train only,
**no hyperparameter tuning, no feature engineering beyond the cleanup above**:

- Per-fold AUC: [0.8797, 0.8812, 0.8931, 0.8857, 0.8945], mean **0.8869 ± 0.0060**
- OOF AUC (pooled): **0.8868**
- At the F1-optimal threshold (t≈0.19): binary-F1 **0.3841**, macro-F1 **0.6786**
- At naive t=0.5: binary-F1 0.1777, macro-F1 0.5791

This is a floor established with zero feature engineering and a weaker-than-final GBDT library.
Ideas should target beating OOF AUC ~0.887 / tuned macro-F1 ~0.68 with real GBDT libraries,
proper hyperparameter search, categorical target-encoding, and row-aggregate features. Report
expected gains relative to these numbers specifically.

---

## Rules that constrain every idea

- ❌ No external datasets — only `train.csv`/`test.csv`/`sample_submission.csv`.
- ❌ No test-set tampering or label reverse-engineering — instant disqualification.
- ❌ Do not generate target values with an LLM — explicitly banned.
- ✅ SMOTE / synthetic augmentation / feature engineering **on training data only** is allowed.
- ⚠️ Pre-trained models are allowed only with disclosure in the writeup.
- ✅ An inference notebook is mandatory and must reproduce submitted predictions
  **deterministically** — every seed fixed, fold split pinned, model artifacts saved and
  reloadable. Any idea that introduces non-determinism (unseeded randomness, non-reproducible
  library behavior, wall-clock-dependent logic) must explicitly say how it stays deterministic.

## Environment constraints

- Local dev has pandas/numpy/scikit-learn/scipy only — no lightgbm/xgboost/catboost locally.
  Ideas should be written assuming Kaggle's notebook environment (which has all three) but
  should remain smoke-testable locally with `HistGradientBoostingClassifier` as a stand-in.
- Don't propose GPU-dependent tree training — at this data size CPU is faster and Kaggle CPU
  quota is unlimited while GPU is capped ~30h/week. GPU is only worth proposing for something
  that genuinely needs it (e.g. a neural approach), and should say so explicitly with a cost
  justification.

---

## What to prioritize

Given ~4 days remaining and the scoring split (80% combined private+hidden test, only 10%
public LB, 10% presentation/code), rank ideas favoring:

1. Cheap, high-confidence wins already implied by the measured findings above (the 44-column
   cleanup, sentinel handling, threshold tuning, categorical unseen-level fallback) — these
   should be baseline-solution requirements, not optional ideas, but call out any nuance.
2. Feature engineering with concrete measured justification (row-aggregates for the 252
   zero-inflated columns, target/frequency encoding for the 3 high-cardinality categoricals with
   proper unseen-level handling).
3. Model/ensemble choices appropriate for a ~4-day sprint (single well-tuned GBDT vs. a small
   seed-averaged ensemble vs. blending multiple GBDT libraries) — weigh against the
   generalization requirement, not just OOF score.
4. Robustness ideas addressing the measured 0.5742 adversarial-validation shift, since the
   hidden-test component (40%, completely unseen data) rewards generalization specifically.
5. Do **not** spend idea slots on: deduplication (none exists), leak-hunting (none exists,
   already exhaustively checked), external data, or re-deriving the 83-vs-44 droppable-column
   discrepancy (already resolved — use 44).
