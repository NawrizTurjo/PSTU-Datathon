# dataset_exploration — Stage 1 findings (PSTU Data Thon 2026 Vol-1)

Numbered, idempotent scripts. Each writes its own plain-text (+ CSV) report. Run individually,
any order, any number of times — none mutate `pstu-data-thon-2026-vol-1/`. All numbers below
are **measured**, not carried over from `works.old/` or from CLAUDE.md's first-pass profile.

Local env: pandas/numpy/scikit-learn only. No lightgbm/xgboost/catboost — scripts 07/09 use
`RandomForestClassifier`/`HistGradientBoostingClassifier` as stand-ins per CLAUDE.md convention.

## Scripts

| # | Script | Answers |
|---|---|---|
| 01 | `01_schema_dtypes.py` | shape, dtypes, missing values, target balance, column-order checks |
| 02 | `02_categorical_deep_dive.py` | cardinality + train/test level overlap for the 6 string cols |
| 03 | `03_numeric_profile_sentinels.py` | describe(), sentinel hunt, zero-inflation ratios |
| 04 | `04_constant_duplicate_cols.py` | constant columns, exact-duplicate columns, near-duplicate (corr) columns |
| 05 | `05_adversarial_validation.py` | train-vs-test discriminability, out-of-range features |
| 06 | `06_duplicate_rows.py` | duplicate rows, label conflicts, train/test row overlap |
| 07 | `07_metric_behaviour.py` | binary-F1 vs macro-F1 degenerate floors, threshold sweep |
| 08 | `08_leak_diagnostics.py` | row-index leak, single-feature AUC, capacity test, dup-label consistency |
| 09 | `09_honest_baseline.py` | 5-fold OOF HistGradientBoosting baseline to anchor Stage 2/3 |

## Headline findings

**Shape / schema** — confirmed exactly as CLAUDE.md's first pass stated: train 76,020×351
(350 `feat_*` + `TARGET`), test 60,654×351 (350 `feat_*` + `id` last), zero missing values,
212 int64 / 132 float64 / 6 object columns, positive rate 3.9569% (3,008/76,020).

**Categorical columns** — the 6 string columns' cardinality and train/test overlap match
CLAUDE.md's first pass exactly (feat_142: 2,333 train levels / 55 test-only; feat_325: 1,710 /
27; feat_157: 627 / 8; feat_320/337/318: 0 test-only). Every column's values match its expected
prefix 100%. All 6 have some test rows with unseen levels — any encoding needs an explicit
unseen-level fallback (see report for a per-scheme recipe).

**Sentinels — one new finding CLAUDE.md missed.** `feat_109 == -999999` confirmed as the sole
carrier of that fingerprint (116 train rows / 89 test rows, ~0.15%). `feat_169`'s extreme min
(~-1.11e8) is **not** a sentinel — it's a continuum of large-magnitude values with no repeated
constant, just genuine heavy-tailed data.
**New:** a second sentinel, exactly **`9999999999`** (1e10−1), appears across **23 columns**
(`feat_11, feat_21, feat_26, feat_30, feat_31, feat_36, feat_74, feat_77, feat_96, feat_124,
feat_135, feat_144, feat_149, feat_158, feat_171, feat_196, feat_204, feat_226, feat_301,
feat_315, feat_330, feat_336, feat_340`) — the classic Santander `delta_imp_*` fingerprint.
Same treat-as-missing logic should apply. **This is not in CLAUDE.md and should be added.**

**Zero-inflation** — 252 of the **344 numeric** columns (not 350 — 6 of the 350 total feature
columns are the categorical strings) are ≥90% zero. CLAUDE.md's phrasing ("252 of 350 numeric
columns") is imprecise; the denominator should read 344.

**Constant + duplicate columns — CLAUDE.md's headline number does not reproduce.**
- Constant-in-both train+test: **28** — matches CLAUDE.md exactly, same column list.
- Constant-in-test-only (kept, varies in train): **14** — matches CLAUDE.md exactly.
- **Exact-duplicate columns** (row-for-row identical values, excluding the constant columns
  above): **16 groups, all size 2, 16 redundant columns.** Combined droppable = 28 + 16 = **44**,
  not CLAUDE.md's claimed 83 (28 + 55 across 20 groups).
- A looser **near-duplicate check** (`|corr| > 0.999`, columns that are scaled/linear copies of
  each other rather than byte-identical) finds 36 groups / 49 redundant columns → 77 combined,
  closer to but still short of 83. CLAUDE.md's 55/20 figure was likely produced by a
  correlation-based method at a different threshold and is not exactly reproducible.
- **Recommendation:** drop the 44 exact-duplicate/constant columns safely. Treat the
  corr>0.999 near-duplicate groups (`04_constant_duplicate_report.txt`) as dimensionality-
  reduction candidates only, not automatic drops — a scaled copy can still carry distinct
  information a tree model exploits.

**Adversarial validation** — 5-fold CV AUC discriminating train vs test = **0.5742 ± 0.0027**.
This is a **real, moderate covariate shift**, not the ~0.50 "safe iid split" CLAUDE.md's
category didn't measure. `feat_182` alone carries by far the most train/test separability
(importance 0.17, ~2.4x the next feature). 93/344 numeric columns have test values outside
their train range (`feat_116`, `feat_44`, `feat_334` worst offenders — all also top adversarial
features). **Recommendation:** watch CV-vs-LB gap; consider adversarial-validation sample
weighting or capping/clipping the worst out-of-range features before modeling.

**Duplicate rows / label conflicts** — confirmed **zero** duplicate feature-rows in train,
**zero** in test, **zero** rows shared between train and test on features alone. Matches
CLAUDE.md's first pass exactly. No label-conflict diagnostic needed (nothing to conflict).

**Metric behaviour** — degenerate floors reproduce CLAUDE.md's table exactly:

| Submission | Binary F1 | Macro F1 |
|---|---|---|
| all zeros | 0.0000 | 0.4899 |
| all ones | 0.0761 | 0.0381 |

The all-zeros LB probe (≈0.49 → macro F1, ≈0.00 → binary F1) is still the right first move.
On a quick RF baseline (OOF AUC 0.8485), the optimal threshold for both binary-F1 and macro-F1
landed at the same point (t=0.75 in that run) — the two objectives don't necessarily require
different operating points, but re-tune per model since this is not guaranteed to hold.

**Leak diagnostics — no leak found, real ceiling confirmed.**
- Row-index AUC: 0.5047 (no leak from row order; positive rate is flat ~3.4–4.5% across all
  10 index blocks).
- Best single-feature AUC: 0.6986 (`feat_175`) — no magic feature (would expect >0.85 if leaked).
- Capacity test (unregularized `DecisionTreeClassifier`): train AUC 1.0000 (expected — full
  memorization), held-out val AUC 0.5916, gap 0.41. Val AUC collapses relative to train →
  normal overfitting, **no deterministic rule exists** → the score has a genuine ceiling to
  push against, unlike the old withdrawn dataset.
- Duplicate-row label consistency: N/A (zero duplicate rows exist to conflict).

**Honest baseline** — `HistGradientBoostingClassifier`, 5-fold stratified OOF, 306/350 columns
(44 exact-constant/duplicate dropped), sentinels → NaN, categoricals ordinal-encoded on train
only:

- Per-fold AUC: [0.8797, 0.8812, 0.8931, 0.8857, 0.8945], mean **0.8869 ± 0.0060**
- OOF AUC (pooled): **0.8868**
- Tuned threshold (t=0.19): binary_f1 **0.3841**, macro_f1 **0.6786** (vs naive t=0.5:
  binary_f1 0.1777, macro_f1 0.5791 — threshold tuning alone is worth +0.21 binary_f1 /
  +0.10 macro_f1 here)

This is a floor, not a ceiling — no hyperparameter tuning, no row-aggregate features, no
categorical target-encoding, and a sklearn stand-in instead of real LightGBM/XGBoost/CatBoost.

## Corrections filed against CLAUDE.md

1. "252 of 350 numeric columns are ≥90% zero" → denominator should be **344** (the numeric-only
   count; 350 is the total feature count including the 6 categoricals).
2. "83 droppable columns: 28 constant in both + 55 redundant across 20 duplicate groups" →
   exact-duplicate measurement gives **28 + 16 = 44** droppable. The 55/20 figure does not
   reproduce under exact matching; a correlation-threshold method gets closer (49/36) but still
   not exact. Use 44 as the safe drop list.
3. **New finding to add:** a second sentinel value, `9999999999`, spans 23 columns and should
   be treated as missing alongside `feat_109`'s `-999999`.
4. **New finding to add:** adversarial validation AUC is 0.5742 (moderate real shift), not
   assumed ~0.50 — CLAUDE.md had not yet measured this.

## Implications for Stage 2/3 (ideas + solution)

- Drop the 44 exact-constant/duplicate columns; treat the 36 near-duplicate groups as feature-
  selection candidates, not automatic drops.
- Replace `-999999` (feat_109) and `9999999999` (23 columns) with NaN/missing before modeling.
- Encode the 6 categorical columns with an explicit unseen-level fallback (frequency/target
  encoding with a global-mean bucket, or ordinal + `handle_unknown='ignore'`) — never fit the
  encoder on train+test concatenated.
- Run the all-zeros LB probe immediately to settle binary-vs-macro F1 before any further
  threshold tuning work.
- Budget CV-vs-LB monitoring for the measured covariate shift (AUC 0.5742); `feat_182`,
  `feat_44`, `feat_116`, `feat_306`, `feat_97` are the columns driving it and are good
  candidates for clipping/robustification or exclusion if LB tracks worse than CV.
- OOF AUC ~0.887 / tuned macro-F1 ~0.68 from an untuned baseline is the number to beat with
  real GBDT + row-aggregate features + hyperparameter search on Kaggle.
