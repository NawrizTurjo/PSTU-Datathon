# Run-3 — "Winning_Solution_PSTU" V1 (external/alternate pipeline)

**Not built from `solution/pstu_train.py`.** This is a separately authored notebook
(`Winning_Solution_PSTU.ipynb`) — a different, more elaborate pipeline than the project's own
shift-aware LightGBM approach (run-1/run-2). Kept here as a reference data point, not merged
into `solution/`.

## What it does

- **Cleaning:** drop 28 zero-variance columns + ~30 duplicate-pair columns (`|corr|`/exact-match
  heuristic, not the project's own exact-dedup method — count doesn't match the 44 measured in
  `dataset_exploration/`).
- **Categorical encoding:** stratified 5-fold **out-of-fold target encoding** for all 6
  categorical columns (`feat_142`, `feat_157`, `feat_318`, `feat_320`, `feat_325`, `feat_337`).
- **Feature engineering:** 22 row-wise statistical features (mean/std/skew/kurtosis/percentiles/
  zero-count etc.) + 28 KMeans cluster features (4+8+16 clusters, distances to centers) + the 6
  target-encoded columns.
- **Preprocessing:** `QuantileTransformer(output='normal')` then `PCA(50)`.
- **Imbalance handling:** SMOTE, `sampling_strategy=0.5` (→ ~2:1 post-SMOTE ratio).
- **Models:** LightGBM + XGBoost + CatBoost, each with early stopping, **stratified 10-fold CV**.
- **Calibration:** affine probability shift so each model's *optimal* F1 threshold lands exactly
  at 0.5 (`p_new = clip(p_old + (0.5 - opt_threshold), 0.001, 0.999)`).
- **Ensemble:** weighted blend by OOF F1, with a rank-average fallback.

## Result

| | Value |
|---|---|
| Ensemble OOF/CV F1 (per run-4's post-mortem) | **0.3624** |
| Best single-model OOF (CatBoost) | 0.373 |
| **Public LB** (`submission_binary.csv`, screenshot 2026-08-10) | **0.195681511470** |
| CV→LB gap | **~46% relative drop** |

The `Winning_Solution_PSTU.ipynb` file in this folder has **no saved outputs** (every cell's
`execution_count` is `null`) — it was run on Kaggle and only the resulting CSVs were brought
back here, so the 0.3624/0.373 CV numbers above are taken from `run-4/run-4.ipynb`'s own
post-mortem write-up of "V1", not from re-reading this notebook's own output cells.

## Diagnosed root causes of the CV→LB gap

(As identified in `run-4/run-4.ipynb`'s "V1 Post-Mortem" markdown cell — this is the reasoning
that produced run-4, not this session's own analysis.)

| Root cause | Issue |
|---|---|
| Calibration shift leakage | affine shifts of +0.12 to +0.28 fit the OOF distribution and don't transfer to test |
| Target-encoding leakage | smoothing=10 with up to 2,333 categories memorizes rare labels |
| SMOTE over-aggression | 0.5 ratio → ~30k synthetic samples, poor generalization |
| KMeans on combined train+test | cluster distances peek at the test distribution |
| Weak regularization | `num_leaves=96`, `min_child=40` |
| Model-diversity trap | 3 different model families + per-model calibration destabilizes the blend |

## Files

- `Winning_Solution_PSTU.ipynb` — the 13-cell pipeline (no stored outputs, see above).
- `eda.md` — an auto-generated strategy report (EDA summary + the plan the notebook implements).
- `convert_submission.py` — thresholds `submission.csv` at 0.5 to produce `submission_binary.csv`.
- `submission.csv` / `submission_binary.csv` — the two candidate files; **`submission_binary.csv`
  is the one that scored 0.195681511470** on the public LB.

## To improve on this later

See `../run-4/README.md` — run-4 is a direct iteration on this notebook that already addresses
every root cause in the table above and measurably improved the LB score (0.1957 → 0.2258).
Any further work on this lineage should start from run-4, not from here.
