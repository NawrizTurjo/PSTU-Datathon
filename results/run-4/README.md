# Run-4 — "V2 leakage-free" fix of run-3 (external/alternate pipeline)

Direct iteration on `../run-3/` (`Winning_Solution_PSTU.ipynb`), same lineage — **still not
built from `solution/pstu_train.py`**. `run-4.ipynb`'s own markdown documents this as a
post-mortem-driven rewrite of run-3, explicitly targeting run-3's 46% CV→LB drop.

## Fixes vs run-3 (from the notebook's own post-mortem table)

| Root cause (V1) | V1 issue | V2 fix |
|---|---|---|
| Calibration shift leakage | affine shifts (+0.12 to +0.28) overfit OOF, don't transfer | **no calibration** — raw probabilities |
| Target-encoding leakage | smoothing=10 with 2,333 cats memorizes rare labels | **native categorical handling** inside CatBoost |
| SMOTE over-aggression | 0.5 ratio → ~30k synthetic samples | **0.3 ratio** + `scale_pos_weight` |
| KMeans on combined train+test | cluster distances peek at test | **removed** — pure row-wise stats only |
| Weak regularization | `num_leaves=96`, `min_child=40` | `depth=5`, `l2_leaf_reg=5.0`, `min_data_in_leaf=50` |
| Model-diversity trap | 3 model families + per-model calibration | **3-seed CatBoost-only ensemble**, same architecture |

## Config

- **Model:** CatBoost only, 3-seed ensemble (seeds 42/123/456). `depth=5`, `l2_leaf_reg=5.0`,
  `iterations=5000`, `learning_rate=0.015`, `random_strength=1.5`, `bagging_temperature=0.8`,
  `min_data_in_leaf=50`, `one_hot_max_size=10`, `od_wait=150`, `auto_class_weights='Balanced'`,
  `eval_metric='F1'`.
- **Categoricals:** label-encoded then passed to CatBoost via `cat_features` (native handling,
  not target-encoded) — dropped the leakage path entirely.
- **Imbalance:** SMOTE(0.3) **and** `scale_pos_weight=12.0` together ("complementary").
- **Features:** drop 28 zero-variance + 15 duplicate columns (301 numeric kept) + 6 raw
  categoricals + 6 row-wise stats (mean/std/iqr/zero-count/skew/kurtosis) = 313 total. No KMeans,
  no PCA. `QuantileTransformer(output='normal')` on the 307 truly-numeric columns only.
- **CV:** 5-fold stratified (down from V1's 10-fold, "faster iteration, similar reliability").
- **No calibration** — raw probabilities submitted, thresholded at 0.5 by the grader.

## Result

| | Value |
|---|---|
| OOF F1 @ 0.5, seed 42 | 0.27829 |
| OOF F1 @ 0.5, seed 123 | 0.27092 |
| OOF F1 @ 0.5, seed 456 | 0.26833 |
| Ensemble, rank-average | 0.13809 ← **collapsed, not used** |
| Ensemble, probability-average | **0.28724** ← used |
| Test predicted-positive rate | **7.05%** vs train's 3.96% (**1.78×**, inflated) |
| **Public LB** (screenshot label "FixIssuesv2 - Version 4", 2026-08-10) | **0.225758329189** |
| OOF→LB ratio | 0.2258 / 0.2872 ≈ **0.79** (a real gap, much smaller than run-3's ≈0.54) |

Rank-averaging the 3 CatBoost seeds is a **large regression** here (0.138 vs 0.287
probability-averaging) — the notebook's own selection logic catches this (picks whichever OOF
score is higher) and correctly falls back to probability-averaging, but it's worth flagging: for
this pipeline, rank-average is not a safe default.

The test positive rate running *above* train (1.78×) is the **opposite direction** from the
project's own `solution/pstu_train.py` runs (run-1: ratio 0.87 below train; run-2: ratio ~0.86
below train — see `../run-1/`, `../run-2/`, and `CLAUDE.md`). Consistent with removing the
calibration shift here: raw LogLoss-trained probabilities plus `scale_pos_weight=12` push more
rows above the 0.5 cut than the model would naturally place there.

## Comparison across all measured runs so far

| Run | Pipeline | OOF/CV F1 | Public LB |
|---|---|---|---|
| run-1 | `solution/pstu_train.py`, LightGBM, pre-search | 0.3954 (t=0.1736) | 0.1849 |
| run-2 | `solution/pstu_train.py`, LightGBM, shift-aware arm search | 0.3957 (t=0.1830) | not submitted |
| run-3 | external, LGB+XGB+CatBoost blend + calibration | 0.3624 | 0.1957 |
| run-4 | external, CatBoost-only, no calibration | 0.2872 | **0.2258** ← best measured so far |

Run-4 is currently the best real leaderboard score measured across every run in this project,
external or internal, despite having the lowest OOF number of the four — the clearest evidence
yet that on this dataset, an honest (lower) OOF paired with less leakage/overfitting beats a
higher OOF built on techniques that don't survive the train→test shift. This mirrors, from a
completely independent pipeline, the same lesson the project's own `ideas/04-shift-robustness/`
was written around.

## To improve on this later

Not attempted here (this file only reports what was measured):

- The 1.78× positive-rate inflation vs train suggests 0.5 is not this model's best operating
  point even before considering LB — a plateau-centred threshold search (as in the project's own
  `01-threshold-engine`) applied to this pipeline's raw probabilities, rather than submitting at
  the grader's fixed 0.5 cut, is the most obvious next lever.
- Rank-average collapsing to 0.138 is unexplained — worth understanding before trusting
  rank-blending in any future ensemble (relevant if `solution/`'s own
  `ideas/05-ensemble-diversity/` is ever implemented).
- No adversarial-shift diagnostic was run on this pipeline (unlike `solution/pstu_train.py`'s
  section 5) — unknown whether `feat_182` or the other shift-driving columns identified in
  `dataset_exploration/05_adversarial_validation_report.txt` were part of what V2 fixed, or
  whether that headroom is still on the table here too.

## Files

- `run-4.ipynb` — the 5-cell-visible pipeline described above, **with outputs saved** (ran via
  papermill on Kaggle) — all numbers in this file were read directly from its stored cell
  outputs, not reconstructed.
- `submission.csv` / `submission_binary.csv` — `submission.csv` holds raw probabilities;
  `submission_binary.csv` is the same thresholded at 0.5. The screenshot's "FixIssuesv2 -
  Version 4" LB score (0.225758329189) corresponds to this run's Kaggle-committed submission.
