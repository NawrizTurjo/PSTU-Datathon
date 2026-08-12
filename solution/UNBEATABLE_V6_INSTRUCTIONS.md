# How to run `Unbeatable_V6.ipynb` on Kaggle

Single notebook, trains everything and writes both submission files directly. Not built from
`pstu_train.py`/`run-5-raw.py` — see `Unbeatable_V6.py`'s header markdown for why (no SMOTE, no
class weighting, adversarial-drop + PCA feature reduction, pseudo-labeling).

## Setup

1. **New Notebook** → **File → Import Notebook** → upload `Unbeatable_V6.ipynb`.
2. **Add Data** → *Competitions* → **PSTU Data Thon 2026 Vol-1**. Mounts at
   `/kaggle/input/competitions/pstu-data-thon-2026-vol-1/`; the notebook also probes
   `/kaggle/input/pstu-data-thon-2026-vol-1/`, so either layout works unedited.
3. **Settings** → Accelerator **None**, Internet **Off**. XGBoost/LightGBM/CatBoost are all
   preinstalled on Kaggle's default CPU image — no `!pip install` needed.
4. **Run All.**

Expect **35–70 minutes** on a free CPU kernel (5 folds × 3 models × 2 stages, plus a quick
adversarial-validation pass). CatBoost is the slowest of the three per fold. If you're iterating
and want a fast correctness check first, set `SMOKE_TEST = True` in the first code cell (drops to
an 8k-row subsample and tiny models, runs in under a minute), confirm it completes and writes
both submission files, then set it back to `False` and Run All again for the real run.

## What to check in the output, top to bottom

| Print | Expect | If not |
|---|---|---|
| `lightgbm ... \| xgboost ... \| catboost ...` | version strings for all three | an import failed — check Settings → Internet is off but the packages are still preinstalled; if one is genuinely missing, `!pip install <name>` in a cell above the import |
| `constant/duplicate columns dropped ...` | **44** | data has changed since Stage-1 EDA — investigate before trusting anything downstream |
| unseen-level lines per categorical column | small counts (feat_142 ~55, feat_325 ~27, feat_157 ~8, others 0) | matches Stage-1 EDA; large unexpected counts mean a different test file |
| `f1_threshold_search verified against sklearn...` | prints and does not raise | if this assertion fails, stop — nothing downstream is trustworthy |
| `adversarial train/test AUC` | **~0.55–0.62** (Stage-1 measured: 0.5742) | much higher (>0.7) suggests a leak in the adversarial features themselves; much lower (~0.50) suggests train/test are less separable than previously measured — still fine, just note it |
| top-20 drift feature list | `feat_182` and friends near the top | expected per EDA (`feat_182` importance ~0.17, ~2.4x next) |
| `PCA: N components \| variance covered ...` | N somewhere in the 30s–60s range for 0.95 coverage | wildly different N isn't wrong, just means the quantile-transformed features are more/less collinear than expected |
| 5× `fold i [stage 1]: blended AUC ...` | ~0.87–0.90 | |
| `[stage 1 / pre-pseudo] OOF AUC ... best threshold ... -> F1 ...` | F1 somewhere in **0.30–0.42** | far outside this range — check the fold AUCs above didn't collapse |
| `pseudo-labeled rows: N / 60654 (...)` | typically a few thousand to ~15–20k, mostly negative (confident-0 is easier to hit than confident-1 at this positive rate) | if pseudo-labeled count is 0, `PSEUDO_POS_THRESH`/`PSEUDO_NEG_THRESH` are too strict for this run's calibration — loosen them (e.g. 0.85/0.10) |
| 5× `fold i [stage 2]: blended AUC ...` | | |
| `[stage 2 / post-pseudo] OOF AUC ... -> F1 ...` | compare directly against stage 1's F1 | |
| `FINAL MODEL: stage2_pseudo_labeled` or `stage1_baseline` | either is a valid, expected outcome — the notebook picks whichever honest OOF F1 is higher | if it always says `stage1_baseline` across repeated runs, pseudo-labeling isn't helping on this data; that's a legitimate finding, not a bug |
| `submission.csv: 60654 rows \| N positives \| rate X%` | rate roughly **3–8%** | outside `[1%, 15%]` triggers an explicit `WARNING:` — read it before submitting |

## Outputs (in `/kaggle/working/`)

| File | Contents |
|---|---|
| `submission.csv` | **submit this** — `id,TARGET` with `TARGET` already 0/1 at the tuned threshold |
| `submission_prob.csv` | `id,TARGET` with `TARGET` as the raw blended probability (0.0–1.0), for re-thresholding later without a re-run |
| `adversarial_importance.csv` | every feature's adversarial-validation importance, full ranking |
| `threshold_curve_stage1.csv` / `threshold_curve_stage2.csv` | F1 at all 50 grid points (0.01–0.50) for each stage |
| `run_summary.json` | every number printed during the run, machine-readable |

Submit `submission.csv` from the notebook's **Output** tab, or **Save Version → Save & Run All
(Commit)** first if you want the run archived.

## Config knobs (top of the first code cell / `Unbeatable_V6.py`)

| Setting | Default | Notes |
|---|---|---|
| `SEED` | `42` | single seed throughout — fixed for determinism, not searched |
| `N_FOLDS` | `5` | shared by stage 1 and stage 2 (same split, so the two stages are directly comparable) |
| `ADV_CV_FOLDS` | `5` | folds for the train-vs-test adversarial classifier |
| `DRIFT_DROP_N` | `20` | numeric features dropped by adversarial-importance rank; raise/lower and re-run to A/B |
| `PCA_VARIANCE` | `0.95` | fraction of variance PCA must retain; passed straight to `sklearn.decomposition.PCA(n_components=...)` |
| `PSEUDO_POS_THRESH` / `PSEUDO_NEG_THRESH` | `0.90` / `0.05` | confidence cutoffs for pseudo-labeling; tighter = fewer, safer pseudo rows |
| `N_ESTIMATORS` / `CAT_ITERATIONS` | `2000` | trees/iterations cap for XGBoost+LightGBM / CatBoost — early stopping usually triggers well before this |
| `EARLY_STOPPING_ROUNDS` | `100` | |
| `SMOKE_TEST` | `False` | `True` → 8k-row subsample + tiny models, for a fast correctness pass |

There is no `scale_pos_weight`, `auto_class_weights`, or SMOTE knob anywhere in this notebook —
that's deliberate, see the notebook's own header markdown.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train.csv not found` | competition data not attached | Add Data → Competitions → PSTU Data Thon 2026 Vol-1 |
| `ModuleNotFoundError: catboost` (or xgboost/lightgbm) | unusual/stripped-down Kaggle image | `!pip install -q catboost` (etc.) in a new first cell |
| Notebook times out | slow kernel / CatBoost dominating | drop `CAT_ITERATIONS` to ~1000, or `N_FOLDS` to 3 for a faster (less rigorous) pass |
| `pseudo-labeled rows: 0` | thresholds too strict for this run's probability calibration | loosen `PSEUDO_POS_THRESH`/`PSEUDO_NEG_THRESH` |
| `WARNING: predicted-positive rate ... outside the ~3-6% ballpark` | threshold/pseudo-labeling picked an extreme cut | inspect `threshold_curve_stage{1,2}.csv` — if F1 is nearly flat across a wide threshold range, the argmax may have picked a noisy edge; consider manually picking a threshold from the plateau instead of the printed argmax |
| Submission rejected | wrong column names / row count | the assertions right before `submission.to_csv(...)` should already catch this before it gets that far — if the notebook completed, the file is valid |

## After submitting

Compare the public LB score against the two best measured results so far in this project
(`CLAUDE.md`):

| Run | Pipeline | Public LB |
|---|---|---|
| run-1 | `pstu_train.py`, LightGBM, threshold-tuned, no pseudo-labeling | 0.1849 |
| run-4 | external CatBoost + SMOTE(0.3) + `scale_pos_weight=12`, raw probs @ 0.5 | **0.2258** ← best so far |

Unbeatable V6's hypothesis is that an honestly-calibrated ensemble (no SMOTE/weighting) plus an
exact threshold search plus pseudo-labeling can beat both. Record the result — whatever it is —
back into `CLAUDE.md` / a `results/run-N/README.md`, the way every prior run in this project has
been.
