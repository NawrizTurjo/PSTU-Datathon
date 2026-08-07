# Solution Ideas — index & priority

Every number below is **measured** on this dataset (5-fold OOF on `converted_train.csv`),
not estimated. Where something is a guess, it says so.

## The competitive landscape (measured)

| Strategy | Composite score |
|---|---|
| Predict all `0` | **0.3028** |
| Predict all `1` | **0.4388** |
| RandomForest baseline, tuned threshold | **0.5167** |
| HistGradientBoosting (≈LightGBM), tuned threshold | **0.5269** |
| Realistic well-executed ceiling (estimate) | **~0.54–0.56** |

Two things follow, and they should shape everything you do:

1. **The floor is high.** Submitting all-ones scores 0.4388 without a model. A decent
   GBDT scores ~0.527. So the entire competitive range is roughly **0.50–0.56** — the
   gaps between leaderboard positions will be small. A +0.005 improvement is meaningful,
   and a single careless mistake (bad threshold, broken submission) costs more than any
   modelling cleverness gains.
2. **You cannot win this with a bigger model.** 48,128 × 286 is a small tabular
   dataset with 2,406 positives. The bottleneck is ranking quality and metric handling,
   not compute.

## Priority order

Do them in this order. The first two are cheap and account for most of the realistic gain.

| # | Idea | Effort | Expected gain | Confidence |
|---|---|---|---|---|
| [00](00-foundation/) | **Foundation** — preprocessing, CV protocol, metric function | 1h | prerequisite | — |
| [03](03-threshold-engine/) | **Threshold engine** — optimize the real composite | 1–2h | **+0.015–0.020** | high (measured) |
| [01](01-gbdt-core/) | **GBDT core** — LightGBM/CatBoost/XGBoost | 2–4h | **+0.010** over RF | high (measured) |
| [02](02-feature-engineering/) | **Feature engineering** — Santander row-aggregates + domain ratios | 4–8h | +0.005–0.015 | medium |
| [04](04-ensemble-diversity/) | **Ensemble & diversity** — rank-averaging, stacking | 3–6h | +0.003–0.008 | medium |
| [05](05-label-noise/) | **Label-noise handling** — 3.3% conflicting rows | 2–4h | 0 to +0.005 | low (speculative) |
| [06](06-dead-ends/) | **Dead ends** — measured negatives, read to save time | 10min read | saves hours | high (measured) |

**Start with 00 → 03 → 01.** Threshold optimization is listed above the model work
deliberately: it's worth more (+0.018 measured) than upgrading RandomForest to a
gradient-boosted model (+0.010 measured), and takes a fraction of the time.

## Read this before touching the GPU

Kaggle gives you far more hardware than this problem needs:

- The cleaned dataset is **63 MB** (`converted_train.csv`). It fits in RAM many times over.
- LightGBM on 48k × 286 trains in **~30–90 seconds on CPU**. A full 5-fold CV is a few minutes.
- **You do not need the GPU for the tree models.** 16 GB of VRAM is irrelevant to
  gradient-boosted trees at this size; `device="gpu"` will likely be *slower* than CPU
  here because of kernel launch overhead on such a small dataset.
- The only thing that would use the GPU is a neural tabular model (TabNet / FT-Transformer),
  which would consume well under 2 GB of the 16 GB and — see
  [04-ensemble-diversity](04-ensemble-diversity/) — is unlikely to beat the GBDTs on
  its own. Spend GPU quota only if you've exhausted the ideas above.

Practical implication: you can iterate entirely in CPU notebooks and never burn GPU quota.

## Folder map

- **[00-foundation/](00-foundation/)** — shared preprocessing, CV protocol, the reusable
  metric function, submission-format traps. Everything else assumes this.
  - [metric-decomposition.md](00-foundation/metric-decomposition.md) — what the scoring
    formula actually rewards. **Read this one; it's the most useful document here.**
- **[01-gbdt-core/](01-gbdt-core/)** — the workhorse models.
- **[02-feature-engineering/](02-feature-engineering/)** — feature blocks to try, each
  independently validatable.
- **[03-threshold-engine/](03-threshold-engine/)** — converting probabilities into the
  binary column without leaving points on the table.
- **[04-ensemble-diversity/](04-ensemble-diversity/)** — blending, stacking, and an honest
  assessment of neural tabular models here.
- **[05-label-noise/](05-label-noise/)** — the 3.3% conflicting-label rows.
- **[06-dead-ends/](06-dead-ends/)** — things that were tested and **do not work**.
  Reading this costs 10 minutes and saves several hours.

## Source of the numbers

All EDA findings referenced here come from `dataset_exploration/` (scripts `01`–`10`
and its `README.md`). The additional measurements quoted in these idea docs
(metric decomposition, degenerate floors, HistGradientBoosting baseline, exact-match
lookup evaluation) were produced while writing them and are reproducible from the
protocol in [00-foundation/](00-foundation/).
