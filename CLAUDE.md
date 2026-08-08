# CLAUDE.md — project context

> ## ⛔ READ FIRST — THE DATASET IS BEING REPLACED
>
> The organizers announced (2026-08-08) that **the released dataset contained leaks** and
> that **the entire dataset will be re-uploaded**.
>
> **Every measured number in this repo was derived from the leaky dataset and must be
> treated as unverified until re-run against the new data.** The *tooling* carries over;
> the *findings* do not. See [What survives the re-upload](#what-survives-the-re-upload).
>
> First action next session: re-download the data, then re-run
> `dataset_exploration/01`–`10` and diff the reports against the ones committed here.

## Project

Kaggle competition: [PSTU Data Craft](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact).
Predict 7-day critical failure in off-grid solar water-pumping stations in coastal Bangladesh.
Binary classification, custom weighted composite metric, submission needs both a binary
label and a probability.

**Status:** EDA pipeline, solution roadmap and a runnable Kaggle notebook are all built and
smoke-tested — but against the withdrawn dataset. Nothing has been submitted to the
leaderboard. Work is paused pending the re-upload.

---

## What happened in the last session (important context)

The user shared the public leaderboard. The score distribution was sharply **trimodal**:

| Rank | Score | Note |
|---|---|---|
| 1 | **1.000000000000** | marked **Disqualified** |
| 2–5 | 0.967 – 0.983 | 4 teams |
| 6–7 | 0.641 – 0.649 | 2 teams |
| 8–12 | 0.535 – 0.538 | 5 teams |

Our honest pipeline scored **0.5309 OOF** — landing exactly in the bottom cluster. That
clustering was the tell: three discrete plateaus, not a continuum, means three different
*discoveries*, not three levels of modelling skill.

An investigation was run before the announcement arrived, and its conclusions are worth
keeping because they are about the **structure of the problem**, not the specific data:

1. **No row-order/index leak.** Positive rate flat (~0.05) across all 20 index blocks;
   AUC using raw row index = 0.4884; target autocorrelation ≈ 0 at all lags.
2. **No magic feature.** Best single-feature AUC = **0.7005**
   (`base_station_installation_age_years`).
3. **No learnable deterministic rule.** An unregularized deep model reached
   **train AUC 0.9986 but validation AUC 0.7724** — *worse* than the regularized model's
   0.828. Capacity bought memorization of noise, not signal.
4. **The target was provably not a function of the features.** 3.3% of train rows sat in
   duplicate groups with byte-identical features and *opposite* labels.

Conclusion at the time: a 0.98 score was **not reachable from the features**, so it had to
come from something structural. The organizers' leak announcement confirms that read.

**A metric-implementation hypothesis was mid-test when the session stopped** — the idea that
the scorer might use `average='weighted'`, which would make conservative submissions score
~0.89 by accident. **That hypothesis is superseded and should not be carried forward as a
finding.** The simpler explanation now fits: under the metric exactly as documented,
near-perfect predictions score ~1.0 (verified: top-5% predictions with an AUC-1.0
probability column give 0.9999). A leak alone explains the leaderboard. There is **no
evidence the metric is misimplemented.**

---

## What survives the re-upload

### ✅ Reusable — tooling and method

- **All 10 EDA scripts** in `dataset_exploration/`. They are generic: they *detect* schema,
  sentinels, constants, duplicates and shift rather than hardcoding them. Re-run as-is.
- **The whole notebook pipeline** (`solution/pstu_kaggle_solution.ipynb`): streaming Bengali
  decoder, CV harness, feature-engineering blocks, exhaustive O(n) cut-point optimiser,
  submission validator. *(One caveat — see Landmines below.)*
- **The metric decomposition** (pure algebra on the published formula — see below).
- **Kaggle runtime knowledge**: dataset is small, CPU beats GPU for trees, CPU notebooks
  have no weekly quota.
- **The diagnostic playbook** from the investigation above: index-leak test, single-feature
  AUC scan, capacity/memorization test, duplicate-group consistency test. Run these on the
  new data early — they cheaply reveal whether a leak still exists.

### ❌ Invalidated — every measured number

Re-verify all of these; do not cite them until re-run:

- `-999999` sentinel in `base_number_of_dependent_farmers` (was 66 train / 23 test rows)
- The 12 zero-information columns (6 constant + 6 duplicate)
- 63 Bengali boolean-text columns
- 5.00% positive rate; 48,128 train / 12,032 test
- Adversarial validation AUC 0.4985 (train/test iid)
- "No recoverable station id" finding
- 7.35% duplicate rows / 3.3% conflicting labels
- All benchmark scores (RF 0.5167, HGB 0.5269, HGB+FE 0.5309)
- **The entire `ideas/06-dead-ends/` list** — those were measured against leaky data. In
  particular the train→test exact-match finding is likely to behave completely differently.

### ⚠️ Landmines to fix before the next run

1. **`solution/pstu_kaggle_solution.py` hardcodes `CONSTANT_COLS` and `DUPLICATE_COLS`**
   (12 column names). On the new dataset these will very likely be wrong, and the notebook
   will silently drop the wrong columns or `errors="ignore"` past them.
   **Fix:** replace both lists with auto-detection — a column is droppable if
   `nunique() <= 1` in *both* train and test, or if it hashes identically to an earlier
   column. ~15 lines, and it makes the notebook dataset-agnostic.
2. **`FLAG_PAIRS` and `RISK_FLAGS`** in the same file assume the current column names.
   Guarded by `if col in df.columns`, so they degrade quietly rather than crash — check the
   printed feature count to confirm the blocks actually fired.
3. **The Bengali decoder assumes `হ্যাঁ`/`না` prefixes.** It auto-detects which columns are
   text-boolean, so it adapts, but verify the detected count is sane after re-download.

---

## The metric (durable — algebra, not measurement)

Official:
`0.30·F1 + 0.25·AUC + 0.15·Precision + 0.15·Recall + 0.10·BalancedAccuracy + 0.05·Specificity`

Substituting `BalAcc = (R+S)/2` collapses it **exactly** (verified to machine precision) to:

```
0.30·F1 + 0.25·AUC + 0.15·Precision + 0.20·Recall + 0.10·Specificity
```

**Recall outweighs precision** (0.20 vs 0.15) — the official form hides this by splitting
recall's weight across two terms.

Durable consequences:
- The two submission columns are scored **independently**: `Target_Probability` → AUC only;
  `Target_Binary` → the other 0.75. Optimize them separately.
- Cut-point tuning is worth a lot (was +0.018 to +0.028). Never submit a naive 0.5 threshold.
- The optimal cut-point is found **exhaustively in O(n)** by sorting once and taking
  cumulative sums (`cutoff_curve()` in the notebook) — no grid search, no Bayesian optimizer.
  This was verified against brute force to 1e-9.
- Degenerate floors depend on the class balance, so **recompute them** on the new data. On
  the old data: all-zeros 0.3028, all-ones 0.4388.

*(Only re-derive this if the organizers also change the metric. Check `overview.md` against
the new competition page.)*

---

## Repository map

| Path | What it is | Status after re-upload |
|---|---|---|
| `dataset/` | Raw competition CSVs (gitignored) | **replace** |
| `dataset_exploration/` | 10 EDA scripts + 16 reports + findings README | scripts ✅ / reports ❌ |
| `ideas/` | 7-folder prioritized solution roadmap | method ✅ / numbers ❌ |
| `solution/pstu_kaggle_solution.ipynb` | Runnable Kaggle notebook (26 cells) | ✅ after landmine fix |
| `solution/pstu_kaggle_solution.py` | Same source, `# %%` cell-marked | ✅ after landmine fix |
| `overview.md`, `dataset_description.md` | Competition-provided docs | **re-download** |
| `missing-exploration.md` | Gap-analysis checklist (all items addressed) | ✅ historical |
| `prompt.md` | Idea-generation prompt, pre-loaded with findings | ⚠️ contains stale numbers |
| `readme.md` | Public-facing README | ⚠️ contains stale numbers |

---

## Conventions (durable)

- **Never load the raw `train.csv` directly into pandas.** The boolean columns are stored as
  long Bengali sentences, which is why the file was 910 MB for 48k rows. Use the streaming
  decoder in the notebook, or `dataset_exploration/02_convert_to_numeric.py`.
- `converted_train.csv` / `converted_test.csv` are **generated artifacts** and are *not* in
  `.gitignore` (which only lists `dataset/`). Regenerate them; don't commit them. **Delete
  the stale ones when the new dataset lands** — they are from the withdrawn data.
- Submission format: exactly `id,Target_Binary,Target_Probability`; `id` = 0..n-1 in test row
  order; probabilities **strictly inside (0,1)** (clip to `1e-6`). An "Evaluation Error"
  scores nothing at all.
- **Don't enable the GPU for tree models** — at this data size it is slower than CPU, and
  Kaggle's CPU notebooks have no weekly quota (GPU has ~30 h/week).
- Local environment has `pandas`, `numpy`, `scikit-learn`, `scipy` but **not** lightgbm /
  xgboost / catboost. Kaggle has all three. Use `HistGradientBoostingClassifier` as the local
  stand-in — `scratchpad/smoke_test.py` patches the notebook to do exactly this.

---

## Next session — suggested order

1. **Re-download** the new dataset and the refreshed `overview.md` /
   `dataset_description.md`. Confirm whether the metric or submission format changed.
2. **Delete** stale `dataset_exploration/converted_*.csv` and the old report outputs.
3. **Re-run `dataset_exploration/01`–`10`** and read the regenerated reports fresh. Do not
   assume any prior finding holds.
4. **Run the leak diagnostics** from the investigation section above (index leak,
   single-feature AUC, capacity/memorization test, duplicate-label consistency). These
   directly answer "did they actually fix the leak?" and take minutes.
5. **Fix the hardcoded column lists** in the notebook (Landmine #1) so it is dataset-agnostic.
6. **Re-baseline**: run the notebook, get a fresh OOF composite, and recompute the
   all-zeros / all-ones floors for the new class balance.
7. Only then revisit `ideas/` — rewrite the measured numbers, and **rebuild
   `ideas/06-dead-ends/` from scratch** rather than trusting it.

**Expectation management:** with the leak removed, the leaderboard will almost certainly
compress toward the honest-modelling range. The old 0.535 cluster is the realistic
neighbourhood for competent tabular ML on this problem; a 0.90+ target was only ever
reachable through the leak. Set goals against the *new* leaderboard once it repopulates,
not against the withdrawn one.
