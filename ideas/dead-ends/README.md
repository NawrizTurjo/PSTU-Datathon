# Dead ends — measured negatives

**Read this before proposing anything.** Every entry was checked against the real data in Stage 1
(`../../dataset_exploration/`) and ruled out. Ten minutes here saves hours of re-deriving things
that are already settled.

Two categories: things that were **measured and found absent**, and things that are
**prohibited or structurally impossible**.

---

## 1. Leak hunting — exhaustively checked, nothing there

Four independent probes, all negative (`08_leak_diagnostics_report.txt`):

| Probe | Result | Meaning |
|---|---|---|
| Row-index AUC | **0.5047** | row order carries nothing; positive rate flat (3.4–4.5%) across all 10 index blocks |
| Best single-feature AUC | **0.6986** (`feat_175`) | no magic feature — a leak would show >0.85–0.90 |
| Capacity test (unregularized tree) | train **1.0000** → held-out **0.5916** | no deterministic rule exists |
| Duplicate-group label consistency | N/A | zero duplicate rows to check |

**Do not propose:** "find the hidden formula", magic-feature hunts, row-order/index features,
`id`-derived features, or any variant of reverse-engineering the label. The capacity test is the
decisive one — an unregularized tree that can memorize the training set perfectly still gets
0.59 on held-out data. There is no rule to find. **The score has a genuine ceiling, and the way
to approach it is ordinary modelling plus a correct operating point.**

The good news buried in this: because there is no leak *and* no label noise, the ceiling is
real but honest. Unlike the previous (withdrawn) competition dataset, higher scores here are
genuinely earnable.

---

## 2. Duplicate-row deduplication and label-noise cleanup — nothing to clean

Measured (`06_duplicate_rows_report.txt`):

- **0** duplicate feature-rows in train
- **0** in test
- **0** rows shared between train and test on features alone

**Do not propose:** deduplication, label-conflict resolution, noise-robust losses motivated by
duplicate conflicts, or "clean the training set" ideas. There is nothing to deduplicate and no
measured label noise to model. (Noise-robust losses may still be defensible on general grounds —
but not with *this* justification, and they are not recommended in a four-day sprint.)

---

## 3. `feat_169` as a sentinel — checked, it is not one

The first-pass profile flagged `feat_169`'s minimum of ≈ **−1.11e8** as a possible second
sentinel. Measured (`03_numeric_profile_report.txt`): it is a **genuine heavy-tailed
continuum** — the minimum is −111,227,971.87, the second-smallest distinct value is
−108,184,164.08, and no value repeats at the extreme. A sentinel is a repeated constant; this
is not one.

**Do not propose:** treating `feat_169` as missing, or special-casing its extreme values as
encoded nulls. Ordinary robust scaling or clipping is sufficient.

*(Contrast: the **real** sentinels — `-999999` in `feat_109`, and `9999999999` across 23
columns — are handled in [`../00-foundation/`](../00-foundation/). The second one was **not** in
the first-pass profile and is easy to miss.)*

---

## 4. The "83 droppable columns" figure — does not reproduce, use 44

An early estimate claimed 83 droppable columns (28 constant + 55 redundant across 20 duplicate
groups). Measured directly (`04_constant_duplicate_report.txt`):

| Method | Groups | Redundant | Total droppable |
|---|---|---|---|
| **Exact row-for-row duplicate** | 16 | 16 | **44** (with the 28 constants) |
| Pearson `\|corr\| > 0.999` | 36 | 49 | 77 |
| *Original estimate* | *20* | *55* | *83* |

The 55/20 figure does not reproduce under either method. **Use 44 as the safe drop list.**

**Do not propose:** re-deriving the 83 figure, or hunting for the "missing" 39 columns. It was
a first-pass artifact. The corr>0.999 groups are documented in the Stage 1 report and are
legitimate *dimensionality-reduction candidates* — but not safe automatic drops, since a scaled
copy of a column can still carry distinct information a tree model exploits.

One nuance worth keeping: **14 columns are constant in test but vary in train.** Keep them. They
contribute nothing to test-time discrimination but a model can still learn from their
train-time variance, and dropping them is a small, unnecessary loss.

---

## 5. External data, Santander joins, and anything rule-adjacent

The data's fingerprint (76,020 rows, exactly 3,008 positives, the `-999999` sentinel) matches
the public **Santander Customer Satisfaction** dataset, and test `id`s run 0–75,817 in line with
that competition's indices.

**This is recorded to justify structural modelling choices only** — extreme sparsity, sentinel
handling, and row-aggregate features being effective ([03](../03-feature-engineering/)).

**Do not propose, under any framing:**

- joining external data of any kind — **explicitly prohibited by the rules**
- looking up Santander labels or IDs — **instant disqualification**, and moot anyway since that
  competition's true test labels were never public
- pre-trained models without explicit disclosure
- generating target values with an LLM — **explicitly banned**

There is no upside available here and the downside is disqualification.

---

## 6. GPU training for the tree models

**Do not propose.** At 76,020 × ~306 columns, CPU LightGBM/XGBoost trains a 5-fold model in
minutes; GPU adds host↔device transfer overhead at this size and burns the ~30 h/week Kaggle GPU
cap. Kaggle **CPU** notebooks have no weekly quota. GPU is only defensible for something that
genuinely needs it (a neural approach), which brings us to:

---

## 7. Deep learning / neural tabular models

Not measured — ruled out on cost/benefit for a four-day sprint, so treat this as a **judgement
call rather than a measured negative.**

The reasoning: 76,020 rows with 3,008 positives and 344 mostly-sparse numeric features is
squarely GBDT territory. A TabNet/FT-Transformer/MLP would need its own preprocessing pipeline
(scaling, embedding the 2,333-level categoricals), its own tuning budget, GPU hours against a
capped quota, and its own determinism work for the mandatory inference notebook — to most likely
land below a tuned LightGBM. If someone has spare time on day 3, the highest-value use of it is
[05](../05-ensemble-diversity/), not a new model family.

**Only reconsider if:** LightGBM tuning plateaus well below expectations *and* there is a full
day spare *and* the inference notebook is already verified.

---

## 8. Tuning against the public leaderboard

**Do not.** The public LB is **10% of test data** (~6,065 rows, ~240 positives) and only **10%
of the final grade**. F1 on ~240 positives has sampling noise on the order of ±0.02–0.03 — larger
than the entire expected gain from [02](../02-gbdt-core/) and [03](../03-feature-engineering/)
combined.

Use the public LB for exactly two things:

1. The **all-zeros metric probe** (binary vs macro F1 — a decisive, unambiguous signal).
2. A **coarse CV-vs-LB gap check** ([04](../04-shift-robustness/) step 1).

Select models on OOF, not on LB. Private (40%) + hidden (40%) dwarf it.

---

## 9. SMOTE / synthetic oversampling — allowed, but not recommended first

Permitted by the rules (on training data only), so not a dead end in the strict sense — but
deprioritized deliberately.

Reasoning: GBDTs handle imbalance through `scale_pos_weight`/`is_unbalance` far more cheaply,
and that is already a measured A/B arm in [02](../02-gbdt-core/) step 3. More importantly,
**the threshold engine already recovers most of what oversampling is trying to achieve** —
reweighting mostly rescales the probability distribution, which the tuned cut then undoes.
Measured evidence for that mechanism: moving from a 0.5 cut to the tuned cut is worth +0.207
binary-F1, dwarfing anything class-rebalancing typically delivers.

SMOTE also interpolates between neighbours in a **252-of-344-columns-are-≥90%-zero** feature
space, where "between two rows" is not a meaningful point, and it adds a fitted, stateful step
to the pipeline feeding [06](../06-inference-notebook/).

**Revisit only if** all three imbalance arms in [02](../02-gbdt-core/) prove inadequate and
there is time left over.
