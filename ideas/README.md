# Solution ideas — index & priority

Every number in this folder is **measured** on the real competition data (Stage 1 EDA,
`../dataset_exploration/`), not estimated — unless it explicitly says *estimate* or *guess*.
Expected-gain figures are honest ranges with stated confidence; several are "could be zero."

**Today is 2026-08-09. Final submission 13 Aug 18:00 GMT+6 — about 4 days.**

---

## The competitive landscape

| Strategy | Binary F1 | Macro F1 | OOF AUC |
|---|---|---|---|
| Submit all zeros | **0.0000** ← confirmed on public LB | 0.4899 | — |
| Submit all ones | 0.0761 | 0.0381 | — |
| Baseline GBDT (HistGBM), probs submitted raw (0.5 cut) | 0.1777 | 0.5791 | 0.8868 |
| Baseline GBDT (HistGBM), tuned operating point (t≈0.19) | 0.3841 | 0.6786 | 0.8868 |
| **Real LightGBM run-1 (pre-search) — OOF** | 0.3954 (t=0.1736) | — | 0.8920 |
| **Real LightGBM run-1 (pre-search) — public LB** | **0.1849** ← measured, 2026-08-09 | — | — |
| **Real LightGBM run-2 (arm search: `drop_top5_shift`) — OOF** | 0.3957 (t=0.1830) | 0.6843 | 0.8928 |
| **Real LightGBM run-2 (shift holdout F1)** | 0.4068 *(confounded, higher pos rate)* | 0.6873 | — |
| **Real LightGBM run-2 — public LB** | *pending submission* | — | — |
| Top public-LB team (all competitors, at time of writing) | 0.2955 | — | — |
| ~~Realistic well-executed ceiling~~ ~~0.42–0.45~~ | **~0.28–0.32** *(revised estimate)* | ~0.70–0.72 *(estimate, unchanged)* | ~0.89–0.90 |

OOF baseline = `HistGradientBoostingClassifier`, 5-fold stratified OOF, 306 columns after the
44-column cleanup, sentinels → NaN, no tuning, no feature engineering.
See `../dataset_exploration/09_honest_baseline_report.txt`.

**The LB row is a real submission from `solution/pstu_train.ipynb` on Kaggle** (confirmed
`backend: lightgbm`, not the local stand-in). Submission positive rate 2,597/60,654 = 0.0428
vs its own OOF rate 0.0494 (ratio 0.87) — same ~0.86–0.87 rate-shrinkage pattern measured
locally with the HistGBM stand-in, now confirmed with the real model too. That ratio alone does
not explain a roughly 2× drop in binary-F1 from OOF to LB, though — most of the gap is real
generalization loss, not a rate artifact, and it lines up with the measured adversarial
train/test AUC of **0.5742**. This is the textbook case `../ideas/04-shift-robustness/`'s
step 1 describes: **LB well below OOF → the shift is real, proceed to that idea's steps 2–4**
(the `feat_182` audit, clipping — already applied by default in `pstu_train.py` — and
adversarial weighting as a last resort).

**Ceiling estimate revised down, on real evidence.** The original ~0.42–0.45 guess was made
before any submission existed. With a live leaderboard now visible — top entry **0.2955**, six
teams clustered in **0.20–0.30** — that guess was too optimistic by a wide margin. **~0.28–0.32
binary-F1 is the realistic target**, not ~0.44. This matters for planning: don't chase
+0.10–0.15 F1 through feature engineering alone; +0.02–0.05 from closing the OOF↔LB gap
([04](04-shift-robustness/)) plus incremental tuning ([02](02-gbdt-core/)) is what's actually on
the table, and it's still enough to move several places on this board.

**Four things follow, and they should shape everything:**

1. **The operating point is still worth more than the model.** Submitting the same model's raw
   probabilities against the grader's fixed 0.5 cut scores **0.1777** binary-F1 (HistGBM
   baseline); submitting hard labels at the tuned cut scores **0.3841** OOF / **0.1849** on the
   real LB with real LightGBM. The mechanic's OOF-measured value (+0.207) still dwarfs every
   modelling idea in this folder — the fact that LB came in lower doesn't change that the
   mechanic itself is correct and mandatory; it means the *model* needs to close a gap the
   mechanic can't.
2. **The OOF↔LB gap is now confirmed real, not hypothetical.** [04-shift-robustness](04-shift-robustness/)
   moves up in priority accordingly — it's no longer a "maybe, low confidence" item, it's
   responding to a measured 2× score drop.
3. **Modelling gains from here are still incremental, not transformational.** Untuned HistGBM
   hits OOF AUC 0.8868; real LightGBM got to ~0.887–0.895. The remaining headroom is in closing
   the generalization gap, not in finding a bigger OOF number.
4. **60% of the grade is not the public leaderboard.** Private LB 40% + hidden test 40% +
   presentation/code 10% vs public LB 10%. The confirmed shift means even private/hidden may
   undershoot OOF similarly — budget expectations accordingly. A broken or non-deterministic
   inference notebook forfeits 40% outright — that is still worth more than every score idea
   here.

---

## Priority order

Ordered by (points at stake ÷ hours), not by novelty.

| # | Idea | Effort | Expected gain | Confidence |
|---|---|---|---|---|
| [00](00-foundation/) | **Foundation** — preprocessing contract, CV protocol, metric fns | 1–2h | prerequisite | — |
| [01](01-threshold-engine/) | **Threshold engine** — the operating point & submission mechanic | 1–2h | **+0.207 OOF vs naive**; done, see caveat below | **high (measured)** |
| [06](06-inference-notebook/) | **Inference notebook** — mandatory, worth 40% of the grade | 3–4h | protects **40%** of final mark | **high** |
| [04](04-shift-robustness/) | **Shift robustness** — closing the confirmed OOF→LB gap | 2–4h | **+0.02–0.06 F1 on LB** *(upgraded — gap is now measured, not hypothetical)* | **medium-high** |
| [02](02-gbdt-core/) | **GBDT core** — real LightGBM/XGBoost/CatBoost + tuning | 3–5h | +0.005–0.020 F1 | medium-high |
| [03](03-feature-engineering/) | **Feature engineering** — row-aggregates, categorical encoding | 4–8h | +0.005–0.020 F1 | medium |
| [05](05-ensemble-diversity/) | **Ensemble & diversity** — seed averaging, rank blending | 2–4h | +0.003–0.010 F1 | medium |
| [dead-ends](dead-ends/) | **Dead ends** — measured negatives, read before proposing anything | 10 min read | saves hours | high (measured) |

**01's caveat:** the mechanic is proven and non-negotiable (never submit raw probabilities), but
its OOF-measured gain doesn't fully transfer — LB came in at 0.1849 against an OOF binary-F1 of
~0.39–0.40. That's not a flaw in the mechanic; it's the same OOF↔LB gap idea 04 addresses.

**04 moved up** from its original "low-medium confidence, maybe zero" ranking. It was written
against a *measured but unexercised* signal (adversarial AUC 0.5742); a real submission has now
confirmed LB lands well below OOF, which is exactly the trigger condition idea 04's step 1
describes for proceeding to steps 2–4.

### Status (2026-08-09) and revised order

```
Day 1   00 → 01 → 06 skeleton → real-LightGBM submission → metric probe   ✅ DONE
                  LB confirmed: binary F1, 0.1849, OOF→LB gap confirmed real
Day 2   04 (close the confirmed gap: feat_182 audit, clipping check, resubmit)
Day 3   02 (tuning pass), 03 (features) — only after 04 shows what survives the shift
Day 4   05 (ensemble) if time remains, freeze, finalize 06 + presentation
```

**04 now comes before 02/03.** Spending a day tuning or engineering features against an OOF
metric that's already known to overstate LB by ~2x risks optimizing the wrong thing — find out
which features/techniques survive the shift first, *then* tune on top of that.

---

## Before anything else: run the metric probe

The competition page contradicts itself — evaluation section says "F1 Score", submission section
says thresholded-at-0.5 "**Macro F1**". These are wildly different objectives at a 3.96%
positive rate.

**Submit all zeros as your first submission.** Measured floors:

- LB ≈ **0.4899** → grader uses **macro F1**
- LB ≈ **0.0000** → grader uses **binary F1**

One submission resolves it. Until it is resolved, tune and report **both** metrics — every idea
in this folder is written to work either way, but the optimal aggressiveness of the operating
point differs between them, so do not skip this.

*(Convenient measured coincidence: on the baseline model both metrics peaked at the same
threshold, t=0.19. Do not rely on that holding after the model changes — re-check each time.)*

---

## What the data is (one-screen summary)

Full detail in [`../dataset_exploration/README.md`](../dataset_exploration/README.md).

- 76,020 train × 350 features; 60,654 test. **No missing values.** Positive rate **3.9569%**
  (3,008 / 76,020).
- 344 numeric + **6 categorical string** columns (`feat_142` 2,333 levels, `feat_325` 1,710,
  `feat_157` 627, `feat_320` 119, `feat_337` 39, `feat_318` 12). All have clean prefixes;
  three have unseen levels in test (≤0.15% of rows) — **encodings need an unseen fallback.**
- **Two sentinels:** `-999999` in `feat_109` only, and **`9999999999` across 23 columns.**
  Both → missing.
- **252 of 344 numeric columns are ≥90% zero.** Heavily sparse.
- **44 columns safely droppable** (28 constant in both splits + 16 exact duplicates).
- **Adversarial train-vs-test AUC 0.5742** — real, moderate covariate shift. `feat_182` drives
  it (~2.4× the next feature).
- **No leaks.** Row-index AUC 0.5047; best single-feature AUC 0.6986; unregularized tree hits
  train AUC 1.0 but held-out 0.5916. There is a genuine generalization ceiling — see
  [`dead-ends/`](dead-ends/).
- **Zero duplicate rows, zero label conflicts.** Nothing to deduplicate.

---

## Hard rules (violating any of these ends the run)

- ❌ No external datasets. Only the three provided files.
- ❌ No test-set tampering or label reverse-engineering — instant disqualification.
- ❌ Do not generate target values with an LLM — explicitly banned.
- ⚠️ Pre-trained models only with disclosure.
- ✅ SMOTE / augmentation / feature engineering **on training data** is allowed.
- ✅ Inference notebook mandatory, must be **deterministic** — every seed fixed, fold split
  pinned, artifacts saved.
