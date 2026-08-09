# 00 — Foundation

**Prerequisite, not an idea.** Everything else assumes these exist. ~1–2 hours.
Nothing here is optional and nothing here is speculative — every step is a direct consequence
of a measured Stage 1 finding.

## What it is

A single preprocessing function, a single CV protocol, and a single metric module, all written
once and reused by every experiment. The point is that every idea in this folder gets compared
on identical footing, and that the preprocessing used at training time is *the same code
object* used at inference time — which is what makes the mandatory inference notebook
reproducible.

## Why it should work (measured evidence)

Not a score idea — a correctness idea. But three of its five steps are directly worth score:

- The 44-column drop and sentinel handling are what took the baseline to OOF AUC 0.8868.
- The categorical unseen-level fallback is the difference between the hidden-test inference run
  working and **crashing** — measured: `feat_142`/`feat_325`/`feat_157` have levels in test that
  never appear in train (0.1467% / 0.0775% / 0.0181% of test rows). The hidden test is a
  *different* 40% of unseen data, so assume it has unseen levels too, likely more.
- A pinned fold split is what lets you compare idea 02 against idea 03 and believe the
  difference.

## Concrete steps

### 1. Path auto-detection

```python
CANDIDATE_DIRS = [
    "/kaggle/input/pstu-data-thon-2026-vol-1",
    "pstu-data-thon-2026-vol-1",
    "../input/pstu-data-thon-2026-vol-1",
]
DATA_DIR = next(d for d in CANDIDATE_DIRS if os.path.exists(os.path.join(d, "train.csv")))
```

Same notebook must run locally and on Kaggle without edits.

### 2. Column contract — constants, defined once

```python
CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]

SENTINEL_NEG = {"feat_109": -999999}          # measured: 116 train / 89 test rows
SENTINEL_BIG_COLS = [                          # measured: 9999999999 across 23 columns
    "feat_11","feat_21","feat_26","feat_30","feat_31","feat_36","feat_74","feat_77",
    "feat_96","feat_124","feat_135","feat_144","feat_149","feat_158","feat_171","feat_196",
    "feat_204","feat_226","feat_301","feat_315","feat_330","feat_336","feat_340",
]
```

`DROP_COLS` — the 44 safe drops — should be **read from
`../dataset_exploration/04_constant_duplicate_report.txt`**, or recomputed from train inside the
notebook. Do not hand-transcribe 44 column names; that is exactly where a typo hides.

### 3. Preprocessing function — one function, both splits

```python
def preprocess(df, drop_cols, cat_maps=None):
    """cat_maps=None -> fit mode (train). cat_maps=dict -> transform mode (test/hidden)."""
```

It must:

- drop `drop_cols`
- replace `-999999` in `feat_109` and `9999999999` in the 23 columns with `np.nan`
- encode the 6 categorical columns, **mapping any unseen level to a reserved code** (`-1`, or
  the train global mean for target encoding — never `NaN`-by-accident and never an error)
- return `(X, cat_maps)` so the fitted maps can be pickled and reloaded by the inference notebook

**Critical:** fit the categorical maps on **train only**. Fitting on `pd.concat([train, test])`
leaks test-set level identity and — more practically — cannot be reproduced on the hidden test,
which you will never see at training time.

Note the sentinel replacement is worth doing even for tree models that handle raw values: with
`9999999999` left in, a split threshold has to sit between a genuine value and a 10-billion
sentinel, wasting a split and distorting the histogram binning that LightGBM/HistGBM use.

### 4. Pinned CV protocol

```python
SEED = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

Stratified is required — 3.96% positives means an unstratified fold can swing the per-fold
positive count meaningfully. Generate the fold indices **once**, save them, and reuse across
every experiment. Measured baseline per-fold AUC spread was ±0.0060; that is the noise floor
you are comparing ideas against, so any "improvement" under ~0.006 AUC needs seed-averaging
before you believe it.

### 5. Metric module

```python
def score_both(y_true, y_pred_binary):
    return {
        "binary_f1": f1_score(y_true, y_pred_binary, average="binary", zero_division=0),
        "macro_f1":  f1_score(y_true, y_pred_binary, average="macro",  zero_division=0),
    }
```

Report both, always, until the LB probe resolves the ambiguity. Also log OOF AUC — not because
it is graded (it is not) but because it is the lowest-variance signal for "did this model
actually get better," which F1-at-a-threshold is not.

### 6. Submission builder + validator

```python
sub = sample_submission.copy()
sub["TARGET"] = preds          # test.csv row order matches sample_submission — measured
assert len(sub) == 60654
assert list(sub.columns) == ["id", "TARGET"]
assert sub["TARGET"].isin([0, 1]).all()
assert sub["id"].equals(sample_submission["id"])
```

`id` comes from `test.csv`'s **last column** and is neither `0..n-1` nor contiguous. Never
regenerate it with `range()`. Run the validator on every submission, every time.

## Kaggle cost

Minutes. `pd.read_csv` on the 128 MB train file takes seconds; no streaming needed
(measured ~235 MB in RAM). CPU only.

## Honest expected gain

**Prerequisite — no standalone gain.** But skipping the categorical fallback or the submission
validator has a measured downside of *the entire 40% hidden-test component*.

## When to abandon

Never. If any part of this is taking more than 2 hours, simplify it (drop fewer columns, use
plain ordinal encoding) and move on to [01](../01-threshold-engine/) — but do not skip the
unseen-level fallback or the submission validator under any time pressure.
