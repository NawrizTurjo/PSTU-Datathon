# 06 — Inference notebook (40% of the grade)

**Numbered 06 but scheduled third.** Build the skeleton on day 1, right after
[01](../01-threshold-engine/). ~3–4 hours total. This is the single highest-stakes item in the
roadmap and it is not a leaderboard activity.

## What it is

A standalone, mandatory notebook that loads saved model artifacts, runs on an arbitrary test
file it has never seen, and writes `submission.csv` — **deterministically**, producing exactly
the predictions you submitted. It is graded on the hidden test: 40% of *completely unseen* data,
run by the organizers, not by you.

## Why it should work (measured evidence)

This is a forfeit-avoidance item, so the "evidence" is about what breaks:

- **Unseen categorical levels are measured to exist.** `feat_142` has 55 levels in test that are
  absent from train (0.1467% of test rows), `feat_325` has 27, `feat_157` has 8. The hidden test
  is a *different, larger* slice of unseen data — assume proportionally more. A `LabelEncoder`
  that raises on unseen input, or a target-encoding map lookup that silently yields `NaN` into a
  model that cannot accept it, fails the run. This is the most likely single cause of a
  zero on the hidden test.
- **The measured 0.5742 adversarial AUC** means the hidden slice may shift further still. Range
  assumptions baked into preprocessing (e.g. hand-coded clip bounds) can be violated: 93 of 344
  numeric columns already have test values outside their train min/max.
- **`id` is not `0..n-1` and not contiguous** (measured). Any submission builder that regenerates
  ids with `range()` produces a correctly-shaped, completely misaligned file — which scores
  approximately zero and looks like a modelling failure rather than a plumbing one.
- **`TARGET` must be present in train but absent from test** (measured). Preprocessing code that
  assumes a `TARGET` column exists will crash on the hidden file.

## Concrete steps

### 1. Save every artifact at training time

```python
artifacts = {
    "models":     fitted_models,        # list, one per fold (or one refit on all data)
    "cat_maps":   cat_maps,             # fitted on TRAIN only
    "drop_cols":  drop_cols,            # the 44
    "threshold":  t_star,               # the chosen operating point
    "feature_order": list(X.columns),   # exact column order the model was fitted on
    "seed": SEED,
}
joblib.dump(artifacts, "artifacts.joblib")
```

`feature_order` is not optional. Reindexing the inference frame to it (`X = X[feature_order]`)
is what protects you from a column-order mismatch producing silently garbage predictions rather
than an error.

### 2. Write the inference notebook to take the test path as a variable

```python
TEST_PATH = os.path.join(DATA_DIR, "test.csv")   # organizers may point this elsewhere
test = pd.read_csv(TEST_PATH)
ids = test["id"].copy()                          # capture BEFORE any preprocessing
X = preprocess(test.drop(columns=["id"]), drop_cols, cat_maps=cat_maps)
X = X[feature_order]
```

Call **the same `preprocess()` function** from [00](../00-foundation/), in transform mode. Do not
reimplement preprocessing in the inference notebook — a divergence between the two copies is
undetectable until it costs you the 40%.

### 3. Determinism checklist

- Fix `random_state` / `seed` on **every** estimator, splitter, and any sampling step.
- Set `PYTHONHASHSEED`, `np.random.seed(SEED)`, `random.seed(SEED)`.
- For LightGBM/XGBoost set `deterministic=True` / `num_threads` pinned — multithreaded histogram
  construction can produce run-to-run float differences that move borderline predictions across
  the threshold.
- **Do not refit anything in the inference notebook.** It loads and predicts, nothing else.
- No wall-clock, no `time`-seeded anything, no unpinned `pd.factorize` on the inference frame
  (factorize order depends on row order — use the *saved train map*, always).

### 4. Prove it reproduces

```python
assert (new_preds == saved_train_time_preds).all(), "inference notebook diverged"
```

Run the inference notebook against the ordinary `test.csv`, and assert its output is
**bit-identical** to the `submission.csv` you actually uploaded. If it is not, you do not have a
reproducible pipeline — find out now, not on 13 Aug.

### 5. Run the full submission validator (from [00](../00-foundation/))

Row count 60,654, columns exactly `["id","TARGET"]`, values in {0,1}, ids equal to the input
file's ids in the input file's order. Plus the plausibility check from
[01](../01-threshold-engine/): positive count in roughly 1,500–4,500.

### 6. Defensive fallbacks, in the notebook itself

```python
X = X.replace([np.inf, -np.inf], np.nan)      # unexpected division artifacts from FE
unseen = ~test[c].isin(cat_maps[c].keys())     # log it, don't crash on it
print(f"{c}: {unseen.sum()} unseen levels mapped to fallback")
```

Print the unseen-level counts. If the hidden test has far more than the measured ~0.15%, that
line in the output log is the only way anyone will ever know why the score dropped.

## Kaggle cost

Minutes to run. A few hours to write and verify. CPU only — the notebook loads and predicts.

Keep the artifact file small enough to attach as a Kaggle dataset: 5 LightGBM models at these
dimensions is single-digit MB. If seed-averaging ([05](../05-ensemble-diversity/)) inflates that
to dozens of models, consider saving a single model refit on all data instead — and measure
whether that costs anything before committing.

## Honest expected gain

**Protects 40% of the final grade.** There is no score to be gained here, only a catastrophic
loss to be avoided. Expected value dominates every modelling idea in this folder by an order of
magnitude: a +0.02 F1 improvement is worth a fraction of one leaderboard component; a notebook
that fails to run is worth −40 points of 100.

Confidence: **high**, in the sense that the failure modes listed above are measured properties
of this dataset, not hypotheticals.

## When to abandon

Never. If time runs out on everything else, submit a simpler model with a working inference
notebook. If forced to choose between a better model and a verified notebook, choose the
notebook — the arithmetic is not close.

## Interaction with other ideas

- [03 feature engineering](../03-feature-engineering/) is the main threat here. Every engineered
  feature must be computable from a single test file with no access to train statistics beyond
  the saved artifacts. Row-wise aggregates are safe (per-row, no fitted state); target encoding
  is only safe if the encoding map is saved and reloaded, with a global-mean fallback.
- [05 ensembling](../05-ensemble-diversity/) multiplies the artifacts you must save and reload
  correctly. Rank-averaging in particular is computed *across the test set*, so it must be
  reproduced identically on a differently-sized hidden set — verify that the rank transform is
  applied per-file, not against saved train ranks.
