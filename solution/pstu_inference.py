# %% [markdown]
# # PSTU Data Thon 2026 Vol-1 — Inference notebook (MANDATORY DELIVERABLE)
#
# Loads `artifacts.joblib`, runs on an arbitrary test file, writes `submission.csv`.
# **Deterministic**: it loads and predicts, it never fits anything.
#
# This notebook is graded on the hidden test — 40% of the final mark. Its most likely failure
# modes are measured properties of this dataset, and each has an explicit guard below:
#
# | Failure mode | Measured basis | Guard |
# |---|---|---|
# | Unseen categorical level | `feat_142` has 55 test-only levels (0.15% of rows) | `.fillna()` fallback + logged count |
# | Column-order mismatch | — | reindex to saved `feature_order`, assert |
# | Feature logic drift vs training | — | sha256 of `build_features` asserted equal |
# | `id` regenerated with `range()` | ids are non-contiguous | ids read from the test file, asserted |
# | Values outside train range | 93 of 344 numeric cols | clip to saved train bounds |
#
# **To point this at a different test file, change `TEST_PATH` in the config cell. Nothing else.**

# %%
import os
import gc
import random
import hashlib
import inspect
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = os.environ.get(
    "PSTU_OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
)
os.makedirs(OUT_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")

# %% [markdown]
# ## 1. Locate the artifacts and the test file
#
# On Kaggle, attach the training run's output as a dataset — `artifacts.joblib` then lives under
# `/kaggle/input/<your-dataset-slug>/`. Both that and the local path are probed.

# %%
ARTIFACT_CANDIDATES = [
    "/kaggle/input/pstu-artifacts/artifacts.joblib",
    "/kaggle/working/artifacts.joblib",
    "artifacts.joblib",
    "solution/artifacts.joblib",
]
# any *.joblib inside /kaggle/input, as a fallback for a differently-named dataset
if os.path.isdir("/kaggle/input"):
    for root, _dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.endswith(".joblib"):
                ARTIFACT_CANDIDATES.append(os.path.join(root, fn))

ARTIFACT_PATH = next((p for p in ARTIFACT_CANDIDATES if os.path.exists(p)), None)
if ARTIFACT_PATH is None:
    raise FileNotFoundError(f"artifacts.joblib not found. Looked in: {ARTIFACT_CANDIDATES}")
print("ARTIFACT_PATH =", ARTIFACT_PATH)

DATA_CANDIDATES = [
    "/kaggle/input/competitions/pstu-data-thon-2026-vol-1",
    "/kaggle/input/pstu-data-thon-2026-vol-1",
    "pstu-data-thon-2026-vol-1",
    "../input/competitions/pstu-data-thon-2026-vol-1",
    "../input/pstu-data-thon-2026-vol-1",
    "../pstu-data-thon-2026-vol-1",
]
DATA_DIR = next((d for d in DATA_CANDIDATES if os.path.exists(os.path.join(d, "test.csv"))), None)
if DATA_DIR is None:
    raise FileNotFoundError(f"test.csv not found. Looked in: {DATA_CANDIDATES}")

TEST_PATH = os.path.join(DATA_DIR, "test.csv")   # <-- repoint here for the hidden test
print("TEST_PATH =", TEST_PATH)

art = joblib.load(ARTIFACT_PATH)
print(f"loaded {len(art['models'])} models | backend={art['backend']} | "
      f"OOF AUC={art['oof_auc']:.4f} | threshold={art['threshold']:.6f}")

# %% [markdown]
# ## 2. Feature builder — byte-identical copy of the training function
#
# Do **not** edit this function in isolation. If the training notebook's copy changes, this one
# must be updated to match; the sha256 assertion below is what enforces that.

# %%
TARGET = "TARGET"
ID = "id"
CAT_COLS = art["cat_cols"]
SENTINEL_NEG_COL = list(art["sentinel_neg"].keys())[0]
SENTINEL_NEG_VAL = art["sentinel_neg"][SENTINEL_NEG_COL]
SENTINEL_BIG_COLS = art["sentinel_big_cols"]
SENTINEL_BIG_VAL = art["sentinel_big_val"]


def build_features(df, drop_cols, maps=None, clip_bounds=None):
    """Returns (X, maps, clip_bounds). maps=None -> fit mode."""
    fitting = maps is None
    if fitting:
        maps = {"freq": {}, "code": {}}

    raw = df.drop(columns=[c for c in (TARGET, ID) if c in df.columns])
    out = pd.DataFrame(index=raw.index)

    # --- sentinel indicators (BEFORE nulling) ---
    if SENTINEL_NEG_COL in raw.columns:
        out["sent_neg"] = (raw[SENTINEL_NEG_COL] == SENTINEL_NEG_VAL).astype(np.int8)
    big_cols = [c for c in SENTINEL_BIG_COLS if c in raw.columns]
    out["sent_big_count"] = sum(
        (raw[c] == SENTINEL_BIG_VAL).astype(np.int8) for c in big_cols
    ) if big_cols else np.int8(0)

    # --- sentinels -> NaN ---
    work = raw.copy()
    if SENTINEL_NEG_COL in work.columns:
        work[SENTINEL_NEG_COL] = work[SENTINEL_NEG_COL].replace(SENTINEL_NEG_VAL, np.nan)
    for c in big_cols:
        work[c] = work[c].replace(SENTINEL_BIG_VAL, np.nan)

    # --- drop constant / duplicate / shift-flagged columns ---
    work = work.drop(columns=[c for c in drop_cols if c in work.columns])

    num_cols = [c for c in work.columns if c not in CAT_COLS]
    num = work[num_cols]

    # --- clip to train range (fitted on train, applied everywhere) ---
    if fitting:
        clip_bounds = {"lo": num.min(), "hi": num.max()}
    num = num.clip(lower=clip_bounds["lo"], upper=clip_bounds["hi"], axis=1)

    # --- row-wise aggregates over the sparse numeric block ---
    nonzero = (num != 0)
    out["agg_n_nonzero"] = nonzero.sum(axis=1).astype(np.int16)
    out["agg_n_zero"] = (num == 0).sum(axis=1).astype(np.int16)
    out["agg_n_nan"] = num.isna().sum(axis=1).astype(np.int16)
    out["agg_n_negative"] = (num < 0).sum(axis=1).astype(np.int16)
    out["agg_sum"] = num.sum(axis=1)
    out["agg_mean_nonzero"] = num.where(nonzero).mean(axis=1)
    out["agg_std"] = num.std(axis=1)
    out["agg_max"] = num.max(axis=1)
    out["agg_min"] = num.min(axis=1)

    # --- categorical encodings ---
    for c in CAT_COLS:
        if c not in work.columns:
            continue
        if fitting:
            maps["freq"][c] = work[c].value_counts(normalize=True).to_dict()
            maps["code"][c] = {lvl: i for i, lvl in enumerate(sorted(work[c].unique()))}
        out[f"{c}_freq"] = work[c].map(maps["freq"][c]).fillna(0.0).astype(np.float32)
        out[f"{c}_code"] = work[c].map(maps["code"][c]).fillna(-1).astype(np.int32)

    # --- numeric passthrough ---
    out = pd.concat([out, num], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, maps, clip_bounds


def fn_source_hash(fn):
    """sha256 of a function's source, or None where source is unavailable (bare `exec`)."""
    try:
        return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return None


_hash = fn_source_hash(build_features)
_ref = art.get("feature_fn_hash")
if _hash is None or _ref is None:
    print("WARNING: function source unavailable — drift check skipped. "
          "Verify manually that build_features matches the training notebook.")
else:
    assert _hash == _ref, (
        "build_features has DIVERGED from the training notebook's copy.\n"
        f"  training: {_ref}\n  here:     {_hash}\n"
        "Predictions would be silently wrong. Sync the two functions before proceeding."
    )
    print("build_features matches training copy:", _hash[:16])

# %% [markdown]
# ## 3. Load the test file and report unseen categorical levels
#
# The unseen-level counts are printed deliberately. If the hidden test carries far more than the
# ~0.15% measured on the public test set, this log line is the only way anyone will ever know
# why the score moved.

# %%
test = pd.read_csv(TEST_PATH)
print("test:", test.shape)

if ID not in test.columns:
    raise KeyError(f"'{ID}' column missing from {TEST_PATH}")
test_ids = test[ID].copy()          # captured BEFORE any preprocessing

for c in CAT_COLS:
    if c in test.columns:
        n_unseen = int((~test[c].isin(art["cat_maps"]["code"][c].keys())).sum())
        print(f"  {c}: {n_unseen} rows with unseen levels "
              f"({n_unseen / len(test):.4%}) -> fallback bucket")

# %%
X_test, _, _ = build_features(
    test, art["drop_cols"], maps=art["cat_maps"], clip_bounds=art["clip_bounds"]
)

missing = [c for c in art["feature_order"] if c not in X_test.columns]
extra = [c for c in X_test.columns if c not in art["feature_order"]]
assert not missing, f"features missing at inference: {missing[:10]} ({len(missing)} total)"
if extra:
    print(f"note: dropping {len(extra)} unexpected columns: {extra[:5]}")

X_test = X_test[art["feature_order"]]      # exact training column order
assert list(X_test.columns) == art["feature_order"], "column order mismatch"
print("feature matrix:", X_test.shape)

del test
gc.collect()

# %% [markdown]
# ## 4. Predict
#
# Average every saved fold/seed model, exactly as the training run did, then apply the saved
# threshold. No refitting, no recalibration, no threshold re-selection — those all happened at
# training time and their outputs are in the artifacts.

# %%
proba = np.zeros(len(X_test))
for entry in art["models"]:
    proba += entry["model"].predict_proba(X_test)[:, 1] / len(art["models"])

pred = (proba >= art["threshold"]).astype(int)

rate = pred.mean()
expected = art["expected_pred_pos_rate"]
print(f"predicted positives: {int(pred.sum())} / {len(pred)}  (rate {rate:.4f})")
print(f"expected rate from OOF: {expected:.4f}  |  ratio {rate / expected:.3f}")
if not (0.5 < rate / expected < 2.0):
    print("WARNING: predicted-positive rate is far from the OOF rate. "
          "Check for distribution shift or a preprocessing mismatch.")

# %% [markdown]
# ## 5. Write and validate the submission

# %%
sub = pd.DataFrame({ID: test_ids.values, TARGET: pred})

assert list(sub.columns) == [ID, TARGET], f"bad columns: {list(sub.columns)}"
assert len(sub) == len(test_ids), "row count mismatch"
assert sub[ID].equals(test_ids.reset_index(drop=True)), "ids reordered"
assert sub[ID].is_unique, "duplicate ids"
assert sub[TARGET].isin([0, 1]).all(), "TARGET must be 0/1"
assert not sub.isna().any().any(), "NaNs in submission"
assert 0.01 < sub[TARGET].mean() < 0.15, f"implausible positive rate {sub[TARGET].mean():.4f}"

sub.to_csv(SUBMISSION_PATH, index=False)
print(f"wrote {SUBMISSION_PATH}  ({len(sub)} rows, {int(sub[TARGET].sum())} positives)")
print(sub.head())

# %% [markdown]
# ## 6. Reproduction check
#
# When run against the same `test.csv` the training notebook used, the output must be
# **bit-identical** to the submission that run produced. If this assertion fails, the pipeline is
# not reproducible — find out here, not on the hidden test.
#
# On a genuinely different test file the check is skipped (different row count), which is the
# expected path during the real hidden-test run.

# %%
ref = art.get("train_submission_preds")
if ref is not None and len(ref) == len(pred):
    n_diff = int((np.asarray(ref) != pred).sum())
    if n_diff == 0:
        print(f"REPRODUCTION OK: all {len(pred)} predictions match the training run exactly.")
    else:
        raise AssertionError(
            f"REPRODUCTION FAILED: {n_diff} / {len(pred)} predictions differ from the "
            "training run. The pipeline is not deterministic — do not submit."
        )
else:
    print(f"Different test file ({len(pred)} rows vs {len(ref) if ref is not None else '?'} "
          "at training time) — reproduction check skipped, as expected for the hidden test.")

print("\nInference complete.")
