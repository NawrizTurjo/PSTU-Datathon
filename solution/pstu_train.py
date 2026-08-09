# %% [markdown]
# # PSTU Data Thon 2026 Vol-1 — Training notebook
#
# Trains the model, selects the operating point, writes `submission.csv`, and saves
# `artifacts.joblib` for the mandatory inference notebook.
#
# Built from the Stage-3 roadmap in `ideas/`. The measured facts driving every choice here:
#
# | Fact | Value | Consequence |
# |---|---|---|
# | Positive rate | 3.9569% | grader's fixed 0.5 cut is catastrophic — submit hard labels |
# | Raw probs @ 0.5 vs tuned cut | 0.1777 vs 0.3841 binary-F1 (OOF) | **+0.207** from the mechanic alone |
# | F1 plateau | t in [0.15, 0.21] all within 0.005 | pick plateau centre, not OOF argmax |
# | Per-fold AUC noise | +/-0.0060 | anything under that is not an improvement |
# | Sentinels | `-999999` in feat_109; `9999999999` in 23 cols | both -> NaN |
# | Droppable columns | 44 (28 constant + 16 exact dupes) | recomputed here, never hand-typed |
# | Unseen categorical levels | up to 0.15% of test rows | every encoder needs a fallback |
# | Adversarial train/test AUC | 0.5742 | real shift — clip to train range at inference |
# | **run-1, real LightGBM (2026-08-09)** | **OOF binary-F1 0.395 -> public LB 0.1849** | **the OOF/LB gap is now measured, not hypothetical** |
#
# **Metric — resolved.** All-zeros probe scored public LB 0.0000000, matching the measured
# binary-F1 floor exactly -> the grader uses **binary F1**. `TARGET_METRIC` below defaults to it.
#
# **New in this version:** section 5 builds a *shift-aware holdout* (the train rows an
# adversarial train-vs-test classifier says look most like test) and section 7 uses it to pick
# between candidate feature/imbalance configurations — something plain cross-validation cannot
# do, because every CV fold looks equally "trainy". This directly targets the measured run-1
# gap. It cannot guarantee a specific leaderboard score — the hidden test's true distribution
# isn't observable locally — but it optimizes against the actual measured failure mode instead
# of against an OOF number already known to overstate LB by roughly 2x.

# %%
import os
import gc
import json
import random
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- configuration
SEED = 42
N_FOLDS = 5
SEEDS = [42, 1337, 2026]        # seed averaging (idea 05); 3 is a good cost/benefit point
TARGET_METRIC = "binary_f1"     # confirmed via all-zeros LB probe (public score 0.0000000)
PLATEAU_TOL = 0.005             # ~= measured per-fold noise; width of the "flat" region
USE_NATIVE_CATEGORICAL = False  # LightGBM native cats. Off by default: safer, see instructions
SMOKE_TEST = False              # True -> subsample + tiny models, for local pipeline checks

HOLDOUT_FRAC = 0.15             # fraction of train rows, most test-like, held out for arm search
SKIP_ARM_SEARCH = False         # True -> force keep_all/none, skip straight to final CV

OUT_DIR = os.environ.get(
    "PSTU_OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
)
os.makedirs(OUT_DIR, exist_ok=True)
ARTIFACT_PATH = os.path.join(OUT_DIR, "artifacts.joblib")
SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")

# ---------------------------------------------------------------- determinism
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# %%
# Backend: real LightGBM on Kaggle, HistGradientBoosting locally (not installed in dev env).
try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    BACKEND = "histgbm"

print(f"backend: {BACKEND}")
print(f"smoke test: {SMOKE_TEST}")

# %% [markdown]
# ## 1. Load data
#
# Path auto-detection so the same notebook runs on Kaggle and locally with no edits.
# Kaggle mounts competition data under `/kaggle/input/competitions/<slug>/`; some setups use
# `/kaggle/input/<slug>/`. Both are probed, plus the local checkout.
#
# The 5-fold split is pinned here, using only `y`'s shape — not tied to any particular feature
# matrix — so the exact same fold indices are reused by the arm search (section 7) and the final
# run (section 8). Comparing arms against each other, and against the final model, means
# something only because the split underneath never changes.

# %%
CANDIDATE_DIRS = [
    "/kaggle/input/competitions/pstu-data-thon-2026-vol-1",
    "/kaggle/input/pstu-data-thon-2026-vol-1",
    "pstu-data-thon-2026-vol-1",
    "../input/competitions/pstu-data-thon-2026-vol-1",
    "../input/pstu-data-thon-2026-vol-1",
    "../pstu-data-thon-2026-vol-1",
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(f"train.csv not found in any of: {CANDIDATE_DIRS}")
print("DATA_DIR =", DATA_DIR)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("train:", train.shape, "| test:", test.shape, "| sample_sub:", sample_sub.shape)

if SMOKE_TEST:
    train = train.sample(n=8000, random_state=SEED).reset_index(drop=True)
    test = test.head(4000).reset_index(drop=True)
    sample_sub = sample_sub.head(4000).reset_index(drop=True)
    SEEDS = [42]
    print("SMOKE: reduced to", train.shape, test.shape)

TARGET = "TARGET"
ID = "id"
y = train[TARGET].values.astype(int)
test_ids = test[ID].copy()

print(f"positive rate: {y.mean():.6f}  ({y.sum()} / {len(y)})")

folds = list(
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)
)

# %% [markdown]
# ## 2. Column contract
#
# The 6 categorical columns and both sentinel families are **measured** (Stage 1).
# The 44 droppable columns are **recomputed from train**, never hand-transcribed — a typo in a
# 44-name list is exactly the kind of silent bug that costs a competition.

# %%
CAT_COLS = ["feat_142", "feat_157", "feat_318", "feat_320", "feat_325", "feat_337"]

SENTINEL_NEG_COL = "feat_109"
SENTINEL_NEG_VAL = -999999

SENTINEL_BIG_VAL = 9999999999
SENTINEL_BIG_COLS = [
    "feat_11", "feat_21", "feat_26", "feat_30", "feat_31", "feat_36", "feat_74", "feat_77",
    "feat_96", "feat_124", "feat_135", "feat_144", "feat_149", "feat_158", "feat_171",
    "feat_196", "feat_204", "feat_226", "feat_301", "feat_315", "feat_330", "feat_336",
    "feat_340",
]

FEAT_COLS = [c for c in train.columns if c != TARGET]
NUMERIC_COLS = [c for c in FEAT_COLS if c not in CAT_COLS]

assert [c for c in test.columns if c != ID] == FEAT_COLS, "train/test feature columns differ"
assert list(test.columns)[-1] == ID, "id is not the last column of test.csv"


def compute_droppable(df, numeric_cols):
    """Constant-in-train columns + exact row-for-row duplicate columns.

    Constant columns are excluded from the duplicate search first: every all-zero column is
    trivially 'equal' to every other, which would otherwise merge them into one giant bogus
    group (this bit the Stage-1 script before it was fixed).
    """
    const = [c for c in numeric_cols if df[c].nunique(dropna=False) == 1]
    search = [c for c in numeric_cols if c not in const]

    buckets = {}
    for c in search:
        h = pd.util.hash_pandas_object(df[c], index=False).sum()
        buckets.setdefault(h, []).append(c)

    redundant, seen = set(), set()
    for cols in buckets.values():
        if len(cols) < 2:
            continue
        remaining = list(cols)
        while remaining:
            base = remaining.pop(0)
            if base in seen:
                continue
            group, still = [base], []
            for c in remaining:
                if df[base].equals(df[c]):      # confirm real equality (hash collisions)
                    group.append(c)
                    seen.add(c)
                else:
                    still.append(c)
            remaining = still
            if len(group) > 1:
                seen.add(base)
                _keep, *drop = sorted(group)
                redundant.update(drop)
    return sorted(set(const) | redundant)


BASE_DROP_COLS = compute_droppable(train, NUMERIC_COLS)
print(f"droppable columns: {len(BASE_DROP_COLS)} (Stage-1 measured 44 on the full train set)")

# %% [markdown]
# ## 3. Feature engineering
#
# One function, used for train, test, and the hidden test. Fit-mode returns the fitted maps;
# transform-mode consumes them. Everything it produces is either per-row (no fitted state) or
# driven by a saved map with an explicit unseen-level fallback.
#
# - **Sentinel indicators** are computed *before* sentinels become `NaN`, or the information is
#   destroyed.
# - **Row aggregates** target the measured sparsity (252 of 344 numeric cols are >=90% zero).
#   A tree cannot build "count of nonzero across 252 columns" itself — it needs 252 simultaneous
#   splits. Hand it the feature.
# - **Frequency encoding** handles the high-cardinality cats (2,333 / 1,710 / 627 levels).
#   Unseen level -> frequency 0.0, which is both honest and a de-facto "new category" flag.
# - **Ordinal codes** carry raw identity; unseen -> -1.
# - **Clipping to train range** addresses the measured 0.5742 adversarial AUC: 93 of 344 numeric
#   columns have test values outside their train range, and trees extrapolate badly there.
#   (Clipping was already active in run-1; it alone did not close the OOF/LB gap — see section 5.)
#
# `drop_cols` is a parameter, not hard-coded inside this function: section 7 calls it with
# different drop lists to build candidate feature-arm matrices without touching this function's
# body, which keeps the sha256 identity check against `pstu_inference.py` valid regardless of
# which arm ends up winning.

# %%
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


# Hash the feature-builder source. The inference notebook must carry a byte-identical copy of
# this function; it re-hashes its own and asserts a match. A silent divergence between the two
# copies is otherwise undetectable until it has already cost the hidden-test component.
import hashlib
import inspect


def fn_source_hash(fn):
    """sha256 of a function's source, or None where source is unavailable (bare `exec`)."""
    try:
        return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return None


FEATURE_FN_HASH = fn_source_hash(build_features)
print("build_features sha256:", (FEATURE_FN_HASH or "unavailable")[:16])

X_base, CAT_MAPS_base, CLIP_base = build_features(train, BASE_DROP_COLS)
X_test_base, _, _ = build_features(test, BASE_DROP_COLS, maps=CAT_MAPS_base, clip_bounds=CLIP_base)
X_test_base = X_test_base[X_base.columns]      # enforce identical column order

CODE_COLS = [f"{c}_code" for c in CAT_COLS if f"{c}_code" in X_base.columns]

print(f"baseline feature matrix (44-drop, keep_all arm): {X_base.shape}  "
      f"(from {len(FEAT_COLS)} raw columns)")
print(f"engineered: {X_base.shape[1] - (len(FEAT_COLS) - len(BASE_DROP_COLS))} new columns")

# %% [markdown]
# ## 4. Metrics and the threshold engine
#
# `cutoff_curve` is an exhaustive O(n log n) sweep: sort once, take cumulative sums, and the full
# confusion matrix at **every** distinct cut point falls out. No grid, no missed optimum.
#
# It is verified against sklearn's `f1_score` by brute force in the cell below — the whole
# submission hinges on this function being right.
#
# Selection picks the **plateau centre, not the argmax**. Measured: every cut in t in [0.15, 0.21]
# scored within 0.005 binary-F1 of the peak, which is inside the +/-0.0060 per-fold noise. The
# argmax simply won a coin flip on one OOF sample; its neighbours transfer just as well and the
# centre is less likely to be a fold-noise artifact.

# %%
def cutoff_curve(y_true, scores):
    """Confusion matrix + F1s at every distinct cut. Returns a DataFrame, one row per cut."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")       # stable -> deterministic on ties
    s = scores[order]
    yy = y_true[order]

    P = int(y_true.sum())
    N = len(y_true) - P

    tp = np.cumsum(yy)                       # tp[k-1] = true positives in the top k
    k = np.arange(1, len(yy) + 1)
    fp = k - tp
    fn = P - tp
    tn = N - fp

    # Only cut between distinct scores, otherwise `>= t` would include more rows than k.
    valid = np.empty(len(s), dtype=bool)
    valid[:-1] = s[:-1] != s[1:]
    valid[-1] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where((2 * tn + fn + fp) > 0, 2 * tn / (2 * tn + fn + fp), 0.0)

    return pd.DataFrame({
        "threshold": s[valid],
        "n_pred_pos": k[valid],
        "tp": tp[valid], "fp": fp[valid], "fn": fn[valid], "tn": tn[valid],
        "binary_f1": f1_pos[valid],
        "macro_f1": ((f1_pos + f1_neg) / 2.0)[valid],
    })


# --- verify against sklearn brute force (must pass, or nothing downstream is trustworthy) ---
_rng = np.random.default_rng(0)
_y = (_rng.random(4000) < 0.04).astype(int)
_s = np.clip(_y * 0.3 + _rng.random(4000) * 0.7, 0, 1).round(3)   # deliberate ties
_curve = cutoff_curve(_y, _s)
_probe = _curve.sample(n=min(40, len(_curve)), random_state=0)
for _, r in _probe.iterrows():
    _pred = (_s >= r["threshold"]).astype(int)
    assert abs(f1_score(_y, _pred, average="binary", zero_division=0) - r["binary_f1"]) < 1e-9
    assert abs(f1_score(_y, _pred, average="macro", zero_division=0) - r["macro_f1"]) < 1e-9
    assert _pred.sum() == r["n_pred_pos"]
print(f"cutoff_curve verified against sklearn on {len(_probe)} cut points (tol 1e-9)")
del _rng, _y, _s, _curve, _probe


def select_threshold(y_true, scores, metric="binary_f1", tol=PLATEAU_TOL):
    """Plateau-centred cut selection. Returns (threshold, diagnostics dict)."""
    curve = cutoff_curve(y_true, scores)
    peak = curve[metric].max()
    plateau = curve[curve[metric] >= peak - tol]

    # centre by predicted-positive count: monotone in the cut and comparable across models,
    # unlike the raw threshold value which depends entirely on model calibration.
    k_star = int(np.median(plateau["n_pred_pos"]))
    row = curve.iloc[(curve["n_pred_pos"] - k_star).abs().argmin()]
    argmax_row = curve.loc[curve[metric].idxmax()]

    diag = {
        "metric": metric,
        "threshold": float(row["threshold"]),
        "n_pred_pos": int(row["n_pred_pos"]),
        "pred_pos_rate": float(row["n_pred_pos"]) / len(y_true),
        "binary_f1": float(row["binary_f1"]),
        "macro_f1": float(row["macro_f1"]),
        "argmax_threshold": float(argmax_row["threshold"]),
        "argmax_score": float(argmax_row[metric]),
        "plateau_lo": float(plateau["threshold"].min()),
        "plateau_hi": float(plateau["threshold"].max()),
        "plateau_width_k": int(plateau["n_pred_pos"].max() - plateau["n_pred_pos"].min()),
        "cost_vs_argmax": float(argmax_row[metric] - row[metric]),
    }
    return float(row["threshold"]), diag

# %% [markdown]
# ## 5. Shift diagnostic: adversarial validation + a shift-aware holdout
#
# **Why this exists:** run-1 of this pipeline (2026-08-09, real LightGBM on Kaggle) scored OOF
# binary-F1 ~0.395 but public LB only **0.1849** — roughly half. The predicted-positive rate
# ratio (submission 0.0428 vs OOF 0.0494, ratio 0.87) only explains a small part of that; most
# of the gap is real generalization loss, consistent with the measured adversarial train/test
# AUC of 0.5742 (Stage 1). Plain random-fold CV cannot see this: every fold is drawn from the
# same distribution as every other fold, so nothing in ordinary CV would ever penalize a
# configuration for failing to generalize to *shifted* data specifically.
#
# **What this builds:** an adversarial classifier (train vs test, 5-fold — well, `ADV_CV`-fold —
# out-of-fold to avoid overfitting bias) scores every TRAIN row by how much it looks like a TEST
# row. The top `HOLDOUT_FRAC` of train rows by that score become a standing **shift holdout** —
# not removed from training, just tracked by index. Two uses:
#
#   1. Section 7 scores each candidate feature/imbalance configuration by its F1 restricted to
#      this holdout, not by its F1 on ordinary CV — picking whichever config survives the shift
#      best, instead of whichever overfits the in-distribution folds hardest.
#   2. Section 9 reports the FINAL model's F1 restricted to this same holdout as a second,
#      shift-aware OOF number that should track the real LB more closely than the standard
#      (optimistic) OOF figure — worth comparing against the next real submission's LB score.

# %%
ADV_CV = 2 if SMOKE_TEST else 3
ADV_N_ESTIMATORS = 30 if SMOKE_TEST else 150

_adv_X = pd.concat([X_base, X_test_base], ignore_index=True).fillna(-1.0)
_adv_y = np.array([0] * len(X_base) + [1] * len(X_test_base))
_adv_clf = RandomForestClassifier(
    n_estimators=ADV_N_ESTIMATORS, max_depth=6, n_jobs=-1, random_state=SEED
)
_adv_oof = cross_val_predict(
    _adv_clf, _adv_X, _adv_y, cv=ADV_CV, method="predict_proba", n_jobs=1
)[:, 1]
ADV_AUC = roc_auc_score(_adv_y, _adv_oof)
p_test_train = _adv_oof[: len(X_base)]

n_holdout = max(200, int(HOLDOUT_FRAC * len(y)))
shift_holdout_idx = np.argsort(-p_test_train)[:n_holdout]
holdout_pos_rate = y[shift_holdout_idx].mean()

print(f"adversarial train/test AUC (OOF, {ADV_CV}-fold): {ADV_AUC:.4f}  "
      f"(Stage-1 measured: 0.5742 -- should be close)")
print(f"shift holdout: {n_holdout} rows ({n_holdout / len(y):.1%} of train), "
      f"positive rate {holdout_pos_rate:.4f}  (overall: {y.mean():.4f})")

del _adv_X, _adv_y, _adv_clf, _adv_oof
gc.collect()

# %% [markdown]
# ## 6. Model
#
# LightGBM on Kaggle; `HistGradientBoostingClassifier` locally so the pipeline is smoke-testable
# without the GBDT libraries installed.
#
# **CPU only, deliberately.** At 76,020 x ~320 columns CPU LightGBM trains a fold in seconds;
# GPU adds transfer overhead at this size and burns the capped ~30h/week quota for no gain.
# See the instructions file.
#
# `fit_predict` takes `spw` (scale_pos_weight) and `quick` (reduced estimators / early-stopping
# patience) as explicit arguments rather than reading module globals, so the same function
# serves both the fast arm search (section 7) and the full final run (section 8) without any
# global-state ordering hazard between them.

# %%
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.1,
    lambda_l2=1.0,
    n_estimators=50 if SMOKE_TEST else 3000,
    n_jobs=-1,
    verbose=-1,
    deterministic=True,
    force_row_wise=True,        # silences the threading-dependent binning path
)

HGB_PARAMS = dict(
    max_iter=50 if SMOKE_TEST else 300,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=1.0,
)

ARM_SEARCH_N_ESTIMATORS = 30 if SMOKE_TEST else 1200
ARM_SEARCH_EARLY_STOP = 20 if SMOKE_TEST else 100


def make_lgb_params(seed, spw=None, quick=False):
    params = dict(LGB_PARAMS, random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    if quick:
        params["n_estimators"] = ARM_SEARCH_N_ESTIMATORS
    if spw:
        params["scale_pos_weight"] = spw
    return params


def fit_predict(X_tr, y_tr, X_va, y_va, seed, X_te=None, spw=None, quick=False):
    """Fit one fold. Returns (val_proba, test_proba_or_None, model)."""
    if BACKEND == "lightgbm":
        params = make_lgb_params(seed, spw=spw, quick=quick)
        model = lgb.LGBMClassifier(**params)
        stop_rounds = ARM_SEARCH_EARLY_STOP if quick else 200
        fit_kw = dict(eval_set=[(X_va, y_va)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(stop_rounds, verbose=False),
                                 lgb.log_evaluation(0)])
        if USE_NATIVE_CATEGORICAL and CODE_COLS:
            fit_kw["categorical_feature"] = CODE_COLS
        model.fit(X_tr, y_tr, **fit_kw)
    else:
        hgb_params = dict(HGB_PARAMS)
        if quick:
            hgb_params["max_iter"] = min(hgb_params["max_iter"], ARM_SEARCH_N_ESTIMATORS)
        model = HistGradientBoostingClassifier(random_state=seed, **hgb_params)
        model.fit(X_tr, y_tr)
    va_proba = model.predict_proba(X_va)[:, 1]
    te_proba = model.predict_proba(X_te)[:, 1] if X_te is not None else None
    return va_proba, te_proba, model

# %% [markdown]
# ## 7. Shift-aware configuration search
#
# Operationalizes `ideas/04-shift-robustness/` steps 2-3 automatically: instead of guessing
# whether to drop the shift-driving features or reweight for imbalance, both are tried and
# scored on the shift holdout from section 5 — the closest local proxy available for real LB
# behaviour. Ordinary CV can't make this choice; it would cheerfully reward a config that
# overfits the exact shift that's hurting the real score.
#
# 3 feature arms x 2 imbalance arms, each scored with a fast, reduced-estimator CV over the
# first few pinned folds (a relative-ranking pass, not a final fit). The winner is retrained
# properly with the full seed x fold budget in section 8.
#
# `feat_182` is kept as a candidate to drop despite being the 3rd-strongest single feature in
# the dataset (AUC 0.6824, Stage 1) — it is *also* by far the largest driver of the measured
# train/test shift (importance ~2.4x the next feature), so it is exactly the awkward case this
# search exists to resolve empirically rather than by argument.

# %%
NEG_POS_RATIO = float((len(y) - y.sum()) / y.sum())   # measured on this run's train set

FEATURE_ARMS = {
    "keep_all": [],
    "drop_top1_feat182": ["feat_182"],
    "drop_top5_shift": ["feat_182", "feat_44", "feat_116", "feat_306", "feat_97"],
}
IMBALANCE_ARMS = {
    "none": None,
    "scale_pos_weight": NEG_POS_RATIO,
}
ARM_SEARCH_N_FOLDS = 1 if SMOKE_TEST else 3

_arm_X_cache = {"keep_all": X_base}


def get_arm_features(feat_arm_name, extra_drop):
    if feat_arm_name not in _arm_X_cache:
        drop_cols_arm = sorted(set(BASE_DROP_COLS) | set(extra_drop))
        X_arm, _, _ = build_features(train, drop_cols_arm)
        _arm_X_cache[feat_arm_name] = X_arm
    return _arm_X_cache[feat_arm_name]


arm_results = []
if SKIP_ARM_SEARCH:
    winner = {"feature_arm": "keep_all", "imbalance_arm": "none", "extra_drop": [], "spw": None}
    print("SKIP_ARM_SEARCH=True -- using keep_all/none without searching.")
else:
    for feat_name, extra_drop in FEATURE_ARMS.items():
        X_arm = get_arm_features(feat_name, extra_drop)
        for imb_name, spw in IMBALANCE_ARMS.items():
            arm_oof = np.zeros(len(y))
            for fold_idx in range(ARM_SEARCH_N_FOLDS):
                tr_idx, va_idx = folds[fold_idx]
                va_p, _, _ = fit_predict(
                    X_arm.iloc[tr_idx], y[tr_idx], X_arm.iloc[va_idx], y[va_idx],
                    seed=SEED, spw=spw, quick=True,
                )
                arm_oof[va_idx] = va_p

            scored_idx = np.concatenate([folds[i][1] for i in range(ARM_SEARCH_N_FOLDS)])
            _, arm_diag = select_threshold(y[scored_idx], arm_oof[scored_idx], metric=TARGET_METRIC)
            arm_thr = arm_diag["threshold"]

            holdout_in_scored = np.intersect1d(shift_holdout_idx, scored_idx)
            if len(holdout_in_scored) > 20:
                holdout_pred = (arm_oof[holdout_in_scored] >= arm_thr).astype(int)
                holdout_f1 = f1_score(y[holdout_in_scored], holdout_pred,
                                       average="binary", zero_division=0)
            else:
                holdout_f1 = float("nan")

            arm_results.append({
                "feature_arm": feat_name, "imbalance_arm": imb_name,
                "extra_drop": extra_drop, "spw": spw,
                "quick_full_f1": arm_diag["binary_f1"], "shift_holdout_f1": holdout_f1,
                "n_holdout_scored": len(holdout_in_scored),
            })
            print(f"  [{feat_name:18s} x {imb_name:18s}] "
                  f"quick_full_f1={arm_diag['binary_f1']:.4f}  "
                  f"shift_holdout_f1={holdout_f1:.4f}  (n_holdout_scored={len(holdout_in_scored)})")

    arm_df = pd.DataFrame(arm_results).sort_values("shift_holdout_f1", ascending=False)
    arm_df.to_csv(os.path.join(OUT_DIR, "arm_search.csv"), index=False)
    winner = arm_df.iloc[0].to_dict()
    # Re-derive spw/extra_drop from the source dicts rather than trusting the DataFrame
    # round-trip: a column mixing None with floats (scale_pos_weight) gets silently coerced to
    # NaN by pandas, and `if spw:` in make_lgb_params() treats NaN as truthy -- that combination
    # would apply a bogus scale_pos_weight even when the "none" arm won. Caught by smoke test.
    winner["spw"] = IMBALANCE_ARMS[winner["imbalance_arm"]]
    winner["extra_drop"] = FEATURE_ARMS[winner["feature_arm"]]
    print(f"\nwinning config: feature_arm={winner['feature_arm']} "
          f"imbalance_arm={winner['imbalance_arm']}  "
          f"(shift_holdout_f1={winner['shift_holdout_f1']:.4f})")

DROP_COLS = sorted(set(BASE_DROP_COLS) | set(winner["extra_drop"]))
SCALE_POS_WEIGHT = winner["spw"]

if winner["feature_arm"] == "keep_all":
    X, CAT_MAPS, CLIP_BOUNDS = X_base, CAT_MAPS_base, CLIP_base
    X_test = X_test_base
else:
    X, CAT_MAPS, CLIP_BOUNDS = build_features(train, DROP_COLS)
    X_test, _, _ = build_features(test, DROP_COLS, maps=CAT_MAPS, clip_bounds=CLIP_BOUNDS)
    X_test = X_test[X.columns]

FEATURE_ORDER = list(X.columns)
CODE_COLS = [f"{c}_code" for c in CAT_COLS if f"{c}_code" in X.columns]

print(f"\nfinal feature matrix: {X.shape}  (dropped {len(DROP_COLS)} columns total: "
      f"{len(BASE_DROP_COLS)} base + {len(DROP_COLS) - len(BASE_DROP_COLS)} shift-driven)")
print(f"final imbalance setting: scale_pos_weight={SCALE_POS_WEIGHT}")

del train, test, _arm_X_cache
gc.collect()

# %% [markdown]
# ## 8. Cross-validation with seed averaging
#
# The fold split is **pinned** (defined once in section 1) and identical across every seed —
# only the model seed varies. Mixing both sources of variance would make the seed-averaging
# comparison uninterpretable.
#
# Measured per-fold AUC spread on the run-1 baseline was +/-0.0060. That is the noise floor: any
# "improvement" smaller than it is not one.

# %%
oof_proba = np.zeros(len(y))
test_proba = np.zeros(len(X_test))
models = []
per_seed_auc = {}
per_fold_thresholds = []

for seed in SEEDS:
    seed_oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        va_p, te_p, model = fit_predict(
            X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx],
            seed=seed, X_te=X_test, spw=SCALE_POS_WEIGHT, quick=False,
        )
        seed_oof[va_idx] = va_p
        test_proba += te_p / (len(SEEDS) * N_FOLDS)
        models.append({"seed": seed, "fold": fold, "model": model})

        auc = roc_auc_score(y[va_idx], va_p)
        fold_aucs.append(auc)

        # per-fold threshold stability (idea 01 step 4) -- measured on the first seed only
        if seed == SEEDS[0]:
            _, d = select_threshold(y[va_idx], va_p, metric=TARGET_METRIC)
            per_fold_thresholds.append(d["pred_pos_rate"])

        print(f"  seed {seed} fold {fold}: AUC {auc:.4f}")

    oof_proba += seed_oof / len(SEEDS)
    per_seed_auc[seed] = roc_auc_score(y, seed_oof)
    print(f"seed {seed}: OOF AUC {per_seed_auc[seed]:.4f} "
          f"(folds {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")

OOF_AUC = roc_auc_score(y, oof_proba)
print(f"\nseed-averaged OOF AUC: {OOF_AUC:.4f}")
print(f"per-fold optimal predicted-positive rate: "
      f"{np.round(per_fold_thresholds, 4)}  (spread {np.ptp(per_fold_thresholds):.4f})")

# %% [markdown]
# ## 9. Operating point
#
# The single highest-value decision in the whole pipeline. `TARGET_METRIC` (binary F1, confirmed
# by the LB probe) selects the cut; macro F1 is still reported alongside it as a sanity check.
#
# The shift-holdout metric printed at the end of this section is the number to trust most when
# judging whether this run is likely to beat run-1's LB 0.1849 — it is computed the same way the
# arm search judged candidates in section 7, just applied to the final, fully-trained model.

# %%
THRESHOLD, DIAG = select_threshold(y, oof_proba, metric=TARGET_METRIC)

curve = cutoff_curve(y, oof_proba)
naive = (oof_proba >= 0.5).astype(int)
naive_scores = {
    "binary_f1": f1_score(y, naive, average="binary", zero_division=0),
    "macro_f1": f1_score(y, naive, average="macro", zero_division=0),
}

print(f"=== operating point ({TARGET_METRIC}) ===")
for k, v in DIAG.items():
    print(f"  {k}: {v}")
print(f"\n=== what the mechanic is worth ===")
print(f"  raw probabilities @ grader's 0.5 cut : "
      f"binary_f1 {naive_scores['binary_f1']:.4f} | macro_f1 {naive_scores['macro_f1']:.4f} "
      f"({int(naive.sum())} positives)")
print(f"  hard labels @ tuned cut              : "
      f"binary_f1 {DIAG['binary_f1']:.4f} | macro_f1 {DIAG['macro_f1']:.4f} "
      f"({DIAG['n_pred_pos']} positives)")
print(f"  gain: binary_f1 {DIAG['binary_f1'] - naive_scores['binary_f1']:+.4f} | "
      f"macro_f1 {DIAG['macro_f1'] - naive_scores['macro_f1']:+.4f}")

print(f"\n=== degenerate floors (for the LB probe) ===")
for name, const in (("all zeros", 0), ("all ones", 1)):
    p = np.full(len(y), const)
    print(f"  {name:10s}: binary_f1 {f1_score(y, p, average='binary', zero_division=0):.4f} | "
          f"macro_f1 {f1_score(y, p, average='macro', zero_division=0):.4f}")

print(f"\n=== shift-holdout metrics ({len(shift_holdout_idx)} most test-like train rows) ===")
holdout_pred = (oof_proba[shift_holdout_idx] >= THRESHOLD).astype(int)
HOLDOUT_BINARY_F1 = f1_score(y[shift_holdout_idx], holdout_pred, average="binary", zero_division=0)
HOLDOUT_MACRO_F1 = f1_score(y[shift_holdout_idx], holdout_pred, average="macro", zero_division=0)
print(f"  shift-holdout binary_f1: {HOLDOUT_BINARY_F1:.4f}  "
      f"(full-train OOF binary_f1: {DIAG['binary_f1']:.4f})")
print(f"  shift-holdout macro_f1:  {HOLDOUT_MACRO_F1:.4f}")
print(f"  adversarial train/test AUC: {ADV_AUC:.4f}")
print("  This should track real LB more closely than the full OOF figure above --")
print("  compare it against the next submission's public LB score.")

curve.to_csv(os.path.join(OUT_DIR, "threshold_curve.csv"), index=False)

# %% [markdown]
# ## 10. Submission
#
# `test.csv` row order already matches `sample_submission.csv` (measured), but ids are taken from
# `test.csv` itself and asserted equal rather than assumed. `id` is **not** `0..n-1` and **not**
# contiguous — regenerating it with `range()` produces a correctly-shaped, totally misaligned
# file that scores ~0 and looks like a modelling failure.

# %%
test_pred = (test_proba >= THRESHOLD).astype(int)

sub = pd.DataFrame({ID: test_ids.values, TARGET: test_pred})


def validate_submission(sub_df, reference_ids, n_expected):
    assert list(sub_df.columns) == [ID, TARGET], f"bad columns: {list(sub_df.columns)}"
    assert len(sub_df) == n_expected, f"bad row count: {len(sub_df)} != {n_expected}"
    assert sub_df[ID].equals(pd.Series(reference_ids).reset_index(drop=True)), "id mismatch"
    assert sub_df[TARGET].isin([0, 1]).all(), "TARGET must be 0/1"
    assert sub_df[ID].is_unique, "duplicate ids"
    assert not sub_df.isna().any().any(), "NaNs in submission"
    rate = sub_df[TARGET].mean()
    assert 0.01 < rate < 0.15, f"implausible positive rate {rate:.4f} (expected ~0.044)"
    return rate


rate = validate_submission(sub, sample_sub[ID].values, len(sample_sub))
sub.to_csv(SUBMISSION_PATH, index=False)

print(f"wrote {SUBMISSION_PATH}")
print(f"  rows: {len(sub)} | positives: {int(sub[TARGET].sum())} | rate: {rate:.4f}")
print(f"  OOF predicted-positive rate for comparison: {DIAG['pred_pos_rate']:.4f}")
print(sub.head())

# %% [markdown]
# ## 11. Save artifacts for the inference notebook
#
# Everything the inference notebook needs to reproduce these exact predictions, and nothing it
# would have to recompute. `feature_order` in particular is what turns a column-order mismatch
# into a loud failure instead of silently garbage predictions. `drop_cols` now carries whatever
# section 7 decided (base 44, possibly plus shift-driving columns) — `build_features`'s body is
# unchanged, so the sha256 identity check against `pstu_inference.py` stays valid either way.

# %%
artifacts = {
    "models": models,
    "cat_maps": CAT_MAPS,
    "clip_bounds": CLIP_BOUNDS,
    "drop_cols": DROP_COLS,
    "base_drop_cols": BASE_DROP_COLS,
    "feature_order": FEATURE_ORDER,
    "code_cols": CODE_COLS,
    "threshold": THRESHOLD,
    "target_metric": TARGET_METRIC,
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "seed": SEED,
    "backend": BACKEND,
    "oof_auc": float(OOF_AUC),
    "diagnostics": DIAG,
    "expected_pred_pos_rate": DIAG["pred_pos_rate"],
    "cat_cols": CAT_COLS,
    "feature_fn_hash": FEATURE_FN_HASH,
    "train_submission_preds": test_pred,   # for the inference-notebook reproduction check
    "sentinel_neg": {SENTINEL_NEG_COL: SENTINEL_NEG_VAL},
    "sentinel_big_cols": SENTINEL_BIG_COLS,
    "sentinel_big_val": SENTINEL_BIG_VAL,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "winning_arm": {"feature_arm": winner["feature_arm"], "imbalance_arm": winner["imbalance_arm"],
                     "extra_drop_cols": winner["extra_drop"]},
    "adversarial_auc": float(ADV_AUC),
    "shift_holdout_binary_f1": float(HOLDOUT_BINARY_F1),
    "shift_holdout_macro_f1": float(HOLDOUT_MACRO_F1),
    "shift_holdout_n": int(len(shift_holdout_idx)),
}
joblib.dump(artifacts, ARTIFACT_PATH, compress=3)

size_mb = os.path.getsize(ARTIFACT_PATH) / 1e6
print(f"wrote {ARTIFACT_PATH} ({size_mb:.1f} MB, {len(models)} models)")

with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
    json.dump({
        "backend": BACKEND, "oof_auc": float(OOF_AUC), "threshold": THRESHOLD,
        "target_metric": TARGET_METRIC, "diagnostics": DIAG,
        "naive_half_cut": naive_scores, "per_seed_auc": {str(k): float(v)
                                                          for k, v in per_seed_auc.items()},
        "submission_positive_rate": float(rate), "n_features": len(FEATURE_ORDER),
        "n_dropped_columns": len(DROP_COLS), "base_dropped_columns": len(BASE_DROP_COLS),
        "winning_arm": {"feature_arm": winner["feature_arm"],
                         "imbalance_arm": winner["imbalance_arm"],
                         "extra_drop_cols": winner["extra_drop"]},
        "adversarial_auc": float(ADV_AUC),
        "shift_holdout_binary_f1": float(HOLDOUT_BINARY_F1),
        "shift_holdout_macro_f1": float(HOLDOUT_MACRO_F1),
        "shift_holdout_n": int(len(shift_holdout_idx)),
    }, f, indent=2)

print("\nDone. Next: run pstu_inference.ipynb and confirm it reproduces submission.csv exactly.")
