# %% [markdown]
# # PSTU Data Thon 2026 Vol-1 — run-5 (raw, not yet run on Kaggle)
#
# **Goal: beat run-4's public LB 0.225758329189**, the best real score measured anywhere in this
# project so far (see `CLAUDE.md` and `results/run-4/README.md`).
#
# This is a **training notebook only** — no inference notebook this round, per this session's
# request. Named `run-5-raw` because it has not been run on Kaggle yet; once it is, copy its
# outputs into `results/run-5/` alongside a short README like `results/run-3/` and `results/run-4/`.
#
# ## What this combines
#
# **From run-4 (`results/run-4/`, the higher-scoring external lineage, LB 0.2258):**
# - CatBoost, **native categorical handling** (no target-encoding leakage) — proven to beat
#   run-3's LightGBM+XGBoost+CatBoost blend with target encoding (LB 0.1957).
# - SMOTE(0.3) + `scale_pos_weight` together ("complementary imbalance handling").
# - No KMeans-on-combined-data, no PCA, no per-model calibration shift — run-3's post-mortem
#   named all three as leakage/instability sources; run-4 removed them and LB improved.
# - A single-model-family ensemble (3 seeds), not a multi-family blend — run-3's "model
#   diversity trap" (3 different models + per-model calibration destabilizing the blend) is not
#   repeated here.
#
# **From run-3 (`results/run-3/`), the one idea worth keeping despite its flawed execution:**
# - Richer row-wise aggregate features than run-4's 6 — but computed the way this project's own
#   `solution/pstu_train.py` already does it (9 aggregates, no cross-row/train-test-combined
#   statistics), not run-3's version (which relied on KMeans over combined train+test, since
#   flagged as a leakage risk in its own post-mortem).
#
# **From this project's own measured findings (`CLAUDE.md`, `dataset_exploration/`), used by
# neither external run:**
# 1. **Sentinel handling** — `-999999` in `feat_109`, `9999999999` across 23 columns → both
#    become `NaN` before anything else touches them. Neither run-3 nor run-4 special-cased these;
#    both fed `9999999999` into the model as a genuine numeric value, and both filled all `NaN`
#    (including these sentinels, once they exist) with `0` or the column median blindly.
# 2. **The precise 44-column safe-drop list** (28 constant + 16 exact-duplicate groups),
#    recomputed here exactly as `solution/pstu_train.py` does — not run-3/4's looser
#    hash/variance heuristics (which found ~58 and ~43 respectively, neither matching 44).
# 3. **The threshold/operating-point mechanic** (`ideas/01-threshold-engine/`) — submit **hard
#    0/1 labels at a plateau-centred cut chosen from OOF**, instead of raw probabilities
#    thresholded at the grader's fixed 0.5. This is the single biggest lever measured anywhere in
#    this project (+0.207 OOF binary-F1 in `solution/pstu_train.py`) and **neither external run
#    used it correctly** — run-3's attempt via a per-model affine calibration shift is exactly
#    what its own post-mortem blames for leaking OOF-specific structure into the submission, and
#    run-4 dropped the idea entirely (its 7.05%-vs-3.96% predicted-positive-rate mismatch at the
#    fixed 0.5 cut, reported in `results/run-4/README.md`, is the direct symptom of never tuning
#    an operating point).
# 4. **The shift-aware feature-arm search** (`ideas/04-shift-robustness/`) — reuses the exact
#    adversarial-validation-derived shift holdout and the same three feature arms
#    (`keep_all` / `drop_top1_feat182` / `drop_top5_shift`) that `solution/pstu_train.py`'s run-2
#    already measured, where `drop_top5_shift` won.
# 5. **Clip numeric features to the saved train range** — the same shift-robustness guard already
#    validated in this project's own pipeline (93/344 numeric columns exceed their train range on
#    the public test).
#
# This cannot guarantee a specific leaderboard score — no local number can promise a public/
# private/hidden LB outcome — but every change above targets either a measured failure mode
# (sentinel corruption, no operating-point tuning) or reuses a technique this project has already
# validated (the arm search, the threshold engine), rather than guessing.

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
SEEDS = [42, 1337, 2026]        # seed averaging, same convention as solution/pstu_train.py
TARGET_METRIC = "binary_f1"     # confirmed via all-zeros LB probe (public score 0.0000000)
PLATEAU_TOL = 0.005             # ~= measured per-fold noise; width of the "flat" region
SMOKE_TEST = False              # True -> subsample + tiny models, for local pipeline checks

HOLDOUT_FRAC = 0.15             # fraction of train rows, most test-like, held out for arm search
SMOTE_STRATEGY = 0.3            # run-4's proven ratio (~3.3:1 post-SMOTE), not the 0.5 run-3 used

RUN4_LB_REFERENCE = 0.225758329189   # the score this run is trying to beat

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
# Backend: real CatBoost + real SMOTE on Kaggle. Local stand-ins so the pipeline is
# smoke-testable without either library installed (neither is in the local dev env).
try:
    from catboost import CatBoostClassifier
    MODEL_BACKEND = "catboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    MODEL_BACKEND = "histgbm"

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_BACKEND = "imblearn"
except ImportError:
    SMOTE_BACKEND = "none"

print(f"model backend: {MODEL_BACKEND}")
print(f"smote backend: {SMOTE_BACKEND}"
      + ("" if SMOTE_BACKEND == "imblearn" else "  (WARNING: no oversampling in this run — "
                                                  "structural smoke test only, not representative "
                                                  "of the real Kaggle run)"))
print(f"smoke test: {SMOKE_TEST}")

# %% [markdown]
# ## 1. Load data
#
# Same path auto-detection and pinned-fold pattern as `solution/pstu_train.py`: the 5-fold split
# is fixed here from `y`'s shape alone, before any feature matrix exists, so the arm search
# (section 7) and the final run (section 8) compare against the exact same split.

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
# Identical to `solution/pstu_train.py` — the 6 categorical columns and both sentinel families
# are measured (Stage 1); the 44 droppable columns are recomputed from train, never hand-typed.

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
    """Constant-in-train columns + exact row-for-row duplicate columns."""
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
                if df[base].equals(df[c]):
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
# ## 3. Feature engineering (CatBoost-native categoricals)
#
# Differs from `solution/pstu_train.py`'s `build_features` in one deliberate way: categoricals
# are kept as **integer codes fit on TRAIN ONLY** (with an explicit reserved "unseen" code), not
# frequency/ordinal encoded, so they can be passed to CatBoost's `cat_features` for native
# handling — the change that closed most of run-3 -> run-4's OOF/LB gap. Fitting the vocabulary
# on train alone (not `train+test`, unlike run-4's `LabelEncoder` fit on the concatenation) means
# levels that appear only in the hidden test still fall back cleanly to the reserved code, instead
# of assuming every level the model will ever see is already visible today.
#
# Numeric features get the same treatment as `solution/pstu_train.py`: sentinel indicators
# computed before nulling, sentinels -> `NaN`, clip to the train range, 9 row-wise aggregates
# computed while `NaN` is still informative, then a **train-median fill** for the final matrix —
# needed because SMOTE cannot interpolate through missing values. (CatBoost itself can handle
# `NaN` natively and would not need this; the fill is purely to keep the SMOTE step and the model
# step consistent. Worth revisiting if this run's score doesn't clear run-4's bar — see the
# "ideas to improve" note near the bottom.)

# %%
def build_features_cb(df, drop_cols, maps=None, clip_bounds=None, fill_values=None):
    """Returns (X, maps, clip_bounds, fill_values). maps=None -> fit mode."""
    fitting = maps is None
    if fitting:
        maps = {}

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

    # --- clip to train range ---
    if fitting:
        clip_bounds = {"lo": num.min(), "hi": num.max()}
    num = num.clip(lower=clip_bounds["lo"], upper=clip_bounds["hi"], axis=1)

    # --- row-wise aggregates over the sparse numeric block (NaN-aware, before fill) ---
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

    # --- fill NaN (train median) for the final numeric block ---
    if fitting:
        fill_values = num.median()
    num = num.fillna(fill_values)

    out = pd.concat([out, num], axis=1)

    # --- categoricals: integer codes, vocabulary fit on TRAIN ONLY, reserved unseen code ---
    cat_code_cols = []
    for c in CAT_COLS:
        if c not in raw.columns:
            continue
        col_out = f"{c}_cat"
        cat_code_cols.append(col_out)
        if fitting:
            levels = sorted(raw[c].dropna().unique())
            code_map = {lvl: i for i, lvl in enumerate(levels)}
            maps[c] = {"code_map": code_map, "unseen_code": len(code_map)}
        cm = maps[c]
        out[col_out] = raw[c].map(cm["code_map"]).fillna(cm["unseen_code"]).astype(np.int32)

    out = out.replace([np.inf, -np.inf], 0.0)
    return out, maps, clip_bounds, fill_values


X_base, CAT_MAPS_base, CLIP_base, FILL_base = build_features_cb(train, BASE_DROP_COLS)
X_test_base, _, _, _ = build_features_cb(
    test, BASE_DROP_COLS, maps=CAT_MAPS_base, clip_bounds=CLIP_base, fill_values=FILL_base
)
X_test_base = X_test_base[X_base.columns]

CAT_CODE_COLS = [f"{c}_cat" for c in CAT_COLS if f"{c}_cat" in X_base.columns]

for c in CAT_COLS:
    unseen_n = int((X_test_base[f"{c}_cat"] == CAT_MAPS_base[c]["unseen_code"]).sum())
    print(f"  {c}: {len(CAT_MAPS_base[c]['code_map'])} train levels, "
          f"{unseen_n} unseen rows in test ({unseen_n / len(X_test_base):.4%})")

print(f"baseline feature matrix (44-drop, keep_all arm): {X_base.shape}  "
      f"(from {len(FEAT_COLS)} raw columns)")

# %% [markdown]
# ## 4. Metrics and the threshold engine
#
# Identical to `solution/pstu_train.py`'s `cutoff_curve` / `select_threshold` — an exhaustive
# O(n log n) sweep verified against sklearn's `f1_score` to 1e-9, picking the **plateau centre**
# rather than the OOF argmax. This is the mechanic run-3 approximated (via leakage-prone
# per-model calibration) and run-4 skipped entirely.

# %%
def cutoff_curve(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y_true[order]

    P = int(y_true.sum())
    N = len(y_true) - P

    tp = np.cumsum(yy)
    k = np.arange(1, len(yy) + 1)
    fp = k - tp
    fn = P - tp
    tn = N - fp

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


_rng = np.random.default_rng(0)
_y = (_rng.random(4000) < 0.04).astype(int)
_s = np.clip(_y * 0.3 + _rng.random(4000) * 0.7, 0, 1).round(3)
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
    curve = cutoff_curve(y_true, scores)
    peak = curve[metric].max()
    plateau = curve[curve[metric] >= peak - tol]

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
# Identical mechanism to `solution/pstu_train.py` section 5 — an adversarial train-vs-test
# classifier scores every train row by how test-like it is; the top `HOLDOUT_FRAC` become a
# standing shift holdout used by the arm search below. Treat the resulting holdout F1 as useful
# only for *relative* ranking between arms, not as an absolute LB estimate — run-2's audit
# (`CLAUDE.md`) found this metric reads *higher* than full OOF in every measured case, the
# opposite of what would be needed to predict the real OOF->LB drop.

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
# ## 6. Model: CatBoost + SMOTE (run-4's proven recipe)
#
# Params are run-4's as measured (LB 0.2258) — depth=5, strong L2, `min_data_in_leaf=50` — not
# retuned here, since this run's goal is to test whether *this project's* fixes (sentinel
# handling, precise drop list, threshold engine, shift-aware feature arm) move the score on top
# of an already-validated model config, not to also search CatBoost hyperparameters blind.
#
# `scale_pos_weight` is derived from the measured positive rate and halved, matching run-4's
# reasoning ("complementary to SMOTE" — SMOTE already closes most of the imbalance gap to ~3.3:1,
# so the remaining reweighting only needs to cover what's left, not the full ~24:1 raw ratio).
#
# SMOTE runs on the **training fold only**, after filling `NaN` with the train-fit median
# (fit inside each fold on that fold's own training rows — no validation-fold leakage). Its
# categorical columns interpolate to non-integer values; they are rounded and clipped back to a
# valid code (including the reserved "unseen" code) before fitting, the same fix run-4 used.

# %%
NEG_POS_RATIO = float((len(y) - y.sum()) / y.sum())
SCALE_POS_WEIGHT = NEG_POS_RATIO / 2.0
print(f"NEG_POS_RATIO={NEG_POS_RATIO:.3f}  SCALE_POS_WEIGHT={SCALE_POS_WEIGHT:.3f}")

CB_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="F1",
    depth=5,
    l2_leaf_reg=5.0,
    random_strength=1.5,
    bagging_temperature=0.8,
    border_count=254,
    grow_policy="SymmetricTree",
    min_data_in_leaf=50,
    one_hot_max_size=10,
    od_type="Iter",
    thread_count=-1,
    verbose=0,
    allow_writing_files=False,
)

ARM_SEARCH_ITERATIONS = 60 if SMOKE_TEST else 1200
ARM_SEARCH_OD_WAIT = 15 if SMOKE_TEST else 100
FINAL_ITERATIONS = 80 if SMOKE_TEST else 5000
FINAL_OD_WAIT = 15 if SMOKE_TEST else 150
FINAL_LR = 0.015
ARM_SEARCH_N_FOLDS = 1 if SMOKE_TEST else 3


def smote_fold(X_tr, y_tr, seed):
    """Fillna(median) -> SMOTE -> round/clip categorical codes back to valid integers.

    Numeric-like columns are derived from X_tr's own columns each call, not a module-level
    list -- the arm search (section 7) calls this with feature matrices that drop different
    columns per arm, and a stale global here would silently fill/SMOTE the wrong column set.
    """
    numeric_like_cols = [c for c in X_tr.columns if c not in CAT_CODE_COLS]
    fill = X_tr[numeric_like_cols].median()
    X_filled = X_tr.copy()
    X_filled[numeric_like_cols] = X_filled[numeric_like_cols].fillna(fill)

    if SMOTE_BACKEND != "imblearn":
        return X_filled, y_tr   # stand-in: no oversampling, rely on class weighting only

    sm = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=seed)
    X_res, y_res = sm.fit_resample(X_filled, y_tr)
    X_res = pd.DataFrame(np.asarray(X_res), columns=X_filled.columns)
    for c in CAT_CODE_COLS:
        unseen_code = CAT_MAPS_base[c.replace("_cat", "")]["unseen_code"]
        X_res[c] = X_res[c].round().clip(0, unseen_code).astype(np.int32)
    return X_res, np.asarray(y_res)


def fit_predict(X_tr, y_tr, X_va, y_va, seed, X_te=None, quick=False):
    """Fit one fold (SMOTE + CatBoost/HistGBM). Returns (val_proba, test_proba_or_None, model)."""
    X_tr_sm, y_tr_sm = smote_fold(X_tr, y_tr, seed)

    if MODEL_BACKEND == "catboost":
        params = dict(CB_PARAMS, random_seed=seed, scale_pos_weight=SCALE_POS_WEIGHT)
        params["iterations"] = ARM_SEARCH_ITERATIONS if quick else FINAL_ITERATIONS
        params["od_wait"] = ARM_SEARCH_OD_WAIT if quick else FINAL_OD_WAIT
        params["learning_rate"] = 0.03 if quick else FINAL_LR
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr_sm, y_tr_sm, cat_features=CAT_CODE_COLS,
            eval_set=(X_va, y_va), early_stopping_rounds=params["od_wait"],
        )
    else:
        # HistGBM's categorical_features caps cardinality at 255; CatBoost has no such limit.
        # High-cardinality columns (feat_142, feat_325 -> 1000+ levels) fall back to plain
        # numeric for this LOCAL STAND-IN ONLY -- structural smoke test, not a real-model concern.
        safe_cat_cols = [c for c in CAT_CODE_COLS if X_tr_sm[c].max() <= 254]
        model = HistGradientBoostingClassifier(
            random_state=seed,
            max_iter=(ARM_SEARCH_ITERATIONS if quick else FINAL_ITERATIONS) // 4,
            learning_rate=0.05, max_depth=6, l2_regularization=1.0,
            categorical_features=safe_cat_cols, class_weight="balanced",
        )
        model.fit(X_tr_sm, y_tr_sm)

    va_proba = model.predict_proba(X_va)[:, 1]
    te_proba = model.predict_proba(X_te)[:, 1] if X_te is not None else None
    return va_proba, te_proba, model

# %% [markdown]
# ## 7. Shift-aware feature-arm search
#
# Same three arms `solution/pstu_train.py`'s run-2 measured (`keep_all` /
# `drop_top1_feat182` / `drop_top5_shift`), scored on the shift holdout from section 5. Unlike
# run-2, the imbalance handling is **not** part of this search — SMOTE(0.3) + scale_pos_weight is
# fixed throughout, since that combination is already the proven, LB-validated part of run-4's
# recipe; only the feature-drop choice is being re-decided here, now that the sentinel fix and
# precise drop list change what "the features" actually contain.

# %%
FEATURE_ARMS = {
    "keep_all": [],
    "drop_top1_feat182": ["feat_182"],
    "drop_top5_shift": ["feat_182", "feat_44", "feat_116", "feat_306", "feat_97"],
}

_arm_X_cache = {"keep_all": X_base}


def get_arm_features(feat_arm_name, extra_drop):
    if feat_arm_name not in _arm_X_cache:
        drop_cols_arm = sorted(set(BASE_DROP_COLS) | set(extra_drop))
        X_arm, _, _, _ = build_features_cb(train, drop_cols_arm)
        _arm_X_cache[feat_arm_name] = X_arm
    return _arm_X_cache[feat_arm_name]


arm_results = []
for feat_name, extra_drop in FEATURE_ARMS.items():
    X_arm = get_arm_features(feat_name, extra_drop)
    arm_oof = np.zeros(len(y))
    for fold_idx in range(ARM_SEARCH_N_FOLDS):
        tr_idx, va_idx = folds[fold_idx]
        va_p, _, _ = fit_predict(
            X_arm.iloc[tr_idx], y[tr_idx], X_arm.iloc[va_idx], y[va_idx],
            seed=SEED, quick=True,
        )
        arm_oof[va_idx] = va_p

    scored_idx = np.concatenate([folds[i][1] for i in range(ARM_SEARCH_N_FOLDS)])
    _, arm_diag = select_threshold(y[scored_idx], arm_oof[scored_idx], metric=TARGET_METRIC)
    arm_thr = arm_diag["threshold"]

    holdout_in_scored = np.intersect1d(shift_holdout_idx, scored_idx)
    if len(holdout_in_scored) > 20:
        holdout_pred = (arm_oof[holdout_in_scored] >= arm_thr).astype(int)
        holdout_f1 = f1_score(y[holdout_in_scored], holdout_pred, average="binary", zero_division=0)
    else:
        holdout_f1 = float("nan")

    arm_results.append({
        "feature_arm": feat_name, "extra_drop": extra_drop,
        "quick_full_f1": arm_diag["binary_f1"], "shift_holdout_f1": holdout_f1,
        "n_holdout_scored": len(holdout_in_scored),
    })
    print(f"  [{feat_name:18s}] quick_full_f1={arm_diag['binary_f1']:.4f}  "
          f"shift_holdout_f1={holdout_f1:.4f}  (n_holdout_scored={len(holdout_in_scored)})")

arm_df = pd.DataFrame(arm_results).sort_values("shift_holdout_f1", ascending=False)
arm_df.to_csv(os.path.join(OUT_DIR, "arm_search.csv"), index=False)
winner = arm_df.iloc[0].to_dict()
winner["extra_drop"] = FEATURE_ARMS[winner["feature_arm"]]   # re-derive, not trust the round-trip
print(f"\nwinning feature arm: {winner['feature_arm']}  "
      f"(shift_holdout_f1={winner['shift_holdout_f1']:.4f})")

DROP_COLS = sorted(set(BASE_DROP_COLS) | set(winner["extra_drop"]))

if winner["feature_arm"] == "keep_all":
    X, CAT_MAPS, CLIP_BOUNDS, FILL_VALUES = X_base, CAT_MAPS_base, CLIP_base, FILL_base
    X_test = X_test_base
else:
    X, CAT_MAPS, CLIP_BOUNDS, FILL_VALUES = build_features_cb(train, DROP_COLS)
    X_test, _, _, _ = build_features_cb(
        test, DROP_COLS, maps=CAT_MAPS, clip_bounds=CLIP_BOUNDS, fill_values=FILL_VALUES
    )
    X_test = X_test[X.columns]

CAT_CODE_COLS = [f"{c}_cat" for c in CAT_COLS if f"{c}_cat" in X.columns]
FEATURE_ORDER = list(X.columns)

print(f"\nfinal feature matrix: {X.shape}  (dropped {len(DROP_COLS)} columns total: "
      f"{len(BASE_DROP_COLS)} base + {len(DROP_COLS) - len(BASE_DROP_COLS)} shift-driven)")

del _arm_X_cache
gc.collect()

# %% [markdown]
# ## 8. Cross-validation with seed averaging

# %%
oof_proba = np.zeros(len(y))
test_proba = np.zeros(len(X_test))
models = []
per_seed_auc = {}

for seed in SEEDS:
    seed_oof = np.zeros(len(y))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        va_p, te_p, model = fit_predict(
            X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx],
            seed=seed, X_te=X_test, quick=False,
        )
        seed_oof[va_idx] = va_p
        test_proba += te_p / (len(SEEDS) * N_FOLDS)
        models.append({"seed": seed, "fold": fold, "model": model})

        auc = roc_auc_score(y[va_idx], va_p)
        fold_aucs.append(auc)
        print(f"  seed {seed} fold {fold}: AUC {auc:.4f}")

    oof_proba += seed_oof / len(SEEDS)
    per_seed_auc[seed] = roc_auc_score(y, seed_oof)
    print(f"seed {seed}: OOF AUC {per_seed_auc[seed]:.4f} "
          f"(folds {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")

OOF_AUC = roc_auc_score(y, oof_proba)
print(f"\nseed-averaged OOF AUC: {OOF_AUC:.4f}")

# %% [markdown]
# ## 9. Operating point
#
# This is the step neither external run did correctly. Compare "raw probabilities @ 0.5" (what
# run-4 actually submitted) against "hard labels @ tuned cut" (what this run submits) on the same
# OOF predictions — the gap between them is the mechanic's measured value on *this* model.

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
print(f"\n=== what the mechanic is worth on this model ===")
print(f"  raw probabilities @ grader's 0.5 cut (= what run-4 submitted): "
      f"binary_f1 {naive_scores['binary_f1']:.4f} | macro_f1 {naive_scores['macro_f1']:.4f} "
      f"({int(naive.sum())} positives, rate {naive.mean():.4f})")
print(f"  hard labels @ tuned cut (= what this run submits)          : "
      f"binary_f1 {DIAG['binary_f1']:.4f} | macro_f1 {DIAG['macro_f1']:.4f} "
      f"({DIAG['n_pred_pos']} positives, rate {DIAG['pred_pos_rate']:.4f})")
print(f"  gain: binary_f1 {DIAG['binary_f1'] - naive_scores['binary_f1']:+.4f} | "
      f"macro_f1 {DIAG['macro_f1'] - naive_scores['macro_f1']:+.4f}")

print(f"\n=== degenerate floors (for reference) ===")
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
print("  NOTE (see run-2 audit in CLAUDE.md): shift-holdout F1 has measured HIGHER than full")
print("  OOF in every case so far (confounded by the holdout's own elevated positive rate) --")
print("  trust it only for the RELATIVE arm ranking above, not as an absolute LB estimate.")

print(f"\n=== vs. the target this run is trying to beat ===")
print(f"  run-4 public LB: {RUN4_LB_REFERENCE:.6f}")
print(f"  this run's OOF binary_f1 (hard-label, tuned cut): {DIAG['binary_f1']:.4f}")
print("  OOF is not LB -- this is not a prediction of the real score, only a same-units")
print("  reference point until this run is actually submitted.")

curve.to_csv(os.path.join(OUT_DIR, "threshold_curve.csv"), index=False)

# %% [markdown]
# ## 10. Submission
#
# Hard 0/1 labels at the tuned threshold — not raw probabilities. `id` comes from `test.csv`
# itself and is asserted equal to `sample_submission.csv`'s, never regenerated with `range()`.

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
    assert 0.005 < rate < 0.20, f"implausible positive rate {rate:.4f}"
    return rate


rate = validate_submission(sub, sample_sub[ID].values, len(sample_sub))
sub.to_csv(SUBMISSION_PATH, index=False)

print(f"wrote {SUBMISSION_PATH}")
print(f"  rows: {len(sub)} | positives: {int(sub[TARGET].sum())} | rate: {rate:.4f}")
print(f"  OOF predicted-positive rate for comparison: {DIAG['pred_pos_rate']:.4f}")
print(f"  run-4's test positive rate for comparison: 0.0705 (at its raw, untuned 0.5 cut)")
print(sub.head())

# %% [markdown]
# ## 11. Save artifacts
#
# No inference notebook this round (per this session's request) — these are saved anyway so the
# run is reproducible later without re-running the arm search or re-deriving the fold split.

# %%
artifacts = {
    "models": models,
    "cat_maps": CAT_MAPS,
    "clip_bounds": CLIP_BOUNDS,
    "fill_values": FILL_VALUES,
    "drop_cols": DROP_COLS,
    "base_drop_cols": BASE_DROP_COLS,
    "feature_order": FEATURE_ORDER,
    "cat_code_cols": CAT_CODE_COLS,
    "threshold": THRESHOLD,
    "target_metric": TARGET_METRIC,
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "seed": SEED,
    "model_backend": MODEL_BACKEND,
    "smote_backend": SMOTE_BACKEND,
    "oof_auc": float(OOF_AUC),
    "diagnostics": DIAG,
    "cat_cols": CAT_COLS,
    "sentinel_neg": {SENTINEL_NEG_COL: SENTINEL_NEG_VAL},
    "sentinel_big_cols": SENTINEL_BIG_COLS,
    "sentinel_big_val": SENTINEL_BIG_VAL,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "smote_strategy": SMOTE_STRATEGY,
    "winning_feature_arm": winner["feature_arm"],
    "adversarial_auc": float(ADV_AUC),
    "shift_holdout_binary_f1": float(HOLDOUT_BINARY_F1),
    "shift_holdout_macro_f1": float(HOLDOUT_MACRO_F1),
    "shift_holdout_n": int(len(shift_holdout_idx)),
    "run4_lb_reference": RUN4_LB_REFERENCE,
}
joblib.dump(artifacts, ARTIFACT_PATH, compress=3)

size_mb = os.path.getsize(ARTIFACT_PATH) / 1e6
print(f"wrote {ARTIFACT_PATH} ({size_mb:.1f} MB, {len(models)} models)")

with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
    json.dump({
        "model_backend": MODEL_BACKEND, "smote_backend": SMOTE_BACKEND,
        "oof_auc": float(OOF_AUC), "threshold": THRESHOLD,
        "target_metric": TARGET_METRIC, "diagnostics": DIAG,
        "naive_half_cut": naive_scores,
        "per_seed_auc": {str(k): float(v) for k, v in per_seed_auc.items()},
        "submission_positive_rate": float(rate), "n_features": len(FEATURE_ORDER),
        "n_dropped_columns": len(DROP_COLS), "base_dropped_columns": len(BASE_DROP_COLS),
        "winning_feature_arm": winner["feature_arm"],
        "adversarial_auc": float(ADV_AUC),
        "shift_holdout_binary_f1": float(HOLDOUT_BINARY_F1),
        "shift_holdout_macro_f1": float(HOLDOUT_MACRO_F1),
        "shift_holdout_n": int(len(shift_holdout_idx)),
        "run4_lb_reference": RUN4_LB_REFERENCE,
    }, f, indent=2)

print("\nDone. This is 'raw' -- not yet run on Kaggle. After running there:")
print("  1. Copy the notebook + submission.csv + run_summary.json + arm_search.csv +")
print("     threshold_curve.csv into results/run-5/")
print("  2. Add a results/run-5/README.md like results/run-3/ and results/run-4/'s")
print("  3. Record the real public LB score and compare it against "
      f"{RUN4_LB_REFERENCE:.6f}")
