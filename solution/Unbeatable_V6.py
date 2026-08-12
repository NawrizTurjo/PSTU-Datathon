# %% [markdown]
# # Unbeatable V6 — adversarial-validation-driven, pseudo-labeled ensemble
#
# PSTU Data Thon 2026 Vol-1. Binary F1, 3.957% positive rate, 350 anonymized features (344
# numeric + 6 high-cardinality categorical strings). Full measured background: `CLAUDE.md` /
# `eda.md` at the repo root.
#
# This notebook is a **deliberate departure** from this project's earlier LightGBM-only pipeline
# (`pstu_train.py`, `run-5-raw.py`) and from the two external runs archived under `results/run-3`
# / `results/run-4`. Those all used SMOTE and/or `scale_pos_weight`/`auto_class_weights` to fight
# the 3.957% imbalance directly. This one does not — the model sees the real imbalance under
# plain LogLoss, and every unit of "fighting the imbalance" is spent entirely on **choosing the
# decision threshold**, not on reshaping the training distribution.
#
# | Step | What it does |
# |---|---|
# | 1 | Adversarial validation (train-vs-test) drops the numeric features driving covariate shift |
# | 2 | `QuantileTransformer` + PCA feature reduction (ddof=0 variance, per spec) + 6 label-encoded categoricals |
# | 3 | XGBoost + LightGBM + CatBoost, plain LogLoss, no class weighting; exact 0.01-0.50 threshold grid search on blended OOF |
# | 4 | Pseudo-label confident test rows (p>0.90 / p<0.05), fold-safe retrain |
# | 5 | `submission.csv` (hard 0/1 at the tuned threshold) + `submission_prob.csv` (raw probability) |
#
# **Why no SMOTE / class weighting:** every prior run in this project that tuned the operating
# point post-hoc got most of its F1 from the threshold choice, not the model (`CLAUDE.md`'s
# run-1: +0.207 binary-F1 from threshold alone, same model). Oversampling and class-weighting
# both distort the model's own probability calibration — which is exactly what an exhaustive
# threshold search over the *raw* calibration needs to stay accurate. Removing them is a bet that
# an honestly-calibrated model plus a properly-tuned cut beats a recalibrated model whose "0.5"
# has been artificially moved to mean something else.
#
# Grader behaviour (measured, see `CLAUDE.md`): the grader applies a **fixed 0.5 cut** to
# whatever is submitted. `submission.csv` therefore carries pre-thresholded integers — the
# threshold search below is not a diagnostic, it is baked directly into the file that gets
# scored.

# %%
import os
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- configuration
SEED = 42
N_FOLDS = 5
ADV_CV_FOLDS = 5              # folds for the train-vs-test adversarial classifier
DRIFT_DROP_N = 20             # numeric features dropped by adversarial-importance rank
PCA_VARIANCE = 0.95           # PCA keeps components covering this much variance (ddof=0, see below)
PSEUDO_POS_THRESH = 0.90      # test rows above this get pseudo-label 1
PSEUDO_NEG_THRESH = 0.05      # test rows below this get pseudo-label 0
THRESHOLD_GRID = np.round(np.arange(0.01, 0.501, 0.01), 2)   # exact spec: 0.01 .. 0.50 step 0.01

N_ESTIMATORS = 2000           # XGBoost / LightGBM trees (early-stopped, rarely hit the cap)
CAT_ITERATIONS = 2000         # CatBoost iterations (early-stopped)
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100

SMOKE_TEST = False             # True -> tiny subsample + tiny models, for a fast correctness pass

OUT_DIR = os.environ.get(
    "PSTU_OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- determinism
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# %%
import lightgbm as lgb
import xgboost as xgb
import catboost
from catboost import CatBoostClassifier

print(f"lightgbm {lgb.__version__} | xgboost {xgb.__version__} | catboost {catboost.__version__}")

# %% [markdown]
# ## 1. Load data
#
# Path auto-detection so the same notebook runs on Kaggle (either input-mount layout) or locally
# with no edits.

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
    N_ESTIMATORS, CAT_ITERATIONS, EARLY_STOPPING_ROUNDS = 80, 80, 20
    print("SMOKE: reduced to", train.shape, test.shape)

TARGET = "TARGET"
ID = "id"
y_train = train[TARGET].values.astype(int)
test_ids = test[ID].copy()
print(f"positive rate: {y_train.mean():.6f}  ({y_train.sum()} / {len(y_train)})")

assert [c for c in test.columns if c != ID] == [c for c in train.columns if c != TARGET]
assert list(test.columns)[-1] == ID, "id is not the last column of test.csv"

# %% [markdown]
# ## 2. Column contract
#
# The 6 categorical columns and both sentinel families are measured facts (Stage-1 EDA, see
# `eda.md`). The 44 droppable columns (28 constant-in-train + 16 exact row-for-row duplicates)
# are **recomputed from train here**, never hand-transcribed — a typo in a 44-name list is
# exactly the kind of silent bug that costs a competition.

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


def compute_droppable(df, numeric_cols):
    """Constant-in-train columns + exact row-for-row duplicate columns (keep the first, sorted)."""
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
print(f"constant/duplicate columns dropped (Stage-1 measured 44 on the full train set): "
      f"{len(BASE_DROP_COLS)}")


def clean_sentinels(df):
    out = df.copy()
    if SENTINEL_NEG_COL in out.columns:
        out[SENTINEL_NEG_COL] = out[SENTINEL_NEG_COL].replace(SENTINEL_NEG_VAL, np.nan)
    for c in SENTINEL_BIG_COLS:
        if c in out.columns:
            out[c] = out[c].replace(SENTINEL_BIG_VAL, np.nan)
    return out


train_clean = (
    clean_sentinels(train.drop(columns=[TARGET]))
    .drop(columns=BASE_DROP_COLS, errors="ignore")
    .reset_index(drop=True)
)
test_clean = (
    clean_sentinels(test.drop(columns=[ID]))
    .drop(columns=BASE_DROP_COLS, errors="ignore")
    .reset_index(drop=True)
)
test_clean = test_clean[train_clean.columns]

NUMERIC_COLS_CLEAN = [c for c in train_clean.columns if c not in CAT_COLS]
print(f"columns remaining after sentinel cleanup + constant/dup drop: {train_clean.shape[1]} "
      f"({len(NUMERIC_COLS_CLEAN)} numeric + {len(CAT_COLS)} categorical)")


def fit_label_encoders(df, cols):
    """Vocabulary fit on TRAIN ONLY; unseen levels at inference map to one fallback code past the end."""
    return {c: {lvl: i for i, lvl in enumerate(sorted(df[c].dropna().unique()))} for c in cols}


def apply_label_encoders(df, cols, maps):
    out = df.copy()
    for c in cols:
        unseen_code = len(maps[c])
        out[c] = df[c].map(maps[c]).fillna(unseen_code).astype(np.int32)
    return out


CAT_ENCODE_MAPS = fit_label_encoders(train_clean, CAT_COLS)
train_cat_encoded = apply_label_encoders(train_clean, CAT_COLS, CAT_ENCODE_MAPS)
test_cat_encoded = apply_label_encoders(test_clean, CAT_COLS, CAT_ENCODE_MAPS)

for c in CAT_COLS:
    unseen_n = (test_clean[c].isin(CAT_ENCODE_MAPS[c].keys()) == False).sum()
    print(f"  {c}: {len(CAT_ENCODE_MAPS[c])} train levels, {unseen_n} unseen test rows -> fallback code")

# %% [markdown]
# ## 3. Step 1 — Adversarial validation: find and drop covariate-shifted features
#
# Concatenate train+test (features only), label train=0 / test=1, train a quick LightGBM
# classifier to tell them apart via `ADV_CV_FOLDS`-fold CV. The features it leans on hardest are
# the ones whose train/test distributions disagree most — exactly the features most likely to
# hurt generalization from OOF to the real leaderboard. The top `DRIFT_DROP_N` of those (by mean
# CV gain-importance, **numeric candidates only**) are dropped before anything downstream sees
# them. The 6 categorical columns stay in the adversarial model (so its discrimination accuracy
# is not artificially handicapped) but are never candidates for dropping — they are always kept,
# per spec.
#
# Prior measured baseline (Stage-1 EDA, `eda.md` section 5): adversarial train/test AUC 0.5742,
# `feat_182` dominating at ~0.17 importance (~2.4x the next feature). Expect a similar AUC here.

# %%
adv_X = pd.concat([train_cat_encoded, test_cat_encoded], axis=0, ignore_index=True)
adv_y = np.array([0] * len(train_cat_encoded) + [1] * len(test_cat_encoded))

adv_fill = adv_X[NUMERIC_COLS_CLEAN].median()
adv_X_filled = adv_X.copy()
adv_X_filled[NUMERIC_COLS_CLEAN] = adv_X_filled[NUMERIC_COLS_CLEAN].fillna(adv_fill)

adv_importances = np.zeros(adv_X_filled.shape[1])
adv_aucs = []
adv_folds = StratifiedKFold(n_splits=ADV_CV_FOLDS, shuffle=True, random_state=SEED)
for fold_i, (tr_idx, va_idx) in enumerate(adv_folds.split(adv_X_filled, adv_y)):
    adv_model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold_i,
        n_jobs=-1, verbosity=-1,
    )
    adv_model.fit(adv_X_filled.iloc[tr_idx], adv_y[tr_idx])
    va_pred = adv_model.predict_proba(adv_X_filled.iloc[va_idx])[:, 1]
    adv_aucs.append(roc_auc_score(adv_y[va_idx], va_pred))
    adv_importances += adv_model.feature_importances_ / ADV_CV_FOLDS

print(f"adversarial train/test AUC: {np.mean(adv_aucs):.4f} +/- {np.std(adv_aucs):.4f}")

adv_importance_df = pd.DataFrame({
    "feature": adv_X_filled.columns,
    "importance": adv_importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)
adv_importance_df.to_csv(os.path.join(OUT_DIR, "adversarial_importance.csv"), index=False)

drift_candidates = adv_importance_df[adv_importance_df["feature"].isin(NUMERIC_COLS_CLEAN)]
DRIFT_DROP_COLS = drift_candidates.head(DRIFT_DROP_N)["feature"].tolist()
covered = drift_candidates.head(DRIFT_DROP_N)["importance"].sum() / adv_importance_df["importance"].sum()
print(f"dropping top {DRIFT_DROP_N} shift-driving numeric features "
      f"({covered:.1%} of total adversarial importance):")
for _, row in drift_candidates.head(DRIFT_DROP_N).iterrows():
    print(f"  {row['feature']:<14s} importance={row['importance']:.1f}")

NUMERIC_COLS_FINAL = [c for c in NUMERIC_COLS_CLEAN if c not in DRIFT_DROP_COLS]
print(f"numeric columns remaining after adversarial drop: {len(NUMERIC_COLS_FINAL)}")

# %% [markdown]
# ## 4. Step 2 — QuantileTransformer + PCA (feature reduction)
#
# `QuantileTransformer(output_distribution="normal")` maps every remaining numeric feature onto
# a common Gaussian-shaped scale (robust to the heavy zero-inflation and outliers measured in
# Stage-1 EDA — 252 of 344 numeric columns were >=90% zero). PCA then extracts a compact latent
# representation, keeping only the 6 label-encoded categoricals alongside it — no large feature
# explosion.
#
# **MATH FIX (per spec):** sklearn's `PCA.explained_variance_` divides by `n_samples - 1`
# (ddof=1). This has **no effect on the PCA components themselves** — they come from the SVD of
# the centred data, which is invariant to the ddof convention used to describe variance — but it
# does change what "variance explained" means when reported. We recompute explained variance
# under `ddof=0` (population variance, divide by `n`) below, and use `ddof=0` explicitly in every
# other manual variance/std computation in this notebook.

# %%
train_num_fill = train_clean[NUMERIC_COLS_FINAL].median()
X_train_num = train_clean[NUMERIC_COLS_FINAL].fillna(train_num_fill)
X_test_num = test_clean[NUMERIC_COLS_FINAL].fillna(train_num_fill)

quantile = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=min(1000, len(X_train_num)),
    random_state=SEED,
)
X_train_quant = quantile.fit_transform(X_train_num)
X_test_quant = quantile.transform(X_test_num)

pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)
X_train_pca = pca.fit_transform(X_train_quant)
X_test_pca = pca.transform(X_test_quant)

n_samples_fit = X_train_quant.shape[0]
total_var_ddof0 = X_train_quant.var(axis=0, ddof=0).sum()           # ddof=0, per spec
explained_var_ddof0 = (pca.singular_values_ ** 2) / n_samples_fit   # per-component variance, ddof=0
explained_var_ratio_ddof0 = explained_var_ddof0 / total_var_ddof0
print(f"PCA: {X_train_pca.shape[1]} components | variance covered "
      f"{explained_var_ratio_ddof0.sum():.4f} (ddof=0) vs "
      f"{pca.explained_variance_ratio_.sum():.4f} (sklearn default, ddof=1)")

pca_cols = [f"pca_{i:03d}" for i in range(X_train_pca.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols)

CAT_CODE_COLS = [f"{c}_code" for c in CAT_COLS]
train_cat_codes = train_cat_encoded[CAT_COLS].reset_index(drop=True)
train_cat_codes.columns = CAT_CODE_COLS
test_cat_codes = test_cat_encoded[CAT_COLS].reset_index(drop=True)
test_cat_codes.columns = CAT_CODE_COLS

X_train_final = pd.concat([X_train_pca_df, train_cat_codes], axis=1)
X_test_final = pd.concat([X_test_pca_df, test_cat_codes], axis=1)
CAT_FEATURE_IDX = [X_train_final.columns.get_loc(c) for c in CAT_CODE_COLS]

print(f"final model feature matrix: {X_train_final.shape} "
      f"({len(pca_cols)} PCA components + {len(CAT_CODE_COLS)} categorical codes)")

# %% [markdown]
# ## 5. Step 3 — Pure learning (no imbalance handling) + exact threshold search
#
# XGBoost, LightGBM, and CatBoost, each trained with plain LogLoss on the natural 3.957%
# positive rate — **no `scale_pos_weight`, no `auto_class_weights`, no oversampling.** The three
# models vote by simple average of their probabilities; the operating point that turns that
# blended probability into a 0/1 label is chosen entirely separately, by an exact grid search
# over `t in [0.01, 0.50]` step `0.01` (50 points) against the blended out-of-fold predictions,
# keeping whichever exact value maximizes binary F1.

# %%
def f1_threshold_search(y_true, scores, grid=THRESHOLD_GRID):
    """Exact grid search over thresholds; returns (best_threshold, best_f1, full curve df)."""
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append((float(t), f1, int(preds.sum())))
    curve = pd.DataFrame(rows, columns=["threshold", "f1", "n_pred_pos"])
    best = curve.loc[curve["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"]), curve


# --- sanity check: f1_threshold_search must reproduce sklearn's f1_score exactly ---
_rng = np.random.default_rng(0)
_y = (_rng.random(3000) < 0.04).astype(int)
_s = np.clip(_y * 0.3 + _rng.random(3000) * 0.7, 0, 1)
_t, _f1, _ = f1_threshold_search(_y, _s)
assert abs(f1_score(_y, (_s >= _t).astype(int)) - _f1) < 1e-9
print("f1_threshold_search verified against sklearn.metrics.f1_score (tol 1e-9)")
del _rng, _y, _s, _t, _f1


def make_models(seed):
    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=6, learning_rate=LEARNING_RATE,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, num_leaves=31, learning_rate=LEARNING_RATE,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30, reg_lambda=1.0,
        objective="binary", metric="binary_logloss",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    cat_model = CatBoostClassifier(
        iterations=CAT_ITERATIONS, depth=6, learning_rate=LEARNING_RATE, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="Logloss", cat_features=CAT_FEATURE_IDX,
        random_seed=seed, verbose=False, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        allow_writing_files=False,
    )
    return xgb_model, lgb_model, cat_model


def fit_one_fold(X_tr, y_tr, X_va, y_va, X_test, seed):
    xgb_model, lgb_model, cat_model = make_models(seed)

    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    va_preds = {
        "xgb": xgb_model.predict_proba(X_va)[:, 1],
        "lgb": lgb_model.predict_proba(X_va)[:, 1],
        "cat": cat_model.predict_proba(X_va)[:, 1],
    }
    test_preds = {
        "xgb": xgb_model.predict_proba(X_test)[:, 1],
        "lgb": lgb_model.predict_proba(X_test)[:, 1],
        "cat": cat_model.predict_proba(X_test)[:, 1],
    }
    return va_preds, test_preds


# %%
folds = list(
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(y_train)), y_train
    )
)

oof = {"xgb": np.zeros(len(y_train)), "lgb": np.zeros(len(y_train)), "cat": np.zeros(len(y_train))}
test_pred_folds = {"xgb": [], "lgb": [], "cat": []}

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X_train_final.iloc[tr_idx], X_train_final.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    va_preds, test_preds = fit_one_fold(X_tr, y_tr, X_va, y_va, X_test_final, seed=SEED)
    for name in ("xgb", "lgb", "cat"):
        oof[name][va_idx] = va_preds[name]
        test_pred_folds[name].append(test_preds[name])

    fold_blend = np.mean([va_preds["xgb"], va_preds["lgb"], va_preds["cat"]], axis=0)
    print(f"fold {fold_i} [stage 1]: blended AUC {roc_auc_score(y_va, fold_blend):.4f}")

oof_blend_stage1 = np.mean([oof["xgb"], oof["lgb"], oof["cat"]], axis=0)
test_blend_stage1 = np.mean(
    [np.mean(test_pred_folds[name], axis=0) for name in ("xgb", "lgb", "cat")], axis=0
)

stage1_threshold, stage1_f1, stage1_curve = f1_threshold_search(y_train, oof_blend_stage1)
stage1_curve.to_csv(os.path.join(OUT_DIR, "threshold_curve_stage1.csv"), index=False)
print(f"[stage 1 / pre-pseudo] OOF AUC {roc_auc_score(y_train, oof_blend_stage1):.4f} | "
      f"best threshold {stage1_threshold:.2f} -> F1 {stage1_f1:.4f} "
      f"({int((oof_blend_stage1 >= stage1_threshold).sum())} predicted positive)")

# %% [markdown]
# ## 6. Step 4 — Pseudo-labeling
#
# Take the stage-1 ensemble's test-set probabilities. Rows it is very confident about
# (p > `PSEUDO_POS_THRESH` for class 1, p < `PSEUDO_NEG_THRESH` for class 0) become extra labeled
# rows. They are appended to every fold's **training** partition only — never to a validation
# fold — so the stage-2 OOF score below stays an honest read on genuinely-labeled rows, not
# inflated by the model grading its own pseudo-labels. The final decision threshold is likewise
# taken from this honest stage-2 OOF, never from anything touching pseudo-labeled rows.

# %%
pseudo_pos_mask = test_blend_stage1 > PSEUDO_POS_THRESH
pseudo_neg_mask = test_blend_stage1 < PSEUDO_NEG_THRESH
pseudo_mask = pseudo_pos_mask | pseudo_neg_mask

pseudo_X = X_test_final[pseudo_mask].reset_index(drop=True)
pseudo_y = pseudo_pos_mask[pseudo_mask].astype(int)

print(f"pseudo-labeled rows: {pseudo_mask.sum()} / {len(test_blend_stage1)} "
      f"({pseudo_mask.mean():.4%}) | positive: {int(pseudo_y.sum())} | "
      f"negative: {int((pseudo_y == 0).sum())}")

# %% [markdown]
# ## 7. Stage 2 — honest retrain on train + pseudo-labels
#
# Same 5-fold split as stage 1 (so the two stages are directly comparable). Each fold's training
# set gets every pseudo-labeled row appended; each fold still validates only on real, held-out
# train rows.

# %%
oof2 = {"xgb": np.zeros(len(y_train)), "lgb": np.zeros(len(y_train)), "cat": np.zeros(len(y_train))}
test_pred_folds2 = {"xgb": [], "lgb": [], "cat": []}

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    X_tr = pd.concat([X_train_final.iloc[tr_idx], pseudo_X], axis=0, ignore_index=True)
    y_tr = np.concatenate([y_train[tr_idx], pseudo_y])
    X_va, y_va = X_train_final.iloc[va_idx], y_train[va_idx]

    va_preds, test_preds = fit_one_fold(X_tr, y_tr, X_va, y_va, X_test_final, seed=SEED)
    for name in ("xgb", "lgb", "cat"):
        oof2[name][va_idx] = va_preds[name]
        test_pred_folds2[name].append(test_preds[name])

    fold_blend2 = np.mean([va_preds["xgb"], va_preds["lgb"], va_preds["cat"]], axis=0)
    print(f"fold {fold_i} [stage 2]: blended AUC {roc_auc_score(y_va, fold_blend2):.4f}")

oof_blend_stage2 = np.mean([oof2["xgb"], oof2["lgb"], oof2["cat"]], axis=0)
test_blend_stage2 = np.mean(
    [np.mean(test_pred_folds2[name], axis=0) for name in ("xgb", "lgb", "cat")], axis=0
)

stage2_threshold, stage2_f1, stage2_curve = f1_threshold_search(y_train, oof_blend_stage2)
stage2_curve.to_csv(os.path.join(OUT_DIR, "threshold_curve_stage2.csv"), index=False)
print(f"[stage 2 / post-pseudo] OOF AUC {roc_auc_score(y_train, oof_blend_stage2):.4f} | "
      f"best threshold {stage2_threshold:.2f} -> F1 {stage2_f1:.4f} "
      f"({int((oof_blend_stage2 >= stage2_threshold).sum())} predicted positive)")

# %% [markdown]
# ## 8. Final model selection
#
# Pseudo-labeling is not guaranteed to help — it can also inject confidently-wrong labels and
# hurt generalization. Guard against that automatically: keep whichever stage has the higher
# honest OOF F1, and print which one was chosen so it's never ambiguous which model produced the
# submission.

# %%
if stage2_f1 >= stage1_f1:
    FINAL_STAGE = "stage2_pseudo_labeled"
    final_test_blend = test_blend_stage2
    final_threshold = stage2_threshold
    final_oof_f1 = stage2_f1
else:
    FINAL_STAGE = "stage1_baseline"
    final_test_blend = test_blend_stage1
    final_threshold = stage1_threshold
    final_oof_f1 = stage1_f1

print(f"FINAL MODEL: {FINAL_STAGE}  (stage1 F1={stage1_f1:.4f} | stage2 F1={stage2_f1:.4f})")
print(f"final threshold: {final_threshold:.2f} | final OOF F1: {final_oof_f1:.4f}")

# %% [markdown]
# ## 9. Step 5 — Submission
#
# `submission.csv` carries the hard 0/1 label from the threshold chosen above — the grader
# applies its own fixed 0.5 cut to whatever is submitted, so pre-thresholding here is what makes
# the threshold search matter (see `CLAUDE.md`: raw probabilities at 0.5 vs a tuned hard cut was
# a +0.207 binary-F1 swing on this same competition). `submission_prob.csv` carries the raw
# blended probability, for re-thresholding later without re-running the notebook.

# %%
final_preds_int = (final_test_blend >= final_threshold).astype(int)

submission = sample_sub.copy()
submission[TARGET] = final_preds_int
assert list(submission.columns) == [ID, TARGET]
assert len(submission) == len(sample_sub)
assert set(submission[TARGET].unique()) <= {0, 1}
assert (submission[ID].values == test_ids.values).all()

submission_prob = sample_sub.copy()
submission_prob[TARGET] = final_test_blend
assert list(submission_prob.columns) == [ID, TARGET]
assert (submission_prob[ID].values == test_ids.values).all()

SUBMISSION_PATH = os.path.join(OUT_DIR, "submission.csv")
SUBMISSION_PROB_PATH = os.path.join(OUT_DIR, "submission_prob.csv")
submission.to_csv(SUBMISSION_PATH, index=False)
submission_prob.to_csv(SUBMISSION_PROB_PATH, index=False)

pos_rate = final_preds_int.mean()
print(f"submission.csv: {len(submission)} rows | {int(final_preds_int.sum())} positives | "
      f"rate {pos_rate:.4%}")
print(f"submission_prob.csv: {len(submission_prob)} rows | raw probabilities, "
      f"min={final_test_blend.min():.4f} max={final_test_blend.max():.4f}")
if not (0.01 <= pos_rate <= 0.15):
    print(f"WARNING: predicted-positive rate {pos_rate:.4%} is well outside the ~3-6% ballpark "
          f"measured for train (3.957%) — sanity-check the threshold/pseudo-labeling before submitting.")

run_summary = {
    "final_stage": FINAL_STAGE,
    "adversarial_auc_mean": float(np.mean(adv_aucs)),
    "adversarial_auc_std": float(np.std(adv_aucs)),
    "drift_dropped_columns": DRIFT_DROP_COLS,
    "n_pca_components": int(X_train_pca.shape[1]),
    "pca_variance_covered_ddof0": float(explained_var_ratio_ddof0.sum()),
    "stage1_oof_auc": float(roc_auc_score(y_train, oof_blend_stage1)),
    "stage1_threshold": stage1_threshold,
    "stage1_f1": stage1_f1,
    "stage2_oof_auc": float(roc_auc_score(y_train, oof_blend_stage2)),
    "stage2_threshold": stage2_threshold,
    "stage2_f1": stage2_f1,
    "pseudo_labeled_rows": int(pseudo_mask.sum()),
    "pseudo_positive": int(pseudo_y.sum()),
    "pseudo_negative": int((pseudo_y == 0).sum()),
    "final_threshold": final_threshold,
    "final_oof_f1": final_oof_f1,
    "submission_positive_rate": float(pos_rate),
    "submission_n_positive": int(final_preds_int.sum()),
}
with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
    json.dump(run_summary, f, indent=2)
print(json.dumps(run_summary, indent=2))
