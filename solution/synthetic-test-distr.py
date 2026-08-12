# %% [markdown]
# # Synthetic test-distribution augmentation — PSTU Data Thon 2026 Vol-1
#
# Companion to `Unbeatable_V6.py`. Full background: `CLAUDE.md` / `next-gen.md` at the repo root.
#
# **Hypothesis under test** (teammate, 2026-08-12): the measured train/test covariate shift
# (`CLAUDE.md`: adversarial AUC 0.5742) may be why models trained purely on train's distribution
# underperform on LB. This notebook generates a **synthetic dataset whose feature distribution
# matches test** (not train), pseudo-labels it with a model trained on real train, and adds it to
# training — so the model sees a training mixture that already resembles what it will be scored
# on. This uses only the provided train/test files (no external data), never touches or infers
# real test labels, and pseudo-labeling is model-generated (not LLM-generated) — all explicitly
# permitted per `CLAUDE.md`'s Rules section.
#
# | Step | What it does |
# |---|---|
# | 1 | Same column contract + adversarial-drop + QT/PCA(ddof=0) feature pipeline as `Unbeatable_V6.py` |
# | 2 | **New** — Gaussian-copula generator fit on TEST ONLY: reproduces test's marginals + dominant correlation structure; categoricals sampled from per-cluster empirical test frequencies |
# | 3 | Generator quality check: adversarial AUC of synthetic-vs-test (want ~0.50) and synthetic-vs-train (want clearly >0.50) |
# | 4 | Baseline ensemble (XGB+LGB+CatBoost, plain LogLoss, no SMOTE/weighting — same stance as V6) trains on real train, predicts OOF + test + synthetic |
# | 5 | Pseudo-label synthetic rows, with a negative-domination guard (see below) |
# | 6 | Fold-safe augmented retrain (synthetic rows only ever enter training folds, never validation) |
# | 7 | Pick whichever stage has the higher honest real-train OOF F1 |
# | 8 | **Gap-closing diagnostic** — does the augmented train-vs-test adversarial AUC move toward 0.50? This is the actual test of the hypothesis, not just a hoped-for side effect |
# | 9 | `submission.csv` (hard 0/1) + `submission_prob.csv` + `synthetic_test_distribution.csv` |
#
# **Lesson carried in from `results/unbeatable-6/run_summary.json`** (see `next-gen.md`): that
# run's pseudo-labeling produced `pseudo_positive: 0` — a plain symmetric probability threshold at
# a ~4% base rate almost always finds far more confident negatives than positives, silently
# turning "pseudo-labeling" into "add thousands of extra negatives," which likely explains why its
# LB (0.109-0.124) came in below even the un-augmented run-1 baseline (0.1849). This notebook
# guards against repeating that: pseudo-label thresholds relax (within a floor) until a minimum
# count of *each* class is found, accepted negatives are capped at a fixed ratio to accepted
# positives, and — if zero positives survive even after relaxing — stage 2 is skipped outright
# rather than silently injecting an all-negative batch.

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
ADV_CV_FOLDS = 5              # folds for the train-vs-test drift-drop adversarial classifier
DRIFT_DROP_N = 20             # numeric features dropped by adversarial-importance rank
PCA_VARIANCE = 0.95           # PCA keeps components covering this much variance (ddof=0 reporting)

SYNTHETIC_MULTIPLIER = 1.0    # n_synthetic = len(test) * this
JITTER_FRAC = 0.03            # multiplicative jitter on nonzero, non-integer-valued numeric cells

PSEUDO_POS_THRESH = 0.90      # synthetic rows above this get pseudo-label 1
PSEUDO_NEG_THRESH = 0.05      # synthetic rows below this get pseudo-label 0
PSEUDO_RELAX_STEP = 0.05      # relax thresholds toward 0.5 by this much per step if too few found
PSEUDO_RELAX_FLOOR_MARGIN = 0.05   # never relax past 0.5 +/- this margin
PSEUDO_MIN_ACCEPT = 20        # stop relaxing once this many rows of a class are found
PSEUDO_MAX_NEG_POS_RATIO = 10.0    # cap accepted negatives at this multiple of accepted positives

THRESHOLD_GRID = np.round(np.arange(0.01, 0.501, 0.01), 2)   # exact 0.01 .. 0.50 step 0.01

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

N_SYNTHETIC = int(round(len(test) * SYNTHETIC_MULTIPLIER))
print(f"N_SYNTHETIC = {N_SYNTHETIC}")

# %% [markdown]
# ## 2. Column contract
#
# Identical to `Unbeatable_V6.py` — the 44-column drop is recomputed here, never hand-transcribed.

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
# Identical mechanics to `Unbeatable_V6.py` section 3. Prior measured baseline (`CLAUDE.md`):
# adversarial train/test AUC 0.5742, `feat_182` dominant.

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

print(f"adversarial train/test AUC (raw feature space): {np.mean(adv_aucs):.4f} +/- {np.std(adv_aucs):.4f}")

adv_importance_df = pd.DataFrame({
    "feature": adv_X_filled.columns,
    "importance": adv_importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)
adv_importance_df.to_csv(os.path.join(OUT_DIR, "adversarial_importance.csv"), index=False)

drift_candidates = adv_importance_df[adv_importance_df["feature"].isin(NUMERIC_COLS_CLEAN)]
DRIFT_DROP_COLS = drift_candidates.head(DRIFT_DROP_N)["feature"].tolist()
covered = drift_candidates.head(DRIFT_DROP_N)["importance"].sum() / adv_importance_df["importance"].sum()
print(f"dropping top {DRIFT_DROP_N} shift-driving numeric features ({covered:.1%} of total importance)")

NUMERIC_COLS_FINAL = [c for c in NUMERIC_COLS_CLEAN if c not in DRIFT_DROP_COLS]
print(f"numeric columns remaining after adversarial drop: {len(NUMERIC_COLS_FINAL)}")

# %% [markdown]
# ## 4. Step 2 — QuantileTransformer + PCA (shared model-input feature space)
#
# Fit on **train** — this is the space every row (train, test, and later synthetic) is pushed
# through before hitting the models. Same ddof=0 explained-variance reporting fix as
# `Unbeatable_V6.py` (the PCA components themselves are ddof-invariant; only the reported
# variance-covered number needs the manual recompute).

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
total_var_ddof0 = X_train_quant.var(axis=0, ddof=0).sum()
explained_var_ddof0 = (pca.singular_values_ ** 2) / n_samples_fit
explained_var_ratio_ddof0 = explained_var_ddof0 / total_var_ddof0
print(f"PCA (train-fit, shared space): {X_train_pca.shape[1]} components | variance covered "
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
# ## 5. Step 3 — Synthetic test-distribution generator
#
# **This went through two failed designs before landing here — both caught by the local plumbing
# dry-run, not left for Kaggle to discover. Recorded in `next-gen.md` in full; summary below.**
#
# Design 1 was a Gaussian-copula generator (test-fit `QuantileTransformer` + test-fit `PCA`,
# sampling independent Gaussians per retained latent component, inverse-transforming back to raw
# units). Its generator-quality check — a quick adversarial AUC of synthetic-vs-real-test in the
# shared feature space — came back at **0.98** (should be ~0.50). Diagnosis: `PCA.inverse_transform`
# places every synthetic point *exactly* on the retained k-dimensional subspace (off-subspace
# residual norm ~0), while real rows scatter off that subspace by the ~5% of variance PCA
# discarded — trivially detectable. Fixing that (re-injecting isotropic residual noise) barely
# moved the number (0.98 -> 0.97), so a second, larger effect was still dominant.
#
# Design 2 replaced the parametric copula with resampling real test rows plus multiplicative
# jitter on nonzero values — still 0.98-0.99. Root cause (found by checking per-column
# integer-valuedness against the raw CSVs): **212 of this run's 271 surviving numeric columns are
# >=99% integer-valued** (measured directly, not assumed — `np.isclose(vals, round(vals))`). ANY
# continuous jitter, however small, moves an integer-valued column off its exact-integer grid, and
# a boosted-tree adversarial classifier trivially learns "is this value a whole number" across
# 200+ columns at once — a huge joint signal invisible in per-feature univariate AUC checks (which
# topped out around 0.66) because it only shows up combined across many columns simultaneously.
#
# **What's actually implemented below**: bootstrap-resample real test rows (with replacement, to
# `N_SYNTHETIC` rows), which by construction reproduces every marginal, every joint correlation,
# and every zero/discreteness pattern in test's raw feature space exactly — then apply a small
# multiplicative jitter (`JITTER_FRAC`) **only to the nonzero entries of genuinely continuous
# columns**, leaving integer-valued and zero entries untouched, so the resampled rows are not
# byte-identical to real test rows (avoiding a pure duplicate-injection) while remaining
# statistically indistinguishable from them. Categorical values come along with the resampled row
# unchanged, so numeric<->categorical joint coupling is preserved automatically — no separate
# clustering step is needed at all.
#
# Measured result (this fix, `next-gen.md`'s plumbing dry-run at ~20-30k row scale): synthetic-vs-
# test AUC dropped from **0.98 -> 0.52** (want ~0.50) while synthetic-vs-train AUC stayed at
# **0.60**, i.e. synthetic rows are now statistically indistinguishable from real test rows while
# still clearly distinct from train — exactly the target behaviour.

# %%
frac_int = test_clean[NUMERIC_COLS_FINAL].apply(
    lambda s: np.isclose(s.dropna(), np.round(s.dropna())).mean() if s.notna().any() else 1.0
)
INT_LIKE_MASK = (frac_int >= 0.99).values
print(f"integer-valued numeric columns (excluded from jitter): "
      f"{INT_LIKE_MASK.sum()} / {len(INT_LIKE_MASK)}")

rng = np.random.default_rng(SEED)
sample_idx = rng.integers(0, len(test_clean), size=N_SYNTHETIC)

synth_num_raw = (
    test_clean[NUMERIC_COLS_FINAL].iloc[sample_idx].reset_index(drop=True).fillna(train_num_fill)
)
synth_cat_df = test_clean[CAT_COLS].iloc[sample_idx].reset_index(drop=True)

vals = synth_num_raw.values.copy()
jitter_eligible = (vals != 0) & (~INT_LIKE_MASK)[np.newaxis, :]
mult = np.ones(vals.shape)
mult[jitter_eligible] = 1.0 + rng.normal(loc=0.0, scale=JITTER_FRAC, size=int(jitter_eligible.sum()))
vals = vals * mult
synth_num_raw = pd.DataFrame(vals, columns=NUMERIC_COLS_FINAL)
print(f"synthetic numeric rows generated (bootstrap + jitter): {synth_num_raw.shape}")

# --- push synthetic rows through the SAME train-fit QT+PCA pipeline as train/test ---
X_synth_quant = quantile.transform(synth_num_raw)
X_synth_pca = pca.transform(X_synth_quant)
X_synth_pca_df = pd.DataFrame(X_synth_pca, columns=pca_cols)

synth_cat_encoded = apply_label_encoders(synth_cat_df, CAT_COLS, CAT_ENCODE_MAPS)
synth_cat_codes = synth_cat_encoded[CAT_COLS].reset_index(drop=True)
synth_cat_codes.columns = CAT_CODE_COLS

X_synth_final = pd.concat(
    [X_synth_pca_df.reset_index(drop=True), synth_cat_codes], axis=1
)[X_train_final.columns.tolist()]

print(f"synthetic final feature matrix: {X_synth_final.shape} (matches train/test feature space)")

# %% [markdown]
# ## 6. Generator quality check
#
# Quick adversarial-validation AUCs in the shared PCA+catcode feature space (same space the real
# models will train on). This is the direct test of whether the generator actually worked, printed
# **before** any expensive model training happens.

# %%
def quick_adv_auc(X_a, X_b, seed=SEED, folds=3):
    """OOF adversarial AUC distinguishing rows of X_a (label 0) from X_b (label 1)."""
    Xc = pd.concat([X_a, X_b], axis=0, ignore_index=True)
    yc = np.array([0] * len(X_a) + [1] * len(X_b))
    aucs = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(Xc, yc):
        m = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, verbosity=-1,
        )
        m.fit(Xc.iloc[tr_idx], yc[tr_idx])
        p = m.predict_proba(Xc.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(yc[va_idx], p))
    return float(np.mean(aucs))


synth_vs_test_auc = quick_adv_auc(X_synth_final, X_test_final)
synth_vs_train_auc = quick_adv_auc(X_synth_final, X_train_final)
baseline_shift_auc = quick_adv_auc(X_train_final, X_test_final)

print("\n--- GENERATOR QUALITY CHECK ---")
print(f"synthetic vs real test AUC:  {synth_vs_test_auc:.4f}  (want close to 0.50 -> synthetic matches test)")
print(f"synthetic vs real train AUC: {synth_vs_train_auc:.4f}  (want clearly above 0.50 -> synthetic does NOT look like train)")
print(f"[reference] real train vs real test AUC, same feature space: {baseline_shift_auc:.4f}")

# %% [markdown]
# ## 7. Step 4 — Baseline ensemble (real train only) + exact threshold search
#
# Same architecture and imbalance stance as `Unbeatable_V6.py`: XGBoost + LightGBM + CatBoost,
# plain LogLoss, **no `scale_pos_weight`, no `auto_class_weights`, no oversampling**. Each fold
# also scores the real test rows and the synthetic rows, so the synthetic rows can be
# pseudo-labeled below without a separate training pass.

# %%
def f1_threshold_search(y_true, scores, grid=THRESHOLD_GRID):
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append((float(t), f1, int(preds.sum())))
    curve = pd.DataFrame(rows, columns=["threshold", "f1", "n_pred_pos"])
    best = curve.loc[curve["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"]), curve


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


def fit_one_fold(X_tr, y_tr, X_va, y_va, extra_sets, seed):
    """extra_sets: dict[name -> DataFrame] of additional feature sets to score every fold."""
    xgb_model, lgb_model, cat_model = make_models(seed)

    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    models = {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model}
    va_preds = {name: m.predict_proba(X_va)[:, 1] for name, m in models.items()}
    extra_preds = {
        key: {name: m.predict_proba(X_extra)[:, 1] for name, m in models.items()}
        for key, X_extra in extra_sets.items()
    }
    return va_preds, extra_preds


# %%
folds = list(
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(y_train)), y_train
    )
)

oof = {"xgb": np.zeros(len(y_train)), "lgb": np.zeros(len(y_train)), "cat": np.zeros(len(y_train))}
test_pred_folds = {"xgb": [], "lgb": [], "cat": []}
synth_pred_folds = {"xgb": [], "lgb": [], "cat": []}

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X_train_final.iloc[tr_idx], X_train_final.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    va_preds, extra_preds = fit_one_fold(
        X_tr, y_tr, X_va, y_va,
        extra_sets={"test": X_test_final, "synth": X_synth_final}, seed=SEED,
    )
    for name in ("xgb", "lgb", "cat"):
        oof[name][va_idx] = va_preds[name]
        test_pred_folds[name].append(extra_preds["test"][name])
        synth_pred_folds[name].append(extra_preds["synth"][name])

    fold_blend = np.mean([va_preds["xgb"], va_preds["lgb"], va_preds["cat"]], axis=0)
    print(f"fold {fold_i} [stage 1]: blended AUC {roc_auc_score(y_va, fold_blend):.4f}")

oof_blend_stage1 = np.mean([oof["xgb"], oof["lgb"], oof["cat"]], axis=0)
test_blend_stage1 = np.mean(
    [np.mean(test_pred_folds[name], axis=0) for name in ("xgb", "lgb", "cat")], axis=0
)
synth_blend_stage1 = np.mean(
    [np.mean(synth_pred_folds[name], axis=0) for name in ("xgb", "lgb", "cat")], axis=0
)

stage1_threshold, stage1_f1, stage1_curve = f1_threshold_search(y_train, oof_blend_stage1)
stage1_curve.to_csv(os.path.join(OUT_DIR, "threshold_curve_stage1.csv"), index=False)
print(f"[stage 1 / baseline] OOF AUC {roc_auc_score(y_train, oof_blend_stage1):.4f} | "
      f"best threshold {stage1_threshold:.2f} -> F1 {stage1_f1:.4f}")

# %% [markdown]
# ## 8. Step 5 — Pseudo-label synthetic rows (negative-domination guard)
#
# Thresholds start at `PSEUDO_POS_THRESH` / `PSEUDO_NEG_THRESH` and relax toward 0.5 in
# `PSEUDO_RELAX_STEP` increments (never past `0.5 +/- PSEUDO_RELAX_FLOOR_MARGIN`) until at least
# `PSEUDO_MIN_ACCEPT` rows of each class are found or the floor is hit. Accepted negatives are then
# capped at `PSEUDO_MAX_NEG_POS_RATIO` times the accepted positive count. If zero positives survive
# even after relaxing, stage 2 is skipped entirely rather than injecting an all-negative batch —
# see `next-gen.md` for why (`results/unbeatable-6` did exactly that and its LB score dropped
# below the un-augmented baseline).

# %%
pos_thresh_cur = PSEUDO_POS_THRESH
while (
    (synth_blend_stage1 > pos_thresh_cur).sum() < PSEUDO_MIN_ACCEPT
    and pos_thresh_cur - PSEUDO_RELAX_STEP >= 0.5 + PSEUDO_RELAX_FLOOR_MARGIN
):
    pos_thresh_cur = round(pos_thresh_cur - PSEUDO_RELAX_STEP, 4)

neg_thresh_cur = PSEUDO_NEG_THRESH
while (
    (synth_blend_stage1 < neg_thresh_cur).sum() < PSEUDO_MIN_ACCEPT
    and neg_thresh_cur + PSEUDO_RELAX_STEP <= 0.5 - PSEUDO_RELAX_FLOOR_MARGIN
):
    neg_thresh_cur = round(neg_thresh_cur + PSEUDO_RELAX_STEP, 4)

pos_idx = np.where(synth_blend_stage1 > pos_thresh_cur)[0]
neg_idx = np.where(synth_blend_stage1 < neg_thresh_cur)[0]
print(f"pseudo-label thresholds after relaxation: pos>{pos_thresh_cur} (started {PSEUDO_POS_THRESH}), "
      f"neg<{neg_thresh_cur} (started {PSEUDO_NEG_THRESH})")
print(f"raw confident counts: positive={len(pos_idx)} | negative={len(neg_idx)}")

if len(neg_idx) > PSEUDO_MAX_NEG_POS_RATIO * max(len(pos_idx), 1):
    cap = int(PSEUDO_MAX_NEG_POS_RATIO * max(len(pos_idx), 1))
    neg_idx = rng.choice(neg_idx, size=cap, replace=False)
    print(f"capped negative pseudo-labels to {cap} ({PSEUDO_MAX_NEG_POS_RATIO}:1 ratio guard)")

SYNTH_AUGMENTATION_SKIPPED = len(pos_idx) == 0
if SYNTH_AUGMENTATION_SKIPPED:
    print("WARNING: zero confident-positive synthetic rows even after threshold relaxation -- "
          "stage 2 (augmented retrain) will be SKIPPED, falling back to the stage-1 baseline.")

pseudo_label_col = np.full(N_SYNTHETIC, -1, dtype=int)
pseudo_label_col[pos_idx] = 1
pseudo_label_col[neg_idx] = 0
accepted_mask = pseudo_label_col != -1

accepted_idx = np.concatenate([pos_idx, neg_idx]) if not SYNTH_AUGMENTATION_SKIPPED else np.array([], dtype=int)
accepted_synth_X = X_synth_final.iloc[accepted_idx].reset_index(drop=True)
accepted_synth_y = pseudo_label_col[accepted_idx]

print(f"accepted synthetic pseudo-labels: {len(accepted_idx)} / {N_SYNTHETIC} "
      f"({len(accepted_idx) / N_SYNTHETIC:.4%}) | positive={int((accepted_synth_y == 1).sum())} "
      f"| negative={int((accepted_synth_y == 0).sum())}")

# %% [markdown]
# ## 9. Step 6 — Fold-safe augmented retrain (stage 2)
#
# Same 5-fold split as stage 1. Accepted synthetic pseudo-rows are appended to every fold's
# **training** partition only; validation always uses real, held-out train rows — same fold-safety
# pattern as `Unbeatable_V6.py` section 7, so the stage-2 OOF F1 stays an honest read on real
# labels. Skipped entirely if `SYNTH_AUGMENTATION_SKIPPED`.

# %%
if SYNTH_AUGMENTATION_SKIPPED:
    stage2_f1 = None
    stage2_threshold = None
    oof_blend_stage2 = None
    test_blend_stage2 = None
else:
    oof2 = {"xgb": np.zeros(len(y_train)), "lgb": np.zeros(len(y_train)), "cat": np.zeros(len(y_train))}
    test_pred_folds2 = {"xgb": [], "lgb": [], "cat": []}

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        X_tr = pd.concat([X_train_final.iloc[tr_idx], accepted_synth_X], axis=0, ignore_index=True)
        y_tr = np.concatenate([y_train[tr_idx], accepted_synth_y])
        X_va, y_va = X_train_final.iloc[va_idx], y_train[va_idx]

        va_preds, extra_preds = fit_one_fold(
            X_tr, y_tr, X_va, y_va, extra_sets={"test": X_test_final}, seed=SEED,
        )
        for name in ("xgb", "lgb", "cat"):
            oof2[name][va_idx] = va_preds[name]
            test_pred_folds2[name].append(extra_preds["test"][name])

        fold_blend2 = np.mean([va_preds["xgb"], va_preds["lgb"], va_preds["cat"]], axis=0)
        print(f"fold {fold_i} [stage 2]: blended AUC {roc_auc_score(y_va, fold_blend2):.4f}")

    oof_blend_stage2 = np.mean([oof2["xgb"], oof2["lgb"], oof2["cat"]], axis=0)
    test_blend_stage2 = np.mean(
        [np.mean(test_pred_folds2[name], axis=0) for name in ("xgb", "lgb", "cat")], axis=0
    )

    stage2_threshold, stage2_f1, stage2_curve = f1_threshold_search(y_train, oof_blend_stage2)
    stage2_curve.to_csv(os.path.join(OUT_DIR, "threshold_curve_stage2.csv"), index=False)
    print(f"[stage 2 / synth-augmented] OOF AUC {roc_auc_score(y_train, oof_blend_stage2):.4f} | "
          f"best threshold {stage2_threshold:.2f} -> F1 {stage2_f1:.4f}")

# %% [markdown]
# ## 10. Step 7 — Final model selection
#
# Pseudo-labeling (real or synthetic) is not guaranteed to help. Keep whichever stage has the
# higher honest OOF F1 on real train, same automatic safety net as `Unbeatable_V6.py`.

# %%
if SYNTH_AUGMENTATION_SKIPPED:
    FINAL_STAGE = "stage1_baseline_no_synth"
    final_test_blend = test_blend_stage1
    final_threshold = stage1_threshold
    final_oof_f1 = stage1_f1
elif stage2_f1 >= stage1_f1:
    FINAL_STAGE = "stage2_synth_augmented"
    final_test_blend = test_blend_stage2
    final_threshold = stage2_threshold
    final_oof_f1 = stage2_f1
else:
    FINAL_STAGE = "stage1_baseline"
    final_test_blend = test_blend_stage1
    final_threshold = stage1_threshold
    final_oof_f1 = stage1_f1

print(f"FINAL MODEL: {FINAL_STAGE}  (stage1 F1={stage1_f1:.4f} | "
      f"stage2 F1={'skipped' if stage2_f1 is None else f'{stage2_f1:.4f}'})")
print(f"final threshold: {final_threshold:.2f} | final OOF F1: {final_oof_f1:.4f}")

# %% [markdown]
# ## 11. Step 8 — Gap-closing diagnostic
#
# The actual test of the teammate's hypothesis: does adding the (pseudo-labeled) synthetic rows
# move the train-vs-test adversarial AUC closer to 0.50, in the same shared feature space measured
# in the generator quality check above? This is diagnostic only — it does **not** decide
# `FINAL_STAGE` (that stays OOF-F1-driven) — but it directly answers whether the augmentation
# narrowed the measured covariate shift, independent of whether F1 moved.

# %%
if FINAL_STAGE == "stage2_synth_augmented":
    X_final_train_used = pd.concat([X_train_final, accepted_synth_X], axis=0, ignore_index=True)
else:
    X_final_train_used = X_train_final

final_shift_auc = quick_adv_auc(X_final_train_used, X_test_final)
gap_closed = abs(final_shift_auc - 0.5) < abs(baseline_shift_auc - 0.5)

print("\n--- GAP-CLOSING DIAGNOSTIC ---")
print(f"baseline train-vs-test AUC (real train only):        {baseline_shift_auc:.4f}")
print(f"final train-vs-test AUC (training set actually used): {final_shift_auc:.4f}")
print(f"covariate shift {'NARROWED' if gap_closed else 'did NOT narrow'} "
      f"(distance to 0.50: {abs(baseline_shift_auc - 0.5):.4f} -> {abs(final_shift_auc - 0.5):.4f})")

# %% [markdown]
# ## 12. Step 9 — Submission

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
if not (0.01 <= pos_rate <= 0.15):
    print(f"WARNING: predicted-positive rate {pos_rate:.4%} is well outside the ~3-6% ballpark "
          f"measured for train (3.957%) -- sanity-check before submitting.")

# --- synthetic dataset export, raw units, for inspection / reuse ---
synthetic_export = synth_num_raw.copy()
for c in CAT_COLS:
    synthetic_export[c] = synth_cat_df[c].values
synthetic_export["source_test_row_idx"] = sample_idx
synthetic_export["stage1_prob"] = synth_blend_stage1
synthetic_export["accepted"] = accepted_mask
synthetic_export["pseudo_label"] = pseudo_label_col
synthetic_export.to_csv(os.path.join(OUT_DIR, "synthetic_test_distribution.csv"), index=False)
print(f"synthetic_test_distribution.csv: {len(synthetic_export)} rows written")

run_summary = {
    "final_stage": FINAL_STAGE,
    "adversarial_auc_raw_space_mean": float(np.mean(adv_aucs)),
    "adversarial_auc_raw_space_std": float(np.std(adv_aucs)),
    "drift_dropped_columns": DRIFT_DROP_COLS,
    "n_pca_components_shared": int(X_train_pca.shape[1]),
    "pca_variance_covered_ddof0_shared": float(explained_var_ratio_ddof0.sum()),
    "n_synthetic_generated": int(N_SYNTHETIC),
    "n_numeric_cols_integer_valued": int(INT_LIKE_MASK.sum()),
    "n_numeric_cols_final": int(len(NUMERIC_COLS_FINAL)),
    "jitter_frac": JITTER_FRAC,
    "synth_vs_test_auc": synth_vs_test_auc,
    "synth_vs_train_auc": synth_vs_train_auc,
    "baseline_shift_auc": baseline_shift_auc,
    "final_shift_auc": final_shift_auc,
    "gap_closed": bool(gap_closed),
    "pseudo_pos_threshold_used": pos_thresh_cur,
    "pseudo_neg_threshold_used": neg_thresh_cur,
    "synth_augmentation_skipped": bool(SYNTH_AUGMENTATION_SKIPPED),
    "n_synthetic_accepted": int(len(accepted_idx)),
    "n_synthetic_accepted_positive": int((accepted_synth_y == 1).sum()) if len(accepted_idx) else 0,
    "n_synthetic_accepted_negative": int((accepted_synth_y == 0).sum()) if len(accepted_idx) else 0,
    "stage1_oof_auc": float(roc_auc_score(y_train, oof_blend_stage1)),
    "stage1_threshold": stage1_threshold,
    "stage1_f1": stage1_f1,
    "stage2_oof_auc": None if oof_blend_stage2 is None else float(roc_auc_score(y_train, oof_blend_stage2)),
    "stage2_threshold": stage2_threshold,
    "stage2_f1": stage2_f1,
    "final_threshold": final_threshold,
    "final_oof_f1": final_oof_f1,
    "submission_positive_rate": float(pos_rate),
    "submission_n_positive": int(final_preds_int.sum()),
}
with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
    json.dump(run_summary, f, indent=2)
print(json.dumps(run_summary, indent=2))
