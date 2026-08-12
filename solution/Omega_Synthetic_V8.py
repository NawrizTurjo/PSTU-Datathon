# %% [markdown]
# # OMEGA + SYNTHETIC — V8 — PSTU Data Thon 2026 Vol-1
#
# `results/omega-7/winning-blueprint-omega-v7.ipynb`'s full feature/model architecture
# (zero-var/dup purification, triple cat encoding, Santander-style FE, QT + manual-SVD PCA at
# ddof=0, XGBoost+LightGBM with `scale_pos_weight`, exhaustive threshold search, hard-binary
# multi-threshold submission) **unchanged**, with one addition layered on top: the synthetic
# test-distribution generator from `solution/synthetic-test-distr.py` / `next-gen.md`.
#
# | Stage | Source | What it does |
# |---|---|---|
# | Purification, cat encoding, Santander FE, QT+PCA(ddof=0) | Omega (unchanged) | see `results/omega-7/README`-equivalent markdown below |
# | XGB+LGB, `scale_pos_weight`, 3 seeds x 5 folds, LogLoss | Omega (unchanged) | no SMOTE, no custom loss, no CatBoost |
# | **Synthetic generator** | `synthetic-test-distr.py` | bootstrap-resample real test rows + selective jitter (integer-valued columns untouched) |
# | **Pseudo-label + fold-safe augmented retrain** | `synthetic-test-distr.py` | negative-domination guard (relax thresholds, cap ratio, skip if no positives) |
# | **Gap-closing diagnostic** | `synthetic-test-distr.py` | does augmented train-vs-test adversarial AUC move toward 0.50? |
# | Exhaustive threshold search + hard-binary multi-threshold submission | Omega (unchanged) | applied to whichever stage (baseline or augmented) wins on OOF F1 |
#
# **Why the generator is safe to bolt on**: it only touches test's unlabeled *feature* values
# (no label join, no external data — same compliance basis as `synthetic-test-distr.py`) and it
# went through two failed designs before arriving at bootstrap+selective-jitter — see
# `next-gen.md` for the full account. Short version: **212-281 of ~340 numeric columns are ≥99%
# integer-valued** (measured directly from the CSVs); any continuous jitter on those columns is
# trivially detected by an adversarial classifier across hundreds of columns at once. Fixed by
# resampling real test rows wholesale (exact joint structure, exact zero/discreteness pattern,
# exact categorical coupling — for free, no clustering needed) and jittering **only nonzero cells
# of the minority of genuinely continuous columns**. Measured result: synthetic-vs-test AUC
# 0.98 -> 0.52 (want ~0.50).
#
# **Also carried over — the pseudo-labeling failure diagnosed in `results/unbeatable-6`**: a
# plain symmetric probability threshold at this ~4% base rate almost always finds far more
# confident negatives than positives (`unbeatable-6`'s run had `pseudo_positive: 0`), silently
# turning "pseudo-labeling" into "inject thousands of extra negatives." Guarded against here the
# same way as `synthetic-test-distr.py`: thresholds relax toward 0.5 (within a floor) until a
# minimum count of each class is found, accepted negatives are capped at a fixed ratio to
# accepted positives, and stage 2 (augmented retrain) is skipped outright if zero positives
# survive even after relaxing.
#
# **Compute cost note**: stage 2 (if it runs) repeats Omega's full 3-seeds x 5-folds x 2-models
# training loop on the augmented data — roughly **doubles runtime** vs plain Omega. Set
# `CFG['enable_stage2'] = False` to skip straight to a stage-1-only run if that's a concern; the
# generator-quality check and gap-closing diagnostic still run either way, since those are useful
# signal on their own about whether the augmentation approach is worth pursuing further.

# %%
# ===================================================================
# CELL 1: Imports & Environment Setup
# ===================================================================
import numpy as np
import pandas as pd
import warnings, os, gc, sys, time
from pathlib import Path
warnings.filterwarnings('ignore')

# Core ML
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import skew as skew_fn, kurtosis as kt_fn

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

print("=" * 75)
print("  OMEGA + SYNTHETIC — V8")
print("  Omega FE/model architecture (unchanged) + synthetic test-distribution augmentation")
print("=" * 75)
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy:  {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

T_START = time.time()


class DebugTracker:
    def __init__(self):
        self.log = []
        self.step_count = 0
        self.t_last = time.time()

    def log_step(self, name, shape=None, extra=None):
        self.step_count += 1
        t_now = time.time()
        elapsed = t_now - self.t_last
        self.t_last = t_now
        shape_str = f"  shape={shape}" if shape else ""
        time_str = f"  +{elapsed:.1f}s"
        extra_str = f"  {extra}" if extra else ""
        print(f"  [D{self.step_count:02d}] {name}{shape_str}{time_str}{extra_str}")
        self.log.append({'step': self.step_count, 'name': name, 'shape': shape,
                          'elapsed': elapsed, 'extra': extra})

    def summary(self):
        print("\n" + "=" * 75)
        print("  DEBUG TRACKER SUMMARY")
        print("=" * 75)
        for e in self.log:
            print(f"  [{e['step']:02d}] {e['name']}: shape={e.get('shape', 'N/A')}")


dbg = DebugTracker()

# %% [markdown]
# ## CELL 2: Configuration
#
# Omega's `CFG` unchanged, plus a new `synthetic` block for the generator/pseudo-labeling.

# %%
CFG = {
    # === Paths (Kaggle) ===
    'train_path': '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/train.csv',
    'test_path':  '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/test.csv',
    'sub_path':   '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/sample_submission.csv',

    # === Reproducibility ===
    'seed': 42,
    'ensemble_seeds': [42, 123, 456],

    # === CV ===
    'n_folds': 5,

    # === STEP 1: Data Purification (NO covariate clipping) ===
    'drop_zero_variance': True,
    'drop_exact_duplicates': True,

    # === STEP 1: PCA (ddof=0) ===
    'pca_variance_threshold': 0.95,

    # === STEP 1: Santander-Style Feature Engineering ===
    'santander_sentinels': [-999999, 9999999999],
    'use_zero_counts': True,
    'use_sentinel_counts': True,
    'use_row_moments': True,
    'use_uniqueness_features': True,
    'uniqueness_top_n': 30,

    # === STEP 1: Target Encoding ===
    'te_smoothing': 50,
    'te_n_folds': 5,

    # === STEP 2: Model Parameters (NO custom objectives) ===
    'xgb_params': {
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.020,
        'subsample': 0.80,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.70,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'min_child_weight': 5,
        'scale_pos_weight': 4.0,
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 200,
        'verbosity': 0,
        'random_state': 42,
    },
    'lgb_params': {
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.020,
        'subsample': 0.80,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'num_leaves': 63,
        'min_child_samples': 30,
        'scale_pos_weight': 3.0,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbosity': -1,
    },

    # === STEP 3: Threshold Search & Multi-Submission ===
    'threshold_min': 0.01,
    'threshold_max': 0.99,
    'threshold_step': 0.0025,
    'probe_thresholds': [0.20, 0.25, 0.30, 0.35, 0.40],

    # === NEW: Synthetic test-distribution generator ===
    'synthetic_multiplier': 1.0,      # n_synthetic = len(test) * this
    'jitter_frac': 0.03,              # multiplicative jitter, nonzero+non-integer-valued cells only
    'integer_like_threshold': 0.99,   # column is "integer-valued" if this fraction of values round-trip

    # === NEW: Pseudo-labeling guard (see next-gen.md — unbeatable-6 postmortem) ===
    'pseudo_pos_thresh': 0.90,
    'pseudo_neg_thresh': 0.05,
    'pseudo_relax_step': 0.05,
    'pseudo_relax_floor_margin': 0.05,
    'pseudo_min_accept': 20,
    'pseudo_max_neg_pos_ratio': 10.0,

    'enable_stage2': True,            # False -> skip augmented retrain entirely (Omega baseline only)
}

KNOWN_STRING_CATS = ['feat_142', 'feat_157', 'feat_318', 'feat_320', 'feat_325', 'feat_337']

SMOKE_TEST = False   # True -> tiny subsample + tiny models, for a fast correctness pass

print("\n" + "=" * 75)
print("  CONFIGURATION")
print("=" * 75)
print(f"  Ensemble seeds:       {CFG['ensemble_seeds']}")
print(f"  CV folds:             {CFG['n_folds']}")
print(f"  PCA variance:         {CFG['pca_variance_threshold']} (ddof=0)")
print(f"  Synthetic multiplier: {CFG['synthetic_multiplier']}")
print(f"  Jitter frac:          {CFG['jitter_frac']}")
print(f"  Stage 2 enabled:      {CFG['enable_stage2']}")

# %% [markdown]
# ## CELL 3: Data Loading & Purification (Omega STEP 1a, unchanged)

# %%
DATA_DIR = None
for d in [os.path.dirname(CFG['train_path']),
          '/kaggle/input/pstu-data-thon-2026-vol-1',
          'pstu-data-thon-2026-vol-1',
          '../input/competitions/pstu-data-thon-2026-vol-1',
          '../input/pstu-data-thon-2026-vol-1',
          '../pstu-data-thon-2026-vol-1']:
    if os.path.exists(os.path.join(d, 'train.csv')):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError("train.csv not found in any candidate directory")
CFG['train_path'] = os.path.join(DATA_DIR, 'train.csv')
CFG['test_path'] = os.path.join(DATA_DIR, 'test.csv')
CFG['sub_path'] = os.path.join(DATA_DIR, 'sample_submission.csv')
print("DATA_DIR =", DATA_DIR)

print("\n" + "=" * 75)
print("  CELL 3: DATA LOADING & PURIFICATION")
print("=" * 75)

train_raw = pd.read_csv(CFG['train_path'])
test_raw = pd.read_csv(CFG['test_path'])
sub_raw = pd.read_csv(CFG['sub_path'])

if SMOKE_TEST:
    train_raw = train_raw.sample(n=8000, random_state=CFG['seed']).reset_index(drop=True)
    test_raw = test_raw.head(4000).reset_index(drop=True)
    sub_raw = sub_raw.head(4000).reset_index(drop=True)
    for p in (CFG['xgb_params'], CFG['lgb_params']):
        p['n_estimators'] = 80
        p['early_stopping_rounds'] = 20
    CFG['ensemble_seeds'] = [42]
    print("SMOKE: reduced to", train_raw.shape, test_raw.shape)

print(f"\n  Train shape: {train_raw.shape}")
print(f"  Test shape:  {test_raw.shape}")

y = train_raw['TARGET'].values.astype(int)
n_pos = np.sum(y == 1)
n_neg = np.sum(y == 0)
print(f"\n  Target distribution:")
print(f"    Class 0 (Stable):  {n_neg:,} ({100*n_neg/len(y):.2f}%)")
print(f"    Class 1 (At-Risk):  {n_pos:,} ({100*n_pos/len(y):.2f}%)")
print(f"    Imbalance ratio:    {n_neg/n_pos:.2f}:1")

feature_cols = [c for c in train_raw.columns if c.startswith('feat_')]
print(f"\n  Total feat_* columns: {len(feature_cols)}")

string_cat_features = [c for c in feature_cols if train_raw[c].dtype == 'object']
numerical_features = [c for c in feature_cols if train_raw[c].dtype != 'object']

print(f"\n  STRING categoricals: {len(string_cat_features)}")
for c in string_cat_features:
    print(f"    {c}: {train_raw[c].nunique():,} unique")
print(f"  Numerical features: {len(numerical_features)}")
assert set(string_cat_features) == set(KNOWN_STRING_CATS), \
    f"Cat mismatch! Expected {KNOWN_STRING_CATS}, got {string_cat_features}"
print(f"  All 6 known string cats confirmed.")

print(f"\n  --- Detecting Zero-Variance Features ---")
train_num = train_raw[numerical_features].fillna(0)
test_num = test_raw[numerical_features].fillna(0)

zero_var_features = []
for c in numerical_features:
    if train_num[c].nunique() <= 1 and test_num[c].nunique() <= 1:
        zero_var_features.append(c)
print(f"  Zero-variance features: {len(zero_var_features)}")

print(f"\n  --- Detecting Exact Duplicate Features ---")
train_num_values = train_num.values.astype(np.float64)
test_num_values = test_num.values.astype(np.float64)

duplicate_pairs = []
n_num = len(numerical_features)
already_duplicate = set()

for i in range(n_num):
    if numerical_features[i] in already_duplicate:
        continue
    for j in range(i + 1, n_num):
        if numerical_features[j] in already_duplicate:
            continue
        if (abs(train_num_values[:, i].mean() - train_num_values[:, j].mean()) < 1e-6 and
                abs(train_num_values[:, i].std() - train_num_values[:, j].std()) < 1e-6):
            if np.array_equal(train_num_values[:, i], train_num_values[:, j]):
                if np.array_equal(test_num_values[:, i], test_num_values[:, j]):
                    duplicate_pairs.append((numerical_features[i], numerical_features[j]))
                    already_duplicate.add(numerical_features[j])

dup_features_to_drop = [pair[1] for pair in duplicate_pairs]
all_drop_features = list(set(zero_var_features + dup_features_to_drop))
all_drop_features = [f for f in all_drop_features if f not in string_cat_features]

print(f"  Zero-variance to drop:    {len(zero_var_features)}")
print(f"  Duplicate features (2nd): {len(dup_features_to_drop)}")
print(f"  Total features to DROP:   {len(all_drop_features)}")

keep_numerical = [f for f in numerical_features if f not in all_drop_features]
print(f"\n  Features AFTER purification:")
print(f"    String cats kept:  {len(string_cat_features)}")
print(f"    Numerical kept:    {len(keep_numerical)} / {len(numerical_features)}")
print(f"  NO covariate shift clipping -- keeping all test values as-is")

X_train_num = train_raw[keep_numerical].fillna(0).astype(np.float32)
X_test_num = test_raw[keep_numerical].fillna(0).astype(np.float32)

n_train = X_train_num.shape[0]
n_test = X_test_num.shape[0]
n_feats = X_train_num.shape[1]

dbg.log_step("Data purification done",
             extra=f"Dropped {len(all_drop_features)} redundant "
                   f"({len(zero_var_features)} zero-var + {len(dup_features_to_drop)} dup)")

N_SYNTHETIC = int(round(len(test_raw) * CFG['synthetic_multiplier']))
print(f"\n  N_SYNTHETIC = {N_SYNTHETIC}")

# %% [markdown]
# ## CELL 4: Categorical Engineering — LE + Frequency + Target Encoding (Omega STEP 1b, unchanged)
#
# Fit artifacts (`label_encoders`, `freq_maps`, `te_maps`) are now kept around (Omega's original
# discarded them after building the train/test columns) so the synthetic rows generated in CELL 5
# can be encoded through the exact same fitted mappings as test — no unseen-level risk, since
# synthetic categorical values are literal copies of real test values.

# %%
print("\n" + "=" * 75)
print("  CELL 4: CATEGORICAL ENGINEERING -- LE + FREQ + TARGET ENCODING")
print("=" * 75)

train_cats = {}
test_cats = {}
for c in string_cat_features:
    train_cats[c] = train_raw[c].fillna('MISSING').astype(str).values
    test_cats[c] = test_raw[c].fillna('MISSING').astype(str).values

train_cats_le = pd.DataFrame(index=train_raw.index)
test_cats_le = pd.DataFrame(index=test_raw.index)
label_encoders = {}
for c in string_cat_features:
    le = LabelEncoder()
    all_vals = np.concatenate([train_cats[c], test_cats[c]])
    le.fit(all_vals)
    label_encoders[c] = le
    train_cats_le[f'{c}_le'] = le.transform(train_cats[c]).astype(np.int32)
    test_cats_le[f'{c}_le'] = le.transform(test_cats[c]).astype(np.int32)
print(f"  Label Encoding: {len(string_cat_features)} features")

train_cats_freq = pd.DataFrame(index=train_raw.index)
test_cats_freq = pd.DataFrame(index=test_raw.index)
freq_maps = {}
for c in string_cat_features:
    all_vals = np.concatenate([train_cats[c], test_cats[c]])
    val_counts = pd.Series(all_vals).value_counts()
    freq_map = np.log1p(val_counts).to_dict()
    freq_maps[c] = freq_map
    train_cats_freq[f'{c}_freq'] = pd.Series(train_cats[c]).map(freq_map).fillna(0).astype(np.float32).values
    test_cats_freq[f'{c}_freq'] = pd.Series(test_cats[c]).map(freq_map).fillna(0).astype(np.float32).values
print(f"  Frequency Encoding: {len(string_cat_features)} features (log1p scale)")

train_cats_te = pd.DataFrame(index=train_raw.index)
test_cats_te = pd.DataFrame(index=test_raw.index)
global_mean = y.mean()
smoothing = CFG['te_smoothing']
te_maps = {}

skf_te = StratifiedKFold(n_splits=CFG['te_n_folds'], shuffle=True, random_state=CFG['seed'])
for c in string_cat_features:
    train_vals = train_cats[c]
    train_te_col = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in skf_te.split(train_raw, y):
        tr_series = pd.Series(train_vals[tr_idx])
        tr_y = y[tr_idx]
        cat_sum = pd.Series(tr_y).groupby(tr_series).sum()
        cat_count = tr_series.groupby(tr_series).count()
        smoothed_means = (cat_sum + smoothing * global_mean) / (cat_count + smoothing)
        va_series = pd.Series(train_vals[va_idx])
        train_te_col[va_idx] = va_series.map(smoothed_means).fillna(global_mean).values
    train_cats_te[f'{c}_te'] = train_te_col.astype(np.float32)

    tr_series_full = pd.Series(train_vals)
    cat_sum_full = pd.Series(y).groupby(tr_series_full).sum()
    cat_count_full = tr_series_full.groupby(tr_series_full).count()
    smoothed_full = (cat_sum_full + smoothing * global_mean) / (cat_count_full + smoothing)
    te_maps[c] = smoothed_full

    test_series = pd.Series(test_cats[c])
    test_cats_te[f'{c}_te'] = test_series.map(smoothed_full).fillna(global_mean).astype(np.float32).values
print(f"  Target Encoding: {len(string_cat_features)} features (5-fold CV, smoothing={smoothing})")

dbg.log_step("Cat encoding done",
             extra=f"LE+{train_cats_le.shape[1]} Freq+{train_cats_freq.shape[1]} TE+{train_cats_te.shape[1]}")

# %% [markdown]
# ## CELL 5 — NEW: Synthetic test-distribution generator
#
# See the design-principles markdown at the top for why this is bootstrap-resample + selective
# jitter rather than a parametric copula. Operates on `keep_numerical` / `string_cat_features` —
# the exact same purified column set every downstream Omega step uses — so synthetic rows slot
# into the rest of the pipeline (Santander FE, QT+PCA, cat encoding) exactly like a real row.

# %%
print("\n" + "=" * 75)
print("  CELL 5: SYNTHETIC TEST-DISTRIBUTION GENERATOR")
print("=" * 75)

frac_int = test_raw[keep_numerical].apply(
    lambda s: np.isclose(s.dropna(), np.round(s.dropna())).mean() if s.notna().any() else 1.0
)
INT_LIKE_MASK = (frac_int >= CFG['integer_like_threshold']).values
print(f"  Integer-valued numeric columns (excluded from jitter): "
      f"{INT_LIKE_MASK.sum()} / {len(INT_LIKE_MASK)}")

rng = np.random.default_rng(CFG['seed'])
sample_idx = rng.integers(0, len(test_raw), size=N_SYNTHETIC)

synth_num_prejitter = test_raw[keep_numerical].iloc[sample_idx].reset_index(drop=True).fillna(0)
synth_cat_raw = {c: test_raw[c].fillna('MISSING').astype(str).values[sample_idx] for c in string_cat_features}

vals = synth_num_prejitter.values.astype(np.float64).copy()
jitter_eligible = (vals != 0) & (~INT_LIKE_MASK)[np.newaxis, :]
mult = np.ones(vals.shape)
mult[jitter_eligible] = 1.0 + rng.normal(loc=0.0, scale=CFG['jitter_frac'], size=int(jitter_eligible.sum()))
vals = vals * mult
synth_num_raw = pd.DataFrame(vals, columns=keep_numerical)

X_synth_num = synth_num_raw.astype(np.float32)
print(f"  Synthetic numeric rows generated (bootstrap + jitter): {X_synth_num.shape}")

# --- encode synthetic categoricals through the SAME fitted maps as test (CELL 4) ---
synth_cats_le = pd.DataFrame(index=range(N_SYNTHETIC))
synth_cats_freq = pd.DataFrame(index=range(N_SYNTHETIC))
synth_cats_te = pd.DataFrame(index=range(N_SYNTHETIC))
for c in string_cat_features:
    vals_c = synth_cat_raw[c]
    synth_cats_le[f'{c}_le'] = label_encoders[c].transform(vals_c).astype(np.int32)
    synth_cats_freq[f'{c}_freq'] = pd.Series(vals_c).map(freq_maps[c]).fillna(0).astype(np.float32).values
    synth_cats_te[f'{c}_te'] = pd.Series(vals_c).map(te_maps[c]).fillna(global_mean).astype(np.float32).values
print(f"  Synthetic categoricals encoded via fitted LE/Freq/TE maps (no unseen-level risk -- "
      f"values are literal copies of real test rows)")

dbg.log_step("Synthetic generator done", shape=X_synth_num.shape)

# %% [markdown]
# ## CELL 6: Santander-Style Feature Engineering (Omega STEP 1c, unchanged formulas)
#
# Refactored into a function so the exact same feature formulas apply to train, test, **and**
# synthetic rows. `top_indices` / `val_to_count` / `singleton_vals` are fit once from train+test
# combined (matching Omega's original transductive design) and then applied to synthetic rows
# via the same `.get(v, 1)` fallback Omega already used for any value not seen during fitting.
#
# **Second finding from the local dry-run**: computing synthetic rows' Santander features from
# the *jittered* numeric values (the natural-looking choice) made the Santander block alone
# score adversarial AUC **1.0000** against real test, even after CELL 5's integer-protection fix.
# Cause: this dataset has extreme-magnitude continuous columns (EDA, `CLAUDE.md`: `feat_169` min
# ~-1.11e8). A 3% multiplicative jitter on a column at that scale produces an absolute delta of
# millions — swamping `row_sum`/`row_mean`/`row_std`/etc, which aggregate across *all* numeric
# columns, and making jittered rows trivially identifiable regardless of how careful the
# per-column jitter otherwise is. Fix: Santander features for synthetic rows are computed from
# `synth_num_prejitter` (the exact resampled real values, before jitter) instead of the jittered
# `X_synth_num`. This is correct, not a workaround — jitter's only job is to keep individual
# per-feature values from being byte-identical to a real row; a resampled row's row-level
# aggregate profile is already, by construction, a real profile and needs no perturbing. Measured
# fix: Santander-block-only synth-vs-test AUC dropped from 1.0000 to **0.47**.

# %%
print("\n" + "=" * 75)
print("  CELL 6: SANTANDER-STYLE FEATURE ENGINEERING")
print("=" * 75)

X_all = np.vstack([X_train_num.values, X_test_num.values]).astype(np.float64)
print(f"  Combined (train+test) data for fitting FE stats: {X_all.shape}")

sentinels = CFG['santander_sentinels']

if CFG['use_uniqueness_features']:
    top_n = min(CFG['uniqueness_top_n'], n_feats)
    variances = np.var(X_all, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    val_to_count_per_feat = {}
    singleton_per_feat = {}
    for feat_idx in top_indices:
        col_all = X_all[:, feat_idx]
        unique_vals, counts = np.unique(col_all, return_counts=True)
        val_to_count_per_feat[feat_idx] = dict(zip(unique_vals, counts))
        singleton_per_feat[feat_idx] = set(unique_vals[counts == 1])
    print(f"  Uniqueness features fit on top-{top_n} highest-variance columns")
else:
    top_indices = np.array([], dtype=int)
    val_to_count_per_feat, singleton_per_feat = {}, {}


def ensure_2d(arr):
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def santander_features_for(X_num_vals):
    """X_num_vals: (n, n_feats) float array in keep_numerical column order. Returns (n, k) float32."""
    n = X_num_vals.shape[0]
    parts = []

    if CFG['use_zero_counts']:
        parts.append(ensure_2d(np.sum(X_num_vals == 0, axis=1).astype(np.float32)))
    else:
        parts.append(ensure_2d(np.zeros(n, dtype=np.float32)))

    if CFG['use_sentinel_counts']:
        sentinel_cols = [np.sum(X_num_vals == s, axis=1).astype(np.float32) for s in sentinels]
        if sentinel_cols:
            parts.append(np.column_stack(sentinel_cols).astype(np.float32))

    if CFG['use_row_moments']:
        sentinel_mask = np.zeros(X_num_vals.shape, dtype=bool)
        for s in sentinels:
            sentinel_mask |= (X_num_vals == s)
        valid = (X_num_vals != 0) & (~sentinel_mask)
        n_valid = valid.sum(axis=1)
        masked = np.where(valid, X_num_vals, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_mean = np.nan_to_num(np.nanmean(masked, axis=1), nan=0.0).astype(np.float32)
            row_std = np.nan_to_num(np.nanstd(masked, axis=1, ddof=0), nan=0.0).astype(np.float32)
            row_sum = np.nan_to_num(np.nansum(masked, axis=1), nan=0.0).astype(np.float32)
            row_skew = np.nan_to_num(skew_fn(masked, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)
            row_kurt = np.nan_to_num(kt_fn(masked, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)
        for arr in (row_mean, row_std, row_sum, row_skew, row_kurt, n_valid.astype(np.float32)):
            parts.append(ensure_2d(arr))

    if CFG['use_uniqueness_features']:
        for feat_idx in top_indices:
            val_to_count = val_to_count_per_feat[feat_idx]
            singleton_vals = singleton_per_feat[feat_idx]
            col = X_num_vals[:, feat_idx]
            freq = np.array([np.log1p(val_to_count.get(v, 1)) for v in col], dtype=np.float32)
            sing = np.array([1.0 if v in singleton_vals else 0.0 for v in col], dtype=np.float32)
            parts.append(ensure_2d(freq))
            parts.append(ensure_2d(sing))

    return np.column_stack(parts).astype(np.float32)


X_santander_train = santander_features_for(X_train_num.values.astype(np.float64))
X_santander_test = santander_features_for(X_test_num.values.astype(np.float64))
X_santander_synth = santander_features_for(synth_num_prejitter.values.astype(np.float64))  # pre-jitter, see markdown above

print(f"\n  Santander features: {X_santander_train.shape[1]} total")
print(f"    train: {X_santander_train.shape}  test: {X_santander_test.shape}  "
      f"synth: {X_santander_synth.shape}")

del X_all
gc.collect()

dbg.log_step("Santander FE done", shape=X_santander_train.shape,
             extra=f"{X_santander_train.shape[1]} engineered features (train/test/synth)")

# %% [markdown]
# ## CELL 7: QT -> PCA(ddof=0) -> Feature Assembly (Omega STEP 1d, unchanged formulas)
#
# `QuantileTransformer` and the manual-SVD PCA are still fit on train+test combined only
# (Omega's original transductive design — unchanged). Synthetic rows are **projected** through
# the already-fitted QT and the retained right-singular-vectors `Vt[:n_pca]` (standard PCA
# projection: `X_new_centered @ Vt.T`), never used to fit either transform — keeping the fitted
# basis identical to plain Omega.

# %%
print("\n" + "=" * 75)
print("  CELL 7: QT -> PCA(ddof=0) -> FEATURE ASSEMBLY")
print("=" * 75)

X_tr_vals = X_train_num.values.astype(np.float64)
X_te_vals = X_test_num.values.astype(np.float64)
X_sy_vals = X_synth_num.values.astype(np.float64)

qt = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, n_train),
    random_state=CFG['seed'],
    subsample=200_000,
)
X_train_qt = qt.fit_transform(X_tr_vals).astype(np.float32)
X_test_qt = qt.transform(X_te_vals).astype(np.float32)
X_synth_qt = qt.transform(X_sy_vals).astype(np.float32)
print(f"  QT train: {X_train_qt.shape}  QT test: {X_test_qt.shape}  QT synth: {X_synth_qt.shape}")

dbg.log_step("QT done", shape=X_train_qt.shape)

print(f"\n  PCA with ddof=0 ({CFG['pca_variance_threshold']:.0%} variance) via manual SVD")
X_qt_all = np.vstack([X_train_qt, X_test_qt])
n_total = X_qt_all.shape[0]
qt_mean = X_qt_all.mean(axis=0)
X_centered = X_qt_all - qt_mean

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
explained_variance_ddof0 = (S ** 2) / n_total
explained_variance_ratio = explained_variance_ddof0 / explained_variance_ddof0.sum()
cumsum_var = np.cumsum(explained_variance_ratio)

n_pca = int(np.searchsorted(cumsum_var, CFG['pca_variance_threshold']) + 1)
n_pca = min(n_pca, X_qt_all.shape[1])
print(f"  PCA components: {n_pca}/{X_qt_all.shape[1]} (retains {cumsum_var[n_pca-1]*100:.1f}% variance, ddof=0)")

X_train_pca = (U[:n_train, :n_pca] * S[:n_pca]).astype(np.float32)
X_test_pca = (U[n_train:, :n_pca] * S[:n_pca]).astype(np.float32)

PCA_VT = Vt[:n_pca].copy()          # retained for projecting synthetic (new, unfitted) rows
X_synth_centered = (X_synth_qt.astype(np.float64) - qt_mean)
X_synth_pca = (X_synth_centered @ PCA_VT.T).astype(np.float32)
print(f"  PCA train: {X_train_pca.shape}  PCA test: {X_test_pca.shape}  PCA synth: {X_synth_pca.shape}")

del X_qt_all, X_centered, U, S, Vt, X_train_qt, X_test_qt, X_synth_qt, X_synth_centered
gc.collect()

dbg.log_step("PCA done", shape=X_train_pca.shape, extra=f"{n_pca} components, ddof=0")

# --- Feature Assembly: PCA + Santander + LE cats + Frequency cats + Target Encoding cats ---
X_tr_base = np.hstack([
    X_train_pca, X_santander_train,
    train_cats_le.values.astype(np.float32),
    train_cats_freq.values.astype(np.float32),
    train_cats_te.values.astype(np.float32),
]).astype(np.float32)

X_te_base = np.hstack([
    X_test_pca, X_santander_test,
    test_cats_le.values.astype(np.float32),
    test_cats_freq.values.astype(np.float32),
    test_cats_te.values.astype(np.float32),
]).astype(np.float32)

X_sy_base = np.hstack([
    X_synth_pca, X_santander_synth,
    synth_cats_le.values.astype(np.float32),
    synth_cats_freq.values.astype(np.float32),
    synth_cats_te.values.astype(np.float32),
]).astype(np.float32)

n_total_features = X_tr_base.shape[1]
print(f"\n  Final feature matrix: train={X_tr_base.shape}  test={X_te_base.shape}  synth={X_sy_base.shape}")
assert X_tr_base.shape[1] == X_te_base.shape[1] == X_sy_base.shape[1]

dbg.log_step("Feature assembly done", shape=X_tr_base.shape, extra=f"{n_total_features} features")

# %% [markdown]
# ## CELL 8 — NEW: Generator quality check
#
# **Finding from the local dry-run, worth recording**: an early version of this notebook ran
# `quick_adv_auc` on the full `X_*_base` (everything, including Target Encoding) and got
# **AUC 1.0000** for real-train-vs-real-test — nonsensical, since the measured baseline
# (`CLAUDE.md`) is ~0.57. Block-by-block isolation traced it to the **Target Encoding columns
# alone** (AUC 1.0000 in isolation; every other block — PCA, Santander, LE, Freq — sat at a sane
# ~0.51). Root cause: Omega's TE (CELL 4, unchanged) computes **train**'s encoding via 5-fold
# leakage-safe OOF (each row gets one of 5 slightly different fold-conditional smoothed means),
# but **test**'s encoding via a single full-fit smoothed mean — standard, correct, leakage-safe
# target-encoding practice, *not a bug*. But it means a category's train-side encoded values
# cluster around ~5 distinct nearby floats while its test-side value is one single precise float,
# which a boosted tree trivially learns to tell apart across 6 TE columns x thousands of repeated
# category values. Omega itself never surfaces this (it never runs any adversarial check), so it
# was invisible until this notebook's own diagnostics needed a meaningful number.
#
# **Fix, scoped narrowly**: build a second feature matrix — `X_*_diag`, everything `X_*_base` has
# *except* the 6 TE columns — used **only** by `quick_adv_auc` below and the gap-closing
# diagnostic in CELL 13. The actual models (CELL 9, 12) still train on the full `X_*_base`
# including TE, completely unchanged from Omega. This keeps Omega's architecture untouched while
# making the new diagnostics report something real instead of a TE-encoding artifact.

# %%
X_tr_diag = np.hstack([
    X_train_pca, X_santander_train,
    train_cats_le.values.astype(np.float32),
    train_cats_freq.values.astype(np.float32),
]).astype(np.float32)
X_te_diag = np.hstack([
    X_test_pca, X_santander_test,
    test_cats_le.values.astype(np.float32),
    test_cats_freq.values.astype(np.float32),
]).astype(np.float32)
X_sy_diag = np.hstack([
    X_synth_pca, X_santander_synth,
    synth_cats_le.values.astype(np.float32),
    synth_cats_freq.values.astype(np.float32),
]).astype(np.float32)


def quick_adv_auc(X_a, X_b, seed=CFG['seed'], folds=3):
    """OOF adversarial AUC distinguishing rows of X_a (label 0) from X_b (label 1).
    Always called with the TE-excluded *_diag matrices -- see markdown above for why."""
    Xc = np.vstack([X_a, X_b])
    yc = np.array([0] * len(X_a) + [1] * len(X_b))
    aucs = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(Xc, yc):
        m = LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=seed,
                            n_jobs=-1, verbosity=-1)
        m.fit(Xc[tr_idx], yc[tr_idx])
        p = m.predict_proba(Xc[va_idx])[:, 1]
        aucs.append(roc_auc_score(yc[va_idx], p))
    return float(np.mean(aucs))


synth_vs_test_auc = quick_adv_auc(X_sy_diag, X_te_diag)
synth_vs_train_auc = quick_adv_auc(X_sy_diag, X_tr_diag)
baseline_shift_auc = quick_adv_auc(X_tr_diag, X_te_diag)

print("\n" + "=" * 75)
print("  CELL 8: GENERATOR QUALITY CHECK  (TE columns excluded -- see markdown above)")
print("=" * 75)
print(f"  synthetic vs real test AUC:  {synth_vs_test_auc:.4f}  (want ~0.50 -> synthetic matches test)")
print(f"  synthetic vs real train AUC: {synth_vs_train_auc:.4f}  (want clearly >0.50 -> not train-like)")
print(f"  [reference] real train vs real test AUC: {baseline_shift_auc:.4f}")

dbg.log_step("Generator quality check done",
             extra=f"synth_vs_test={synth_vs_test_auc:.4f} synth_vs_train={synth_vs_train_auc:.4f}")

# %% [markdown]
# ## CELL 9: Stage 1 — Robust Training (Omega STEP 2, unchanged model config)
#
# XGBoost + LightGBM, `scale_pos_weight`, plain LogLoss, 3 seeds x 5 folds. Extended only to
# additionally score the synthetic rows every fold (needed to pseudo-label them below) — the
# model architecture, hyperparameters, and training logic are Omega's, byte-for-byte.

# %%
print("\n" + "=" * 75)
print("  CELL 9: STAGE 1 TRAINING -- XGBOOST + LIGHTGBM")
print("  NO custom loss | NO SMOTE | NO shuffling | scale_pos_weight only")
print("=" * 75)

n_seeds = len(CFG['ensemble_seeds'])
n_base = len(y)

oof_xgb = np.zeros((n_base, n_seeds), dtype=np.float32)
oof_lgb = np.zeros((n_base, n_seeds), dtype=np.float32)
test_xgb = np.zeros((n_test, n_seeds), dtype=np.float32)
test_lgb = np.zeros((n_test, n_seeds), dtype=np.float32)
synth_xgb = np.zeros((N_SYNTHETIC, n_seeds), dtype=np.float32)
synth_lgb = np.zeros((N_SYNTHETIC, n_seeds), dtype=np.float32)

print(f"\n  TRAINING: {n_seeds} seeds x {CFG['n_folds']} folds x 2 models = "
      f"{n_seeds * CFG['n_folds'] * 2} fits")

for seed_idx, seed in enumerate(CFG['ensemble_seeds']):
    print(f"\n{'='*60}\n  ENSEMBLE SEED {seed} ({seed_idx+1}/{n_seeds})\n{'='*60}")
    skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_base, y)):
        t_fold_start = time.time()
        print(f"\n  --- Fold {fold+1}/{CFG['n_folds']} ---")

        X_tr_fold, y_tr_fold = X_tr_base[tr_idx], y[tr_idx]
        X_va_fold, y_va_fold = X_tr_base[va_idx], y[va_idx]

        xgb_params = CFG['xgb_params'].copy()
        xgb_params['random_state'] = seed
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)], verbose=False)

        oof_xgb[va_idx, seed_idx] = xgb_model.predict_proba(X_va_fold)[:, 1].astype(np.float32)
        test_xgb[:, seed_idx] += xgb_model.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']
        synth_xgb[:, seed_idx] += xgb_model.predict_proba(X_sy_base)[:, 1].astype(np.float32) / CFG['n_folds']

        f1_fold = f1_score(y_va_fold, (oof_xgb[va_idx, seed_idx] >= 0.5).astype(int))
        print(f"    [XGBoost seed={seed}] F1@0.5={f1_fold:.5f}")
        del xgb_model
        gc.collect()

        lgb_params = CFG['lgb_params'].copy()
        lgb_params['random_state'] = seed
        lgb_model = LGBMClassifier(**lgb_params)
        lgb_model.fit(X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)],
                      callbacks=[early_stopping(200), log_evaluation(0)])

        oof_lgb[va_idx, seed_idx] = lgb_model.predict_proba(X_va_fold)[:, 1].astype(np.float32)
        test_lgb[:, seed_idx] += lgb_model.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']
        synth_lgb[:, seed_idx] += lgb_model.predict_proba(X_sy_base)[:, 1].astype(np.float32) / CFG['n_folds']

        f1_fold_lgb = f1_score(y_va_fold, (oof_lgb[va_idx, seed_idx] >= 0.5).astype(int))
        print(f"    [LightGBM seed={seed}] F1@0.5={f1_fold_lgb:.5f}  [{time.time()-t_fold_start:.0f}s]")
        del lgb_model
        gc.collect()

    s1_xgb = f1_score(y, (np.nan_to_num(oof_xgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    s1_lgb = f1_score(y, (np.nan_to_num(oof_lgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    print(f"\n  Seed {seed} cumulative OOF@0.5: XGB={s1_xgb:.5f}  LGB={s1_lgb:.5f}")

STAGE1_FOLDS = {seed: list(StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True,
                random_state=seed).split(X_tr_base, y)) for seed in CFG['ensemble_seeds']}

elapsed_train = time.time() - T_START
print(f"\n  STAGE 1 TRAINING COMPLETE -- {n_seeds * CFG['n_folds'] * 2} models. "
      f"Elapsed: {elapsed_train/60:.1f} min")
dbg.log_step("Stage 1 training done", extra=f"Elapsed: {elapsed_train/60:.1f} min")

# %% [markdown]
# ## CELL 10: Stage 1 Threshold Optimization (Omega STEP 3a, unchanged)

# %%
def optimize_threshold(oof_arr, y_true, cfg):
    thresholds = np.arange(cfg['threshold_min'], cfg['threshold_max'] + cfg['threshold_step']/2,
                            cfg['threshold_step'])
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        binary = (oof_arr >= t).astype(int)
        if np.sum(binary) == 0:
            continue
        f1 = f1_score(y_true, binary)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


oof_xgb_avg = np.nan_to_num(np.mean(oof_xgb, axis=1), nan=0.0)
oof_lgb_avg = np.nan_to_num(np.mean(oof_lgb, axis=1), nan=0.0)
oof_ensemble_stage1 = np.nan_to_num(np.mean(np.hstack([oof_xgb, oof_lgb]), axis=1), nan=0.0)

stage1_threshold, stage1_f1 = optimize_threshold(oof_ensemble_stage1, y, CFG)
f1_at_05_stage1 = f1_score(y, (oof_ensemble_stage1 >= 0.5).astype(int))

print("\n" + "=" * 75)
print("  CELL 10: STAGE 1 THRESHOLD OPTIMIZATION")
print("=" * 75)
print(f"  Optimal threshold t_opt: {stage1_threshold:.4f}")
print(f"  OOF F1 at t_opt:         {stage1_f1:.5f}")
print(f"  F1@0.5 (baseline):       {f1_at_05_stage1:.5f}")

test_xgb_avg = np.mean(test_xgb, axis=1)
test_lgb_avg = np.mean(test_lgb, axis=1)
test_ensemble_stage1 = np.mean(np.hstack([test_xgb, test_lgb]), axis=1)

synth_xgb_avg = np.mean(synth_xgb, axis=1)
synth_lgb_avg = np.mean(synth_lgb, axis=1)
synth_ensemble_stage1 = np.mean(np.hstack([synth_xgb, synth_lgb]), axis=1)

dbg.log_step("Stage 1 threshold optimization done",
             extra=f"t_opt={stage1_threshold:.4f}, OOF F1={stage1_f1:.5f}")

# %% [markdown]
# ## CELL 11 — NEW: Pseudo-label synthetic rows (negative-domination guard)
#
# Same guard as `synthetic-test-distr.py` / `next-gen.md`'s postmortem on `results/unbeatable-6`:
# thresholds relax toward 0.5 (never past the floor margin) until a minimum count of each class
# is found, accepted negatives are capped at a fixed ratio to accepted positives, and stage 2 is
# skipped outright if zero positives survive relaxation.

# %%
print("\n" + "=" * 75)
print("  CELL 11: PSEUDO-LABEL SYNTHETIC ROWS")
print("=" * 75)

pos_thresh_cur = CFG['pseudo_pos_thresh']
while ((synth_ensemble_stage1 > pos_thresh_cur).sum() < CFG['pseudo_min_accept']
       and pos_thresh_cur - CFG['pseudo_relax_step'] >= 0.5 + CFG['pseudo_relax_floor_margin']):
    pos_thresh_cur = round(pos_thresh_cur - CFG['pseudo_relax_step'], 4)

neg_thresh_cur = CFG['pseudo_neg_thresh']
while ((synth_ensemble_stage1 < neg_thresh_cur).sum() < CFG['pseudo_min_accept']
       and neg_thresh_cur + CFG['pseudo_relax_step'] <= 0.5 - CFG['pseudo_relax_floor_margin']):
    neg_thresh_cur = round(neg_thresh_cur + CFG['pseudo_relax_step'], 4)

pos_idx = np.where(synth_ensemble_stage1 > pos_thresh_cur)[0]
neg_idx = np.where(synth_ensemble_stage1 < neg_thresh_cur)[0]
print(f"  thresholds after relaxation: pos>{pos_thresh_cur} (started {CFG['pseudo_pos_thresh']}), "
      f"neg<{neg_thresh_cur} (started {CFG['pseudo_neg_thresh']})")
print(f"  raw confident counts: positive={len(pos_idx)} | negative={len(neg_idx)}")

if len(neg_idx) > CFG['pseudo_max_neg_pos_ratio'] * max(len(pos_idx), 1):
    cap = int(CFG['pseudo_max_neg_pos_ratio'] * max(len(pos_idx), 1))
    neg_idx = rng.choice(neg_idx, size=cap, replace=False)
    print(f"  capped negative pseudo-labels to {cap} ({CFG['pseudo_max_neg_pos_ratio']}:1 ratio guard)")

SYNTH_AUGMENTATION_SKIPPED = (len(pos_idx) == 0) or (not CFG['enable_stage2'])
if len(pos_idx) == 0:
    print("  WARNING: zero confident-positive synthetic rows even after relaxation -- "
          "stage 2 will be SKIPPED, falling back to stage-1 baseline.")
elif not CFG['enable_stage2']:
    print("  CFG['enable_stage2']=False -- stage 2 SKIPPED by configuration.")

pseudo_label_col = np.full(N_SYNTHETIC, -1, dtype=int)
pseudo_label_col[pos_idx] = 1
pseudo_label_col[neg_idx] = 0
accepted_mask = pseudo_label_col != -1

accepted_idx = np.concatenate([pos_idx, neg_idx]) if not SYNTH_AUGMENTATION_SKIPPED else np.array([], dtype=int)
accepted_synth_X = X_sy_base[accepted_idx]         # full feature space (incl. TE) -- used for training
accepted_synth_X_diag = X_sy_diag[accepted_idx]    # TE-excluded space -- used only by the gap diagnostic
accepted_synth_y = pseudo_label_col[accepted_idx]

print(f"  accepted synthetic pseudo-labels: {len(accepted_idx)} / {N_SYNTHETIC} "
      f"({len(accepted_idx)/N_SYNTHETIC:.4%}) | positive={int((accepted_synth_y==1).sum())} "
      f"| negative={int((accepted_synth_y==0).sum())}")

dbg.log_step("Pseudo-labeling done", extra=f"accepted={len(accepted_idx)}, skipped={SYNTH_AUGMENTATION_SKIPPED}")

# %% [markdown]
# ## CELL 12 — NEW: Stage 2 — fold-safe augmented retrain
#
# Same 3-seeds x 5-folds structure and model configs as stage 1 (Omega's, unchanged). Accepted
# synthetic pseudo-rows are appended to every fold's **training** partition only — validation
# always uses real, held-out train rows — so stage-2 OOF F1 stays an honest read on real labels.
# Reuses the exact same fold splits as stage 1 (`STAGE1_FOLDS`) for direct comparability.

# %%
if SYNTH_AUGMENTATION_SKIPPED:
    stage2_f1 = None
    stage2_threshold = None
    oof_ensemble_stage2 = None
    test_ensemble_stage2 = None
    print("\n  STAGE 2 SKIPPED (see CELL 11).")
else:
    print("\n" + "=" * 75)
    print("  CELL 12: STAGE 2 -- FOLD-SAFE AUGMENTED RETRAIN")
    print("=" * 75)

    oof_xgb2 = np.zeros((n_base, n_seeds), dtype=np.float32)
    oof_lgb2 = np.zeros((n_base, n_seeds), dtype=np.float32)
    test_xgb2 = np.zeros((n_test, n_seeds), dtype=np.float32)
    test_lgb2 = np.zeros((n_test, n_seeds), dtype=np.float32)

    for seed_idx, seed in enumerate(CFG['ensemble_seeds']):
        print(f"\n{'='*60}\n  [STAGE 2] SEED {seed} ({seed_idx+1}/{n_seeds})\n{'='*60}")

        for fold, (tr_idx, va_idx) in enumerate(STAGE1_FOLDS[seed]):
            t_fold_start = time.time()
            print(f"\n  --- Fold {fold+1}/{CFG['n_folds']} ---")

            X_tr_fold = np.vstack([X_tr_base[tr_idx], accepted_synth_X])
            y_tr_fold = np.concatenate([y[tr_idx], accepted_synth_y])
            X_va_fold, y_va_fold = X_tr_base[va_idx], y[va_idx]

            xgb_params = CFG['xgb_params'].copy()
            xgb_params['random_state'] = seed
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)], verbose=False)
            oof_xgb2[va_idx, seed_idx] = xgb_model.predict_proba(X_va_fold)[:, 1].astype(np.float32)
            test_xgb2[:, seed_idx] += xgb_model.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']
            f1_fold = f1_score(y_va_fold, (oof_xgb2[va_idx, seed_idx] >= 0.5).astype(int))
            print(f"    [XGBoost seed={seed}] F1@0.5={f1_fold:.5f}")
            del xgb_model
            gc.collect()

            lgb_params = CFG['lgb_params'].copy()
            lgb_params['random_state'] = seed
            lgb_model = LGBMClassifier(**lgb_params)
            lgb_model.fit(X_tr_fold, y_tr_fold, eval_set=[(X_va_fold, y_va_fold)],
                          callbacks=[early_stopping(200), log_evaluation(0)])
            oof_lgb2[va_idx, seed_idx] = lgb_model.predict_proba(X_va_fold)[:, 1].astype(np.float32)
            test_lgb2[:, seed_idx] += lgb_model.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']
            f1_fold_lgb = f1_score(y_va_fold, (oof_lgb2[va_idx, seed_idx] >= 0.5).astype(int))
            print(f"    [LightGBM seed={seed}] F1@0.5={f1_fold_lgb:.5f}  [{time.time()-t_fold_start:.0f}s]")
            del lgb_model
            gc.collect()

    oof_ensemble_stage2 = np.nan_to_num(np.mean(np.hstack([oof_xgb2, oof_lgb2]), axis=1), nan=0.0)
    test_ensemble_stage2 = np.mean(np.hstack([test_xgb2, test_lgb2]), axis=1)
    stage2_threshold, stage2_f1 = optimize_threshold(oof_ensemble_stage2, y, CFG)
    print(f"\n  [stage 2] OOF F1 at t_opt={stage2_threshold:.4f}: {stage2_f1:.5f}")

    dbg.log_step("Stage 2 training done", extra=f"OOF F1={stage2_f1:.5f}")

# %% [markdown]
# ## CELL 13 — NEW: Final stage selection + gap-closing diagnostic
#
# Keep whichever stage has the higher honest OOF F1 on real train (same automatic safety net as
# `Unbeatable_V6.py` / `synthetic-test-distr.py`). The gap-closing diagnostic is informational
# only — it does **not** decide the final stage — it answers whether the augmentation (when used)
# actually narrowed the measured train-vs-test covariate shift.

# %%
if SYNTH_AUGMENTATION_SKIPPED:
    FINAL_STAGE = "stage1_baseline_no_synth"
    final_test_ensemble = test_ensemble_stage1
    OPTIMAL_THRESHOLD = stage1_threshold
    BEST_OOF_F1 = stage1_f1
    X_final_train_used_diag = X_tr_diag
elif stage2_f1 >= stage1_f1:
    FINAL_STAGE = "stage2_synth_augmented"
    final_test_ensemble = test_ensemble_stage2
    OPTIMAL_THRESHOLD = stage2_threshold
    BEST_OOF_F1 = stage2_f1
    X_final_train_used_diag = np.vstack([X_tr_diag, accepted_synth_X_diag])
else:
    FINAL_STAGE = "stage1_baseline"
    final_test_ensemble = test_ensemble_stage1
    OPTIMAL_THRESHOLD = stage1_threshold
    BEST_OOF_F1 = stage1_f1
    X_final_train_used_diag = X_tr_diag

print("\n" + "=" * 75)
print("  CELL 13: FINAL STAGE SELECTION")
print("=" * 75)
print(f"  FINAL STAGE: {FINAL_STAGE}")
print(f"  stage1 F1={stage1_f1:.5f} | stage2 F1={'skipped' if stage2_f1 is None else f'{stage2_f1:.5f}'}")
print(f"  final threshold: {OPTIMAL_THRESHOLD:.4f} | final OOF F1: {BEST_OOF_F1:.5f}")

final_shift_auc = quick_adv_auc(X_final_train_used_diag, X_te_diag)
gap_closed = abs(final_shift_auc - 0.5) < abs(baseline_shift_auc - 0.5)
print(f"\n  GAP-CLOSING DIAGNOSTIC")
print(f"  baseline train-vs-test AUC:            {baseline_shift_auc:.4f}")
print(f"  final train-vs-test AUC (used training set): {final_shift_auc:.4f}")
print(f"  covariate shift {'NARROWED' if gap_closed else 'did NOT narrow'} "
      f"(distance to 0.50: {abs(baseline_shift_auc-0.5):.4f} -> {abs(final_shift_auc-0.5):.4f})")

dbg.log_step("Final stage + gap diagnostic done", extra=f"FINAL_STAGE={FINAL_STAGE}, gap_closed={gap_closed}")

# %% [markdown]
# ## CELL 14: Hard Binary Submission + Multi-Threshold Probing (Omega STEP 3b, unchanged)

# %%
print("\n" + "=" * 75)
print("  CELL 14: HARD BINARY SUBMISSION + MULTI-THRESHOLD PROBING")
print("=" * 75)

output_dir = Path('/kaggle/working') if os.path.isdir('/kaggle/working') else Path('.')
output_dir.mkdir(exist_ok=True)


def make_submission(probs, threshold, filename):
    binary = (probs >= threshold).astype(int)
    sub = sub_raw.copy()
    sub['TARGET'] = binary
    path = output_dir / filename
    sub.to_csv(path, index=False)
    n_pos_sub = np.sum(binary)
    return path, n_pos_sub, n_pos_sub / len(binary) * 100


sub_path, n_pos_opt, pos_rate_opt = make_submission(final_test_ensemble, OPTIMAL_THRESHOLD, 'submission.csv')
print(f"\n  PRIMARY SUBMISSION: submission.csv")
print(f"  Threshold: {OPTIMAL_THRESHOLD:.4f} | Stage: {FINAL_STAGE}")
print(f"  Class 1: {n_pos_opt:,} ({pos_rate_opt:.2f}%)")

sub_check = pd.read_csv(sub_path)
assert set(sub_check['TARGET'].unique()).issubset({0, 1}), "Submission must be integer 0/1!"
print(f"  Verified: integer 0/1 only")

sub_prob_path = output_dir / 'submission_prob.csv'
sub_prob = sub_raw.copy()
sub_prob['TARGET'] = final_test_ensemble
sub_prob.to_csv(sub_prob_path, index=False)
print(f"  submission_prob.csv written (raw blended probability)")

print(f"\n  PROBE SUBMISSIONS")
for t_probe in CFG['probe_thresholds']:
    fname = f'submission_t{t_probe:.2f}.csv'.replace('.', '_')
    path, n_pos, pos_rate = make_submission(final_test_ensemble, t_probe, fname)
    print(f"  {fname:30s}: threshold={t_probe:.2f}, pos_count={n_pos:,}, pos_rate={pos_rate:.2f}%")

dbg.log_step("Submissions saved", extra=f"Primary: submission.csv (t={OPTIMAL_THRESHOLD:.4f})")

# %% [markdown]
# ## CELL 15 — NEW: Synthetic dataset export + run summary

# %%
synthetic_export = synth_num_raw.copy()
for c in string_cat_features:
    synthetic_export[c] = synth_cat_raw[c]
synthetic_export['source_test_row_idx'] = sample_idx
synthetic_export['stage1_prob'] = synth_ensemble_stage1
synthetic_export['accepted'] = accepted_mask
synthetic_export['pseudo_label'] = pseudo_label_col
synthetic_export.to_csv(output_dir / 'synthetic_test_distribution.csv', index=False)
print(f"\n  synthetic_test_distribution.csv: {len(synthetic_export)} rows written")

import json
run_summary = {
    'final_stage': FINAL_STAGE,
    'n_pca_components': int(n_pca),
    'pca_variance_covered_ddof0': float(cumsum_var[n_pca - 1]),
    'n_synthetic_generated': int(N_SYNTHETIC),
    'n_numeric_cols_integer_valued': int(INT_LIKE_MASK.sum()),
    'n_numeric_cols_total': int(len(keep_numerical)),
    'jitter_frac': CFG['jitter_frac'],
    'synth_vs_test_auc': synth_vs_test_auc,
    'synth_vs_train_auc': synth_vs_train_auc,
    'baseline_shift_auc': baseline_shift_auc,
    'final_shift_auc': final_shift_auc,
    'gap_closed': bool(gap_closed),
    'pseudo_pos_threshold_used': pos_thresh_cur,
    'pseudo_neg_threshold_used': neg_thresh_cur,
    'synth_augmentation_skipped': bool(SYNTH_AUGMENTATION_SKIPPED),
    'n_synthetic_accepted': int(len(accepted_idx)),
    'n_synthetic_accepted_positive': int((accepted_synth_y == 1).sum()) if len(accepted_idx) else 0,
    'n_synthetic_accepted_negative': int((accepted_synth_y == 0).sum()) if len(accepted_idx) else 0,
    'stage1_threshold': stage1_threshold,
    'stage1_f1': stage1_f1,
    'stage2_threshold': stage2_threshold,
    'stage2_f1': stage2_f1,
    'final_threshold': OPTIMAL_THRESHOLD,
    'final_oof_f1': BEST_OOF_F1,
    'submission_positive_rate': float(pos_rate_opt) / 100.0,
    'submission_n_positive': int(n_pos_opt),
}
with open(output_dir / 'run_summary.json', 'w') as f:
    json.dump(run_summary, f, indent=2)
print(json.dumps(run_summary, indent=2))

dbg.summary()
total_time = time.time() - T_START
print(f"\n  OMEGA + SYNTHETIC V8 -- COMPLETE")
print(f"  Total runtime: {total_time/60:.1f} min")
print(f"  Primary submission: submission.csv (stage={FINAL_STAGE}, t={OPTIMAL_THRESHOLD:.4f})")

# %% [markdown]
# # DESIGN PRINCIPLES — OMEGA + SYNTHETIC V8
#
# | # | Principle | Source |
# |---|-----------|--------|
# | 1 | Standard LogLoss, `scale_pos_weight`, hard binary output, no CatBoost, no covariate clipping, triple cat encoding | Omega (unchanged) |
# | 2 | Bootstrap-resample real test rows + jitter only nonzero cells of genuinely continuous (non-integer-valued) columns | `synthetic-test-distr.py` — two earlier parametric-copula designs failed a local adversarial-AUC check (0.98 instead of ~0.50) before this one passed it (0.52) |
# | 3 | Generator-quality check runs before any model training | catches a broken generator in seconds, not after a multi-hour Kaggle run |
# | 4 | Pseudo-label threshold relaxation + negative-ratio cap + skip-if-no-positives | `results/unbeatable-6` measured `pseudo_positive: 0` from a naive symmetric threshold — this prevents repeating that |
# | 5 | Stage selection by honest real-train OOF F1, never by the gap-closing diagnostic | augmentation is not assumed to help; it has to prove it on real labels |
# | 6 | Gap-closing diagnostic reported but not used to pick the final stage | answers the actual research question (did shift narrow?) independent of whether F1 moved |
