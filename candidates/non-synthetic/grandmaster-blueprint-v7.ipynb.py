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
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

print("=" * 75)
print("  🏆 GRANDMASTER BLUEPRINT FINAL — Pathway Alpha")
print("  Data Purification | Covariate Clipping | PCA(ddof=0) | Santander FE")
print("  Per-Fold Class-Conditional Shuffling | Soft F1 Loss | Rank-Shifting")
print("=" * 75)
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy:  {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

T_START = time.time()

# ===================================================================
# Debug Tracker
# ===================================================================
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
            print(f"  [{e['step']:02d}] {e['name']}: shape={e.get('shape','N/A')}")

dbg = DebugTracker()

# ===================================================================
# CELL 2: Configuration
# ===================================================================
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

    # === STEP 1: Data Purification ===
    'drop_zero_variance': True,
    'drop_exact_duplicates': True,
    'clip_lower_percentile': 0.1,
    'clip_upper_percentile': 99.9,

    # === STEP 2: PCA (ddof=0) ===
    'pca_variance_threshold': 0.95,

    # === STEP 2: Santander-Style Feature Engineering ===
    'santander_sentinels': [-999999, 9999999999],
    'use_zero_counts': True,
    'use_sentinel_counts': True,
    'use_row_moments': True,
    'use_uniqueness_features': True,
    'uniqueness_top_n': 30,

    # === STEP 3: Class-Conditional Shuffling (NO SMOTE) ===
    'shuffle_upsample_factor': 8,
    'shuffle_random_state': 42,

    # === STEP 4: Model Parameters ===
    # XGBoost — with Soft F1 custom objective
    'xgb_params': {
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.015,
        'subsample': 0.80,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.70,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 200,
        'verbosity': 0,
        'random_state': 42,
        # DELIBERATELY OMITTED: scale_pos_weight
    },

    # LightGBM — with Soft F1 custom objective
    'lgb_params': {
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.015,
        'subsample': 0.80,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'num_leaves': 63,
        'min_child_samples': 30,
        'random_state': 42,
        'verbosity': -1,
        # DELIBERATELY OMITTED: scale_pos_weight
    },

    # CatBoost — LogLoss + F1 eval
    'catboost_params': {
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.020,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.0,
        'bagging_temperature': 0.5,
        'border_count': 254,
        'min_data_in_leaf': 30,
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False,
        'early_stopping_rounds': 200,
        'eval_metric': 'F1',
        # DELIBERATELY OMITTED: auto_class_weights, scale_pos_weight
    },

    # === Soft F1 Loss Configuration ===
    'use_soft_f1_loss': True,
    'soft_f1_eps': 1e-8,

    # === STEP 5: Threshold Search & Recalibration ===
    'threshold_min': 0.01,
    'threshold_max': 0.99,
    'threshold_step': 0.0025,
}

KNOWN_STRING_CATS = ['feat_142', 'feat_157', 'feat_318', 'feat_320', 'feat_325', 'feat_337']

print("\n" + "=" * 75)
print("  CONFIGURATION — Pathway Alpha")
print("=" * 75)
print(f"  Ensemble seeds:          {CFG['ensemble_seeds']}")
print(f"  CV folds:                {CFG['n_folds']}")
print(f"  PCA variance:            {CFG['pca_variance_threshold']} (ddof=0)")
print(f"  Shuffle upsample factor: {CFG['shuffle_upsample_factor']}× (PER FOLD)")
print(f"  Clip percentiles:        [{CFG['clip_lower_percentile']}, {CFG['clip_upper_percentile']}]")
print(f"  Sentinels:               {CFG['santander_sentinels']}")
print(f"  Soft F1 loss:            {'✅ Enabled' if CFG['use_soft_f1_loss'] else '❌ LogLoss fallback'}")
print(f"  Threshold range:         [{CFG['threshold_min']}, {CFG['threshold_max']}] "
      f"step={CFG['threshold_step']}")
print(f"  Total fits:              {len(CFG['ensemble_seeds']) * CFG['n_folds'] * 3}")

# ===================================================================
# CELL 3: Data Loading & Redundant Feature Detection
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 3: DATA LOADING & PURIFICATION (STEP 1a)")
print("=" * 75)

train_raw = pd.read_csv(CFG['train_path'])
test_raw  = pd.read_csv(CFG['test_path'])
sub_raw   = pd.read_csv(CFG['sub_path'])

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

# --- 3A: Detect string vs numerical by dtype ---
string_cat_features = [c for c in feature_cols if train_raw[c].dtype == 'object']
numerical_features  = [c for c in feature_cols if train_raw[c].dtype != 'object']

print(f"\n  STRING categoricals: {len(string_cat_features)}")
for c in string_cat_features:
    print(f"    {c}: {train_raw[c].nunique():,} unique")
print(f"\n  Numerical features: {len(numerical_features)}")
assert set(string_cat_features) == set(KNOWN_STRING_CATS), \
    f"Cat mismatch! Expected {KNOWN_STRING_CATS}, got {string_cat_features}"
print(f"  ✓ All 6 known string cats confirmed.")

# --- 3B: Detect Zero-Variance Features ---
print(f"\n  --- Detecting Zero-Variance Features ---")
train_num = train_raw[numerical_features].fillna(0)
test_num  = test_raw[numerical_features].fillna(0)

zero_var_features = []
for c in numerical_features:
    if train_num[c].nunique() <= 1 and test_num[c].nunique() <= 1:
        zero_var_features.append(c)

print(f"  Zero-variance features detected: {len(zero_var_features)}")
if zero_var_features:
    print(f"    First 10: {zero_var_features[:10]}")

# --- 3C: Detect Exact Duplicate Features ---
print(f"\n  --- Detecting Exact Duplicate Features ---")
train_num_values = train_num.values.astype(np.float64)
test_num_values  = test_num.values.astype(np.float64)

duplicate_pairs = []
n_num = len(numerical_features)
already_duplicate = set()

print(f"  Scanning {n_num} features for duplicates...")
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

if dup_features_to_drop:
    print(f"\n  Duplicate pairs (keeping 1st, dropping 2nd):")
    for p1, p2 in duplicate_pairs[:5]:
        print(f"    {p1} ≈ {p2}")
    if len(duplicate_pairs) > 5:
        print(f"    ... and {len(duplicate_pairs) - 5} more pairs")

keep_numerical = [f for f in numerical_features if f not in all_drop_features]
print(f"\n  ▶ Features AFTER purification:")
print(f"    String cats kept:  {len(string_cat_features)}")
print(f"    Numerical kept:    {len(keep_numerical)} / {len(numerical_features)}")
print(f"    Total kept:        {len(string_cat_features) + len(keep_numerical)}")

dbg.log_step("Data purification done",
             extra=f"Dropped {len(all_drop_features)} redundant "
                   f"({len(zero_var_features)} zero-var + {len(dup_features_to_drop)} dup)")

# ===================================================================
# CELL 4: Categorical Handling with <UNK> Bucket
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 4: CATEGORICAL HANDLING — <UNK> BUCKET (STEP 1b)")
print("=" * 75)

train_cats = {}
test_cats = {}
cat_vocabs = {}
cat_unk_counts = {}

for c in string_cat_features:
    train_vals = train_raw[c].fillna('MISSING').astype(str)
    test_vals  = test_raw[c].fillna('MISSING').astype(str)
    vocab = set(train_vals.unique())
    cat_vocabs[c] = vocab
    test_mapped = test_vals.apply(lambda x: x if x in vocab else '<UNK>')
    n_unk = (test_mapped == '<UNK>').sum()
    train_cats[c] = train_vals.values
    test_cats[c] = test_mapped.values
    cat_unk_counts[c] = n_unk
    print(f"  {c}: vocab_size={len(vocab):,}  test_<UNK>={n_unk} "
          f"({100*n_unk/len(test_raw):.2f}%)")

# Label encode (fit on train + mapped test, including <UNK>)
train_cats_le = pd.DataFrame(index=train_raw.index)
test_cats_le  = pd.DataFrame(index=test_raw.index)
label_encoders = {}

for c in string_cat_features:
    le = LabelEncoder()
    all_vals = np.concatenate([train_cats[c], test_cats[c]])
    le.fit(all_vals)
    label_encoders[c] = le
    train_cats_le[f'{c}_le'] = le.transform(train_cats[c]).astype(np.int32)
    test_cats_le[f'{c}_le']  = le.transform(test_cats[c]).astype(np.int32)

print(f"\n  ✓ Label-encoded {len(string_cat_features)} categoricals (with <UNK> bucket)")
print(f"  LE feature shape: train={train_cats_le.shape}, test={test_cats_le.shape}")

# Raw string DataFrames for CatBoost (explicitly object dtype)
train_cats_raw = pd.DataFrame({
    c: train_cats[c].astype(str).astype('object') for c in string_cat_features
})
test_cats_raw = pd.DataFrame({
    c: test_cats[c].astype(str).astype('object') for c in string_cat_features
})

# Verify object dtype
for c in string_cat_features:
    assert train_cats_raw[c].dtype == 'object', f"{c} not object dtype!"
    assert test_cats_raw[c].dtype == 'object', f"{c} not object dtype!"
print(f"  ✓ All {len(string_cat_features)} cat columns verified as object dtype")

dbg.log_step("Cat encoding with <UNK> done",
             extra=f"Total test <UNK> mappings: {sum(cat_unk_counts.values())}")

# ===================================================================
# CELL 5: Covariate Shift — Percentile Clipping (STEP 1c)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 5: COVARIATE SHIFT — CLIP TEST TO TRAIN PERCENTILES (STEP 1c)")
print("=" * 75)

X_train_num = train_raw[keep_numerical].fillna(0).astype(np.float32)
X_test_num  = test_raw[keep_numerical].fillna(0).astype(np.float32)

print(f"\n  Clean numerical features: {X_train_num.shape[1]}")
print(f"  Train range: [{X_train_num.values.min():.4f}, {X_train_num.values.max():.4f}]")
print(f"  Test range:  [{X_test_num.values.min():.4f}, {X_test_num.values.max():.4f}]")

lo_pct = CFG['clip_lower_percentile']
hi_pct = CFG['clip_upper_percentile']

print(f"\n  Computing train [{lo_pct}, {hi_pct}] percentiles...")
train_lower = np.percentile(X_train_num.values, lo_pct, axis=0)
train_upper = np.percentile(X_train_num.values, hi_pct, axis=0)

X_test_clipped = X_test_num.values.copy()
for j in range(X_test_clipped.shape[1]):
    X_test_clipped[:, j] = np.clip(X_test_clipped[:, j], train_lower[j], train_upper[j])

n_clipped_low  = np.sum(X_test_num.values < train_lower[np.newaxis, :])
n_clipped_high = np.sum(X_test_num.values > train_upper[np.newaxis, :])
n_total_cells  = X_test_num.size

print(f"\n  Clipping statistics:")
print(f"    Values clipped (low):  {n_clipped_low:,} / {n_total_cells:,} "
      f"({100*n_clipped_low/n_total_cells:.4f}%)")
print(f"    Values clipped (high): {n_clipped_high:,} / {n_total_cells:,} "
      f"({100*n_clipped_high/n_total_cells:.4f}%)")
print(f"    Total cells modified:  {n_clipped_low + n_clipped_high:,} "
      f"({100*(n_clipped_low+n_clipped_high)/n_total_cells:.4f}%)")

X_train_clipped = X_train_num.values.copy()

print(f"\n  After clipping test to train percentiles:")
print(f"    Train range: [{X_train_clipped.min():.4f}, {X_train_clipped.max():.4f}]")
print(f"    Test range:  [{X_test_clipped.min():.4f}, {X_test_clipped.max():.4f}]")

assert X_test_clipped.min() >= train_lower.min() - 1e-6, "Test below train floor!"
assert X_test_clipped.max() <= train_upper.max() + 1e-6, "Test above train ceiling!"
print(f"  ✓ Test values bounded within train [{lo_pct}, {hi_pct}] percentile range")

del X_train_num, X_test_num, train_lower, train_upper; gc.collect()

dbg.log_step("Covariate clipping done",
             extra=f"Clipped {n_clipped_low+n_clipped_high:,}/{n_total_cells:,} test cells "
                   f"({100*(n_clipped_low+n_clipped_high)/n_total_cells:.4f}%)")

# ===================================================================
# CELL 6: Santander-Style Feature Engineering (STEP 2a)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 6: SANTANDER-STYLE FEATURE ENGINEERING (STEP 2a)")
print("=" * 75)

n_train = X_train_clipped.shape[0]
n_test  = X_test_clipped.shape[0]
n_feats = X_train_clipped.shape[1]

X_all = np.vstack([X_train_clipped, X_test_clipped]).astype(np.float64)
print(f"  Combined data: {X_all.shape}")

# --- 6A: Zero Count per Row ---
if CFG['use_zero_counts']:
    print(f"\n  --- 6A: Zero Counts ---")
    n_zeros_train = np.sum(X_train_clipped == 0, axis=1).astype(np.float32)
    n_zeros_test  = np.sum(X_test_clipped == 0, axis=1).astype(np.float32)
    print(f"    Train zero count: mean={n_zeros_train.mean():.1f}, "
          f"med={np.median(n_zeros_train):.1f}, max={n_zeros_train.max():.0f}")
else:
    n_zeros_train = np.zeros(n_train, dtype=np.float32)
    n_zeros_test  = np.zeros(n_test, dtype=np.float32)

# --- 6B: Sentinel Value Counts ---
sentinel_feats_train = []
sentinel_feats_test  = []
if CFG['use_sentinel_counts']:
    print(f"\n  --- 6B: Sentinel Counts ---")
    for sentinel in CFG['santander_sentinels']:
        s_train = np.sum(X_train_clipped == sentinel, axis=1).astype(np.float32)
        s_test  = np.sum(X_test_clipped == sentinel, axis=1).astype(np.float32)
        n_sentinel = np.sum(X_all == sentinel)
        print(f"    Sentinel {sentinel}: {n_sentinel:,} occurrences total "
              f"({100*n_sentinel/X_all.size:.4f}% of cells)")
        sentinel_feats_train.append(s_train)
        sentinel_feats_test.append(s_test)

sentinel_train = np.column_stack(sentinel_feats_train).astype(np.float32) if sentinel_feats_train else np.zeros((n_train, 0), dtype=np.float32)
sentinel_test  = np.column_stack(sentinel_feats_test).astype(np.float32) if sentinel_feats_test else np.zeros((n_test, 0), dtype=np.float32)

# --- 6C: Row Moments on Non-Zero, Non-Sentinel Values ---
moment_feats_train = []
moment_feats_test  = []
moment_names = []

if CFG['use_row_moments']:
    print(f"\n  --- 6C: Row Moments (non-zero, non-sentinel) ---")
    sentinel_mask_train = np.zeros(X_train_clipped.shape, dtype=bool)
    sentinel_mask_test  = np.zeros(X_test_clipped.shape, dtype=bool)
    for s in CFG['santander_sentinels']:
        sentinel_mask_train |= (X_train_clipped == s)
        sentinel_mask_test  |= (X_test_clipped == s)

    valid_train = (X_train_clipped != 0) & (~sentinel_mask_train)
    valid_test  = (X_test_clipped != 0) & (~sentinel_mask_test)

    n_valid_train = valid_train.sum(axis=1)
    n_valid_test  = valid_test.sum(axis=1)
    print(f"    Valid cells per row: train mean={n_valid_train.mean():.1f}, "
          f"test mean={n_valid_test.mean():.1f}")

    masked_train = np.where(valid_train, X_train_clipped, np.nan)
    masked_test  = np.where(valid_test, X_test_clipped, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        row_mean_train = np.nan_to_num(np.nanmean(masked_train, axis=1), 0).astype(np.float32)
        row_mean_test  = np.nan_to_num(np.nanmean(masked_test, axis=1), 0).astype(np.float32)

        row_std_train = np.nan_to_num(np.nanstd(masked_train, axis=1, ddof=0), 0).astype(np.float32)
        row_std_test  = np.nan_to_num(np.nanstd(masked_test, axis=1, ddof=0), 0).astype(np.float32)

        row_sum_train = np.nan_to_num(np.nansum(masked_train, axis=1), 0).astype(np.float32)
        row_sum_test  = np.nan_to_num(np.nansum(masked_test, axis=1), 0).astype(np.float32)

        row_skew_train = np.nan_to_num(skew_fn(masked_train, axis=1, nan_policy='omit'), 0).astype(np.float32)
        row_skew_test  = np.nan_to_num(skew_fn(masked_test, axis=1, nan_policy='omit'), 0).astype(np.float32)

        row_kurt_train = np.nan_to_num(kt_fn(masked_train, axis=1, nan_policy='omit'), 0).astype(np.float32)
        row_kurt_test  = np.nan_to_num(kt_fn(masked_test, axis=1, nan_policy='omit'), 0).astype(np.float32)

    moment_feats_train = [row_mean_train, row_std_train, row_sum_train,
                           row_skew_train, row_kurt_train, n_valid_train.astype(np.float32)]
    moment_feats_test  = [row_mean_test, row_std_test, row_sum_test,
                           row_skew_test, row_kurt_test, n_valid_test.astype(np.float32)]
    moment_names = ['row_mean_valid', 'row_std_valid', 'row_sum_valid',
                    'row_skew_valid', 'row_kurtosis_valid', 'n_valid_cells']
    print(f"    Generated {len(moment_names)} row-moment features")

# --- 6D: Value Uniqueness & Frequency Features ---
uniq_feats_train = []
uniq_feats_test  = []
uniq_names = []

if CFG['use_uniqueness_features']:
    print(f"\n  --- 6D: Value Uniqueness & Frequency ---")
    top_n = min(CFG['uniqueness_top_n'], n_feats)
    variances = np.var(X_all, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    print(f"    Analyzing top-{top_n} highest-variance features for uniqueness...")

    for feat_idx in top_indices:
        col_train = X_train_clipped[:, feat_idx]
        col_test  = X_test_clipped[:, feat_idx]
        col_all   = X_all[:, feat_idx]

        unique_vals, counts = np.unique(col_all, return_counts=True)
        singleton_mask = counts == 1
        val_to_count = dict(zip(unique_vals, counts))
        singleton_vals = set(unique_vals[singleton_mask])

        freq_train = np.array([np.log1p(val_to_count.get(v, 1)) for v in col_train], dtype=np.float32)
        freq_test  = np.array([np.log1p(val_to_count.get(v, 1)) for v in col_test], dtype=np.float32)

        sing_train = np.array([1.0 if v in singleton_vals else 0.0 for v in col_train], dtype=np.float32)
        sing_test  = np.array([1.0 if v in singleton_vals else 0.0 for v in col_test], dtype=np.float32)

        uniq_feats_train.extend([freq_train, sing_train])
        uniq_feats_test.extend([freq_test, sing_test])
        uniq_names.extend([f'freq_feat_{feat_idx}', f'singleton_feat_{feat_idx}'])

    print(f"    Generated {len(uniq_names)} uniqueness features ({top_n} features × 2)")

# --- 6E: Assemble Santander Features ---
def ensure_2d(arr):
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

santander_parts_tr = [ensure_2d(n_zeros_train)]
santander_parts_te = [ensure_2d(n_zeros_test)]
if sentinel_train.shape[1] > 0:
    santander_parts_tr.append(sentinel_train)
    santander_parts_te.append(sentinel_test)
for arr in moment_feats_train:
    santander_parts_tr.append(ensure_2d(arr))
for arr in moment_feats_test:
    santander_parts_te.append(ensure_2d(arr))
for arr in uniq_feats_train:
    santander_parts_tr.append(ensure_2d(arr))
for arr in uniq_feats_test:
    santander_parts_te.append(ensure_2d(arr))

X_santander_train = np.column_stack(santander_parts_tr).astype(np.float32)
X_santander_test  = np.column_stack(santander_parts_te).astype(np.float32)

print(f"\n  ▶ Santander features: {X_santander_train.shape[1]} total")
print(f"    Zero counts:         1")
print(f"    Sentinel counts:     {sentinel_train.shape[1]}")
print(f"    Row moments:         {len(moment_names)}")
print(f"    Uniqueness features: {len(uniq_names)}")

del X_all, masked_train, masked_test, valid_train, valid_test; gc.collect()

dbg.log_step("Santander FE done", shape=X_santander_train.shape,
             extra=f"{X_santander_train.shape[1]} engineered features")

# ===================================================================
# CELL 7: QT → PCA(ddof=0) → Feature Assembly (STEP 2b)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 7: QT → PCA(ddof=0) → FEATURE ASSEMBLY (STEP 2b)")
print("=" * 75)

# --- 7A: QuantileTransformer ---
print(f"\n  --- 7A: QuantileTransformer ---")
qt = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, n_train),
    random_state=CFG['seed'],
    subsample=200_000
)

X_train_qt = qt.fit_transform(X_train_clipped).astype(np.float32)
X_test_qt  = qt.transform(X_test_clipped).astype(np.float32)

print(f"  QT train: {X_train_qt.shape}  range=[{X_train_qt.min():.3f}, {X_train_qt.max():.3f}]")
print(f"  QT test:  {X_test_qt.shape}  range=[{X_test_qt.min():.3f}, {X_test_qt.max():.3f}]")

dbg.log_step("QT done", shape=X_train_qt.shape)

# --- 7B: PCA with ddof=0 (Manual SVD) ---
print(f"\n  --- 7B: PCA with ddof=0 ({CFG['pca_variance_threshold']:.0%} variance) ---")

# Stack train + test for joint PCA
X_qt_all = np.vstack([X_train_qt, X_test_qt])
n_total = X_qt_all.shape[0]

# Center the data
qt_mean = X_qt_all.mean(axis=0)
X_centered = X_qt_all - qt_mean

# Manual SVD
print(f"    Computing SVD on {X_centered.shape}...")
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# ddof=0: divide by n (NOT n-1 like sklearn)
explained_variance_ddof0 = (S ** 2) / n_total
explained_variance_ratio = explained_variance_ddof0 / explained_variance_ddof0.sum()
cumsum_var = np.cumsum(explained_variance_ratio)

# Find n_components for target variance
n_pca = int(np.searchsorted(cumsum_var, CFG['pca_variance_threshold']) + 1)
n_pca = min(n_pca, X_qt_all.shape[1])

print(f"    Variance explained (ddof=0):")
for k in [10, 25, 50, 100, n_pca]:
    if k <= len(cumsum_var):
        print(f"      Top-{k:3d}: {cumsum_var[k-1]:.4f} ({cumsum_var[k-1]*100:.1f}%)")

# Transform: project onto principal components
X_train_pca = (U[:n_train, :n_pca] * S[:n_pca]).astype(np.float32)
X_test_pca  = (U[n_train:, :n_pca] * S[:n_pca]).astype(np.float32)

print(f"\n    PCA components kept: {n_pca}/{X_qt_all.shape[1]} "
      f"(retains {cumsum_var[n_pca-1]*100:.1f}% variance with ddof=0)")
print(f"    PCA train: {X_train_pca.shape}  range=[{X_train_pca.min():.3f}, {X_train_pca.max():.3f}]")
print(f"    PCA test:  {X_test_pca.shape}  range=[{X_test_pca.min():.3f}, {X_test_pca.max():.3f}]")

# Clean up large intermediates
del X_qt_all, X_centered, U, S, Vt, X_train_qt, X_test_qt; gc.collect()

dbg.log_step("PCA done", shape=X_train_pca.shape, extra=f"{n_pca} components, ddof=0")

# --- 7C: Feature Assembly ---
# Pipeline A: XGBoost/LightGBM (all float32 numpy)
X_tr_base = np.hstack([
    X_train_pca,
    X_santander_train,
    train_cats_le.values.astype(np.float32)
]).astype(np.float32)

X_te_base = np.hstack([
    X_test_pca,
    X_santander_test,
    test_cats_le.values.astype(np.float32)
]).astype(np.float32)

n_total_features = X_tr_base.shape[1]
print(f"\n  ▶ Final feature matrix: {X_tr_base.shape}")
print(f"    PCA components:      {n_pca}")
print(f"    Santander features:  {X_santander_train.shape[1]}")
print(f"    LE cat features:     {train_cats_le.shape[1]}")
print(f"    TOTAL:               {n_total_features}")

# --- 7D: CatBoost DataFrame ---
# Numerical base: PCA + Santander (float32)
cb_train = pd.DataFrame(X_train_pca, columns=[f'pca_{i}' for i in range(n_pca)])
cb_test  = pd.DataFrame(X_test_pca,  columns=[f'pca_{i}' for i in range(n_pca)])

# Add Santander features (numerical)
for i in range(X_santander_train.shape[1]):
    cb_train[f'santander_{i}'] = X_santander_train[:, i]
    cb_test[f'santander_{i}']  = X_santander_test[:, i]

# Add raw STRING categoricals for CatBoost (explicitly object dtype)
cb_cat_indices = []
for c in string_cat_features:
    col_idx = cb_train.shape[1]
    cb_train[c] = train_cats_raw[c].values  # already object dtype
    cb_test[c]  = test_cats_raw[c].values   # already object dtype
    cb_cat_indices.append(col_idx)

# Verify dtypes
for ci in cb_cat_indices:
    col_name = cb_train.columns[ci]
    assert cb_train[col_name].dtype == 'object', \
        f"CB cat column '{col_name}' must be object, got {cb_train[col_name].dtype}"
print(f"\n  CatBoost DataFrame:")
print(f"    Train: {cb_train.shape}  Test: {cb_test.shape}")
print(f"    cat_features: {len(cb_cat_indices)} indices {cb_cat_indices}")
print(f"    cat_names: {[cb_train.columns[i] for i in cb_cat_indices]}")
print(f"    ✓ All cat columns verified as object dtype")

dbg.log_step("Feature assembly done", shape=X_tr_base.shape,
             extra=f"{n_total_features} features, {len(cb_cat_indices)} CB cats")

# ===================================================================
# CELL 8: Per-Fold Shuffling + Model Training (STEPS 3+4)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 8: PER-FOLD SHUFFLING + SOFT F1 TRAINING (STEPS 3+4)")
print("=" * 75)

# --- 8A: Soft F1 Custom Objective Functions ---

def soft_f1_objective_xgb(y_true, y_pred):
    """
    Custom Soft F1 objective for XGBoost.
    y_true: binary labels (0/1)  — float32
    y_pred: raw margin scores (before sigmoid) — float32
    Returns: (gradient, hessian) as float32 arrays

    CRITICAL: Gradients are scaled by batch_size to prevent float32 underflow.
    Raw Soft F1 gradients are O(1/n) — with n=76k this is ~1e-7,
    which underflows float32 (precision ~7 decimal digits).
    Scaling by n brings them to logloss-like magnitude (~1.0).
    """
    n = len(y_true)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    S = np.sum(p)
    T = np.sum(y_true)
    TP = np.sum(y_true * p)
    D = S + T + CFG['soft_f1_eps']

    # dL/dp = -2*(y_i*D - TP) / D²
    dL_dp = -2.0 * (y_true * D - TP) / (D * D)
    # Chain rule: dp/dmargin = p*(1-p)
    dp_dm = p * (1.0 - p)
    # Scale by n to prevent float32 underflow (~1e-7 → ~1.0)
    grad = dL_dp * dp_dm * n
    hess = np.abs(grad) + 1e-6

    return grad.astype(np.float32), hess.astype(np.float32)


def soft_f1_objective_lgb(y_true, y_pred):
    """
    Custom Soft F1 objective for LightGBM.
    y_pred is raw margin scores (BEFORE sigmoid) — same as XGBoost.
    LightGBM passes raw margins to custom objectives, NOT probabilities.

    CRITICAL: Gradients are scaled by batch_size to prevent underflow.
    Raw Soft F1 gradients are O(1/n) — scaling by n brings them to
    logloss-like magnitude.
    """
    n = len(y_true)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # LightGBM passes raw margins — must apply sigmoid + chain rule (same as XGBoost)
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    S = np.sum(p)
    T = np.sum(y_true)
    TP = np.sum(y_true * p)
    D = S + T + CFG['soft_f1_eps']

    dL_dp = -2.0 * (y_true * D - TP) / (D * D)
    # Chain rule through sigmoid: dp/dm = p*(1-p)
    dp_dm = p * (1.0 - p)
    # Scale by n to prevent underflow (~1e-7 → ~1.0)
    grad = dL_dp * dp_dm * n
    hess = np.abs(grad) + 1e-6

    return grad.astype(np.float64), hess.astype(np.float64)


print(f"  ✓ Soft F1 custom objectives defined")
print(f"    eps = {CFG['soft_f1_eps']}")

# --- 8B: Training Loop ---
n_seeds = len(CFG['ensemble_seeds'])
n_base = len(y)

# OOF storage (ONLY for original samples, indexed by position in y)
oof_xgb = np.zeros((n_base, n_seeds), dtype=np.float32)
oof_cb  = np.zeros((n_base, n_seeds), dtype=np.float32)
oof_lgb = np.zeros((n_base, n_seeds), dtype=np.float32)

# Test prediction storage
n_test_final = len(test_raw)
test_xgb = np.zeros((n_test_final, n_seeds), dtype=np.float32)
test_cb  = np.zeros((n_test_final, n_seeds), dtype=np.float32)
test_lgb = np.zeros((n_test_final, n_seeds), dtype=np.float32)

fold_scores = {'xgb': {s: [] for s in CFG['ensemble_seeds']},
               'cb':  {s: [] for s in CFG['ensemble_seeds']},
               'lgb': {s: [] for s in CFG['ensemble_seeds']}}

upsample_factor = CFG['shuffle_upsample_factor']

print(f"\n{'='*75}")
print(f"  TRAINING: {n_seeds} seeds × {CFG['n_folds']} folds × 3 models = "
      f"{n_seeds * CFG['n_folds'] * 3} fits")
print(f"  Shuffling: {upsample_factor}× PER FOLD (no leakage)")
print(f"{'='*75}")

for seed_idx, seed in enumerate(CFG['ensemble_seeds']):
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE SEED {seed} ({seed_idx+1}/{n_seeds})")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_base, y)):
        t_fold_start = time.time()
        print(f"\n  --- Fold {fold+1}/{CFG['n_folds']} ---")

        # ================================================================
        # PER-FOLD CLASS-CONDITIONAL FEATURE SHUFFLING
        # ================================================================
        X_tr_fold_orig = X_tr_base[tr_idx]      # float32
        y_tr_fold_orig = y[tr_idx]

        # Isolate positive class in this fold's training data
        pos_mask_fold = (y_tr_fold_orig == 1)
        X_pos_fold = X_tr_fold_orig[pos_mask_fold]
        n_pos_fold = X_pos_fold.shape[0]
        n_synthetic_fold = n_pos_fold * upsample_factor

        if n_pos_fold > 0:
            rng = np.random.RandomState(CFG['shuffle_random_state'] + seed + fold)
            X_synthetic_fold = np.zeros((n_synthetic_fold, n_total_features), dtype=np.float32)

            for j in range(n_total_features):
                col_pool = X_pos_fold[:, j]
                random_indices = rng.randint(0, n_pos_fold, size=n_synthetic_fold)
                X_synthetic_fold[:, j] = col_pool[random_indices]

            # Augment fold training data
            X_tr_fold = np.vstack([X_tr_fold_orig, X_synthetic_fold]).astype(np.float32)
            y_tr_fold = np.hstack([y_tr_fold_orig, np.ones(n_synthetic_fold, dtype=np.int32)])
        else:
            X_tr_fold = X_tr_fold_orig
            y_tr_fold = y_tr_fold_orig
            n_synthetic_fold = 0

        # Validation: ORIGINAL samples only (no synthetic)
        X_va_fold = X_tr_base[va_idx]
        y_va_fold = y[va_idx]

        print(f"    Train: {X_tr_fold.shape[0]:,} rows "
              f"({len(y_tr_fold_orig):,} orig + {n_synthetic_fold:,} synth), "
              f"pos={np.sum(y_tr_fold==1):,} ({100*np.mean(y_tr_fold==1):.2f}%)")
        print(f"    Valid: {X_va_fold.shape[0]:,} rows (ALL original), "
              f"pos={np.sum(y_va_fold==1):,} ({100*np.mean(y_va_fold==1):.2f}%)")

        # --- CatBoost per-fold augmentation ---
        cb_tr_fold_orig = cb_train.iloc[tr_idx].reset_index(drop=True)
        cb_va_fold = cb_train.iloc[va_idx].reset_index(drop=True)

        if n_synthetic_fold > 0:
            # Build synthetic rows for CatBoost
            cb_synthetic_rows = []
            # PCA features
            for j in range(n_pca):
                cb_synthetic_rows.append(X_synthetic_fold[:, j])
            # Santander features
            sant_offset = n_pca
            for j in range(X_santander_train.shape[1]):
                cb_synthetic_rows.append(X_synthetic_fold[:, sant_offset + j])
            # String cats: randomly sample from fold's positive class
            cb_pos_fold = cb_tr_fold_orig.iloc[pos_mask_fold]
            rng_cb = np.random.RandomState(CFG['shuffle_random_state'] + seed + fold)
            cb_cat_values = {}
            for c in string_cat_features:
                pos_vals = cb_pos_fold[c].values
                random_idx = rng_cb.randint(0, len(pos_vals), size=n_synthetic_fold)
                cb_cat_values[c] = pos_vals[random_idx]

            # Build DataFrame
            cb_synthetic = pd.DataFrame(index=range(n_synthetic_fold))
            for j in range(n_pca):
                cb_synthetic[f'pca_{j}'] = X_synthetic_fold[:, j].astype(np.float32)
            for j in range(X_santander_train.shape[1]):
                cb_synthetic[f'santander_{j}'] = X_synthetic_fold[:, n_pca + j].astype(np.float32)
            for c in string_cat_features:
                cb_synthetic[c] = cb_cat_values[c].astype(str).astype('object')

            cb_tr_fold = pd.concat([cb_tr_fold_orig, cb_synthetic], ignore_index=True)
        else:
            cb_tr_fold = cb_tr_fold_orig

        # ================================================================
        # XGBOOST — float32 input, custom Soft F1 objective
        # ================================================================
        t0 = time.time()
        xgb_params = CFG['xgb_params'].copy()
        xgb_params['random_state'] = seed

        # Verify float32
        assert X_tr_fold.dtype == np.float32, f"XGB input must be float32, got {X_tr_fold.dtype}"
        assert X_va_fold.dtype == np.float32
        assert X_te_base.dtype == np.float32

        xgb = XGBClassifier(**xgb_params)

        if CFG['use_soft_f1_loss']:
            xgb.set_params(objective=soft_f1_objective_xgb)

        xgb.fit(X_tr_fold, y_tr_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                verbose=False)

        oof_xgb[va_idx, seed_idx] = xgb.predict_proba(X_va_fold)[:, 1].astype(np.float32)
        test_xgb[:, seed_idx] += xgb.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']

        va_probs = oof_xgb[va_idx, seed_idx]
        f1_fold = f1_score(y_va_fold, (va_probs >= 0.5).astype(int))
        fold_scores['xgb'][seed].append(f1_fold)
        dt = time.time() - t0
        print(f"    [XGBoost seed={seed}] F1@0.5={f1_fold:.5f}  "
              f"mean_p={va_probs.mean():.4f}  pos@0.5={np.mean(va_probs >= 0.5)*100:.1f}%  [{dt:.0f}s]")

        del xgb; gc.collect()

        # ================================================================
        # CATBOOST — object-type cats, built-in F1 eval, LogLoss internally
        # ================================================================
        t0 = time.time()
        cb_params = CFG['catboost_params'].copy()
        cb_params['random_seed'] = seed

        # Verify cat columns are object dtype
        for ci in cb_cat_indices:
            assert cb_tr_fold.iloc[:, ci].dtype == 'object', \
                f"CB col {ci} dtype={cb_tr_fold.iloc[:, ci].dtype}, expected object"

        cb = CatBoostClassifier(**cb_params, cat_features=cb_cat_indices)
        cb.fit(cb_tr_fold, y_tr_fold,
               eval_set=(cb_va_fold, y_va_fold),
               verbose=False)

        oof_cb[va_idx, seed_idx] = cb.predict_proba(cb_va_fold)[:, 1].astype(np.float32)
        test_cb[:, seed_idx] += cb.predict_proba(cb_test)[:, 1].astype(np.float32) / CFG['n_folds']

        va_probs_cb = oof_cb[va_idx, seed_idx]
        f1_fold_cb = f1_score(y_va_fold, (va_probs_cb >= 0.5).astype(int))
        fold_scores['cb'][seed].append(f1_fold_cb)
        dt_cb = time.time() - t0
        print(f"    [CatBoost seed={seed}] F1@0.5={f1_fold_cb:.5f}  "
              f"mean_p={va_probs_cb.mean():.4f}  [{dt_cb:.0f}s]")

        del cb; gc.collect()

        # ================================================================
        # LIGHTGBM — float32 input, custom Soft F1 objective
        # ================================================================
        t0 = time.time()
        lgb_params = CFG['lgb_params'].copy()
        lgb_params['random_state'] = seed

        lgb = LGBMClassifier(**lgb_params)

        if CFG['use_soft_f1_loss']:
            lgb.set_params(objective=soft_f1_objective_lgb)

        lgb.fit(X_tr_fold, y_tr_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                eval_metric='logloss',
                callbacks=[early_stopping(200), log_evaluation(0)])

        # LightGBM with custom objective returns 1D array (positive class probs)
        lgb_va_probs = lgb.predict_proba(X_va_fold)
        if lgb_va_probs.ndim == 2:
            lgb_va_probs = lgb_va_probs[:, 1]
        oof_lgb[va_idx, seed_idx] = lgb_va_probs.astype(np.float32)

        lgb_te_probs = lgb.predict_proba(X_te_base)
        if lgb_te_probs.ndim == 2:
            lgb_te_probs = lgb_te_probs[:, 1]
        test_lgb[:, seed_idx] += lgb_te_probs.astype(np.float32) / CFG['n_folds']

        va_probs_lgb = oof_lgb[va_idx, seed_idx]
        f1_fold_lgb = f1_score(y_va_fold, (va_probs_lgb >= 0.5).astype(int))
        fold_scores['lgb'][seed].append(f1_fold_lgb)
        dt_lgb = time.time() - t0
        print(f"    [LightGBM seed={seed}] F1@0.5={f1_fold_lgb:.5f}  "
              f"mean_p={va_probs_lgb.mean():.4f}  [{dt_lgb:.0f}s]")

        del lgb; gc.collect()

        t_fold_total = time.time() - t_fold_start
        print(f"    Fold total: {t_fold_total:.0f}s")

    # End of folds for this seed — cumulative OOF
    # oof_*[:, seed_idx] is 1D (all rows for this seed) — no mean needed
    s1_xgb = f1_score(y, (np.nan_to_num(oof_xgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    s1_cb  = f1_score(y, (np.nan_to_num(oof_cb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    s1_lgb = f1_score(y, (np.nan_to_num(oof_lgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    print(f"\n  Seed {seed} cumulative OOF@0.5: "
          f"XGB={s1_xgb:.5f}  CB={s1_cb:.5f}  LGB={s1_lgb:.5f}")

elapsed_train = time.time() - T_START
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE — {n_seeds * CFG['n_folds'] * 3} models trained")
print(f"  Elapsed: {elapsed_train/60:.1f} min ({elapsed_train:.0f}s)")
print(f"{'='*60}")

dbg.log_step("Model training done", extra=f"Elapsed: {elapsed_train/60:.1f} min")

# ===================================================================
# CELL 9: Threshold Optimization & Probability Recalibration (STEP 5)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 9: THRESHOLD OPTIMIZATION & RECALIBRATION (STEP 5)")
print("=" * 75)

# --- 9A: Average OOF across seeds ---
oof_xgb_avg = np.nan_to_num(np.mean(oof_xgb, axis=1), nan=0.0)
oof_cb_avg  = np.nan_to_num(np.mean(oof_cb, axis=1), nan=0.0)
oof_lgb_avg = np.nan_to_num(np.mean(oof_lgb, axis=1), nan=0.0)

# Ensemble strategies
all_oof_seeds = np.hstack([oof_xgb, oof_cb, oof_lgb])
oof_uniform = np.nan_to_num(np.mean(all_oof_seeds, axis=1), nan=0.0)
oof_avg     = (oof_xgb_avg + oof_cb_avg + oof_lgb_avg) / 3.0

print(f"\n  OOF prediction stats (original samples):")
for label, arr in [('XGBoost', oof_xgb_avg), ('CatBoost', oof_cb_avg),
                    ('LightGBM', oof_lgb_avg), ('Uniform Ensemble', oof_uniform),
                    ('Avg Ensemble', oof_avg)]:
    print(f"    {label:20s}: mean={np.mean(arr):.5f}  med={np.median(arr):.5f}  "
          f"p90={np.percentile(arr, 90):.4f}  p95={np.percentile(arr, 95):.4f}  "
          f"p99={np.percentile(arr, 99):.4f}")

# --- 9B: Exhaustive Threshold Search ---
thresholds = np.arange(CFG['threshold_min'], CFG['threshold_max'] + CFG['threshold_step']/2,
                        CFG['threshold_step'])

print(f"\n  Grid searching {len(thresholds)} thresholds [{CFG['threshold_min']:.4f}, "
      f"{CFG['threshold_max']:.4f}] step={CFG['threshold_step']:.4f}...")

best_results = {}
for label, oof_arr in [
    ('XGBoost', oof_xgb_avg),
    ('CatBoost', oof_cb_avg),
    ('LightGBM', oof_lgb_avg),
    ('Uniform Ensemble', oof_uniform),
    ('Avg Ensemble', oof_avg),
]:
    best_f1, best_t, best_prec, best_rec, best_pos = 0, 0.5, 0, 0, 0
    for t in thresholds:
        binary = (oof_arr >= t).astype(int)
        if np.sum(binary) == 0:
            continue
        f1 = f1_score(y, binary)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            best_prec = precision_score(y, binary, zero_division=0)
            best_rec  = recall_score(y, binary, zero_division=0)
            best_pos  = np.mean(binary) * 100

    best_results[label] = {
        'threshold': best_t, 'f1': best_f1,
        'precision': best_prec, 'recall': best_rec,
        'pos_rate': best_pos,
    }

print(f"\n  {'='*75}")
print(f"  OPTIMAL THRESHOLDS (maximizing OOF F1)")
print(f"  {'='*75}")
print(f"  {'Model':20s} {'Best T':>10s} {'F1':>10s} {'Precision':>10s} {'Recall':>10s} {'Pos%':>8s}")
print(f"  {'-'*70}")
for label, res in best_results.items():
    print(f"  {label:20s} {res['threshold']:10.4f} {res['f1']:10.5f} "
          f"{res['precision']:10.4f} {res['recall']:10.4f} {res['pos_rate']:7.2f}%")

# --- 9C: Select Best Ensemble ---
best_label = 'Uniform Ensemble'
best_f1 = best_results['Uniform Ensemble']['f1']
best_t_opt = best_results['Uniform Ensemble']['threshold']

for label, res in best_results.items():
    if res['f1'] > best_f1:
        best_f1, best_label, best_t_opt = res['f1'], label, res['threshold']

print(f"\n  ▶ SELECTED: {best_label}")
print(f"    Optimal threshold t_opt: {best_t_opt:.4f}")
print(f"    OOF F1 at t_opt:         {best_f1:.5f}")

f1_at_05 = f1_score(y, (oof_uniform >= 0.5).astype(int))
print(f"    F1@0.5 (no optimization): {f1_at_05:.5f}")
print(f"    Gain from optimization:   {best_f1 - f1_at_05:+.5f}")

# --- 9D: Piecewise Linear Recalibration (Rank-Shifting) ---
print(f"\n  --- Probability Recalibration (Rank-Shifting) ---")
print(f"  Mapping t_opt={best_t_opt:.4f} → 0.5 via piecewise linear transform")

def recalibrate_probabilities(probs, t_opt):
    """
    Piecewise linear transformation mapping t_opt → 0.5.

    If p ≤ t_opt:   p_new = (p / t_opt) × 0.5
    If p > t_opt:   p_new = 0.5 + ((p − t_opt) / (1 − t_opt)) × 0.5
    """
    p = np.asarray(probs, dtype=np.float64)
    p_new = np.zeros_like(p)
    mask_low = p <= t_opt
    mask_high = p > t_opt

    if t_opt > 0:
        p_new[mask_low] = (p[mask_low] / t_opt) * 0.5
    else:
        p_new[mask_low] = 0.0

    if t_opt < 1.0:
        p_new[mask_high] = 0.5 + ((p[mask_high] - t_opt) / (1.0 - t_opt)) * 0.5
    else:
        p_new[mask_high] = 1.0

    return p_new.astype(np.float64)

# Verify on OOF
oof_calibrated = recalibrate_probabilities(oof_uniform, best_t_opt)
print(f"\n  Recalibration verification on OOF:")
print(f"    Before: mean={oof_uniform.mean():.5f}  "
      f"pos@0.5={np.mean(oof_uniform >= 0.5)*100:.2f}%  "
      f"pos@{best_t_opt:.4f}={np.mean(oof_uniform >= best_t_opt)*100:.2f}%")
print(f"    After:  mean={oof_calibrated.mean():.5f}  "
      f"pos@0.5={np.mean(oof_calibrated >= 0.5)*100:.2f}%")

f1_calibrated = f1_score(y, (oof_calibrated >= 0.5).astype(int))
f1_raw_at_topt = f1_score(y, (oof_uniform >= best_t_opt).astype(int))
print(f"    F1(raw@{best_t_opt:.4f})  = {f1_raw_at_topt:.5f}")
print(f"    F1(calibrated@0.5) = {f1_calibrated:.5f}")
print(f"    Match: {'✅ PERFECT' if abs(f1_calibrated - f1_raw_at_topt) < 1e-6 else '⚠️ MISMATCH'}")

OPTIMAL_THRESHOLD = best_t_opt
BEST_OOF_F1 = best_f1

dbg.log_step("Threshold & recalibration done",
             extra=f"t_opt={OPTIMAL_THRESHOLD:.4f}, OOF F1={BEST_OOF_F1:.5f}")

# ===================================================================
# CELL 10: Final Test Predictions & Submission
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 10: FINAL TEST PREDICTIONS & SUBMISSION")
print("=" * 75)

# Average test predictions across all seeds
test_xgb_avg = np.mean(test_xgb, axis=1)
test_cb_avg  = np.mean(test_cb, axis=1)
test_lgb_avg = np.mean(test_lgb, axis=1)
test_uniform = np.mean(np.hstack([test_xgb, test_cb, test_lgb]), axis=1)
test_avg     = (test_xgb_avg + test_cb_avg + test_lgb_avg) / 3.0

# Map best_label to test prediction array
test_label_map = {
    'XGBoost': test_xgb_avg,
    'CatBoost': test_cb_avg,
    'LightGBM': test_lgb_avg,
    'Uniform Ensemble': test_uniform,
    'Avg Ensemble': test_avg,
}
raw_test_probs = test_label_map.get(best_label, test_uniform)

print(f"\n  Using ensemble: {best_label}")
print(f"\n  Raw test prediction stats:")
for label, probs in [
    ('XGBoost', test_xgb_avg),
    ('CatBoost', test_cb_avg),
    ('LightGBM', test_lgb_avg),
    ('Uniform Ensemble', test_uniform),
    ('Avg Ensemble', test_avg),
]:
    pos_at_topt = np.mean(probs >= OPTIMAL_THRESHOLD) * 100
    print(f"  {label:25s}: mean={np.mean(probs):.5f}  "
          f"pos@t_opt={pos_at_topt:.2f}%  "
          f"range=[{probs.min():.4f}, {probs.max():.4f}]")

# --- Apply Rank-Shifting ---
print(f"\n  --- Applying Rank-Shifting ---")
final_probs = recalibrate_probabilities(raw_test_probs, OPTIMAL_THRESHOLD)

binary_predictions = (final_probs >= 0.5).astype(int)
n_pos_pred = np.sum(binary_predictions)
pos_rate_pred = n_pos_pred / len(binary_predictions) * 100

print(f"\n  After recalibration (continuous probabilities):")
print(f"    Mean prob:         {final_probs.mean():.5f}")
print(f"    Median prob:       {np.median(final_probs):.5f}")
print(f"    Std prob:          {final_probs.std():.5f}")
print(f"    Range:             [{final_probs.min():.4f}, {final_probs.max():.4f}]")

print(f"\n  At Kaggle's 0.5 threshold (≡ our t_opt={OPTIMAL_THRESHOLD:.4f}):")
print(f"    Class 0: {len(binary_predictions) - n_pos_pred:,} "
      f"({100 - pos_rate_pred:.2f}%)")
print(f"    Class 1: {n_pos_pred:,} ({pos_rate_pred:.2f}%)")

if 1.0 < pos_rate_pred < 12.0:
    print(f"    ✅ Positive rate in healthy range (1-12%)")
elif pos_rate_pred < 0.5:
    print(f"    ⚠️  WARNING: Very low positive rate")
elif pos_rate_pred > 50:
    print(f"    ⚠️  WARNING: Very high positive rate")

# --- Create Submission ---
submission = sub_raw.copy()
submission['TARGET'] = final_probs.astype(np.float64)

output_dir = Path('/kaggle/working')
output_dir.mkdir(exist_ok=True)

sub_path = output_dir / 'submission.csv'
submission.to_csv(sub_path, index=False)

print(f"\n  {'='*75}")
print(f"  ✓ SUBMISSION SAVED: {sub_path}")
print(f"  {'='*75}")
print(f"  Shape:  {submission.shape}")
print(f"  Values: CONTINUOUS (rank-shifted probabilities)")
print(f"  t_opt:  {OPTIMAL_THRESHOLD:.4f} → mapped to 0.5")
print(f"  Mean:   {final_probs.mean():.5f}")
print(f"\n  First 10 rows:")
print(submission.head(10).to_string())
print(f"\n  Value distribution:")
print(submission['TARGET'].describe().to_string())

# Also save binary version for comparison
sub_binary = sub_raw.copy()
sub_binary['TARGET'] = binary_predictions.astype(int)
sub_binary_path = output_dir / 'submission_binary.csv'
sub_binary.to_csv(sub_binary_path, index=False)
print(f"\n  ✓ Binary backup saved: {sub_binary_path}")

dbg.log_step("Submission saved",
             extra=f"Continuous recalibrated, pos@0.5={pos_rate_pred:.2f}%")

# ===================================================================
# CELL 11: Grandmaster Performance Dashboard
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 11: GRANDMASTER PERFORMANCE DASHBOARD")
print("=" * 75)

# --- Model Health ---
print(f"\n  {'='*60}")
print(f"  MODEL HEALTH CHECK")
print(f"  {'='*60}")

for model_name, oof_data in [('XGBoost', oof_xgb), ('CatBoost', oof_cb), ('LightGBM', oof_lgb)]:
    oof_avg_m = np.nan_to_num(np.mean(oof_data, axis=1), nan=0.0)
    f1_opt = f1_score(y, (oof_avg_m >= OPTIMAL_THRESHOLD).astype(int))
    f1_05  = f1_score(y, (oof_avg_m >= 0.5).astype(int))
    status = "✅ ALIVE" if f1_opt > 0.05 else "🔴 DEAD"
    print(f"  {status} {model_name:12s}: mean_p={oof_avg_m.mean():.4f}  "
          f"F1@0.5={f1_05:.5f}  F1@{OPTIMAL_THRESHOLD:.3f}={f1_opt:.5f}")

# --- Feature Engineering Summary ---
print(f"\n  {'='*60}")
print(f"  FEATURE ENGINEERING SUMMARY")
print(f"  {'='*60}")
print(f"  Raw feat_* columns:              350")
print(f"  STRING categoricals:              {len(string_cat_features)}")
print(f"  Zero-variance DROPPED:            {len(zero_var_features)}")
print(f"  Duplicate DROPPED:                {len(dup_features_to_drop)}")
print(f"  {'─'*40}")
print(f"  PCA components (ddof=0):          {n_pca}")
print(f"  Santander engineered features:    {X_santander_train.shape[1]}")
print(f"  LE categorical features:          {train_cats_le.shape[1]}")
print(f"  {'─'*40}")
print(f"  TOTAL features:                   {n_total_features}")
print(f"  CatBoost cat_features:            {len(cb_cat_indices)}")

# --- Training Summary ---
print(f"\n  {'='*60}")
print(f"  TRAINING SUMMARY")
print(f"  {'='*60}")
print(f"  Original train size:              {n_base:,}")
print(f"  Per-fold upsampling:              {upsample_factor}× (synthetic/pos ratio depends on fold)")
print(f"  Models trained:                   {n_seeds * CFG['n_folds'] * 3}")
print(f"  Soft F1 loss:                     {'✅' if CFG['use_soft_f1_loss'] else '❌'}")

# --- Key Results ---
print(f"\n  {'='*60}")
print(f"  KEY RESULTS")
print(f"  {'='*60}")
print(f"  Optimal threshold (t_opt):        {OPTIMAL_THRESHOLD:.4f}")
print(f"  Best OOF F1 (at t_opt):           {BEST_OOF_F1:.5f}")
print(f"  OOF F1@0.5 (no optimization):     {f1_at_05:.5f}")
print(f"  Gain from threshold optimization: {BEST_OOF_F1 - f1_at_05:+.5f}")
print(f"  Best model:                       {best_label}")
print(f"  Test predicted pos rate:          {pos_rate_pred:.2f}%")
print(f"  Test predicted pos count:         {n_pos_pred:,}")

# --- Version Comparison ---
print(f"\n  {'='*60}")
print(f"  CROSS-VERSION COMPARISON")
print(f"  {'='*60}")
print(f"  {'Version':15s} {'LB F1':>10s} {'OOF F1':>10s} {'Strategy':>40s}")
print(f"  {'-'*75}")
print(f"  {'V2':15s} {'0.22576':>10s} {'0.2872':>10s} {'CB-only':>40s}")
print(f"  {'V3':15s} {'0.19233':>10s} {'0.2910':>10s} {'SMOTE+3models':>40s}")
print(f"  {'V5':15s} {'---':>10s} {'0.0675':>10s} {'SMOTE+class_weights kills all':>40s}")
print(f"  {'V6 (S2)':15s} {'---':>10s} {'0.2937→?':>10s} {'PureLogLoss+Adv+Pseudo':>40s}")
print(f"  {'▶ ALPHA':15s} {'---':>10s} {f'{BEST_OOF_F1:.4f}':>10s} "
      f"{'PCA+Shuffle+SoftF1+RankShift':>40s}")

# --- Target ---
print(f"\n  {'='*60}")
print(f"  TARGET")
print(f"  {'='*60}")
print(f"  Top team LB F1:       0.30103+")
print(f"  V2 best LB F1:        0.22576")
print(f"  Alpha OOF F1:         {BEST_OOF_F1:.5f}")
print(f"  Alpha target LB:      0.30–0.33")

# --- Debug Summary ---
dbg.summary()

total_time = time.time() - T_START
print(f"\n  {'='*60}")
print(f"  🏆 GRANDMASTER BLUEPRINT FINAL — COMPLETE ✅")
print(f"  Total runtime: {total_time/60:.1f} min ({total_time:.0f}s)")
print(f"  Submit: submission.csv (continuous rank-shifted probabilities)")
print(f"  t_opt = {OPTIMAL_THRESHOLD:.4f} mapped to Kaggle's 0.5 cutoff")
print(f"  {'='*60}")