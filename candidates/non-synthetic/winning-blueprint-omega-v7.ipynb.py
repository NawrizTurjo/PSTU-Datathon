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
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import skew as skew_fn, kurtosis as kt_fn

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

print("=" * 75)
print("  🏆 WINNING BLUEPRINT OMEGA — The #1 Spot Strategy")
print("  Pure FE | QT + PCA(ddof=0) | Santander FE | Multi-Encode Cats")
print("  XGBoost + LightGBM | scale_pos_weight | LogLoss | Hard Binary")
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
# CELL 2: Configuration — Omega Strategy
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
    # XGBoost — native LogLoss, scale_pos_weight for imbalance
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

    # LightGBM — native LogLoss, scale_pos_weight for imbalance
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
    # Extra thresholds for LB probing
    'probe_thresholds': [0.20, 0.25, 0.30, 0.35, 0.40],
}

KNOWN_STRING_CATS = ['feat_142', 'feat_157', 'feat_318', 'feat_320', 'feat_325', 'feat_337']

print("\n" + "=" * 75)
print("  CONFIGURATION — Omega Strategy")
print("=" * 75)
print(f"  Ensemble seeds:          {CFG['ensemble_seeds']}")
print(f"  CV folds:                {CFG['n_folds']}")
print(f"  PCA variance:            {CFG['pca_variance_threshold']} (ddof=0)")
print(f"  XGB scale_pos_weight:    {CFG['xgb_params']['scale_pos_weight']}")
print(f"  LGB scale_pos_weight:    {CFG['lgb_params']['scale_pos_weight']}")
print(f"  Loss function:           Standard LogLoss (NO custom F1 loss)")
print(f"  Upsampling:              NONE (scale_pos_weight only)")
print(f"  CatBoost:                EXCLUDED (XGB+LGB only)")
print(f"  Submission:              HARD BINARY (integer 0/1)")
print(f"  Probe thresholds:        {CFG['probe_thresholds']}")
print(f"  Total fits:              {len(CFG['ensemble_seeds']) * CFG['n_folds'] * 2}")

# ===================================================================
# CELL 3: Data Loading & Purification (STEP 1a)
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
print(f"  Numerical features: {len(numerical_features)}")
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

print(f"  Zero-variance features: {len(zero_var_features)}")
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
print(f"\n  ⚠️  NO covariate shift clipping — keeping all test values as-is")

X_train_num = train_raw[keep_numerical].fillna(0).astype(np.float32)
X_test_num  = test_raw[keep_numerical].fillna(0).astype(np.float32)

n_train = X_train_num.shape[0]
n_test  = X_test_num.shape[0]
n_feats = X_train_num.shape[1]

dbg.log_step("Data purification done",
             extra=f"Dropped {len(all_drop_features)} redundant "
                   f"({len(zero_var_features)} zero-var + {len(dup_features_to_drop)} dup), "
                   f"NO clipping")

# ===================================================================
# CELL 4: Categorical Engineering (STEP 1b)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 4: CATEGORICAL ENGINEERING — LE + FREQ + TARGET ENCODING")
print("=" * 75)

# --- 4A: Prepare categorical values ---
train_cats = {}
test_cats = {}

for c in string_cat_features:
    train_cats[c] = train_raw[c].fillna('MISSING').astype(str).values
    test_cats[c]  = test_raw[c].fillna('MISSING').astype(str).values

# --- 4B: Label Encoding (fit on train+test combined) ---
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

print(f"  ✓ Label Encoding: {len(string_cat_features)} features")

# --- 4C: Frequency Encoding ---
train_cats_freq = pd.DataFrame(index=train_raw.index)
test_cats_freq  = pd.DataFrame(index=test_raw.index)

for c in string_cat_features:
    all_vals = np.concatenate([train_cats[c], test_cats[c]])
    val_counts = pd.Series(all_vals).value_counts()
    freq_map = np.log1p(val_counts).to_dict()

    train_cats_freq[f'{c}_freq'] = pd.Series(train_cats[c]).map(freq_map).fillna(0).astype(np.float32).values
    test_cats_freq[f'{c}_freq']  = pd.Series(test_cats[c]).map(freq_map).fillna(0).astype(np.float32).values

print(f"  ✓ Frequency Encoding: {len(string_cat_features)} features (log1p scale)")

# --- 4D: 5-Fold CV Target Encoding (smoothed) ---
train_cats_te = pd.DataFrame(index=train_raw.index)
test_cats_te  = pd.DataFrame(index=test_raw.index)
global_mean = y.mean()
smoothing = CFG['te_smoothing']

print(f"\n  --- Target Encoding (5-fold CV, smoothing={smoothing}) ---")

skf_te = StratifiedKFold(n_splits=CFG['te_n_folds'], shuffle=True, random_state=CFG['seed'])

for c in string_cat_features:
    train_vals = train_cats[c]
    train_te_col = np.zeros(len(y), dtype=np.float64)

    for tr_idx, va_idx in skf_te.split(train_raw, y):
        tr_series = pd.Series(train_vals[tr_idx])
        tr_y = y[tr_idx]

        # Category means on training fold
        cat_sum = pd.Series(tr_y).groupby(tr_series).sum()
        cat_count = tr_series.groupby(tr_series).count()

        # Smoothed: (n*mean + smoothing*global_mean) / (n + smoothing)
        smoothed_means = (cat_sum + smoothing * global_mean) / (cat_count + smoothing)

        # Map to validation fold
        va_series = pd.Series(train_vals[va_idx])
        train_te_col[va_idx] = va_series.map(smoothed_means).fillna(global_mean).values

    train_cats_te[f'{c}_te'] = train_te_col.astype(np.float32)

    # For test: use full training data smoothed means
    tr_series_full = pd.Series(train_vals)
    cat_sum_full = pd.Series(y).groupby(tr_series_full).sum()
    cat_count_full = tr_series_full.groupby(tr_series_full).count()
    smoothed_full = (cat_sum_full + smoothing * global_mean) / (cat_count_full + smoothing)

    test_series = pd.Series(test_cats[c])
    test_cats_te[f'{c}_te'] = test_series.map(smoothed_full).fillna(global_mean).astype(np.float32).values

print(f"  ✓ Target Encoding: {len(string_cat_features)} features (5-fold CV, smoothing={smoothing})")

print(f"\n  ▶ Cat encoding summary:")
print(f"    LE features:  {train_cats_le.shape[1]}")
print(f"    Freq features: {train_cats_freq.shape[1]}")
print(f"    TE features:  {train_cats_te.shape[1]}")
print(f"    TOTAL cat features: {train_cats_le.shape[1] + train_cats_freq.shape[1] + train_cats_te.shape[1]}")

dbg.log_step("Cat encoding done",
             extra=f"LE+{train_cats_le.shape[1]} Freq+{train_cats_freq.shape[1]} TE+{train_cats_te.shape[1]}")

# ===================================================================
# CELL 5: Santander-Style Feature Engineering (STEP 1c)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 5: SANTANDER-STYLE FEATURE ENGINEERING (STEP 1c)")
print("=" * 75)

X_all = np.vstack([X_train_num.values, X_test_num.values]).astype(np.float64)
print(f"  Combined data: {X_all.shape}")

# --- 5A: Zero Count per Row ---
if CFG['use_zero_counts']:
    print(f"\n  --- 5A: Zero Counts ---")
    n_zeros_train = np.sum(X_train_num.values == 0, axis=1).astype(np.float32)
    n_zeros_test  = np.sum(X_test_num.values == 0, axis=1).astype(np.float32)
    print(f"    Train zero count: mean={n_zeros_train.mean():.1f}, "
          f"med={np.median(n_zeros_train):.1f}")
else:
    n_zeros_train = np.zeros(n_train, dtype=np.float32)
    n_zeros_test  = np.zeros(n_test, dtype=np.float32)

# --- 5B: Sentinel Value Counts ---
sentinel_feats_train = []
sentinel_feats_test  = []
if CFG['use_sentinel_counts']:
    print(f"\n  --- 5B: Sentinel Counts ---")
    for sentinel in CFG['santander_sentinels']:
        s_train = np.sum(X_train_num.values == sentinel, axis=1).astype(np.float32)
        s_test  = np.sum(X_test_num.values == sentinel, axis=1).astype(np.float32)
        n_sentinel = np.sum(X_all == sentinel)
        print(f"    Sentinel {sentinel}: {n_sentinel:,} occurrences "
              f"({100*n_sentinel/X_all.size:.4f}% of cells)")
        sentinel_feats_train.append(s_train)
        sentinel_feats_test.append(s_test)

sentinel_train = np.column_stack(sentinel_feats_train).astype(np.float32) if sentinel_feats_train else np.zeros((n_train, 0), dtype=np.float32)
sentinel_test  = np.column_stack(sentinel_feats_test).astype(np.float32) if sentinel_feats_test else np.zeros((n_test, 0), dtype=np.float32)

# --- 5C: Row Moments on Non-Zero, Non-Sentinel Values ---
moment_feats_train = []
moment_feats_test  = []
moment_names = []

if CFG['use_row_moments']:
    print(f"\n  --- 5C: Row Moments (non-zero, non-sentinel) ---")
    X_tr = X_train_num.values
    X_te = X_test_num.values

    sentinel_mask_train = np.zeros(X_tr.shape, dtype=bool)
    sentinel_mask_test  = np.zeros(X_te.shape, dtype=bool)
    for s in CFG['santander_sentinels']:
        sentinel_mask_train |= (X_tr == s)
        sentinel_mask_test  |= (X_te == s)

    valid_train = (X_tr != 0) & (~sentinel_mask_train)
    valid_test  = (X_te != 0) & (~sentinel_mask_test)

    n_valid_train = valid_train.sum(axis=1)
    n_valid_test  = valid_test.sum(axis=1)

    masked_train = np.where(valid_train, X_tr, np.nan)
    masked_test  = np.where(valid_test, X_te, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        row_mean_train = np.nan_to_num(np.nanmean(masked_train, axis=1), nan=0.0).astype(np.float32)
        row_mean_test  = np.nan_to_num(np.nanmean(masked_test, axis=1), nan=0.0).astype(np.float32)

        row_std_train = np.nan_to_num(np.nanstd(masked_train, axis=1, ddof=0), nan=0.0).astype(np.float32)
        row_std_test  = np.nan_to_num(np.nanstd(masked_test, axis=1, ddof=0), nan=0.0).astype(np.float32)

        row_sum_train = np.nan_to_num(np.nansum(masked_train, axis=1), nan=0.0).astype(np.float32)
        row_sum_test  = np.nan_to_num(np.nansum(masked_test, axis=1), nan=0.0).astype(np.float32)

        row_skew_train = np.nan_to_num(skew_fn(masked_train, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)
        row_skew_test  = np.nan_to_num(skew_fn(masked_test, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)

        row_kurt_train = np.nan_to_num(kt_fn(masked_train, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)
        row_kurt_test  = np.nan_to_num(kt_fn(masked_test, axis=1, nan_policy='omit'), nan=0.0).astype(np.float32)

    moment_feats_train = [row_mean_train, row_std_train, row_sum_train,
                           row_skew_train, row_kurt_train, n_valid_train.astype(np.float32)]
    moment_feats_test  = [row_mean_test, row_std_test, row_sum_test,
                           row_skew_test, row_kurt_test, n_valid_test.astype(np.float32)]
    moment_names = ['row_mean', 'row_std', 'row_sum', 'row_skew', 'row_kurtosis', 'n_valid']
    print(f"    Generated {len(moment_names)} row-moment features")

# --- 5D: Value Uniqueness & Frequency Features ---
uniq_feats_train = []
uniq_feats_test  = []
uniq_names = []

if CFG['use_uniqueness_features']:
    print(f"\n  --- 5D: Value Uniqueness & Frequency ---")
    top_n = min(CFG['uniqueness_top_n'], n_feats)
    variances = np.var(X_all, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    print(f"    Analyzing top-{top_n} highest-variance features...")

    X_tr = X_train_num.values
    X_te = X_test_num.values

    for feat_idx in top_indices:
        col_all = X_all[:, feat_idx]
        unique_vals, counts = np.unique(col_all, return_counts=True)
        val_to_count = dict(zip(unique_vals, counts))
        singleton_vals = set(unique_vals[counts == 1])

        freq_train = np.array([np.log1p(val_to_count.get(v, 1)) for v in X_tr[:, feat_idx]], dtype=np.float32)
        freq_test  = np.array([np.log1p(val_to_count.get(v, 1)) for v in X_te[:, feat_idx]], dtype=np.float32)

        sing_train = np.array([1.0 if v in singleton_vals else 0.0 for v in X_tr[:, feat_idx]], dtype=np.float32)
        sing_test  = np.array([1.0 if v in singleton_vals else 0.0 for v in X_te[:, feat_idx]], dtype=np.float32)

        uniq_feats_train.extend([freq_train, sing_train])
        uniq_feats_test.extend([freq_test, sing_test])
        uniq_names.extend([f'freq_feat_{feat_idx}', f'singleton_feat_{feat_idx}'])

    print(f"    Generated {len(uniq_names)} uniqueness features ({top_n} features × 2)")

# --- 5E: Assemble Santander Features ---
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

del X_all, masked_train, masked_test; gc.collect()

dbg.log_step("Santander FE done", shape=X_santander_train.shape,
             extra=f"{X_santander_train.shape[1]} engineered features")

# ===================================================================
# CELL 6: QT → PCA(ddof=0) → Feature Assembly (STEP 1d)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 6: QT → PCA(ddof=0) → FEATURE ASSEMBLY (STEP 1d)")
print("=" * 75)

# --- 6A: QuantileTransformer ---
print(f"\n  --- 6A: QuantileTransformer ---")
X_tr_vals = X_train_num.values.astype(np.float64)
X_te_vals = X_test_num.values.astype(np.float64)

qt = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, n_train),
    random_state=CFG['seed'],
    subsample=200_000
)

X_train_qt = qt.fit_transform(X_tr_vals).astype(np.float32)
X_test_qt  = qt.transform(X_te_vals).astype(np.float32)

print(f"  QT train: {X_train_qt.shape}  range=[{X_train_qt.min():.3f}, {X_train_qt.max():.3f}]")
print(f"  QT test:  {X_test_qt.shape}  range=[{X_test_qt.min():.3f}, {X_test_qt.max():.3f}]")

dbg.log_step("QT done", shape=X_train_qt.shape)

# --- 6B: PCA with ddof=0 (Manual SVD) ---
print(f"\n  --- 6B: PCA with ddof=0 ({CFG['pca_variance_threshold']:.0%} variance) ---")

X_qt_all = np.vstack([X_train_qt, X_test_qt])
n_total = X_qt_all.shape[0]

qt_mean = X_qt_all.mean(axis=0)
X_centered = X_qt_all - qt_mean

print(f"    Computing SVD on {X_centered.shape}...")
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# ddof=0: divide by n (NOT n-1 like sklearn)
explained_variance_ddof0 = (S ** 2) / n_total
explained_variance_ratio = explained_variance_ddof0 / explained_variance_ddof0.sum()
cumsum_var = np.cumsum(explained_variance_ratio)

n_pca = int(np.searchsorted(cumsum_var, CFG['pca_variance_threshold']) + 1)
n_pca = min(n_pca, X_qt_all.shape[1])

print(f"    Variance explained (ddof=0):")
for k in [10, 25, 50, 100, n_pca]:
    if k <= len(cumsum_var):
        print(f"      Top-{k:3d}: {cumsum_var[k-1]:.4f} ({cumsum_var[k-1]*100:.1f}%)")

X_train_pca = (U[:n_train, :n_pca] * S[:n_pca]).astype(np.float32)
X_test_pca  = (U[n_train:, :n_pca] * S[:n_pca]).astype(np.float32)

print(f"\n    PCA components: {n_pca}/{X_qt_all.shape[1]} "
      f"(retains {cumsum_var[n_pca-1]*100:.1f}% variance)")
print(f"    PCA train: {X_train_pca.shape}  PCA test: {X_test_pca.shape}")

del X_qt_all, X_centered, U, S, Vt, X_train_qt, X_test_qt; gc.collect()

dbg.log_step("PCA done", shape=X_train_pca.shape, extra=f"{n_pca} components, ddof=0")

# --- 6C: Feature Assembly ---
# Combine: PCA + Santander + LE cats + Frequency cats + Target Encoding cats
cat_features_le = train_cats_le.values.astype(np.float32)
cat_features_te_le = test_cats_le.values.astype(np.float32)

cat_features_freq = train_cats_freq.values.astype(np.float32)
cat_features_te_freq = test_cats_freq.values.astype(np.float32)

cat_features_te_enc = train_cats_te.values.astype(np.float32)
cat_features_te_enc_test = test_cats_te.values.astype(np.float32)

X_tr_base = np.hstack([
    X_train_pca,
    X_santander_train,
    cat_features_le,
    cat_features_freq,
    cat_features_te_enc,
]).astype(np.float32)

X_te_base = np.hstack([
    X_test_pca,
    X_santander_test,
    cat_features_te_le,
    cat_features_te_freq,
    cat_features_te_enc_test,
]).astype(np.float32)

n_total_features = X_tr_base.shape[1]
print(f"\n  ▶ Final feature matrix: {X_tr_base.shape}")
print(f"    PCA components:        {n_pca}")
print(f"    Santander features:    {X_santander_train.shape[1]}")
print(f"    LE cat features:       {train_cats_le.shape[1]}")
print(f"    Frequency cat features:{train_cats_freq.shape[1]}")
print(f"    Target Encoding cats:  {train_cats_te.shape[1]}")
print(f"    TOTAL:                 {n_total_features}")

dbg.log_step("Feature assembly done", shape=X_tr_base.shape,
             extra=f"{n_total_features} features")

# ===================================================================
# CELL 7: Robust Training — XGBoost + LightGBM (STEP 2)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 7: ROBUST TRAINING — XGBOOST + LIGHTGBM (STEP 2)")
print("  NO custom loss | NO SMOTE | NO shuffling | scale_pos_weight only")
print("=" * 75)

n_seeds = len(CFG['ensemble_seeds'])
n_base = len(y)

# OOF storage
oof_xgb = np.zeros((n_base, n_seeds), dtype=np.float32)
oof_lgb = np.zeros((n_base, n_seeds), dtype=np.float32)

# Test prediction storage
n_test_final = len(test_raw)
test_xgb = np.zeros((n_test_final, n_seeds), dtype=np.float32)
test_lgb = np.zeros((n_test_final, n_seeds), dtype=np.float32)

fold_scores = {
    'xgb': {s: [] for s in CFG['ensemble_seeds']},
    'lgb': {s: [] for s in CFG['ensemble_seeds']},
}

print(f"\n{'='*75}")
print(f"  TRAINING: {n_seeds} seeds × {CFG['n_folds']} folds × 2 models = "
      f"{n_seeds * CFG['n_folds'] * 2} fits")
print(f"  Strategy: scale_pos_weight ONLY (no upsampling)")
print(f"{'='*75}")

for seed_idx, seed in enumerate(CFG['ensemble_seeds']):
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE SEED {seed} ({seed_idx+1}/{n_seeds})")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_base, y)):
        t_fold_start = time.time()
        print(f"\n  --- Fold {fold+1}/{CFG['n_folds']} ---")

        # Simple train/val split — NO upsampling, NO shuffling
        X_tr_fold = X_tr_base[tr_idx]
        y_tr_fold = y[tr_idx]
        X_va_fold = X_tr_base[va_idx]
        y_va_fold = y[va_idx]

        n_pos_tr = np.sum(y_tr_fold == 1)
        print(f"    Train: {X_tr_fold.shape[0]:,} rows, "
              f"pos={n_pos_tr:,} ({100*n_pos_tr/len(y_tr_fold):.2f}%)")
        print(f"    Valid: {X_va_fold.shape[0]:,} rows, "
              f"pos={np.sum(y_va_fold==1):,} ({100*np.mean(y_va_fold==1):.2f}%)")

        # ================================================================
        # XGBOOST — LogLoss, scale_pos_weight=4.0
        # ================================================================
        t0 = time.time()
        xgb_params = CFG['xgb_params'].copy()
        xgb_params['random_state'] = seed

        assert X_tr_fold.dtype == np.float32, f"XGB input must be float32"

        xgb = XGBClassifier(**xgb_params)
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
        # LIGHTGBM — LogLoss, scale_pos_weight=3.0
        # ================================================================
        t0 = time.time()
        lgb_params = CFG['lgb_params'].copy()
        lgb_params['random_state'] = seed

        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(X_tr_fold, y_tr_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                callbacks=[early_stopping(200), log_evaluation(0)])

        # Standard objective → 2D predict_proba
        oof_lgb[va_idx, seed_idx] = lgb.predict_proba(X_va_fold)[:, 1].astype(np.float32)
        test_lgb[:, seed_idx] += lgb.predict_proba(X_te_base)[:, 1].astype(np.float32) / CFG['n_folds']

        va_probs_lgb = oof_lgb[va_idx, seed_idx]
        f1_fold_lgb = f1_score(y_va_fold, (va_probs_lgb >= 0.5).astype(int))
        fold_scores['lgb'][seed].append(f1_fold_lgb)
        dt_lgb = time.time() - t0
        print(f"    [LightGBM seed={seed}] F1@0.5={f1_fold_lgb:.5f}  "
              f"mean_p={va_probs_lgb.mean():.4f}  pos@0.5={np.mean(va_probs_lgb >= 0.5)*100:.1f}%  [{dt_lgb:.0f}s]")

        del lgb; gc.collect()

        t_fold_total = time.time() - t_fold_start
        print(f"    Fold total: {t_fold_total:.0f}s")

    # Cumulative OOF for this seed
    s1_xgb = f1_score(y, (np.nan_to_num(oof_xgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    s1_lgb = f1_score(y, (np.nan_to_num(oof_lgb[:, seed_idx], nan=0.0) >= 0.5).astype(int))
    print(f"\n  Seed {seed} cumulative OOF@0.5: XGB={s1_xgb:.5f}  LGB={s1_lgb:.5f}")

elapsed_train = time.time() - T_START
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE — {n_seeds * CFG['n_folds'] * 2} models trained")
print(f"  Elapsed: {elapsed_train/60:.1f} min ({elapsed_train:.0f}s)")
print(f"{'='*60}")

dbg.log_step("Training done", extra=f"Elapsed: {elapsed_train/60:.1f} min")

# ===================================================================
# CELL 8: Threshold Optimization on OOF (STEP 3a)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 8: THRESHOLD OPTIMIZATION (STEP 3a)")
print("=" * 75)

# --- 8A: Average OOF across seeds ---
oof_xgb_avg = np.nan_to_num(np.mean(oof_xgb, axis=1), nan=0.0)
oof_lgb_avg = np.nan_to_num(np.mean(oof_lgb, axis=1), nan=0.0)

# Uniform ensemble: average all 6 seed-model combinations
all_oof_seeds = np.hstack([oof_xgb, oof_lgb])
oof_ensemble = np.nan_to_num(np.mean(all_oof_seeds, axis=1), nan=0.0)

print(f"\n  OOF prediction stats:")
for label, arr in [('XGBoost', oof_xgb_avg), ('LightGBM', oof_lgb_avg),
                    ('Ensemble', oof_ensemble)]:
    print(f"    {label:15s}: mean={np.mean(arr):.5f}  med={np.median(arr):.5f}  "
          f"p90={np.percentile(arr, 90):.4f}  p95={np.percentile(arr, 95):.4f}  "
          f"p99={np.percentile(arr, 99):.4f}")

# --- 8B: Exhaustive Threshold Search ---
thresholds = np.arange(CFG['threshold_min'], CFG['threshold_max'] + CFG['threshold_step']/2,
                        CFG['threshold_step'])

print(f"\n  Grid searching {len(thresholds)} thresholds [{CFG['threshold_min']:.4f}, "
      f"{CFG['threshold_max']:.4f}]...")

best_results = {}
for label, oof_arr in [
    ('XGBoost', oof_xgb_avg),
    ('LightGBM', oof_lgb_avg),
    ('Ensemble', oof_ensemble),
]:
    best_f1, best_t, best_prec, best_rec, best_pos = 0, 0.5, 0, 0, 0
    best_binary = None
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
print(f"  {'Model':15s} {'Best T':>10s} {'F1':>10s} {'Precision':>10s} {'Recall':>10s} {'Pos%':>8s}")
print(f"  {'-'*65}")
for label, res in best_results.items():
    print(f"  {label:15s} {res['threshold']:10.4f} {res['f1']:10.5f} "
          f"{res['precision']:10.4f} {res['recall']:10.4f} {res['pos_rate']:7.2f}%")

# --- 8C: Select Best ---
best_label = 'Ensemble'
best_f1 = best_results['Ensemble']['f1']
OPTIMAL_THRESHOLD = best_results['Ensemble']['threshold']

for label, res in best_results.items():
    if res['f1'] > best_f1:
        best_f1 = best_results[label]['f1']
        best_label = label
        OPTIMAL_THRESHOLD = res['threshold']

print(f"\n  ▶ SELECTED: {best_label}")
print(f"    Optimal threshold t_opt: {OPTIMAL_THRESHOLD:.4f}")
print(f"    OOF F1 at t_opt:         {best_f1:.5f}")

f1_at_05 = f1_score(y, (oof_ensemble >= 0.5).astype(int))
print(f"    F1@0.5 (baseline):       {f1_at_05:.5f}")
print(f"    Gain from optimization:  {best_f1 - f1_at_05:+.5f}")

BEST_OOF_F1 = best_f1

dbg.log_step("Threshold optimization done",
             extra=f"t_opt={OPTIMAL_THRESHOLD:.4f}, OOF F1={BEST_OOF_F1:.5f}")

# ===================================================================
# CELL 9: Hard Binary Submission + Multi-Threshold Probing (STEP 3b)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 9: HARD BINARY SUBMISSION + MULTI-THRESHOLD PROBING")
print("=" * 75)

# --- 9A: Average test predictions across all seeds ---
test_xgb_avg = np.mean(test_xgb, axis=1)
test_lgb_avg = np.mean(test_lgb, axis=1)
test_ensemble = np.mean(np.hstack([test_xgb, test_lgb]), axis=1)

print(f"\n  Test prediction stats:")
for label, probs in [
    ('XGBoost', test_xgb_avg),
    ('LightGBM', test_lgb_avg),
    ('Ensemble', test_ensemble),
]:
    pos_at_topt = np.mean(probs >= OPTIMAL_THRESHOLD) * 100
    pos_at_05   = np.mean(probs >= 0.5) * 100
    print(f"  {label:15s}: mean={np.mean(probs):.5f}  "
          f"pos@t_opt={pos_at_topt:.2f}%  pos@0.5={pos_at_05:.2f}%")

# --- 9B: Hard Binary Submission at t_opt ---
output_dir = Path('/kaggle/working')
output_dir.mkdir(exist_ok=True)

def make_submission(probs, threshold, filename):
    """Convert probabilities to hard binary 0/1 at given threshold."""
    binary = (probs >= threshold).astype(int)
    sub = sub_raw.copy()
    sub['TARGET'] = binary
    path = output_dir / filename
    sub.to_csv(path, index=False)
    n_pos_sub = np.sum(binary)
    pos_rate_sub = n_pos_sub / len(binary) * 100
    return path, n_pos_sub, pos_rate_sub

# Primary submission at t_opt
sub_path, n_pos_opt, pos_rate_opt = make_submission(
    test_ensemble, OPTIMAL_THRESHOLD, 'submission.csv'
)

print(f"\n  {'='*75}")
print(f"  PRIMARY SUBMISSION: submission.csv")
print(f"  {'='*75}")
print(f"  Threshold:    {OPTIMAL_THRESHOLD:.4f} (optimal OOF)")
print(f"  Class 0:      {len(test_ensemble) - n_pos_opt:,} ({100 - pos_rate_opt:.2f}%)")
print(f"  Class 1:      {n_pos_opt:,} ({pos_rate_opt:.2f}%)")
print(f"  Values:       INTEGER 0/1 (hard binary, no probabilities)")

# Verify hard binary
sub_check = pd.read_csv(sub_path)
assert sub_check['TARGET'].dtype == 'int64' or set(sub_check['TARGET'].unique()).issubset({0, 1}), \
    "Submission must be integer 0/1!"
print(f"  ✓ Verified:   Integer 0/1 only (unique values: {sorted(sub_check['TARGET'].unique())})")

# --- 9C: Multi-Threshold Probe Submissions ---
print(f"\n  {'='*75}")
print(f"  PROBE SUBMISSIONS (for LB exploration)")
print(f"  {'='*75}")

probe_thresholds = CFG['probe_thresholds']
for t_probe in probe_thresholds:
    fname = f'submission_t{t_probe:.2f}.csv'.replace('.', '_') if '.' in f'{t_probe}' else f'submission_t{t_probe:.2f}.csv'
    # Use standard naming: submission_t0_20.csv etc.
    fname = f'submission_t{t_probe:.2f}.csv'.replace('.', '_')
    path, n_pos, pos_rate = make_submission(test_ensemble, t_probe, fname)
    print(f"  {fname:30s}: threshold={t_probe:.2f}, "
          f"pos_count={n_pos:,}, pos_rate={pos_rate:.2f}%")

# Also generate for the optimal threshold's neighbors
neighbor_offsets = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
for offset in neighbor_offsets:
    t_neighbor = round(OPTIMAL_THRESHOLD + offset, 4)
    if t_neighbor <= 0 or t_neighbor >= 1.0:
        continue
    fname = f'submission_t{t_neighbor:.4f}.csv'.replace('.', '_')
    path, n_pos, pos_rate = make_submission(test_ensemble, t_neighbor, fname)
    # Only print if different enough from existing probes
    if abs(t_neighbor - OPTIMAL_THRESHOLD) < 0.04:
        print(f"  {fname:30s}: threshold={t_neighbor:.4f}, "
              f"pos_count={n_pos:,}, pos_rate={pos_rate:.2f}%")

# --- 9D: Summary ---
print(f"\n  {'='*75}")
print(f"  SUBMISSION SUMMARY")
print(f"  {'='*75}")
print(f"  All files saved to: {output_dir}")
all_subs = sorted(output_dir.glob('submission*.csv'))
for s in all_subs:
    size_kb = s.stat().st_size / 1024
    print(f"    {s.name:<40s} {size_kb:6.1f} KB")

dbg.log_step("Submissions saved",
             extra=f"Primary: submission.csv (t={OPTIMAL_THRESHOLD:.4f}), "
                   f"{len(all_subs)} total files")

# ===================================================================
# CELL 10: Grandmaster Performance Dashboard
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 10: OMEGA PERFORMANCE DASHBOARD")
print("=" * 75)

# --- Model Health ---
print(f"\n  {'='*60}")
print(f"  MODEL HEALTH CHECK")
print(f"  {'='*60}")

for model_name, oof_data in [('XGBoost', oof_xgb), ('LightGBM', oof_lgb)]:
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
print(f"  Santander features:               {X_santander_train.shape[1]}")
print(f"  LE cat features:                  {train_cats_le.shape[1]}")
print(f"  Frequency cat features:           {train_cats_freq.shape[1]}")
print(f"  Target Encoding cat features:     {train_cats_te.shape[1]}")
print(f"  {'─'*40}")
print(f"  TOTAL features:                   {n_total_features}")

# --- Training Summary ---
print(f"\n  {'='*60}")
print(f"  TRAINING SUMMARY")
print(f"  {'='*60}")
print(f"  Models:                           XGBoost + LightGBM (NO CatBoost)")
print(f"  Seeds × Folds:                    {n_seeds} × {CFG['n_folds']}")
print(f"  Models trained:                   {n_seeds * CFG['n_folds'] * 2}")
print(f"  Loss function:                    Standard LogLoss (binary:logistic / binary)")
print(f"  XGB scale_pos_weight:             {CFG['xgb_params']['scale_pos_weight']}")
print(f"  LGB scale_pos_weight:             {CFG['lgb_params']['scale_pos_weight']}")
print(f"  Upsampling:                       NONE")
print(f"  Custom objectives:                NONE")

# --- Key Results ---
print(f"\n  {'='*60}")
print(f"  KEY RESULTS")
print(f"  {'='*60}")
print(f"  Optimal threshold (t_opt):        {OPTIMAL_THRESHOLD:.4f}")
print(f"  Best OOF F1 (at t_opt):           {BEST_OOF_F1:.5f}")
print(f"  OOF F1@0.5 (no optimization):     {f1_at_05:.5f}")
print(f"  Gain from threshold optimization: {BEST_OOF_F1 - f1_at_05:+.5f}")
print(f"  Best model:                       {best_label}")
print(f"  Test pred pos rate (at t_opt):    {pos_rate_opt:.2f}%")
print(f"  Test pred pos count (at t_opt):   {n_pos_opt:,}")

# --- Version Comparison ---
print(f"\n  {'='*60}")
print(f"  CROSS-VERSION COMPARISON")
print(f"  {'='*60}")
print(f"  {'Version':15s} {'LB F1':>10s} {'OOF F1':>10s} {'Strategy':>40s}")
print(f"  {'-'*75}")
print(f"  {'V2':15s} {'0.22576':>10s} {'0.2872':>10s} {'CB-only':>40s}")
print(f"  {'V3':15s} {'0.19233':>10s} {'0.2910':>10s} {'SMOTE+3models':>40s}")
print(f"  {'V6 (S2)':15s} {'---':>10s} {'0.2937→?':>10s} {'PureLogLoss+Adv+Pseudo':>40s}")
print(f"  {'Alpha':15s} {'<0.20':>10s} {'---':>10s} {'SoftF1+Shuffle+RankShift (FAILED)':>40s}")
print(f"  {'▶ OMEGA':15s} {'---':>10s} {f'{BEST_OOF_F1:.4f}':>10s} "
      f"{'LogLoss+PosWeight+HardBinary':>40s}")

# --- Debug Summary ---
dbg.summary()

total_time = time.time() - T_START
print(f"\n  {'='*60}")
print(f"  🏆 WINNING BLUEPRINT OMEGA — COMPLETE ✅")
print(f"  Total runtime: {total_time/60:.1f} min ({total_time:.0f}s)")
print(f"  Primary submission: submission.csv (HARD BINARY 0/1 at t={OPTIMAL_THRESHOLD:.4f})")
print(f"  Probe submissions: {len(probe_thresholds) + len(neighbor_offsets)} threshold variants")
print(f"  {'='*60}")