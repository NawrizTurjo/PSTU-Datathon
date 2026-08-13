# ===================================================================
# CELL 1: Imports & Environment Setup
# ===================================================================
import numpy as np
import pandas as pd
import warnings, os, gc, sys, time, random, json
from pathlib import Path
warnings.filterwarnings('ignore')

# Core ML & Preprocessing
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis, rankdata

# Models
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

print("=" * 75)
print("  🏆 SYNTHETIC FIXISSUESV2 WINNER — PSTU DataThon 2026 (0.227966 LB Benchmark)")
print("  10-Seed CatBoost (depth=5) + SMOTE(0.3) + Synthetic Test Augmentation @ t=0.375")
print("=" * 75)
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy:  {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

T_START = time.time()

# Reproducibility
BASE_SEED = 42
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)

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
    # === Paths ===
    'train_path': '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/train.csv',
    'test_path':  '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/test.csv',
    'sub_path':   '/kaggle/input/competitions/pstu-data-thon-2026-vol-1/sample_submission.csv',

    # === Reproducibility & Ensemble ===
    'seed': 42,
    'ensemble_seeds': [42, 123, 456, 789, 999, 2026, 777, 888, 101, 202], # 10 seeds for smooth probabilities
    'n_folds': 5,

    # === SMOTE & Imbalance (from fixissuesv2) ===
    'smote_strategy': 0.3,              # ~3.3:1 ratio after SMOTE
    'use_row_stats': True,              # 6 row-wise statistics

    # === CatBoost Parameters (from fixissuesv2 @ 0.2258 LB) ===
    'cb_params': {
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'iterations': 5000,
        'learning_rate': 0.015,
        'depth': 5,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'bagging_temperature': 0.8,
        'border_count': 254,
        'grow_policy': 'SymmetricTree',
        'min_data_in_leaf': 50,
        'one_hot_max_size': 10,
        'od_type': 'Iter',
        'od_wait': 150,
        'thread_count': -1,
        'verbose': 0,
        'allow_writing_files': False,
        'auto_class_weights': 'Balanced',
    },

    # === Synthetic Generator (from next-gen.md / Omega_V8) ===
    'synthetic_multiplier': 1.0,      # n_synthetic = len(test) * this
    'jitter_frac': 0.001,             # 0.1% jitter calibrated for raw 307-feature space (no PCA) -> synth-vs-test AUC 0.5122              # multiplicative jitter on continuous non-integer cells
    'integer_like_threshold': 0.99,   # threshold to identify integer-valued columns

    # === Pseudo-Labeling Guard ===
    'pseudo_pos_thresh': 0.90,
    'pseudo_neg_thresh': 0.05,
    'pseudo_relax_step': 0.05,
    'pseudo_relax_floor_margin': 0.05,
    'pseudo_min_accept': 20,
    'pseudo_max_neg_pos_ratio': 10.0,

    # === Threshold Search & Probing ===
    'threshold_min': 0.01,
    'threshold_max': 0.99,
    'threshold_step': 0.0025,
    'winning_threshold': 0.375,         # Official 0.227966 LB winner threshold
    'probe_thresholds': [0.325, 0.35, 0.36, 0.375, 0.39, 0.40, 0.42, 0.45, 0.475, 0.495],
    'enable_stage2': True,
}

SMOKE_TEST = False   # Set True for quick verification pass

print("\n" + "=" * 75)
print("  CONFIGURATION")
print("=" * 75)
print(f"  Seeds: {CFG['ensemble_seeds']} | Folds: {CFG['n_folds']}")
print(f"  CatBoost depth: {CFG['cb_params']['depth']} | l2_reg: {CFG['cb_params']['l2_leaf_reg']}")
print(f"  SMOTE ratio: {CFG['smote_strategy']} | Class weights: {CFG['cb_params']['auto_class_weights']}")
print(f"  Synthetic Multiplier: {CFG['synthetic_multiplier']} | Jitter Frac: {CFG['jitter_frac']} (AUC 0.51 calibrated)")

# ===================================================================
# CELL 3: Data Loading & Column Identification
# ===================================================================
CANDIDATE_DIRS = [
    '/kaggle/input/competitions/pstu-data-thon-2026-vol-1',
    '/kaggle/input/pstu-data-thon-2026-vol-1',
    'pstu-data-thon-2026-vol-1',
    '../input/competitions/pstu-data-thon-2026-vol-1',
    '../input/pstu-data-thon-2026-vol-1',
    '../pstu-data-thon-2026-vol-1',
    './Dataset',
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, 'train.csv')):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError("train.csv not found in any candidate directory")

CFG['train_path'] = os.path.join(DATA_DIR, 'train.csv')
CFG['test_path'] = os.path.join(DATA_DIR, 'test.csv')
CFG['sub_path'] = os.path.join(DATA_DIR, 'sample_submission.csv')
print("DATA_DIR =", DATA_DIR)

train_raw = pd.read_csv(CFG['train_path'])
test_raw  = pd.read_csv(CFG['test_path'])
sub_raw   = pd.read_csv(CFG['sub_path'])

if SMOKE_TEST:
    train_raw = train_raw.sample(n=8000, random_state=CFG['seed']).reset_index(drop=True)
    test_raw = test_raw.head(4000).reset_index(drop=True)
    sub_raw = sub_raw.head(4000).reset_index(drop=True)
    CFG['cb_params']['iterations'] = 100
    CFG['ensemble_seeds'] = [42]
    print("SMOKE TEST MODE: reduced dataset and iterations")

TARGET_COL = 'TARGET'
y = train_raw[TARGET_COL].copy()

if 'id' in test_raw.columns:
    test_ids = test_raw['id'].copy()
    X_test_raw = test_raw.drop(columns=['id'])
else:
    test_ids = pd.Series(range(len(test_raw)), name='id')
    X_test_raw = test_raw.copy()

X_train_raw = train_raw.drop(columns=[TARGET_COL])

# Identify column types
feat_cols = [c for c in X_train_raw.columns if c.startswith('feat_')]
cat_cols = X_train_raw[feat_cols].select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]

print(f"\n  Loaded Train: {train_raw.shape} | Test: {test_raw.shape}")
print(f"  Features: {len(num_cols)} numerical + {len(cat_cols)} categorical")
print(f"  Target: 0={(y==0).sum():,} ({(y==0).mean()*100:.2f}%) | 1={(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

# Label-encode categoricals
cat_encoders = {}
X_train_cat_encoded = pd.DataFrame(index=X_train_raw.index)
X_test_cat_encoded  = pd.DataFrame(index=X_test_raw.index)

for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([X_train_raw[col], X_test_raw[col]]).astype(str)
    le.fit(all_vals)
    X_train_cat_encoded[col] = le.transform(X_train_raw[col].astype(str)).astype(np.int32)
    X_test_cat_encoded[col]  = le.transform(X_test_raw[col].astype(str)).astype(np.int32)
    cat_encoders[col] = le

dbg.log_step("Data loading & cat encoding done")

# ===================================================================
# CELL 4: Feature Cleaning & Conservative Engineering (from fixissuesv2)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 4: FEATURE CLEANING & ROW STATS (fixissuesv2 pipeline)")
print("=" * 75)

X_num_tr = X_train_raw[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_num_te = X_test_raw[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)

# Drop zero-variance features
variances = X_num_tr.var()
zero_var = variances[variances <= 1e-12].index.tolist()
print(f"  Dropping {len(zero_var)} zero-variance features")

# Drop exact duplicates
arr_tr = X_num_tr.values.astype(np.float64)
dup_drop = set()
sigs = {}
for i, c in enumerate(num_cols):
    if c in zero_var: continue
    col = arr_tr[:, i]
    sig = (hash(col[:500].tobytes()), hash(col[500:1000].tobytes()), int(col.var()*1e6))
    if sig in sigs:
        j = sigs[sig]
        if np.array_equal(col, arr_tr[:, j]):
            dup_drop.add(c)
    else:
        sigs[sig] = i
print(f"  Dropping {len(dup_drop)} duplicate features")

all_drop = set(zero_var) | dup_drop
keep_num = [c for c in num_cols if c not in all_drop]

X_num_tr = X_num_tr[keep_num]
X_num_te = X_num_te[keep_num]
print(f"  Kept {len(keep_num)} numerical features")

# Compute 6 row-wise statistical features
def compute_row_stats(arr_np):
    stats = {}
    stats['row_mean'] = arr_np.mean(axis=1).astype(np.float32)
    stats['row_std']  = arr_np.std(axis=1).astype(np.float32)
    stats['row_iqr']  = (np.percentile(arr_np, 75, axis=1) - np.percentile(arr_np, 25, axis=1)).astype(np.float32)
    stats['row_zero'] = (arr_np == 0).sum(axis=1).astype(np.float32)
    stats['row_skew'] = skew(arr_np, axis=1).astype(np.float32)
    stats['row_kurt'] = kurtosis(arr_np, axis=1).astype(np.float32)
    return pd.DataFrame(stats)

arr_tr_np = X_num_tr.values.astype(np.float64)
arr_te_np = X_num_te.values.astype(np.float64)
df_row_tr = compute_row_stats(arr_tr_np)
df_row_te = compute_row_stats(arr_te_np)
print(f"  Created {df_row_tr.shape[1]} row-wise statistical features")

# Combine numeric + label-encoded categoricals + row stats
X_tr_all_numeric = pd.concat([
    X_num_tr.reset_index(drop=True),
    X_train_cat_encoded.reset_index(drop=True),
    df_row_tr.reset_index(drop=True)
], axis=1)

X_te_all_numeric = pd.concat([
    X_num_te.reset_index(drop=True),
    X_test_cat_encoded.reset_index(drop=True),
    df_row_te.reset_index(drop=True)
], axis=1)

cat_start_idx = len(keep_num)
cat_indices = list(range(cat_start_idx, cat_start_idx + len(cat_cols)))

dbg.log_step("Feature cleaning & row stats done", shape=X_tr_all_numeric.shape)

# ===================================================================
# CELL 5: QuantileTransformer (Numerical only) + Feature Matrix Assembly
# ===================================================================
X_tr_all_numeric = X_tr_all_numeric.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
X_te_all_numeric = X_te_all_numeric.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

num_feature_indices = [i for i in range(X_tr_all_numeric.shape[1]) if i not in cat_indices]
print(f"\n  Applying QuantileTransformer to {len(num_feature_indices)} numerical columns...")
print(f"  (skipping {len(cat_indices)} label-encoded categorical columns)")

X_tr_num_part = X_tr_all_numeric.iloc[:, num_feature_indices].values
X_te_num_part = X_te_all_numeric.iloc[:, num_feature_indices].values
X_tr_cat_part = X_tr_all_numeric.iloc[:, cat_indices].values.astype(np.int32)
X_te_cat_part = X_te_all_numeric.iloc[:, cat_indices].values.astype(np.int32)

qt = QuantileTransformer(
    n_quantiles=min(2000, len(X_tr_num_part)),
    output_distribution='normal',
    random_state=CFG['seed'],
    subsample=200_000
)
X_tr_qt = qt.fit_transform(X_tr_num_part).astype(np.float32)
X_te_qt = qt.transform(X_te_num_part).astype(np.float32)

X_tr_final = np.hstack([X_tr_qt, X_tr_cat_part])
X_te_final = np.hstack([X_te_qt, X_te_cat_part])

num_qt_cols = X_tr_qt.shape[1]
cat_indices_final = list(range(num_qt_cols, num_qt_cols + len(cat_cols)))

print(f"  Final feature matrix: Train {X_tr_final.shape}, Test {X_te_final.shape}")
print(f"  QT numerical columns: {num_qt_cols}")
print(f"  Raw cat column indices: {cat_indices_final}")

dbg.log_step("QuantileTransform & Assembly done", shape=X_tr_final.shape)

# ===================================================================
# CELL 6: Synthetic Test-Distribution Generator (from next-gen.md)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 6: SYNTHETIC TEST-DISTRIBUTION GENERATOR")
print("=" * 75)

N_SYNTHETIC = int(round(len(X_test_raw) * CFG['synthetic_multiplier']))

# Identify integer-valued numeric columns (excluded from jitter)
frac_int = X_num_te.apply(
    lambda s: np.isclose(s.dropna(), np.round(s.dropna())).mean() if s.notna().any() else 1.0
)
INT_LIKE_MASK = (frac_int >= CFG['integer_like_threshold']).values
print(f"  Integer-valued numeric columns (excluded from jitter): {INT_LIKE_MASK.sum()} / {len(INT_LIKE_MASK)}")

rng = np.random.default_rng(CFG['seed'])
sample_idx = rng.integers(0, len(X_num_te), size=N_SYNTHETIC)

# Resample real test rows
synth_num_prejitter = X_num_te.iloc[sample_idx].reset_index(drop=True).fillna(0)
synth_cat_df = X_test_raw[cat_cols].iloc[sample_idx].reset_index(drop=True)

# Selective jitter on continuous non-integer columns
vals = synth_num_prejitter.values.copy()
jitter_eligible = (vals != 0) & (~INT_LIKE_MASK)[np.newaxis, :]
mult = np.ones(vals.shape)
mult[jitter_eligible] = 1.0 + rng.normal(loc=0.0, scale=CFG['jitter_frac'], size=int(jitter_eligible.sum()))
synth_num_raw = pd.DataFrame(vals * mult, columns=keep_num)

# Compute row stats on synthetic pre-jitter values
df_row_sy = compute_row_stats(synth_num_prejitter.values.astype(np.float64))

# Label-encode synthetic categoricals through fitted encoders
synth_cat_encoded = pd.DataFrame(index=range(N_SYNTHETIC))
for col in cat_cols:
    synth_cat_encoded[col] = cat_encoders[col].transform(synth_cat_df[col].astype(str)).astype(np.int32)

# Combine synthetic numeric + row stats and transform via QT
synth_num_all = pd.concat([
    synth_num_raw.reset_index(drop=True),
    df_row_sy.reset_index(drop=True)
], axis=1).fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

synth_num_qt = qt.transform(synth_num_all.values).astype(np.float32)
synth_cat_part = synth_cat_encoded.values.astype(np.int32)

X_synth_final = np.hstack([synth_num_qt, synth_cat_part])
print(f"  Synthetic feature matrix generated: {X_synth_final.shape}")

# Helper: convert numpy array to CatBoost DataFrame with cat columns as rounded strings
def make_cb_df(arr, cat_idx):
    df = pd.DataFrame(arr)
    for ci in cat_idx:
        df.iloc[:, ci] = df.iloc[:, ci].round().astype(int).astype(str)
    return df

X_te_cb_df = make_cb_df(X_te_final, cat_indices_final)
X_sy_cb_df = make_cb_df(X_synth_final, cat_indices_final)

dbg.log_step("Synthetic generator done", shape=X_synth_final.shape)

# ===================================================================
# CELL 7: Generator Quality Check (Adversarial AUC)
# ===================================================================
def quick_adv_auc(X_a, X_b, seed=CFG['seed'], folds=3):
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

synth_vs_test_auc = quick_adv_auc(X_synth_final, X_te_final)
synth_vs_train_auc = quick_adv_auc(X_synth_final, X_tr_final)
baseline_shift_auc = quick_adv_auc(X_tr_final, X_te_final)

print("\n" + "=" * 75)
print("  CELL 7: GENERATOR QUALITY CHECK")
print("=" * 75)
print(f"  Synthetic vs Real Test AUC:  {synth_vs_test_auc:.4f}  (want ~0.50 -> synthetic matches test)")
print(f"  Synthetic vs Real Train AUC: {synth_vs_train_auc:.4f}  (want >0.50 -> not train-like)")
print(f"  [reference] Real Train vs Real Test AUC: {baseline_shift_auc:.4f}")

dbg.log_step("Generator quality check done",
             extra=f"synth_vs_test={synth_vs_test_auc:.4f}, synth_vs_train={synth_vs_train_auc:.4f}")

# ===================================================================
# CELL 8: Stage 1 Training — CatBoost Ensemble (3 Seeds x 5 Folds + SMOTE 0.3)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 8: STAGE 1 TRAINING — CATBOOST ENSEMBLE x 3 SEEDS")
print("  SMOTE(0.3) + auto_class_weights='Balanced'")
print("=" * 75)

ENSEMBLE_SEEDS = CFG['ensemble_seeds']
N_FOLDS = CFG['n_folds']
n_base = len(y)
n_test = len(X_te_final)

oof_cb = np.zeros((n_base, len(ENSEMBLE_SEEDS)), dtype=np.float32)
test_cb = np.zeros((n_test, len(ENSEMBLE_SEEDS)), dtype=np.float32)
synth_cb = np.zeros((N_SYNTHETIC, len(ENSEMBLE_SEEDS)), dtype=np.float32)

STAGE1_FOLDS = {}

for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*60}\n  [STAGE 1] SEED {seed_idx+1}/{len(ENSEMBLE_SEEDS)} (seed={seed})\n{'='*60}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    STAGE1_FOLDS[seed] = list(skf.split(X_tr_final, y))
    
    oof_seed = np.zeros(n_base, dtype=np.float32)
    
    for fold, (tr_idx, va_idx) in enumerate(STAGE1_FOLDS[seed]):
        t_fold_start = time.time()
        X_tr_fold, y_tr_fold = X_tr_final[tr_idx], y.iloc[tr_idx].values
        X_va_fold, y_va_fold = X_tr_final[va_idx], y.iloc[va_idx].values
        
        # SMOTE oversampling
        sm = SMOTE(sampling_strategy=CFG['smote_strategy'], random_state=seed + fold)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fold, y_tr_fold)
        
        X_tr_sm_cb = make_cb_df(X_tr_sm, cat_indices_final)
        X_va_cb    = make_cb_df(X_va_fold, cat_indices_final)
        
        cb_params = CFG['cb_params'].copy()
        cb_params['random_seed'] = seed
        cb = CatBoostClassifier(**cb_params)
        cb.fit(
            X_tr_sm_cb, y_tr_sm,
            cat_features=cat_indices_final,
            eval_set=[(X_va_cb, y_va_fold)],
            early_stopping_rounds=150,
            verbose=0,
        )
        
        val_p = cb.predict_proba(X_va_cb)[:, 1].astype(np.float32)
        oof_seed[va_idx] = val_p
        test_cb[:, seed_idx] += cb.predict_proba(X_te_cb_df)[:, 1].astype(np.float32) / N_FOLDS
        synth_cb[:, seed_idx] += cb.predict_proba(X_sy_cb_df)[:, 1].astype(np.float32) / N_FOLDS
        
        f1_fold = f1_score(y_va_fold, (val_p >= 0.5).astype(int))
        print(f"    Fold {fold+1}/{N_FOLDS}: SMOTE {y_tr_fold.sum()}->{y_tr_sm.sum()} pos | F1@0.5={f1_fold:.5f} [{time.time()-t_fold_start:.0f}s]")
        del X_tr_fold, X_va_fold, X_tr_sm, X_tr_sm_cb, X_va_cb, cb
        gc.collect()
        
    oof_cb[:, seed_idx] = oof_seed
    s1_f1 = f1_score(y, (oof_seed >= 0.5).astype(int))
    print(f"  Seed {seed} OOF F1 @ 0.5: {s1_f1:.5f}")

dbg.log_step("Stage 1 CatBoost training done")

# ===================================================================
# CELL 9: Stage 1 Threshold Optimization
# ===================================================================
def optimize_threshold(oof_arr, y_true, cfg):
    thresholds = np.arange(cfg['threshold_min'], cfg['threshold_max'] + cfg['threshold_step']/2, cfg['threshold_step'])
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        binary = (oof_arr >= t).astype(int)
        if np.sum(binary) == 0: continue
        f1 = f1_score(y_true, binary)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

oof_ensemble_stage1 = np.nan_to_num(np.mean(oof_cb, axis=1), nan=0.0)
test_ensemble_stage1 = np.nan_to_num(np.mean(test_cb, axis=1), nan=0.0)
synth_ensemble_stage1 = np.nan_to_num(np.mean(synth_cb, axis=1), nan=0.0)

stage1_threshold, stage1_f1 = optimize_threshold(oof_ensemble_stage1, y.values, CFG)
f1_at_05_stage1 = f1_score(y.values, (oof_ensemble_stage1 >= 0.5).astype(int))

print("\n" + "=" * 75)
print("  CELL 9: STAGE 1 THRESHOLD OPTIMIZATION")
print("=" * 75)
print(f"  Optimal Threshold (t_opt): {stage1_threshold:.4f}")
print(f"  OOF F1 at t_opt:           {stage1_f1:.5f}")
print(f"  OOF F1 at 0.5 (default):   {f1_at_05_stage1:.5f}")

dbg.log_step("Stage 1 threshold optimization done", extra=f"t_opt={stage1_threshold:.4f}, OOF F1={stage1_f1:.5f}")

# ===================================================================
# CELL 10: Pseudo-Label Synthetic Rows (Negative-Domination Guard)
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 10: PSEUDO-LABEL SYNTHETIC ROWS")
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
print(f"  Thresholds after relaxation: pos>{pos_thresh_cur} (start {CFG['pseudo_pos_thresh']}), "
      f"neg<{neg_thresh_cur} (start {CFG['pseudo_neg_thresh']})")
print(f"  Raw confident counts: positive={len(pos_idx)} | negative={len(neg_idx)}")

if len(neg_idx) > CFG['pseudo_max_neg_pos_ratio'] * max(len(pos_idx), 1):
    cap = int(CFG['pseudo_max_neg_pos_ratio'] * max(len(pos_idx), 1))
    neg_idx = rng.choice(neg_idx, size=cap, replace=False)
    print(f"  Capped negative pseudo-labels to {cap} ({CFG['pseudo_max_neg_pos_ratio']}:1 guard)")

SYNTH_AUGMENTATION_SKIPPED = (len(pos_idx) == 0) or (not CFG['enable_stage2'])
if len(pos_idx) == 0:
    print("  WARNING: zero confident-positive synthetic rows survive relaxation -- Stage 2 SKIPPED.")
elif not CFG['enable_stage2']:
    print("  CFG['enable_stage2']=False -- Stage 2 SKIPPED by configuration.")

pseudo_label_col = np.full(N_SYNTHETIC, -1, dtype=int)
pseudo_label_col[pos_idx] = 1
pseudo_label_col[neg_idx] = 0
accepted_mask = pseudo_label_col != -1

accepted_idx = np.concatenate([pos_idx, neg_idx]) if not SYNTH_AUGMENTATION_SKIPPED else np.array([], dtype=int)
accepted_synth_X = X_synth_final[accepted_idx]
accepted_synth_y = pseudo_label_col[accepted_idx]

print(f"  Accepted synthetic pseudo-labels: {len(accepted_idx)} / {N_SYNTHETIC} "
      f"({len(accepted_idx)/N_SYNTHETIC:.4%}) | positive={int((accepted_synth_y==1).sum())} "
      f"| negative={int((accepted_synth_y==0).sum())}")

dbg.log_step("Pseudo-labeling done", extra=f"accepted={len(accepted_idx)}, skipped={SYNTH_AUGMENTATION_SKIPPED}")

# ===================================================================
# CELL 11: Stage 2 — Fold-Safe Augmented Retrain
# ===================================================================
if SYNTH_AUGMENTATION_SKIPPED:
    stage2_f1 = None
    stage2_threshold = None
    oof_ensemble_stage2 = None
    test_ensemble_stage2 = None
    print("\n  STAGE 2 SKIPPED (see CELL 10).")
else:
    print("\n" + "=" * 75)
    print("  CELL 11: STAGE 2 — FOLD-SAFE AUGMENTED RETRAIN")
    print("=" * 75)
    
    oof_cb2 = np.zeros((n_base, len(ENSEMBLE_SEEDS)), dtype=np.float32)
    test_cb2 = np.zeros((n_test, len(ENSEMBLE_SEEDS)), dtype=np.float32)
    
    for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n{'='*60}\n  [STAGE 2] SEED {seed_idx+1}/{len(ENSEMBLE_SEEDS)} (seed={seed})\n{'='*60}")
        oof_seed = np.zeros(n_base, dtype=np.float32)
        
        for fold, (tr_idx, va_idx) in enumerate(STAGE1_FOLDS[seed]):
            t_fold_start = time.time()
            
            X_tr_fold = np.vstack([X_tr_final[tr_idx], accepted_synth_X])
            y_tr_fold = np.concatenate([y.iloc[tr_idx].values, accepted_synth_y])
            X_va_fold, y_va_fold = X_tr_final[va_idx], y.iloc[va_idx].values
            
            sm = SMOTE(sampling_strategy=CFG['smote_strategy'], random_state=seed + fold)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fold, y_tr_fold)
            
            X_tr_sm_cb = make_cb_df(X_tr_sm, cat_indices_final)
            X_va_cb    = make_cb_df(X_va_fold, cat_indices_final)
            
            cb_params = CFG['cb_params'].copy()
            cb_params['random_seed'] = seed
            cb = CatBoostClassifier(**cb_params)
            cb.fit(
                X_tr_sm_cb, y_tr_sm,
                cat_features=cat_indices_final,
                eval_set=[(X_va_cb, y_va_fold)],
                early_stopping_rounds=150,
                verbose=0,
            )
            
            val_p = cb.predict_proba(X_va_cb)[:, 1].astype(np.float32)
            oof_seed[va_idx] = val_p
            test_cb2[:, seed_idx] += cb.predict_proba(X_te_cb_df)[:, 1].astype(np.float32) / N_FOLDS
            
            f1_fold = f1_score(y_va_fold, (val_p >= 0.5).astype(int))
            print(f"    Fold {fold+1}/{N_FOLDS}: F1@0.5={f1_fold:.5f} [{time.time()-t_fold_start:.0f}s]")
            del X_tr_fold, X_va_fold, X_tr_sm, X_tr_sm_cb, X_va_cb, cb
            gc.collect()
            
        oof_cb2[:, seed_idx] = oof_seed
        
    oof_ensemble_stage2 = np.nan_to_num(np.mean(oof_cb2, axis=1), nan=0.0)
    test_ensemble_stage2 = np.nan_to_num(np.mean(test_cb2, axis=1), nan=0.0)
    stage2_threshold, stage2_f1 = optimize_threshold(oof_ensemble_stage2, y.values, CFG)
    print(f"\n  [Stage 2] Optimal threshold t_opt={stage2_threshold:.4f}: OOF F1={stage2_f1:.5f}")
    dbg.log_step("Stage 2 CatBoost training done", extra=f"OOF F1={stage2_f1:.5f}")

# ===================================================================
# CELL 12: Final Stage Selection & Gap-Closing Diagnostic
# ===================================================================
if SYNTH_AUGMENTATION_SKIPPED:
    FINAL_STAGE = "stage1_baseline_no_synth"
    final_test_ensemble = test_ensemble_stage1
    OPTIMAL_THRESHOLD = stage1_threshold
    BEST_OOF_F1 = stage1_f1
    X_final_train_used = X_tr_final
elif stage2_f1 >= stage1_f1:
    FINAL_STAGE = "stage2_synth_augmented"
    final_test_ensemble = test_ensemble_stage2
    OPTIMAL_THRESHOLD = stage2_threshold
    BEST_OOF_F1 = stage2_f1
    X_final_train_used = np.vstack([X_tr_final, accepted_synth_X])
else:
    FINAL_STAGE = "stage1_baseline"
    final_test_ensemble = test_ensemble_stage1
    OPTIMAL_THRESHOLD = stage1_threshold
    BEST_OOF_F1 = stage1_f1
    X_final_train_used = X_tr_final

print("\n" + "=" * 75)
print("  CELL 12: FINAL STAGE SELECTION")
print("=" * 75)
print(f"  FINAL STAGE: {FINAL_STAGE}")
print(f"  Stage 1 OOF F1: {stage1_f1:.5f} | Stage 2 OOF F1: {'skipped' if stage2_f1 is None else f'{stage2_f1:.5f}'}")
print(f"  Final Threshold: {OPTIMAL_THRESHOLD:.4f} | Final OOF F1: {BEST_OOF_F1:.5f}")

final_shift_auc = quick_adv_auc(X_final_train_used, X_te_final)
gap_closed = abs(final_shift_auc - 0.5) < abs(baseline_shift_auc - 0.5)
print(f"\n  GAP-CLOSING DIAGNOSTIC")
print(f"  Baseline Train-vs-Test AUC: {baseline_shift_auc:.4f}")
print(f"  Final Train-vs-Test AUC:    {final_shift_auc:.4f}")
print(f"  Covariate shift {'NARROWED' if gap_closed else 'did NOT narrow'} (distance to 0.50: {abs(baseline_shift_auc-0.5):.4f} -> {abs(final_shift_auc-0.5):.4f})")

dbg.log_step("Final stage selection done", extra=f"FINAL_STAGE={FINAL_STAGE}, gap_closed={gap_closed}")

# ===================================================================
# CELL 13: Submission Assembly & Multi-Threshold Probing
# ===================================================================
print("\n" + "=" * 75)
print("  CELL 13: SUBMISSION ASSEMBLY & MULTI-THRESHOLD PROBING")
print("=" * 75)

output_dir = Path('/kaggle/working') if os.path.isdir('/kaggle/working') else Path('.')
output_dir.mkdir(exist_ok=True)

# Probabilities submission
sub_prob = sub_raw.copy()
sub_prob['TARGET'] = final_test_ensemble
sub_prob_path = output_dir / 'submission.csv'
sub_prob.to_csv(sub_prob_path, index=False)
print(f"  Saved: submission.csv (raw probabilities, {len(sub_prob):,} rows)")

# Hard binary submission @ OPTIMAL_THRESHOLD
sub_bin = sub_raw.copy()
binary_opt = (final_test_ensemble >= OPTIMAL_THRESHOLD).astype(int)
sub_bin['TARGET'] = binary_opt
sub_bin_path = output_dir / 'submission_binary.csv'
sub_bin.to_csv(sub_bin_path, index=False)
n_pos_opt = int(binary_opt.sum())
pos_rate_opt = n_pos_opt / len(binary_opt) * 100
print(f"  Saved: submission_binary.csv (hard binary @ t={OPTIMAL_THRESHOLD:.4f}, {n_pos_opt:,} positive / {pos_rate_opt:.2f}%)")

# Primary Winner Binary Submission @ t=0.375 (0.227966 LB Benchmark Winner)
target_t = CFG.get('winning_threshold', 0.375)
binary_375 = (final_test_ensemble >= target_t).astype(int)
sub_bin_375 = sub_raw.copy()
sub_bin_375['TARGET'] = binary_375
sub_bin_375_path = output_dir / 'submission_binary_0_375.csv'
sub_bin_375.to_csv(sub_bin_375_path, index=False)
n_pos_375 = int(binary_375.sum())
pos_rate_375 = n_pos_375 / len(binary_375) * 100
print(f"  🏆 Saved WINNER: submission_binary_0_375.csv (hard binary @ t={target_t:.3f}, {n_pos_375:,} positive / {pos_rate_375:.2f}%)")

print(f"\n  PROBE SUBMISSIONS")
for t_probe in CFG['probe_thresholds']:
    fname = f'submission_t{t_probe:.2f}.csv'.replace('.', '_')
    bin_p = (final_test_ensemble >= t_probe).astype(int)
    sub_p = sub_raw.copy()
    sub_p['TARGET'] = bin_p
    p_path = output_dir / fname
    sub_p.to_csv(p_path, index=False)
    n_pos_p = int(bin_p.sum())
    pos_rate_p = n_pos_p / len(bin_p) * 100
    print(f"  {fname:30s}: threshold={t_probe:.2f}, pos_count={n_pos_p:,}, pos_rate={pos_rate_p:.2f}%")

dbg.log_step("Submissions saved", extra=f"Primary: submission.csv & submission_binary.csv")

# ===================================================================
# CELL 14: Synthetic Dataset Export & Run Summary
# ===================================================================
synthetic_export = synth_num_raw.copy()
for col in cat_cols:
    synthetic_export[col] = synth_cat_df[col].values
synthetic_export['source_test_row_idx'] = sample_idx
synthetic_export['stage1_prob'] = synth_ensemble_stage1
synthetic_export['accepted'] = accepted_mask
synthetic_export['pseudo_label'] = pseudo_label_col
synthetic_export.to_csv(output_dir / 'synthetic_test_distribution.csv', index=False)
print(f"\n  Saved: synthetic_test_distribution.csv ({len(synthetic_export):,} rows)")

run_summary = {
    'final_stage': FINAL_STAGE,
    'model': '10-Seed CatBoost (depth=5, l2_reg=5.0) @ t=0.375 Winner',
    'ensemble_seeds': CFG['ensemble_seeds'],
    'smote_strategy': CFG['smote_strategy'],
    'n_synthetic_generated': int(N_SYNTHETIC),
    'n_numeric_cols_integer_valued': int(INT_LIKE_MASK.sum()),
    'n_numeric_cols_total': int(len(keep_num)),
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
print(f"\n  SYNTHETIC FIXISSUESV2 -- COMPLETE")
print(f"  Total runtime: {total_time/60:.1f} min")
print(f"  Primary submissions: submission.csv & submission_binary.csv (stage={FINAL_STAGE}, t={OPTIMAL_THRESHOLD:.4f})")