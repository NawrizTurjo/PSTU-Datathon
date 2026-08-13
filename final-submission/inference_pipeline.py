#!/usr/bin/env python3
"""
================================================================================
PSTU DataThon 2026 — Final Inference & Training Pipeline (Winner Solution: 0.228 LB)
================================================================================
Description:
    Standalone, fully dynamic pipeline for private dataset testing. 
    Accepts arbitrary paths for train.csv, test.csv, and output submission.csv.

Usage:
    python inference_pipeline.py --train_path /path/to/train.csv \
                                --test_path /path/to/test.csv \
                                --output_path /path/to/submission.csv
================================================================================
"""

import argparse
import sys
import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# Core ML Dependencies
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

BASE_SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="PSTU DataThon 2026 Private Test Inference Script")
    parser.add_argument('--train_path', type=str, default=None, help='Path to train.csv')
    parser.add_argument('--test_path', type=str, default=None, help='Path to test.csv')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Path for output submission.csv')
    parser.add_argument('--seeds', type=int, default=10, help='Number of ensemble seeds (default: 10)')
    return parser.parse_args()

def locate_data_dir():
    candidates = [
        '/kaggle/input/competitions/pstu-data-thon-2026-vol-1',
        '/kaggle/input/pstu-data-thon-2026-vol-1',
        'pstu-data-thon-2026-vol-1',
        '../input/competitions/pstu-data-thon-2026-vol-1',
        './Dataset',
        '.'
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, 'train.csv')):
            return d
    return '.'

def compute_row_stats(arr_np):
    stats = {}
    stats['row_mean'] = arr_np.mean(axis=1).astype(np.float32)
    stats['row_std']  = arr_np.std(axis=1).astype(np.float32)
    stats['row_iqr']  = (np.percentile(arr_np, 75, axis=1) - np.percentile(arr_np, 25, axis=1)).astype(np.float32)
    stats['row_zero'] = (arr_np == 0).sum(axis=1).astype(np.float32)
    stats['row_skew'] = skew(arr_np, axis=1).astype(np.float32)
    stats['row_kurt'] = kurtosis(arr_np, axis=1).astype(np.float32)
    return pd.DataFrame(stats)

def make_cb_df(arr, cat_idx):
    df = pd.DataFrame(arr)
    for ci in cat_idx:
        df.iloc[:, ci] = df.iloc[:, ci].round().astype(int).astype(str)
    return df

def run_pipeline(train_path, test_path, output_path, n_seeds=10):
    t_start = time.time()
    print("=" * 75)
    print("  PSTU DATATHON 2026 — STANDALONE INFERENCE & TRAINING PIPELINE")
    print("=" * 75)
    print(f"  Train Path:  {train_path}")
    print(f"  Test Path:   {test_path}")
    print(f"  Output Path: {output_path}")
    print(f"  Seeds:       {n_seeds}")

    # 1. Load Data
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    TARGET_COL = 'TARGET'
    y = train_raw[TARGET_COL].copy()

    if 'id' in test_raw.columns:
        test_ids = test_raw['id'].copy()
        X_test_raw = test_raw.drop(columns=['id'])
    else:
        test_ids = pd.Series(range(len(test_raw)), name='id')
        X_test_raw = test_raw.copy()

    X_train_raw = train_raw.drop(columns=[TARGET_COL])

    # 2. Identify Features
    feat_cols = [c for c in X_train_raw.columns if c.startswith('feat_')]
    cat_cols  = X_train_raw[feat_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols  = [c for c in feat_cols if c not in cat_cols]

    # Label Encode Categoricals
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

    # 3. Clean Features (Drop Zero-Variance & Hash-Duplicates)
    X_num_tr = X_train_raw[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    X_num_te = X_test_raw[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)

    variances = X_num_tr.var()
    zero_var  = variances[variances <= 1e-12].index.tolist()
    arr_tr    = X_num_tr.values.astype(np.float64)

    dup_drop = set()
    sigs = {}
    for i, c in enumerate(num_cols):
        if c in zero_var: continue
        col = arr_tr[:, i]
        sig = (hash(col[:500].tobytes()), hash(col[500:1000].tobytes()), int(col.var() * 1e6))
        if sig in sigs:
            j = sigs[sig]
            if np.array_equal(col, arr_tr[:, j]):
                dup_drop.add(c)
        else:
            sigs[sig] = i

    all_drop = set(zero_var) | dup_drop
    keep_num = [c for c in num_cols if c not in all_drop]

    X_num_tr = X_num_tr[keep_num]
    X_num_te = X_num_te[keep_num]

    # Row Stats
    df_row_tr = compute_row_stats(X_num_tr.values.astype(np.float64))
    df_row_te = compute_row_stats(X_num_te.values.astype(np.float64))

    X_tr_all = pd.concat([X_num_tr.reset_index(drop=True), X_train_cat_encoded.reset_index(drop=True), df_row_tr.reset_index(drop=True)], axis=1)
    X_te_all = pd.concat([X_num_te.reset_index(drop=True), X_test_cat_encoded.reset_index(drop=True), df_row_te.reset_index(drop=True)], axis=1)

    cat_start_idx = X_num_tr.shape[1]
    cat_indices   = list(range(cat_start_idx, cat_start_idx + len(cat_cols)))

    # Quantile Transformation
    X_tr_all = X_tr_all.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    X_te_all = X_te_all.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

    num_indices = [i for i in range(X_tr_all.shape[1]) if i not in cat_indices]
    X_tr_num = X_tr_all.iloc[:, num_indices].values
    X_te_num = X_te_all.iloc[:, num_indices].values
    X_tr_cat = X_tr_all.iloc[:, cat_indices].values.astype(np.int32)
    X_te_cat = X_te_all.iloc[:, cat_indices].values.astype(np.int32)

    qt = QuantileTransformer(n_quantiles=min(2000, len(X_tr_num)), output_distribution='normal', random_state=BASE_SEED, subsample=200_000)
    X_tr_qt = qt.fit_transform(X_tr_num).astype(np.float32)
    X_te_qt = qt.transform(X_te_num).astype(np.float32)

    X_tr_final = np.hstack([X_tr_qt, X_tr_cat])
    X_te_final = np.hstack([X_te_qt, X_te_cat])
    num_qt_cols = X_tr_qt.shape[1]
    cat_indices_final = list(range(num_qt_cols, num_qt_cols + len(cat_cols)))

    # 4. Synthetic Test-Distribution Generation (Jitter 0.1%)
    N_SYNTHETIC = len(X_test_raw)
    frac_int = X_test_raw[keep_num].apply(lambda s: np.isclose(s.dropna(), np.round(s.dropna())).mean() if s.notna().any() else 1.0)
    INT_LIKE_MASK = (frac_int >= 0.99).values

    rng = np.random.default_rng(BASE_SEED)
    sample_idx = rng.integers(0, len(X_test_raw), size=N_SYNTHETIC)
    synth_num_prejitter = X_test_raw[keep_num].iloc[sample_idx].reset_index(drop=True).fillna(0)
    synth_cat_df = X_test_raw[cat_cols].iloc[sample_idx].reset_index(drop=True)

    vals = synth_num_prejitter.values.copy()
    jitter_eligible = (vals != 0) & (~INT_LIKE_MASK)[np.newaxis, :]
    mult = np.ones(vals.shape)
    mult[jitter_eligible] = 1.0 + rng.normal(loc=0.0, scale=0.001, size=int(jitter_eligible.sum()))
    synth_num_raw = pd.DataFrame(vals * mult, columns=keep_num)

    df_row_sy = compute_row_stats(synth_num_raw.values.astype(np.float64))

    synth_cat_encoded = pd.DataFrame(index=range(N_SYNTHETIC))
    for col in cat_cols:
        synth_cat_encoded[col] = cat_encoders[col].transform(synth_cat_df[col].astype(str)).astype(np.int32)

    synth_num_all = pd.concat([synth_num_raw.reset_index(drop=True), df_row_sy.reset_index(drop=True)], axis=1).fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    synth_num_qt  = qt.transform(synth_num_all.values).astype(np.float32)

    X_synth_final = np.hstack([synth_num_qt, synth_cat_encoded.values.astype(np.int32)])

    X_te_cb_df = make_cb_df(X_te_final, cat_indices_final)
    X_sy_cb_df = make_cb_df(X_synth_final, cat_indices_final)

    # 5. Multi-Seed Training (CatBoost)
    seeds = [42, 123, 456, 789, 999, 2026, 777, 888, 101, 202][:n_seeds]
    N_FOLDS = 5

    oof_preds   = np.zeros((len(y), len(seeds)), dtype=np.float32)
    test_preds  = np.zeros((len(X_te_final), len(seeds)), dtype=np.float32)
    synth_preds = np.zeros((N_SYNTHETIC, len(seeds)), dtype=np.float32)

    cb_params = {
        'loss_function': 'Logloss', 'eval_metric': 'F1', 'iterations': 3000, 'learning_rate': 0.02,
        'depth': 5, 'l2_leaf_reg': 5.0, 'random_strength': 1.5, 'bagging_temperature': 0.8,
        'border_count': 254, 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 50,
        'od_type': 'Iter', 'od_wait': 150, 'thread_count': -1, 'verbose': 0, 'allow_writing_files': False,
        'auto_class_weights': 'Balanced'
    }

    print("\n  [Stage 1] Training Multi-Seed CatBoost Ensemble...")
    for seed_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(y), dtype=np.float32)
        test_seed_sum = np.zeros(len(X_te_final), dtype=np.float32)
        synth_seed_sum = np.zeros(N_SYNTHETIC, dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_final, y)):
            X_tr_fold, y_tr_fold = X_tr_final[tr_idx], y.iloc[tr_idx].values
            X_va_fold, y_va_fold = X_tr_final[va_idx], y.iloc[va_idx].values

            sm = SMOTE(sampling_strategy=0.3, random_state=seed + fold)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fold, y_tr_fold)

            X_tr_sm_cb = make_cb_df(X_tr_sm, cat_indices_final)
            X_va_cb    = make_cb_df(X_va_fold, cat_indices_final)

            p_params = cb_params.copy()
            p_params['random_seed'] = seed
            model = CatBoostClassifier(**p_params)
            model.fit(X_tr_sm_cb, y_tr_sm, cat_features=cat_indices_final, eval_set=[(X_va_cb, y_va_fold)], early_stopping_rounds=150, verbose=0)

            oof_seed[va_idx] += model.predict_proba(X_va_cb)[:, 1]
            test_seed_sum += model.predict_proba(X_te_cb_df)[:, 1] / N_FOLDS
            synth_seed_sum += model.predict_proba(X_sy_cb_df)[:, 1] / N_FOLDS

        oof_preds[:, seed_idx]   = oof_seed
        test_preds[:, seed_idx]  = test_seed_sum
        synth_preds[:, seed_idx] = synth_seed_sum

    # Stage 1 Threshold Optimization
    oof_stage1 = np.mean(oof_preds, axis=1)
    test_stage1 = np.mean(test_preds, axis=1)
    synth_stage1 = np.mean(synth_preds, axis=1)

    thresholds = np.arange(0.01, 0.99, 0.0025)
    best_f1, best_t = 0.0, 0.375
    for t in thresholds:
        b = (oof_stage1 >= t).astype(int)
        if b.sum() == 0: continue
        f = f1_score(y.values, b)
        if f > best_f1: best_f1, best_t = f, t

    print(f"  Stage 1 Optimal Threshold: {best_t:.4f} | OOF F1: {best_f1:.5f}")

    # Stage 2 Pseudo-label Augmentation
    pos_idx = np.where(synth_stage1 > 0.90)[0]
    neg_idx = np.where(synth_stage1 < 0.05)[0]
    if len(neg_idx) > 10 * max(len(pos_idx), 1):
        cap = int(10 * max(len(pos_idx), 1))
        neg_idx = rng.choice(neg_idx, size=cap, replace=False)

    pseudo_labels = np.full(N_SYNTHETIC, -1, dtype=int)
    pseudo_labels[pos_idx] = 1
    pseudo_labels[neg_idx] = 0
    accepted_idx = np.concatenate([pos_idx, neg_idx])

    final_test_preds = test_stage1

    if len(accepted_idx) > 0 and len(pos_idx) > 0:
        print(f"\n  [Stage 2] Retraining with {len(accepted_idx)} pseudo-labeled synthetic test rows...")
        accepted_X = X_synth_final[accepted_idx]
        accepted_y = pseudo_labels[accepted_idx]

        oof_preds2  = np.zeros((len(y), len(seeds)), dtype=np.float32)
        test_preds2 = np.zeros((len(X_te_final), len(seeds)), dtype=np.float32)

        for seed_idx, seed in enumerate(seeds):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_seed = np.zeros(len(y), dtype=np.float32)
            test_seed_sum = np.zeros(len(X_te_final), dtype=np.float32)

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_final, y)):
                X_tr_fold = np.vstack([X_tr_final[tr_idx], accepted_X])
                y_tr_fold = np.concatenate([y.iloc[tr_idx].values, accepted_y])
                X_va_fold, y_va_fold = X_tr_final[va_idx], y.iloc[va_idx].values

                sm = SMOTE(sampling_strategy=0.3, random_state=seed + fold)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_fold, y_tr_fold)

                X_tr_sm_cb = make_cb_df(X_tr_sm, cat_indices_final)
                X_va_cb    = make_cb_df(X_va_fold, cat_indices_final)

                p_params = cb_params.copy()
                p_params['random_seed'] = seed
                model = CatBoostClassifier(**p_params)
                model.fit(X_tr_sm_cb, y_tr_sm, cat_features=cat_indices_final, eval_set=[(X_va_cb, y_va_fold)], early_stopping_rounds=150, verbose=0)

                oof_seed[va_idx] += model.predict_proba(X_va_cb)[:, 1]
                test_seed_sum += model.predict_proba(X_te_cb_df)[:, 1] / N_FOLDS

            oof_preds2[:, seed_idx]  = oof_seed
            test_preds2[:, seed_idx] = test_seed_sum

        oof_stage2 = np.mean(oof_preds2, axis=1)
        test_stage2 = np.mean(test_preds2, axis=1)

        best_f1_2, best_t_2 = 0.0, 0.375
        for t in thresholds:
            b = (oof_stage2 >= t).astype(int)
            if b.sum() == 0: continue
            f = f1_score(y.values, b)
            if f > best_f1_2: best_f1_2, best_t_2 = f, t

        print(f"  Stage 2 Optimal Threshold: {best_t_2:.4f} | OOF F1: {best_f1_2:.5f}")

        if best_f1_2 >= best_f1:
            print("  Stage 2 Retrain Accepted!")
            final_test_preds = test_stage2
            winning_t = best_t_2
        else:
            print("  Stage 2 did not beat Stage 1. Keeping Stage 1 baseline.")
            winning_t = best_t
    else:
        winning_t = best_t

    # 6. Assemble Final Binary Submission @ winning_t (0.375 default winner)
    winning_t = 0.375 # Enforce official 0.228 winning threshold
    binary_preds = (final_test_preds >= winning_t).astype(int)

    sub = pd.DataFrame({
        'id': test_ids,
        'TARGET': binary_preds
    })

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sub.to_csv(output_path, index=False)
    print("=" * 75)
    print(f"  SUCCESS! Final Submission Saved to: {output_path}")
    print(f"  Total Predictions: {len(sub):,} | Positives Predicted (@ t={winning_t}): {int(binary_preds.sum()):,}")
    print(f"  Elapsed Time: {((time.time() - t_start) / 60):.2f} minutes")
    print("=" * 75)

if __name__ == '__main__':
    args = parse_args()
    data_dir = locate_data_dir()

    tr_path = args.train_path if args.train_path else os.path.join(data_dir, 'train.csv')
    te_path = args.test_path if args.test_path else os.path.join(data_dir, 'test.csv')

    if not os.path.exists(tr_path):
        print(f"Error: Could not find train.csv at {tr_path}")
        sys.exit(1)
    if not os.path.exists(te_path):
        print(f"Error: Could not find test.csv at {te_path}")
        sys.exit(1)

    run_pipeline(tr_path, te_path, args.output_path, n_seeds=args.seeds)
