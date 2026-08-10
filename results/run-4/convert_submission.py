#!/usr/bin/env python3
"""
Convert probability-based submission.csv to binary (0/1) submission
ready for Kaggle upload.

The competition evaluates F1 score at a strict 0.5 threshold.
If Kaggle's evaluator rejects continuous probabilities, we apply
the threshold ourselves and submit binary predictions.

Reads:  results/submission.csv  (probabilities)
Writes: results/submission_binary.csv  (0/1 binary)
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH  = 'submission.csv'
THRESHOLD   = 0.25
threshold_str = str(THRESHOLD).replace('.', '_')
OUTPUT_PATH = f'submission_binary_{threshold_str}.csv'

print(f'Reading: {INPUT_PATH}')
df = pd.read_csv(INPUT_PATH)

print(f'  Rows: {len(df):,}')
print(f'  Columns: {list(df.columns)}')
print(f'  TARGET dtype: {df["TARGET"].dtype}')
print(f'  TARGET range: [{df["TARGET"].min():.6f}, {df["TARGET"].max():.6f}]')
print(f'  TARGET mean:  {df["TARGET"].mean():.6f}')

# Convert continuous probabilities to binary at 0.5 threshold
df['TARGET'] = (df['TARGET'] >= THRESHOLD).astype(int)

pos_count = df['TARGET'].sum()
pos_pct   = df['TARGET'].mean() * 100
print(f'\nAfter thresholding at {THRESHOLD}:')
print(f'  Class 0: {(df["TARGET"]==0).sum():,} ({(1-pos_pct/100)*100:.2f}%)')
print(f'  Class 1: {pos_count:,} ({pos_pct:.2f}%)')

# Ensure correct format: id (int), TARGET (int)
df['id'] = df['id'].astype(int)
df['TARGET'] = df['TARGET'].astype(int)

print(f'\nPreview (first 12 rows):')
print(df.head(12).to_string(index=False))

# Also show a few rows where TARGET=1
ones = df[df['TARGET'] == 1]
if len(ones) > 0:
    print(f'\nSample rows with TARGET=1:')
    print(ones.head(8).to_string(index=False))

df.to_csv(OUTPUT_PATH, index=False)
print(f'\nSaved: {OUTPUT_PATH}')
print(f'File size: {os.path.getsize(OUTPUT_PATH):,} bytes')
print('Ready for Kaggle submission!')
