### Step 1: How Synthetic Data is Generated

The goal is to generate synthetic rows that mirror the **real test set's distribution (covariate shift)** without creating duplicate rows or leaking artificial noise.

```
Real Test CSV ──> Bootstrap Resample ──> Integer Masking ──> 0.1% Jitter (Continuous only) ──> Pre-Jitter Row Stats ──> Adversarial AUC Check
```

1. **Bootstrap Resampling**:
   - We draw $N$ rows ($N = \text{len(test)}$) from the real test CSV with replacement using a fixed seed.
   - This anchors the synthetic dataset directly inside the test feature distribution (capturing test feature correlations, ranges, and missingness).

2. **Integer Column Protection**:
   - In this dataset, ~238 of 301 numerical features are integer-valued (counts, flags, ordinal values).
   - Adding continuous noise to integer columns (e.g. turning `3` into `3.087`) makes synthetic rows instantly detectable by boosted trees (AUC > 0.98).
   - We identify columns where $\ge 99\%$ of non-null values are integers and **freeze them completely** ($0\%$ jitter).

3. **Selective Multiplicative Jitter**:
   - For continuous (non-integer) columns, we apply small multiplicative Gaussian noise:
     $$x_{\text{synth}} = x_{\text{real}} \times (1 + \mathcal{N}(0, \sigma^2))$$
   - Jitter is applied **only to non-zero values** ($x \neq 0$) so zero-inflated distributions remain exact.
   - For the full 307 raw feature space (no PCA), we use $\sigma = 0.001$ (0.1% jitter) so high-dimensional noise does not accumulate across 307 columns into joint tree separability.

4. **Pre-Jitter Row Statistics**:
   - Row-wise aggregate features (`row_mean`, `row_std`, `row_iqr`, `row_zero`, `row_skew`, `row_kurtosis`) are computed from `synth_num_prejitter` (the un-jittered resampled values).
   - This prevents extreme magnitude columns (e.g. `feat_169` min $\approx -1.11 \times 10^8$) from causing aggregate jitter leaks.

5. **Adversarial Quality Check**:
   - We train a 3-fold classifier to distinguish `synthetic vs test` (target AUC $\approx 0.50$) and `synthetic vs train` (target AUC $> 0.50$).
   - This proves the synthetic rows match the test distribution without introducing artificial artifacts.

---

### Step 2: How Synthetic Data is Labeled (Pseudo-Labeling Protocol)

The goal is to assign high-confidence binary labels ($y \in \{0, 1\}$) to synthetic rows while preventing negative domination.

```
Synthetic Rows ──> Stage 1 CatBoost Ensemble ──> Dynamic Threshold Relaxation ──> 10:1 Ratio Cap ──> Fold-Safe Retrain
```

1. **Stage 1 Ensemble Prediction**:
   - The Stage 1 CatBoost ensemble (3 seeds $\times$ 5 folds, trained on real data) outputs continuous probabilities $P(\text{Target}=1)$ for every generated synthetic row.

2. **Dynamic Threshold Relaxation**:
   - We start with strict confidence thresholds:
     - **Positive Pseudo-label ($y=1$)**: $P \ge 0.90$
     - **Negative Pseudo-label ($y=0$)**: $P \le 0.05$
   - If fewer than 20 confident positives survive, the positive threshold relaxes downward in steps of 0.05 (down to floor 0.55).
   - If needed, the negative threshold relaxes upward (up to ceiling 0.45).

3. **Negative-Domination Ratio Guard (10:1 Cap)**:
   - Because the target is heavily imbalanced (~3% positives), raw thresholding produces thousands of confident negatives but very few positives.
   - We randomly sample accepted negatives to enforce a strict **10:1 maximum negative-to-positive ratio**.
   - If 0 positive synthetic rows survive, Stage 2 is automatically **skipped** to avoid polluting the dataset.

4. **Fold-Safe Retraining (Stage 2)**:
   - Accepted synthetic pseudo-rows are appended **ONLY to the training fold partitions** during 5-fold CV retrain.
   - Validation folds remain 100% real train data, ensuring honest OOF F1 evaluation.
   - Stage 2 is accepted **only if real-train OOF F1 improves** (`stage2_f1 >= stage1_f1`).