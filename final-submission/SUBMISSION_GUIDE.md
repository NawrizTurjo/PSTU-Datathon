# 🏆 PSTU DataThon 2026 — Official Top 20 Submission Package

**Team Name:** NawrizTurjo  
**Winning LB Benchmark Score:** `0.227966`  
**Model Name:** `synthetic-fixissuesv2-winner`

---

## 📂 Package Overview

This directory (`final-submission/`) contains all required assets for the final Top 20 evaluation by competition judges:

| File Name | Purpose / Description |
|---|---|
| **`FINAL_SUBMISSION.ipynb`** | **Primary Submission Notebook.** Contains full Markdown documentation (EDA summary, baseline-to-final progression, technical decisions) + complete execution code. Upload this directly to Kaggle Code. |
| **`inference_pipeline.py`** | **Standalone Inference & Retraining Pipeline.** A CLI-ready Python script for organizers to re-test the solution on hidden private datasets. |
| **`PRESENTATION_OUTLINE.md`** | **YouTube Presentation Script & Slide Guide.** Complete section-by-section outline and timing guide for the required video presentation. |
| **`SUBMISSION_GUIDE.md`** | This document. Step-by-step instructions for Kaggle code submission and private dataset evaluation. |

---

## 1. 📥 Kaggle Code Submission Instructions

1. Log into Kaggle using your registered **Team Name** (`NawrizTurjo`).
2. Navigate to the **Code** tab of the PSTU DataThon 2026 competition.
3. Click **New Notebook** and choose **Import Notebook**.
4. Upload **`FINAL_SUBMISSION.ipynb`**.
5. Set environment settings:
   - **Accelerator:** CPU (or GPU - CPU recommended for full 30GB RAM headroom).
   - **Persistence:** Files persisted.
6. Click **Run All** to execute the pipeline.
7. Verify that `submission.csv` is generated in `/kaggle/working/submission.csv` containing binary predictions (`0` or `1`).
8. Save & Publish the notebook as **Public** or **Shared with Competition Host**.

---

## 2. 🧪 Private Dataset Testing Instructions (For Judges)

The organizers can evaluate this solution on a hidden private dataset (`train.csv` and `test.csv`) using the standalone script **`inference_pipeline.py`**.

### Dependencies
Ensure the following standard Python packages are installed:
```bash
pip install numpy pandas scikit-learn imbalanced-learn catboost scipy
```

### Running the Pipeline
Run the script passing the paths to your private dataset:

```bash
python inference_pipeline.py \
    --train_path /path/to/private/train.csv \
    --test_path /path/to/private/test.csv \
    --output_path /path/to/output/submission.csv
```

### Script Execution Logic:
1. **Dynamic Ingestion:** Ingests any valid `train.csv` and `test.csv`.
2. **Automated Cleaning:** Removes 0-variance and identical hash-duplicate columns automatically.
3. **Quantile Normalization & Row Stats:** Computes 6 row-wise statistical features and normalizes continuous columns using `QuantileTransformer`.
4. **Fold-Safe SMOTE Oversampling:** Applies `SMOTE(0.3)` strictly inside 5-fold cross-validation loops across 10 random seeds.
5. **Synthetic Test-Distribution Augmentation:** Bootstraps test rows with calibrated $0.1\%$ multiplicative jitter, pseudo-labels confident test rows, and performs fold-safe retraining.
6. **Threshold Calibration:** Exports final hard binary predictions using the competition-winning decision threshold ($t=0.375$).

---

## 3. 🧠 Summary of Methodology & Key Progressions

```
[ Raw Features (344 cols) ] 
            │
            ▼
[ Feature Cleaning & 6 Row Stats ] ──> (Mean, Std, IQR, Skew, Kurt, Zero Count)
            │
            ▼
[ QuantileTransformer (Normal) ]
            │
            ▼
[ 10-Seed 5-Fold CV + SMOTE(0.3) ] ──> (Fold-Safe Imbalance Handling)
            │
            ▼
[ Synthetic Test Jitter (0.1%) ] ──> (Bootstrapped Test Distr, Jitter Frac=0.001)
            │
            ▼
[ Dynamic Pseudo-Label Guard ] ──> (Strict 10:1 Ratio Cap & Safety Net)
            │
            ▼
[ Optimal Thresholding @ t=0.375 ] ──> [ Final submission.csv (LB 0.228) ]
```

### Core Milestones:
1. **Baseline Failure (LB 0.08 - 0.16):** Standard LightGBM/XGBoost on raw data suffered from extreme F1 penalty due to 96:4 class imbalance and default $t=0.5$ thresholding.
2. **Feature Engineering & Imbalance Fix (LB 0.225):** Introducing 6 row-wise statistics, `QuantileTransformer`, and fold-safe `SMOTE(0.3)` oversampling caused a massive performance jump to `0.2257`.
3. **Synthetic Test-Distribution Augmentation (LB 0.228 Winner):** Adding bootstrapped test-set resamples with $0.1\%$ calibrated multiplicative jitter enabled two-stage pseudo-label retraining. Calibrated jitter prevented noise leakage (verified via Adversarial AUC of ~0.51), yielding our competition-winning score of **`0.227966`**.

---

## 4. 📹 Presentation Guidelines

Top 20 teams must submit a recorded presentation on YouTube. Refer to **`PRESENTATION_OUTLINE.md`** for the complete slide breakdown, script guidelines, and timing recommendations.
