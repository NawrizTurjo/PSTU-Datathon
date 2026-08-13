# Audit of Notebook Strategies & Leaderboard Results

This document provides a comprehensive audit of all major notebook versions, their underlying strategies, LB scores, and an analysis of their strengths and weaknesses.

## 1. Early Baselines (Runs 1 to 5)
* **Folders:** `run-1`, `run-2`, `run-3`, `run-4`, `run-5`
* **Best LB Score:** `~0.185 - 0.192`
* **Strategy Used:**
  * Basic XGBoost/LightGBM/CatBoost architectures.
  * Simple imputation and label encoding.
  * Standard train/validation split (often random split or basic K-Fold).
  * Out-of-the-box thresholding (usually default $t=0.5$).
* **What Went Well:** Established a working pipeline and data loading baseline.
* **What Went Wrong / To Improve:** Failed to address the extreme class imbalance (~3.8% positive class). Did not use specialized feature engineering or probability threshold tuning, leading to many False Negatives (low recall).

## 2. Unbeatable V6 & Synthetic Unbeatable
* **Folders:** `unbeatable-6`, `unbeatable-6 - v2`, `synthetic-unbeatable-6`
* **Best LB Score:** `0.109 - 0.124`
* **Strategy Used:**
  * Attempted to create a highly complex, deep model.
  * Introduced SMOTE but likely misapplied it (e.g., leaking into validation set).
  * The synthetic variant attempted to generate synthetic data without proper noise calibration.
* **What Went Well:** Experimented with data augmentation.
* **What Went Wrong / To Improve:** Severe overfitting to the training distribution. The models failed to generalize to the public LB, likely due to probability distortion from naive SMOTE oversampling and lack of threshold probing.

## 3. Grandmaster V7 & Omega V7
* **Folders:** `grandmaster-7`, `omega-7`
* **Best LB Score:** `0.180 - 0.194`
* **Strategy Used:**
  * Introduced aggressive feature generation (polynomials, heavy interactions).
  * "Omega" tried a two-stage pseudo-labeling system on the test set.
* **What Went Well:** Pseudo-labeling logic was introduced, helping models learn from the test set structure.
* **What Went Wrong / To Improve:** Over-engineered feature space introduced noise. Pseudo-labeling lacked strict "guard rails" (like capping the ratio of positive/negative pseudo-labels), causing the model to over-predict the majority class and crash F1.

## 4. Synthetic Omega V7
* **Folders:** `synthetic-omega-7`
* **Best LB Score:** `0.131 - 0.200`
* **Strategy Used:**
  * Built on Omega V7 but generated a purely synthetic test set by bootstrapping test rows and adding jitter.
  * Retrained on a combination of original train + pseudo-labeled synthetic test data.
* **What Went Well:** The concept of test-distribution augmentation is mathematically sound for combating covariate shift.
* **What Went Wrong / To Improve:** The synthetic generator was *too* perfect. It mimicked the test set so closely (without sufficient calibrated noise) that the model memorized the synthetic rows, causing data leakage and poor public LB generalization (F1 dropped to 0.131 at $t=0.375$).

## 5. The "FixIssuesV2" Core (Best-So-Far)
* **Folders:** `best-so-far` (FixIssuesV2 V4)
* **Best LB Score:** `0.2257`
* **Strategy Used:**
  * Stripped away the over-engineered features from Grandmaster/Omega.
  * **Architecture:** CatBoost (`depth=5`, `l2_reg=5.0`) with `auto_class_weights='Balanced'`.
  * **Feature Engineering:** 6 simple row-wise statistics and a global `QuantileTransformer`.
  * **Imbalance Handling:** SMOTE(0.3) applied *strictly* inside the CV folds.
  * **Thresholding:** Tuned thresholds via probing (found optimal range around 0.375).
* **What Went Well:** Highly robust, clean, and stable. Prevented leakage and optimized directly for F1 on the true data distribution.

## 6. Synthetic FixIssuesV2 (Versions 1-3)
* **Folders:** `synthetic-fixissuesv2-version-1`, `synthetic-fixissuesv2-version-2`, `synthetic-fixissuesv2-version-3`
* **Best LB Score:** `0.2279`
* **Strategy Used:**
  * Combined the rock-solid `FixIssuesV2` baseline with safe synthetic test augmentation.
  * Applied **Calibrated Jitter (0.1%)** only to continuous columns to prevent leakage (verified by Adversarial Validation AUC of ~0.51).
  * Implemented an **OOF F1 Safety Net**: Stage 2 retrain is only kept if it improves or maintains Stage 1 validation F1.
  * **10-Seed Ensemble:** Stabilized probability boundaries.
* **What Went Well:** Achieved our all-time best LB score. Safely utilized test-distribution characteristics without overfitting.
* **What Went Wrong / To Improve:** Reached a plateau. While this is a highly optimized CatBoost model, it lacks the diversity of a multi-model ensemble (which is often necessary to cross the 0.30 barrier).

---

## Key Takeaways for the Final Push (0.22 $\rightarrow$ 0.30)
1. **Simplicity over Complexity:** `FixIssuesV2` proved that clean features and strict fold validation beat over-engineered, complex pipelines (Omega/Grandmaster).
2. **Proper Thresholding is Crucial:** The score jumps massively between $t=0.25$ and $t=0.375$. We must continue exporting fine-grained probe grids.
3. **Model Diversity is Missing:** We have exhaustively optimized CatBoost. To bridge the gap to 0.303, we need to introduce **LightGBM** and **XGBoost** trained on this exact `FixIssuesV2` feature space, and ensemble their predictions. (Completed via `master-extended`).

---

## 🚀 Future Ideas & Brainstorming (Based on the `FixIssuesV2` Core)
Since the `master-extended` notebook might take a few hours to run on Kaggle, here are four alternative notebook strategies we could run in parallel. All of them build on the proven data pipeline (QuantileTransformer + Row Stats + SMOTE + Raw Features) but change the modeling approach:

### Idea 1: Deep Learning / TabNet Ensemble
**The Concept:** Tree models (XGBoost, CatBoost) are great, but they all learn in similar ways (axis-parallel splits). Neural networks learn linear combinations and smooth manifolds.
**The Strategy:** Build a simple Multi-Layer Perceptron (MLP) or use Google's TabNet on the exact same data. Even if the Neural Net scores slightly lower on its own, averaging its predictions with a Tree model usually causes a massive jump in LB score because their errors are totally uncorrelated.

### Idea 2: K-Means Cluster Features (Distance to Minority)
**The Concept:** We have a severe 96:4 imbalance. We want the model to easily identify where the 4% hides.
**The Strategy:** Run a KMeans clustering ($K=10$ or $20$) on the numeric features. Find which cluster centroids contain the highest percentage of the minority class. Add a new feature: `distance_to_deadliest_cluster`. Trees love this because it gives them a direct compass pointing toward the positive class.

### Idea 3: Explicit Feature Interactions (Polynomials done right)
**The Concept:** We stripped out interactions because Omega over-engineered them, but trees still struggle to divide features (e.g., $A / B$).
**The Strategy:** Take *only the Top 20* most important features (measured from CatBoost's `.get_feature_importance()`) and create basic math interactions ($A+B$, $A-B$, $A \times B$, $A \div (B + \epsilon)$). This gives the tree the exact logic it would otherwise need hundreds of splits to learn naturally.

## 7. Master Extended Ensemble & Feature Forge
* **Folders:** `master-final-version-extended-version-1`, `master-final-version-extended-version-2`, `master-final-feature-forge-v4`
* **Best LB Score:** `0.2179` (Forge v4 at $t=0.375$)
* **Strategy Used:** 
  * Replaced CatBoost-only with a 3-model (CatBoost + LightGBM + XGBoost) soft-voting ensemble.
  * *Extended V1* (7h 30m): Scored `0.2127`.
  * *Extended V2* (53m): Leaner config. Scored `0.1944`.
  * *Feature Forge V4* (3h 38m): Included KMeans distance mapping and top-10 math interactions + SMOTENC + Stage 2. Scored `0.2179`.
* **What Went Well:** The code ran successfully, proving our fixes to `SMOTENC` and categorical features held up.
* **What Went Wrong / To Improve:** The scores (`0.194 - 0.217`) failed to beat our simpler, single-model CatBoost `synthetic-fixissuesv2-winner` (`0.2279`). This proves that for this specific dataset and extreme class imbalance, a single highly-regularized CatBoost model with aggressive SMOTE and basic row-stats generalizes better to the hidden public LB than complex ensembles or deep feature engineering.
* **Conclusion:** The `synthetic-fixissuesv2-winner` configuration is officially our best model. We must use it for the final submission.
