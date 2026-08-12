# Final Submission Decisions & Competition Strategy

## 1. Current State & Leaderboard Analysis
- **Our Best Public LB Score:** `0.2279` (`synthetic-fixissuesv2-winner` @ t=0.375)
- **Top Leaderboard Scores:** `~0.303`
- **Gap to 1st Place:** `~0.075` F1 Score.
- **Context:** The private test set (50%) and the hidden test set (40%) make up 90% of the final evaluation. Robustness and avoiding overfitting to the public LB (10%) are the top priorities. Deterministic inference is strictly required.

## 2. Comprehensive Verdict on Tried Architectures

We have tested several distinct pipelines throughout the competition:

### A. Baseline & Early Runs (`run-1`, `run-2`, `PTSU-Datahon-v1`)
- **Scores:** `0.168 - 0.185`
- **Verdict:** Basic architectures that served as good starting points but lacked sophisticated threshold tuning and class imbalance handling.

### B. The "Unbeatable / Grandmaster / Omega" Series
- **Scores:** `0.109 - 0.200`
- **Verdict:** **DISCARD.** These notebooks became overly complex. The `Omega` generator caused severe data leakage by creating synthetic rows that too closely mimicked the test set without properly calibrating the probability distributions. This resulted in poor LB generalization (0.131 F1 for Omega @ t=0.375).

### C. The `FixIssuesV2` Core
- **Scores:** `0.2257`
- **Verdict:** A highly robust, stable baseline. It cleaned up the feature space (QuantileTransformer, 6 row stats), used a strict CatBoost (`depth=5`, `l2_reg=5.0`) architecture, and applied SMOTE(0.3) securely within folds.

### D. The `Synthetic-FixIssuesV2-Winner` (Current Best)
- **Scores:** `0.227966`
- **Verdict:** **KEEP FOR FINAL SUBMISSION (for now).** This architecture took the robust `FixIssuesV2` core and added:
  1. **10-Seed Ensemble:** Drastically reduced probability variance.
  2. **Safe Pseudo-labeling:** Used calibrated 0.1% jitter on continuous columns to safely augment the training set without causing validation leakage.
  3. **Optimal Thresholding:** Found that $t=0.375$ maximizes the F1 score on the highly imbalanced predictions.

---

## 3. Final Submission Requirements Checklist
Based on the official rules, our final submission must include:
- [ ] **Prediction CSV:** Binary target values (0 or 1).
- [ ] **Inference Notebook:** Must run deterministically within 6 hours. Must NOT use external data or modify the test set.
- [ ] **Kaggle Code Markdown:** A markdown cell explaining the EDA, baseline model, improvements, and final architecture.
- [ ] **Presentation (YouTube):** PDF/PPT + short video explaining the approach, feature engineering, and model.

*Our `synthetic-fixissuesv2-winner` notebook already fulfills the deterministic inference requirement by explicitly setting random seeds across all 10 ensemble members.*

---

## 4. "The Last Try": Strategy to Bridge the 0.22 $\rightarrow$ 0.30 Gap
Jumping from 0.22 to 0.30 F1 on a heavily imbalanced dataset is a massive leap. Small tweaks to CatBoost won't get us there. We need a fundamental shift in our modeling approach for this final day:

1. **Model Diversity (The Missing Piece):**
   - We are currently relying *only* on CatBoost. Top Kaggle solutions almost always blend **LightGBM**, **XGBoost**, and **CatBoost**.
   - **Action:** Build a Voting or Stacking Classifier that combines our tuned CatBoost with an optimized LightGBM and XGBoost model.

2. **Advanced Imbalance Handling:**
   - SMOTE(0.3) is good, but algorithms like **BalancedRandomForest** or ensemble methods like **EasyEnsemble** (training many models on different subsamples of the majority class) often outperform SMOTE for extreme imbalance.
   - **Action:** Try undersampling the majority class in a bagging ensemble.

3. **Loss Function Optimization:**
   - Standard Logloss doesn't directly optimize F1.
   - **Action:** Train LightGBM/XGBoost using a custom **Focal Loss** or an F1-approximation objective to force the model to focus on the hard-to-predict positive class.

4. **AutoML Benchmark:**
   - If time is short, running **AutoGluon** or **H2O.ai** on our cleaned feature set (from `FixIssuesV2`) could automatically find the optimal stacking architecture that bridges this gap.

### Conclusion for Today
We will lock in `synthetic-fixissuesv2-winner.ipynb` as our **fallback/guaranteed** submission.
For our final experiments today, we will branch off the `FixIssuesV2` feature engineering pipeline and attempt a **LightGBM + XGBoost + CatBoost Ensemble** with Focal Loss/EasyEnsemble techniques to crack the 0.30 barrier.
