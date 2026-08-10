✅ All 3 Steps Complete — Here's the Full Report
📊 STEP 1: EDA Findings
Metric	Value
Train	76,020 rows × 350 feat + TARGET
Test	60,654 rows × 350 feat + id
Class 0 (Stable)	73,012 (96.04%)
Class 1 (At-Risk)	3,008 (3.96%)
Imbalance Ratio	24.27:1 🔴
Missing Values	None (clean!)
Infinite Values	None
Feature Types	344 numerical + 6 categorical (feat_142, feat_157, feat_318, feat_320, feat_325, feat_337)
Zero-Variance	28 features → DROP
Duplicate Pairs (r=1.0)	30+ detected → DROP one each
**Max	corr
**	skew
PCA 80% variance	~45 components needed
Train/Test shift	Minimal — only 2/30 KS-test significant
Key Insight: The anonymized features have very weak individual predictive power (max |corr| = 0.15). Winning requires non-linear tree ensembles, feature interactions, and row-wise statistical features that capture structural patterns.

🧠 STEP 2: Grand Master Strategy
Cleaning: Drop 28 zero-var + ~30 duplicate features
Encoding: Stratified 5-Fold OOF target encoding for 6 categorical features
Feature Engineering:
22 row-wise statistical features (mean, std, skew, kurtosis, percentiles, zero-count, etc.)
28 KMeans cluster features (4+8+16 clusters + distances to centers)
Target-encoded features (6)
Preprocessing: QuantileTransformer(output='normal') → handles skew + PCA(50)→ preserves ~90% variance
Imbalance: SMOTE with sampling_strategy=0.5 (2:1 post-SMOTE ratio — avoids over-sampling noise)
Models: LightGBM + XGBoost + CatBoost, each with early stopping
CV: Stratified 10-Fold (preserves 3.96% minority ratio)
Calibration: Affine threshold shift to align optimal F1 → exactly 0.5 (critical for Kaggle's strict 0.5 threshold)
Ensemble: Weighted blend (by OOF F1) + Rank-average fallback
📓 STEP 3: Notebook Delivered
Winning_Solution_PSTU.ipynb — 13 cells, ~740 lines of production code:

Cell	Contents
0	Title & Strategy Overview
1	Imports (all libraries)
2	CFG — all hyperparameters in one place
3	Data loading & column identification
4	Zero-variance & duplicate removal
5	OOF Target Encoding (no leakage!)
6	Row-wise stats + KMeans clusters
7	QuantileTransformer + PCA
8	F1 calibration utilities
9	10-Fold CV training (LGB+XGB+CB per fold)
10	Threshold calibration + ensemble blend
11	Test predictions → submission.csv + backup
12	Performance summary + CV plot
Outputs generated: submission.csv (primary) + submission_rank_avg.csv (backup)

🚀 To Run

source .venv/bin/activate
# Install dependencies if needed:
pip install pandas numpy scikit-learn scipy matplotlib seaborn lightgbm xgboost catboost imbalanced-learn

# Run the notebook:
jupyter notebook Winning_Solution_PSTU.ipynb
The notebook will generate submission.csv ready for Kaggle upload. The calibration step ensures your predicted probabilities have their optimal F1 decision boundary centered at exactly 0.5 — matching the competition evaluation.