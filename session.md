# 🚀 PSTU DataThon 2026: End of Session Handoff
**Date/Time of Save:** 2026-08-13 01:20 AM
**Project Status:** READY FOR FINAL KAGGLE SUBMISSIONS

## 📁 Key Deliverables (The Final Notebooks)
Both of the following notebooks have been completely generated, smoke-tested, and heavily patched for safety, stability, and runtime optimization. They are ready to be uploaded to Kaggle as-is.

### 1. `solution/master-final-version-extended.ipynb`
* **Role:** The fast, reliable 3-model baseline. 
* **Architecture:** CatBoost + LightGBM + XGBoost soft-voting ensemble.
* **Features:** Standard preprocessing (QuantileTransformer, LabelEncoding), ~320 columns.
* **Imbalance Handling:** `SMOTENC(0.3)` to handle the 96:4 imbalance without destroying categorical boundaries.
* **Runtime:** ~3 seeds, Stage 2 disabled by default. Expected to run very safely within the 9-hour limit.

### 2. `solution/master-final-feature-forge.ipynb`
* **Role:** The high-variance, heavy-hitting "Kitchen Sink" architecture.
* **Architecture:** Same 3-model ensemble + `SMOTENC(0.3)`.
* **Advanced Features:** 
  - **KMeans Compass:** 10-cluster distance mapping.
  - **ANOVA Math Interactions:** 180 algebraic combinations of the absolute top-10 F-scored features.
  - **Total Width:** Expanded to ~539 columns.
* **Runtime:** ~3 seeds, Stage 2 disabled by default. Handled safely in local CPU smoke tests. Expected to take a few hours on Kaggle.

---

## 🛠️ Crucial Fixes Applied (The "Claude Review" Patches)
To ensure these notebooks do not crash, OOM, or score terribly due to mathematically invalid configurations, the following fixes were applied and synchronized across BOTH notebooks:
1. **The Categorical Interpolation Bug:** Replaced `BorderlineSMOTE` and standard `SMOTE` with `SMOTENC(categorical_features)`. This prevents fractional category codes (e.g., creating a category `3.7`) that corrupted XGBoost and LightGBM inputs.
2. **Double-Imbalance Overcorrection:** Removed `scale_pos_weight: 25.0` from XGBoost and `class_weight='balanced'` from LightGBM. Stacking these on top of the 30% `SMOTENC` generated massive false positives.
3. **Runtime Slashed:** Reduced ensemble seeds from 10 to 3. Turned off `enable_stage2` (Pseudo-label augmentation) by default to halve runtime. 

---

## 📋 Next Steps (For the Morning)
1. **Run the Notebooks:** Upload both `.ipynb` files to Kaggle. **Run on CPU-only** to utilize the 30GB RAM cap, especially for the wide 539-column Feature Forge matrix.
2. **Check the Logs:** At the end of Stage 1, the script will print a runtime breakdown. If Stage 1 finishes incredibly fast, you can manually flip `CFG['enable_stage2'] = True` and re-run to squeeze out pseudo-labeling gains.
3. **Pick the Submissions:** The notebooks output a hard binary submission calibrated to $t=0.375$ (the historical optimal threshold), plus an array of probed thresholds. Compare the positive counts of the probes against the known ~100-200 target on the LB.
4. **Markdown Requirement:** Kaggle requires a Markdown file explaining the EDA and baseline to final model progression. You will need to write this tomorrow. 

See you in the morning! Good luck on the Leaderboard!
