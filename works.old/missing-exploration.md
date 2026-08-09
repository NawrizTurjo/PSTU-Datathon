# 🔍 Missing Dataset Exploration Items

While the initial automated dataset exploration (`dataset_exploration/`) successfully identified column types, compressed Bengali boolean flags (910 MB → 63 MB), discovered the primary `-999999` ghost value in `base_number_of_dependent_farmers`, and computed basic target correlations, several critical exploratory analyses were omitted. 

Below is the list of missing exploration checks necessary to maximize competitive performance.

---

## 1. 🧬 Santander Synthetic Origin & Feature Deduplication
- **Known Finding:** The numeric columns (`num_var*`, `num_op_var*`) follow Santander Customer Satisfaction naming conventions.
- **Missing Exploration:**
  - **Zero-Variance / Constant Columns:** Santander-derived datasets often contain 30+ columns that are identical 0s across all train and test rows. These need explicit detection and removal.
  - **Duplicate Columns:** Check for columns with identical values across all rows (e.g. `ind_var2_0` vs `ind_var2`).
  - **Santander Column Mapping & Un-obfuscation:** Map Santander features back to their original domain definitions (e.g., `num_var38` = financial balance / sales, `var3` = country code sentinel `-999999`).

---

## 2. 🛡️ Adversarial Validation & Train-Test Covariate Shift
- **Missing Exploration:**
  - **Train vs. Test Distribution Shift:** No adversarial validation model (training a classifier to predict `is_test`) was run.
  - **Out-of-Distribution Features:** Check if `test.csv` contains values, ranges, or boolean flag distributions not present in `train.csv`.
  - **Feature Stability Analysis:** Identify features whose distribution differs significantly between train and test sets to avoid overfitting to train-specific noise.

---

## 3. 📍 Implicit Station Clustering & Geographic Entity Recovery
- **Missing Exploration:**
  - **Station ID Recovery:** The dataset lacks an explicit `station_id`. However, stations can likely be reconstructed by grouping static Base Attributes (`base_number_of_dependent_farmers`, `base_station_installation_age_years`, `base_distance_from_coastal_river_km`).
  - **Group Leakage / Group K-Fold Validation:** If multiple rows belong to the same physical station, random `StratifiedKFold` will cause severe data leakage across folds. Exploration is needed to determine if `GroupKFold` by pseudo-station ID is required.
  - **Time-Series / Sequential Row Ordering:** Check if row order in `train.csv` / `test.csv` represents a chronological sequence of logs per station.

---

## 4. 🕳️ Secondary Sentinel Values & Sparsity/Zero-Inflation Profiling
- **Missing Exploration:**
  - **Secondary Sentinels:** In Santander, `-999999` in `var3` is missing country code, but `9999999999` or `-1` or `99` often represent missing values in other numeric columns. Check min/max extreme values across all 287 columns.
  - **Zero-Inflation Ratios:** Many financial and operational count columns are 90%+ sparse. Profiling the exact zero-ratio per column is needed to design sparse-aware feature engineering.

---

## 5. 🔁 Exact & Near-Duplicate Row Detection
- **Missing Exploration:**
  - **Duplicate Rows in Train:** Are there identical feature rows in `train.csv` with conflicting target labels?
  - **Train-Test Overlap:** Are there exact feature matches between `train.csv` and `test.csv`?

---

## 6. 📐 Metric Landscape & Multi-Threshold Simulation
- **Missing Exploration:**
  - **Composite Metric Response:** The competition metric combines 6 scores: $F1 (0.30) + ROC\text{-}AUC (0.25) + Precision (0.15) + Recall (0.15) + BalancedAccuracy (0.10) + Specificity (0.05)$.
  - **Threshold Behavior:** F1, Precision, Recall, Specificity, and Balanced Accuracy are discrete threshold-dependent metrics. Standard thresholding at `0.5` is suboptimal. Simulation of how decision thresholds affect the composite score under extreme class imbalance (5% positive rate) is missing.

---

## 🔬 Recommended Next Exploratory Actions

1. Run **Zero-Variance & Duplicate Column Filter** across `converted_train.csv` and `converted_test.csv`.
2. Run **Adversarial Validation** (`train` vs `test` ROC-AUC score).
3. Attempt **Pseudo-Station Clustering** using `base_*` columns and check target distribution per cluster.
4. Scan all numeric columns for secondary sentinel candidates (`-1`, `999`, `999999`, `9999999999`).
5. Build a **Custom Composite Metric Evaluation Function** with threshold search optimizer.
