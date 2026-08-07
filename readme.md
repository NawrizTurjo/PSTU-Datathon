# 🌊 PSTU Datathon — Predictive Maintenance for Coastal Riverine Water Management Stations

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PSTU%20Data%20Craft-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact)

Predict critical failures in off-grid, solar-powered water management stations across remote riverine and coastal char areas in Bangladesh. Protect vulnerable communities from saline water intrusion and ensure uninterrupted access to fresh water.

---

## 🔗 Competition Link & Dataset Setup

- **Official Competition:** [PSTU Data Craft: Transforming Raw Data Into Impact](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact)

### 📥 Downloading the Dataset

The raw dataset should be placed inside the `dataset/` directory at the project root (not tracked by Git).

#### Option A: Via Kaggle CLI (Recommended)
Make sure you have set up your Kaggle API key (`kaggle.json`).

```bash
# 1. Download the competition files
kaggle competitions download -c pstu-data-craft-transforming-raw-data-into-impact

# 2. Extract into the dataset/ directory
mkdir -p dataset
unzip pstu-data-craft-transforming-raw-data-into-impact.zip -d dataset/
```

#### Option B: Manual Download via Browser
1. Visit the [Kaggle Competition Data Tab](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact/data).
2. Click **Download All** (or download `train.csv` and `test.csv` individually).
3. Move/extract the downloaded files into the `dataset/` directory in this workspace:
   ```text
   PSTU-Datathon/
   └── dataset/
       ├── train.csv
       └── test.csv
   ```

---

## 🗂️ Project Structure & Documentation

| Path | Description |
| :--- | :--- |
| 📄 [`overview.md`](overview.md) | Complete background, domain context, problem statement, and metric formula. |
| 📄 [`dataset_description.md`](dataset_description.md) | Detailed schema breakdown, feature groups, and strict submission format rules. |
| 📁 [`dataset_exploration/`](dataset_exploration/README.md) | Automated EDA scripts: Bengali text decoding (910 MB → 63 MB), schema grouping, ghost value hunt (`-999999`). |
| 📄 [`missing-exploration.md`](missing-exploration.md) | Advanced EDA findings: constant/duplicate feature elimination, adversarial validation proof, station ID debunk, label noise analysis. |
| 📄 [`prompt.md`](prompt.md) | Master AI Architect Prompt for solution architecture and multi-directional ideathon generation. |
| 📁 [`ideas/`](ideas/README.md) | Solution strategy roadmap, metric decomposition, GBDT core, threshold engines, and ensemble designs. |

---

## 📊 Competition Summary

- **Task:** Binary classification + probability estimation for 7-day station failure.
- **Dataset Size:** 48,128 train rows, 12,032 test rows, 286 features + 1 target (`Your_Target_Column`).
- **Class Imbalance:** 5.0% positive failure rate (45,722 Normal vs. 2,406 Failure).
- **Custom Composite Metric:**
  $$\text{FinalScore} = (0.30 \times F1) + (0.25 \times ROCAUC) + (0.15 \times Precision) + (0.15 \times Recall) + (0.10 \times BalancedAccuracy) + (0.05 \times Specificity)$$

---

## 🔬 Key EDA Findings

1. **Bengali Text Flag Encoding:** 63 columns originally contained full Bengali sentences ("না, এই স্টেশনটির..."). Converted to `0`/`1` boolean flags, reducing dataset size from 910 MB to 63 MB with 0 information loss.
2. **Ghost Missing Value Marker:** `-999999` occurs exclusively in `base_number_of_dependent_farmers` (0.14% train / 0.19% test rows). Replaced with `NaN`.
3. **Column Hygiene:** 12 columns dropped with zero loss (6 constant columns with 0 variance, 6 exact duplicate column pairs).
4. **Train-Test Covariate Shift:** Adversarial validation AUC = 0.4985 (chance level), confirming train and test sets are i.i.d.
5. **Validation Strategy:** Pseudo-station clustering proved to be coincidental collisions on dominant fill values rather than real station IDs. **`StratifiedKFold`** is the correct leak-free validation strategy.
6. **Conflicting Label Noise:** 7.35% of train rows are exact feature duplicates, with 3.3% having conflicting target labels (identical features, opposite target), requiring soft-labeling or sample-weight adjustments.
7. **Train-Test Overlap:** 7.3% of test rows match a train row exactly on all features.

---

## 🚀 Quickstart & Pipeline Execution

To run the dataset preprocessing and exploration pipeline:

```bash
# 1. Parse schema and group features
python dataset_exploration/01_explore_schema.py

# 2. Convert Bengali boolean text flags to compact numeric format
python dataset_exploration/02_convert_to_numeric.py

# 3. Profile dataset statistics and target correlations
python dataset_exploration/03_profile_and_ghost_hunt.py

# 4. Search sentinels & inspect boolean pairs
python dataset_exploration/04_ghost_value_and_bool_pairs.py
```
