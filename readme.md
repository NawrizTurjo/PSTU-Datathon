# 🌊 PSTU Datathon — Predictive Maintenance for Coastal Riverine Water Management Stations

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PSTU%20Data%20Craft-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact)

Build a high-performance machine learning solution to predict critical 7-day failures in off-grid, solar-powered water management stations across remote riverine and coastal char (island) regions of Bangladesh. Protecting vulnerable agricultural lands and drinking water sources from tidal saline intrusion.

---

## 🔗 Competition Link & Dataset Setup

- **Official Competition:** [PSTU Data Craft: Transforming Raw Data Into Impact](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact)

### 📥 Downloading the Dataset

Store the raw dataset in the root `dataset/` directory (ignored by `.gitignore`).

#### Option A: Kaggle CLI (Recommended)
```bash
# 1. Download the competition zip file
kaggle competitions download -c pstu-data-craft-transforming-raw-data-into-impact

# 2. Extract into dataset/
mkdir -p dataset
unzip pstu-data-craft-transforming-raw-data-into-impact.zip -d dataset/
```

#### Option B: Manual Download
1. Visit the [Kaggle Competition Data Tab](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact/data).
2. Download `train.csv` and `test.csv` (or the complete ZIP file).
3. Place them in the workspace under `dataset/`:
   ```text
   PSTU-Datathon/
   └── dataset/
       ├── train.csv
       └── test.csv
   ```

---

## 🗂️ Complete Directory & File Architecture

```text
PSTU-Datathon/
├── overview.md                    # Problem domain, background context & scoring rules
├── dataset_description.md         # Schema specification, feature groups & submission format
├── missing-exploration.md         # In-depth EDA: zero-variance drops, iid proof, label noise
├── prompt.md                      # AI Architect Master Prompt for solution ideation
├── dataset_exploration/           # Automated EDA scripts & profiling output
│   ├── 01_explore_schema.py       # Column classification & group assignment
│   ├── 02_convert_to_numeric.py   # Bengali text flag decoder (910 MB -> 63 MB)
│   ├── 03_profile_and_ghost_hunt.py # Outlier, correlation & distribution profiling
│   ├── 04_ghost_value_and_bool_pairs.py # Sentinel validation & boolean-pair correlation
│   ├── README.md                  # Comprehensive findings summary & pipeline guide
│   └── (reports & summaries)      # Schema, profile, ghost-value & boolean pair reports
└── ideas/                         # Measured Solution Roadmap & Strategy Index
    ├── README.md                  # Priority index & benchmark baseline table
    ├── 00-foundation/             # Data hygiene, StratifiedKFold CV & metric decomposition
    │   ├── README.md              # Foundation protocol & submission traps
    │   └── metric-decomposition.md# Deep mathematical breakdown of custom composite score
    ├── 01-gbdt-core/              # LightGBM / CatBoost / XGBoost core architectures
    ├── 02-feature-engineering/    # Santander row-stats, domain physics & ratio features
    ├── 03-threshold-engine/       # Joint composite threshold search (+0.018 score gain)
    ├── 04-ensemble-diversity/     # Rank-averaging, GBDT blending & neural tabular assessment
    ├── 05-label-noise/            # Resolution strategy for 3.3% conflicting-label rows
    └── 06-dead-ends/              # Measured non-working ideas & time-saving traps
```

---

## 📄 Key Documentation Map

| File / Folder | Purpose & Key Details |
| :--- | :--- |
| 📄 [`overview.md`](overview.md) | Competition background, problem statement, evaluation objectives, and official formula. |
| 📄 [`dataset_description.md`](dataset_description.md) | Full breakdown of Base Attributes, Sensor Readings, Financial Metrics, Target, and submission rules. |
| 📁 [`dataset_exploration/`](dataset_exploration/README.md) | Automated EDA pipeline: decodes Bengali boolean flags (`হ্যাঁ`/`না`), identifies `-999999` sentinel in `base_number_of_dependent_farmers`. |
| 📄 [`missing-exploration.md`](missing-exploration.md) | Advanced EDA: identifies 12 zero-information columns (6 constant, 6 duplicates), proves i.i.d train/test split (adversarial AUC 0.4985), debunks pseudo-station IDs, and quantifies label noise. |
| 📄 [`prompt.md`](prompt.md) | Reusable Master AI Architect Prompt to generate system architecture, multi-directional strategy, and solution proposals. |
| 📁 [`ideas/`](ideas/README.md) | Priority-ordered solution strategy (00 Foundation → 03 Threshold Engine → 01 GBDT Core → 02 Feature Eng → 04 Ensembles → 05 Label Noise → 06 Dead Ends). |
| 📄 [`ideas/00-foundation/metric-decomposition.md`](ideas/00-foundation/metric-decomposition.md) | Mathematical decomposition of the 6 sub-metrics and exact submission threshold dynamics. |

---

## 📊 Competition Benchmarks & Metric Landscape

The evaluation metric is a weighted sum of 6 components:
$$\text{FinalScore} = (0.30 \times F1) + (0.25 \times ROCAUC) + (0.15 \times Precision) + (0.15 \times Recall) + (0.10 \times BalancedAccuracy) + (0.05 \times Specificity)$$

### Measured Benchmark Scores (5-Fold CV OOF)

| Model Strategy | Composite Score | Notes |
| :--- | :---: | :--- |
| **All-Zeros Baseline** (`0` for all rows) | `0.3028` | Pure accuracy floor |
| **All-Ones Baseline** (`1` for all rows) | `0.4388` | Non-trivial score floor without modeling |
| **RandomForest Baseline (Naive 0.5 threshold)** | `0.4987` | Standard default cutoff |
| **RandomForest Baseline (Tuned threshold ~0.60)** | `0.5167` | **+0.018 gain** from threshold tuning alone |
| **HistGradientBoosting (LightGBM equivalent)** | `0.5269` | Strong tree baseline with tuned threshold |
| **Realistic Competitive Ceiling (Target)** | **`~0.5400 – 0.5600`** | Tight competitive margin |

---

## 🔬 Core Exploratory Insights

1. **Bengali Text Flag Encoding:** 63 columns were full Bengali text sentences (`"হ্যাঁ..."` / `"না..."`). Decoded into clean `1`/`0` booleans, shrinking train size from 910 MB to 63 MB.
2. **Ghost Missing Value Marker:** Exactly one column (`base_number_of_dependent_farmers`) contains `-999999` (0.14% train / 0.19% test rows). Handled as `NaN`.
3. **Santander Origin & Column Hygiene:** The numeric skeleton (`num_var*`, `num_op_var*`) mirrors Santander Customer Satisfaction. 12 columns dropped with zero loss (6 constant columns, 6 identical column duplicates).
4. **Leak-Free CV Validation:** Adversarial validation score = 0.4985 (i.i.d split). Station ID grouping by `base_*` features was proven to be coincidental collisions on a dominant fill value. **`StratifiedKFold`** is the validated CV strategy.
5. **Label Noise & Exact Overlap:** 7.35% of train rows are exact feature duplicates, with 3.3% having conflicting targets. 7.3% of test rows match a train row exactly.

---

## 🚀 Execution Guide

```bash
# Step 1: Parse dataset schema and group columns
python dataset_exploration/01_explore_schema.py

# Step 2: Convert raw Bengali boolean flags to compact numeric CSVs (910 MB -> 63 MB)
python dataset_exploration/02_convert_to_numeric.py

# Step 3: Run numerical profiling and sentinel detection
python dataset_exploration/03_profile_and_ghost_hunt.py

# Step 4: Validate ghost values and analyze boolean flag pair correlations
python dataset_exploration/04_ghost_value_and_bool_pairs.py
```
