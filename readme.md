# 🌊 PSTU Datathon — Predictive Maintenance for Coastal Riverine Water Management Stations

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PSTU%20Data%20Craft-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/pstu-data-craft-transforming-raw-data-into-impact)

Build a high-performance machine learning solution to predict critical 7-day failures in off-grid, solar-powered water management stations across remote riverine and coastal char (island) regions of Bangladesh. Protecting vulnerable agricultural lands and drinking water sources from tidal saline intrusion.

> ## ⛔ DATASET WITHDRAWN — FINDINGS PENDING RE-VERIFICATION
>
> The organizers announced (2026-08-08) that the released dataset **contained leaks** and
> that the **entire dataset will be re-uploaded**.
>
> **Every measured number below was derived from the withdrawn dataset.** The scripts and
> the notebook pipeline still apply; the specific figures do not. Re-run
> `dataset_exploration/01`–`10` against the new data before trusting anything here.
>
> See [`CLAUDE.md`](CLAUDE.md) for exactly what survives and what must be re-derived.

**Current status:** EDA pipeline, solution roadmap ([`ideas/`](ideas/README.md)) and a runnable Kaggle notebook ([`solution/`](solution/)) are all built and smoke-tested — against the withdrawn dataset. Nothing submitted to the leaderboard. Work paused pending re-upload.

> 🚀 **Running it (after the re-upload):** upload [`solution/pstu_kaggle_solution.ipynb`](solution/pstu_kaggle_solution.ipynb) to Kaggle with the dataset attached and run all cells. CPU only, ~15–30 min, writes `submission.csv`. **First fix the hardcoded `CONSTANT_COLS` / `DUPLICATE_COLS` lists** — they are specific to the old data (see `CLAUDE.md` → Landmines).

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
├── CLAUDE.md                            # Project memory: measured findings, conventions, dead ends
├── overview.md                          # Problem domain, background context & scoring rules
├── dataset_description.md               # Schema specification, feature groups & submission format
├── missing-exploration.md               # Gap-analysis checklist that drove scripts 05–10 (now addressed)
├── prompt.md                            # AI Architect master prompt, grounded in measured EDA findings
├── dataset_exploration/                 # Automated EDA scripts & profiling output
│   ├── 01_explore_schema.py             # Column classification & group assignment
│   ├── 02_convert_to_numeric.py         # Bengali text flag decoder (910 MB -> 63 MB)
│   ├── 03_profile_and_ghost_hunt.py     # Outlier, correlation & distribution profiling
│   ├── 04_ghost_value_and_bool_pairs.py # Sentinel validation & boolean-pair correlation
│   ├── 05_zero_variance_and_duplicate_cols.py  # Constant & duplicate column detection
│   ├── 06_adversarial_validation.py     # Train-vs-test covariate shift test
│   ├── 07_pseudo_station_clustering.py  # Station-ID recovery attempt (result: none exists)
│   ├── 08_secondary_sentinels_and_sparsity.py  # Sentinel scan & zero-inflation profiling
│   ├── 09_duplicate_rows.py             # Duplicate rows, conflicting labels, train-test overlap
│   ├── 10_metric_threshold_simulation.py# Baseline model + composite threshold sweep
│   ├── README.md                        # Comprehensive findings summary & pipeline guide
│   ├── converted_train.csv              # [generated] 63 MB decoded train — use this, not the raw CSV
│   ├── converted_test.csv               # [generated] 16 MB decoded test
│   └── (16 report / CSV outputs)        # schema, profile, ghost-value, adversarial, sparsity,
│                                        #   duplicate-row & threshold-sweep reports
├── ideas/                               # Measured solution roadmap & strategy index
│   ├── README.md                        # Priority index & benchmark baseline table
│   ├── 00-foundation/                   # Data hygiene, StratifiedKFold CV & metric decomposition
│   │   ├── README.md                    # Foundation protocol & submission traps
│   │   └── metric-decomposition.md      # What the composite score actually rewards
│   ├── 01-gbdt-core/                    # LightGBM / CatBoost / XGBoost core architectures
│   ├── 02-feature-engineering/          # Santander row-stats, domain physics & ratio features
│   ├── 03-threshold-engine/             # Composite threshold search (+0.018 measured gain)
│   ├── 04-ensemble-diversity/           # Rank-averaging, GBDT blending & neural tabular assessment
│   ├── 05-label-noise/                  # Strategies for the 3.3% conflicting-label rows
│   └── 06-dead-ends/                    # Measured non-working ideas & time-saving traps
└── solution/                            # Runnable Kaggle deliverable
    ├── pstu_kaggle_solution.ipynb       # Upload this to Kaggle and run all cells
    └── pstu_kaggle_solution.py          # Same source, `# %%` cell-marked (for diffing/editing)
```

> **Note:** `converted_train.csv` / `converted_test.csv` are generated artifacts (79 MB combined)
> and are **not** currently in `.gitignore`. Either add them or regenerate with script `02`
> instead of committing them.

---

## 📄 Key Documentation Map

| File / Folder | Purpose & Key Details |
| :--- | :--- |
| 📄 [`overview.md`](overview.md) | Competition background, problem statement, evaluation objectives, and official formula. |
| 📄 [`dataset_description.md`](dataset_description.md) | Full breakdown of Base Attributes, Sensor Readings, Financial Metrics, Target, and submission rules. |
| 📁 [`dataset_exploration/`](dataset_exploration/README.md) | The 10-script EDA pipeline and its complete findings summary. **Start here to understand the data.** |
| 📄 [`missing-exploration.md`](missing-exploration.md) | Gap-analysis checklist listing 6 exploration areas the first EDA pass skipped. All 6 have since been addressed by scripts `05`–`10`; kept as a record of what was asked and why. |
| 📄 [`prompt.md`](prompt.md) | Reusable master prompt for generating solution architecture, pre-loaded with confirmed EDA findings so the model builds on them instead of re-deriving (or contradicting) them. |
| 📁 [`ideas/`](ideas/README.md) | Priority-ordered solution roadmap (00 Foundation → 03 Threshold Engine → 01 GBDT Core → 02 Feature Eng → 04 Ensembles → 05 Label Noise → 06 Dead Ends). |
| 📄 [`ideas/00-foundation/metric-decomposition.md`](ideas/00-foundation/metric-decomposition.md) | **Highest-value document in the repo.** Algebraic decomposition of the composite metric and what it rewards. |
| 📄 [`ideas/06-dead-ends/`](ideas/06-dead-ends/README.md) | Six tested-and-rejected approaches with the numbers that killed them. 10-minute read, saves hours. |
| 📁 [`solution/`](solution/pstu_kaggle_solution.ipynb) | **The runnable deliverable.** End-to-end Kaggle notebook: streaming Bengali decode → preprocessing → feature engineering → LightGBM/XGBoost/CatBoost → rank-blend → cut-point optimization → validated submission. |
| 📄 [`CLAUDE.md`](CLAUDE.md) | Project memory: every measured finding, settled conventions, and the dead-end list, so a new session resumes without re-deriving anything. |

---

## 📊 Competition Benchmarks & Metric Landscape

The official evaluation metric is a weighted sum of 6 components:

$$\text{FinalScore} = (0.30 \times F1) + (0.25 \times ROCAUC) + (0.15 \times Precision) + (0.15 \times Recall) + (0.10 \times BalancedAccuracy) + (0.05 \times Specificity)$$

### It collapses to five terms

Substituting $\text{BalancedAccuracy} = (Recall + Specificity)/2$ yields an **exactly equivalent** form (verified to machine precision):

$$\text{FinalScore} = 0.30 \, F1 + 0.25 \, AUC + 0.15 \, Precision + \mathbf{0.20} \, Recall + 0.10 \, Specificity$$

**Recall carries more effective weight than Precision (0.20 vs 0.15)** — the official formula obscures this by splitting recall's weight across two terms. At the margin, err toward predicting *more* positives.

### Measured Benchmark Scores

| Model Strategy | OOF ROC-AUC | Composite | Notes |
| :--- | :---: | :---: | :--- |
| **All-Zeros Baseline** (`0` for all rows) | — | `0.3028` | Trivial floor |
| **All-Ones Baseline** (`1` for all rows) | — | `0.4388` | **Any model must beat this** |
| **RandomForest** (naive `0.5` threshold) | `0.8107` | `0.4989` | Standard default cutoff |
| **RandomForest** (tuned threshold `0.60`) | `0.8107` | `0.5167` | **+0.018 from threshold tuning alone** |
| **HistGradientBoosting** (tuned threshold `0.53`) | `0.8189` | `0.5269` | Strong tree baseline |
| **Realistic competitive ceiling** (estimate) | `~0.83–0.84` | **`~0.54 – 0.56`** | Tight competitive margin |

Tree-model scores are 5-fold `StratifiedKFold` out-of-fold. Degenerate baselines are computed on full train assuming a 0.811 AUC probability column.

**Two consequences worth internalizing:** the floor is high (all-ones scores 0.4388 with no model), and the entire competitive band is roughly **0.50–0.56** — so a broken submission or an untuned threshold costs more than any modelling cleverness gains. At the measured optimum, precision is only `0.183`; **precision is the binding constraint**, and it can only be relaxed by better ranking.

---

## 🔬 Core Exploratory Insights

1. **Bengali Text Flag Encoding:** 63 columns were full Bengali text sentences (`"হ্যাঁ..."` / `"না..."`). Decoded into clean `1`/`0` booleans, shrinking train from **910 MB to 63 MB** with zero undecodable values.
2. **Ghost Missing Value Marker:** Exactly one column (`base_number_of_dependent_farmers`) contains `-999999` (0.14% train / 0.19% test rows) — a physically impossible negative farmer count. Handle as `NaN`. An exhaustive scan found **no secondary sentinel** anywhere else.
3. **Santander Origin & Column Hygiene:** The numeric skeleton (`num_var*`, `num_op_var*`) mirrors Santander Customer Satisfaction. **12 columns dropped with zero information loss** (6 constant, 6 exact duplicates). 143 of 223 numeric columns are ≥90% zero.
4. **Leak-Free CV Validation:** Adversarial validation AUC = **0.4985** (train/test are i.i.d., no covariate shift). Station-ID grouping by `base_*` features was proven to be coincidental collisions on a dominant fill value. **`StratifiedKFold` is the validated CV strategy — not `GroupKFold`.**
5. **Label Noise:** 7.35% of train rows are exact feature duplicates; **3.3% sit in groups with conflicting targets** (identical features, different outcome). This caps achievable **precision/F1 but not AUC** — an oracle using per-group means reaches AUC 0.9993, so ranking headroom remains.
6. **Train-Test Overlap is a Dead End:** ~7% of test rows (841) exactly match a train row, but exploiting this **hurts**. The model beats the lookup on precisely those rows (AUC `0.8295` vs `0.7210`), and every blend weight lowers the composite (full override: **−0.026**). Those rows collide because they are the sparse "nothing happened" default profile, not because they are the same station. See [`ideas/06-dead-ends/`](ideas/06-dead-ends/README.md).

---

## 💻 Kaggle Runtime Notes

This problem needs far less hardware than Kaggle provides:

- The cleaned dataset is **63 MB / 48,128 rows × 286 features**. It fits in RAM many times over.
- LightGBM trains in **~30–90 seconds on CPU**; a full 5-fold CV is a few minutes.
- **Do not use the GPU for tree models** — at this data size, GPU LightGBM is typically *slower* than CPU due to kernel launch overhead.
- The only GPU-relevant option is a neural tabular model, which would use **under 2 GB of the 16 GB VRAM** and is unlikely to beat the GBDTs standalone (see [`ideas/04-ensemble-diversity/`](ideas/04-ensemble-diversity/README.md)).

**Practical implication:** iterate entirely in CPU notebooks (no weekly quota) and preserve GPU hours.

---

## 🚀 Execution Guide

Scripts are idempotent and safe to re-run. Steps `01` and `02` are prerequisites for everything after.

```bash
# --- Core pipeline: schema, decoding, profiling, sentinel detection ---
python dataset_exploration/01_explore_schema.py              # Column classification & grouping
python dataset_exploration/02_convert_to_numeric.py          # Bengali flag decode (910 MB -> 63 MB)
python dataset_exploration/03_profile_and_ghost_hunt.py      # Profiling, correlations, outlier scan
python dataset_exploration/04_ghost_value_and_bool_pairs.py  # Sentinel confirmation & boolean pairs

# --- Advanced validation: hygiene, shift, grouping, noise, metric behaviour ---
python dataset_exploration/05_zero_variance_and_duplicate_cols.py  # 12 droppable columns
python dataset_exploration/06_adversarial_validation.py            # Train-vs-test shift test
python dataset_exploration/07_pseudo_station_clustering.py         # Station-ID recovery attempt
python dataset_exploration/08_secondary_sentinels_and_sparsity.py  # Sentinel scan & sparsity
python dataset_exploration/09_duplicate_rows.py                    # Duplicates & conflicting labels
python dataset_exploration/10_metric_threshold_simulation.py       # Baseline + threshold sweep
```

Scripts `06` and `10` fit models and take a few minutes each; the rest complete in seconds.

**Requirements:** `pandas`, `numpy`, `scikit-learn`. Scripts `06` and `10` use `RandomForestClassifier` / `HistGradientBoostingClassifier` from scikit-learn — no LightGBM/XGBoost/CatBoost needed for the EDA pipeline itself.

---

## 🧭 Where to Go Next

Modelling hasn't started. The recommended path, in order:

1. Read [`ideas/00-foundation/metric-decomposition.md`](ideas/00-foundation/metric-decomposition.md) — understand what the score rewards.
2. Skim [`ideas/06-dead-ends/`](ideas/06-dead-ends/README.md) — avoid six approaches already measured as non-working.
3. Implement [`ideas/00-foundation/`](ideas/00-foundation/README.md) — preprocessing, CV protocol, `composite_score()`, submission validator.
4. Implement [`ideas/03-threshold-engine/`](ideas/03-threshold-engine/README.md) — **the cheapest large gain (+0.018)**.
5. Build models per [`ideas/01-gbdt-core/`](ideas/01-gbdt-core/README.md), then layer on [`02`](ideas/02-feature-engineering/README.md) and [`04`](ideas/04-ensemble-diversity/README.md).
