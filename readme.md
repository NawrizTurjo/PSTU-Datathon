# 🏦 PSTU Data Thon 2026 Vol-1 — Account Instability Prediction

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PSTU%20Data%20Thon%202026-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/pstu-data-thon-2026-vol-1/)

Predict whether a financial account will be flagged **at-risk** (`TARGET = 1`) from 350
anonymized behavioural features derived from transaction history, digital engagement and
account metadata.

Binary classification · 76,020 train rows · 3.96% positive rate · evaluated on **F1**.

> **Note on history:** an earlier competition (predictive maintenance for solar water stations)
> was withdrawn by the organizers because its dataset contained leaks. That work is archived in
> [`works.old/`](works.old/) and is **not** valid for this competition — different domain, data,
> and metric. See [`CLAUDE.md`](CLAUDE.md).

---

## ⏱️ Timeline (GMT+6)

| Event | When |
|---|---|
| Competition starts | 9 Aug 2026, 18:00 |
| **Final submission deadline** | **13 Aug 2026, 18:00** |
| Private leaderboard reveal | 13 Aug 2026, 18:30 |
| Inference notebook deadline | 13 Aug 2026, 23:59 |
| Winners announced | 15 Aug 2026 |

## 📊 How you're actually scored

The leaderboard is only half the mark:

| Component | Weight | Notes |
|---|---|---|
| Public LB | 10% | live, on just **10%** of test data — noisy, don't chase it |
| Private LB | 40% | on 50% of test data |
| **Hidden test** | **40%** | 40% of unseen data, executed via your **inference notebook** |
| Presentation + code + format | 10% | |

**Implication:** a reproducible, deterministic inference notebook is worth as much as
leaderboard performance. If it fails to run or gives different results, 40% is gone.

---

## 📥 Data

Place the competition files in `pstu-data-thon-2026-vol-1/`:

```bash
kaggle competitions download -c pstu-data-thon-2026-vol-1
unzip pstu-data-thon-2026-vol-1.zip -d pstu-data-thon-2026-vol-1/
```

```text
pstu-data-thon-2026-vol-1/
├── train.csv               # 76,020 × 351  (feat_1..feat_350 + TARGET)
├── test.csv                # 60,654 × 351  (feat_1..feat_350 + id  <- id is the LAST column)
└── sample_submission.csv   # 60,654 × 2    (id, TARGET)
```

| | |
|---|---|
| Positive rate | **3.957%** (3,008 / 76,020) |
| Missing values | none |
| Feature dtypes | 212 int · 132 float · **6 string** |

---

## 🔍 First-pass findings

Measured on 2026-08-09. Full EDA still to come — see [`CLAUDE.md`](CLAUDE.md) for the plan.

### 1. Six "numerical" features are actually categorical

The competition description states all 350 features are numerical. **It is wrong** — six are
high-cardinality string codes:

| Column | Levels | Prefix | Levels only in test |
|---|---:|---|---:|
| `feat_142` | 2,333 | `PRD_*` | **55** |
| `feat_325` | 1,710 | `SEG_*` | **27** |
| `feat_157` | 627 | `PRV_*` | **8** |
| `feat_320` | 119 | `CH_*` | 0 |
| `feat_337` | 39 | `OFC_*` | 0 |
| `feat_318` | 12 | `PERF_*` | 0 |

Unseen categories appear at inference time, so any encoding needs an explicit fallback or the
hidden-test run will break.

### 2. A `-999999` sentinel hides in `feat_109`

Exactly one column carries it. `feat_169` also reaches ≈ `-1.11e8` and may be a second
sentinel — to be confirmed.

### 3. 83 columns carry zero information

28 constant in both train and test, plus 55 redundant across 20 duplicate-column groups.

### 4. Very sparse

252 of 350 numeric columns are ≥90% zero.

### 5. No duplicate rows, no label conflicts

Unlike the withdrawn dataset (which had 3.3% of rows with identical features and opposite
labels), this data shows **no proven noise ceiling** — higher scores may be genuinely reachable.

---

## ⚠️ The metric is ambiguous — settle it first

The competition page contradicts itself:

- Evaluation section — "evaluated using the **F1 Score**"
- Submission section — "converted to binary predictions using a threshold of 0.5 before
  computing the **Macro F1** score"

At a 3.96% positive rate the two are worlds apart:

| Submission | Binary F1 | Macro F1 |
|---|---:|---:|
| all zeros | **0.0000** | **0.4899** |
| all ones | 0.0761 | 0.0381 |

**Probe it with one early submission:** submit all zeros. Score ≈ 0.49 → macro F1;
≈ 0.00 → binary F1. This determines the whole thresholding strategy, so do it before tuning.

### You still control the threshold

The grader cuts at a fixed 0.5, but you choose the numbers you submit — so submit hard `0`/`1`,
or rank-transform so your chosen operating point lands at 0.5. Since the metric is F1 only
(no AUC term), **only the binary decision earns anything**; ranking matters solely through the
split it produces.

---

## 📤 Submission format

```csv
id,TARGET
3496,0
17271,0
44259,0
```

- `id` comes from `test.csv`'s `id` column — **not contiguous, not `0..n-1`**. Never rebuild it
  with `range()`.
- `test.csv` row order already matches `sample_submission.csv`, so copying the sample and
  overwriting `TARGET` is safe.
- Exactly 60,654 rows plus header, two columns.

---

## 🚦 Rules that shape the solution

| Rule | Status |
|---|---|
| External datasets | ❌ Prohibited — provided files only |
| Test-set tampering / label reverse-engineering | ❌ Instant disqualification |
| Generating targets with an LLM | ❌ Explicitly banned |
| SMOTE / augmentation / feature engineering on train | ✅ Allowed |
| Pre-trained models | ⚠️ Allowed **with disclosure** |
| Inference notebook | ✅ **Mandatory**, must be deterministic |

---

## 🗂️ Repository layout

```text
PSTU-Datathon/
├── CLAUDE.md                    # Project memory: findings, rules, workflow. Read first.
├── readme.md                    # This file
├── overview.md                  # Raw paste from the competition Overview tab
├── dataset_description.md       # Raw paste from the Data + Rules tabs
├── pstu-data-thon-2026-vol-1/   # Competition CSVs (gitignored)
├── tools/
│   └── to_ipynb.py              # Converts a `# %%` cell-marked .py into a .ipynb
└── works.old/                   # Archived work from the WITHDRAWN previous competition
```

Planned, following the process that worked last time:

```text
├── dataset_exploration/         # Stage 1 — numbered EDA scripts + plain-text reports
├── prompt.md                    # Stage 2 — idea-generation prompt, preloaded with findings
├── ideas/                       # Stage 3 — priority-ordered solution roadmap (Opus)
└── solution/                    # Stage 4 — Kaggle notebook + mandatory inference notebook
```

## 🧭 Workflow

1. **Explore** → `dataset_exploration/`: schema, categoricals, sentinels, constants/duplicates,
   adversarial validation, metric behaviour, plus leak diagnostics and an honest baseline.
2. **Prompt** → `prompt.md`: master prompt pre-loaded with Stage 1's measured findings
   (including confirmed negatives, so the model doesn't re-propose them).
3. **Ideate** → `ideas/`: switch to Opus, generate one folder per direction with a priority
   index and honest expected gains.
4. **Build** → `solution/`: end-to-end Kaggle notebook, plus the mandatory deterministic
   inference notebook.

Detailed directions for every stage are in [`CLAUDE.md`](CLAUDE.md).
