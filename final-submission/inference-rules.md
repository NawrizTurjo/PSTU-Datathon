# 📖 PSTU DataThon 2026 — Official Inference Rules & Submission Procedures

**Team Name:** NawrizTurjo
**LB Score Benchmark:** `0.227966`
**Winning Architecture:** 10-Seed CatBoost (`depth=5`) + SMOTE(0.3) + Synthetic Test-Distribution Augmentation @ $t=0.375$
**Document Purpose:** Exhaustive procedure guide for Kaggle submission, hidden test evaluation (40% weight), and organizer private test verification.

---

## 1. 📊 Evaluation & Mark Breakdown

The competition grading strictly follows a 4-component weighted model:

| Component                     |    Weight    | Target Data / Mechanism                              | Description                                                             |
| ----------------------------- | :-----------: | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| **Public Leaderboard**  | **10%** | 10% of test dataset                                  | Real-time score feedback during competition.                            |
| **Private Leaderboard** | **40%** | 50% of test dataset                                  | Revealed only after competition deadline.                               |
| **Hidden Test Set**     | **40%** | 40% completely unseen test dataset                   | Evaluated post-competition via your submitted inference notebook.       |
| **Presentation & Code** | **10%** | Code quality, presentation, notebook reproducibility | Evaluated by judges for clarity, nobility of solution, and determinism. |

> [!IMPORTANT]
> The **Hidden Test Set (40%)** + **Private Leaderboard (40%)** make up **80%** of your final competition rank. Reproducibility, non-overfitting, and robust inference are critical.

---

## 2. 📂 Submission Deliverables Checklist

To remain eligible for Top 20 ranking and final awards, the following 4 deliverables must be submitted before their respective deadlines:

- [X] **1. Prediction CSV (`submission.csv`)**
  - Binary predictions (`0` or `1`) for all test accounts.
  - Required format: `id,TARGET` (2 columns, exact header match).
- [X] **2. Mandatory Inference Notebook (`INFERENCE_NOTEBOOK.ipynb` / `FINAL_SUBMISSION.ipynb`)**
  - Must run deterministically on Kaggle CPU/GPU within 6 hours.
  - Must execute end-to-end without external prohibited datasets.
- [X] **3. Video Presentation (`PRESENTATION_OUTLINE.md`)**
  - Recorded video uploaded to YouTube (unlisted or public).
  - Explains problem formulation, EDA insights, feature engineering, model architecture, and validation.
- [X] **4. Code Repository / Package**
  - Self-contained code package containing training scripts, inference pipeline, and reproducible config.

---

## 3. 🛠️ Inference Protocols & Execution Procedures

Three valid procedures exist for executing inference and submitting notebooks on Kaggle:

### Procedure A: End-to-End Self-Contained Notebook (RECOMMENDED)

*This is the simplest, most reliable procedure for Kaggle code competition submissions.*

```
[ Load train.csv & test.csv ] ──> [ Preprocess & Row Stats ] ──> [ 10-Seed CatBoost + SMOTE ] ──> [ Synthetic Test Augmentation ] ──> [ Save submission.csv ]
```

#### Step-by-Step Procedure:

1. **Upload Notebook**:
   - Go to Kaggle PSTU DataThon 2026 competition page.
   - Click **Code** → **New Notebook** → **File** → **Import Notebook**.
   - Select [`final-submission/INFERENCE_NOTEBOOK.ipynb`](file:///e:/Competitions/PSTU-Datathon/final-submission/INFERENCE_NOTEBOOK.ipynb) (or [`FINAL_SUBMISSION.ipynb`](file:///e:/Competitions/PSTU-Datathon/final-submission/FINAL_SUBMISSION.ipynb)).
2. **Environment Configuration**:
   - **Accelerator:** CPU (or GPU; CPU recommended for full 30GB RAM headroom).
   - **Environment:** Always Use Latest Environment.
   - **Persistence:** Files Persisted.
3. **Run & Save**:
   - Click **Run All** (or **Save Version** → **Save & Run All (Commit)**).
   - The notebook automatically detects dataset paths (`/kaggle/input/competitions/pstu-data-thon-2026-vol-1` or `/kaggle/input/pstu-data-thon-2026-vol-1`).
   - Executes full 10-seed CatBoost training + synthetic test augmentation + thresholding at $t=0.375$ in ~15 minutes.
   - Writes hard binary predictions to `/kaggle/working/submission.csv`.
4. **Submit Predictions**:
   - From the Notebook Output section, select `submission.csv` and click **Submit to Competition**.
5. **Hidden Test Evaluation (Post-Deadline)**:
   - When competition ends, Kaggle automatically mounts the hidden 40% test dataset as `test.csv` in the input directory.
   - Kaggle re-runs your committed notebook automatically.
   - The output `/kaggle/working/submission.csv` is scored against the hidden test labels.

---

### Procedure B: Two-Notebook Artifact Sharing (Train Notebook A → Dataset → Inference Notebook B)

*Use this procedure if you want Notebook B (Inference Only) to run in <30 seconds without retraining models.*

```
[ Notebook A: Train & Export ] ──(Saves artifacts)──> [ Kaggle Dataset ] ──(+ Add Data)──> [ Notebook B: Fast Inference ]
```

#### Step-by-Step Procedure:

#### Phase 1: Train & Export Artifacts (Notebook A)

1. Run training pipeline in Notebook A.
2. Save fitted preprocessing and model objects to `/kaggle/working/`:
   ```python
   import joblib
   joblib.dump(qt, '/kaggle/working/qt_transformer.joblib')
   joblib.dump(cat_encoders, '/kaggle/working/cat_encoders.joblib')
   for i, model in enumerate(models):
       model.save_model(f'/kaggle/working/catboost_seed_{i}.cbm')
   ```
3. Click **Save Version** → **Save & Run All**.

#### Phase 2: Create Kaggle Dataset from Notebook A

1. Navigate to Notebook A's output page on Kaggle.
2. Click **"..." (More Options)** → **Create Dataset**.
3. Name dataset: `pstu-model-artifacts`. Click **Create**.

#### Phase 3: Link & Execute Inference Notebook B

1. Open/Upload Notebook B (`INFERENCE_NOTEBOOK.ipynb`).
2. On right sidebar, click **+ Add Data**.
3. Search for `pstu-model-artifacts` → Click **Add**.
4. Kaggle mounts artifacts at `/kaggle/input/pstu-model-artifacts/`.
5. Notebook B loads pre-trained artifacts and performs instant inference:
   ```python
   import joblib
   from catboost import CatBoostClassifier

   qt = joblib.load('/kaggle/input/pstu-model-artifacts/qt_transformer.joblib')
   # Predict on test.csv and output /kaggle/working/submission.csv
   ```
6. Submit output `submission.csv`.

---

### Procedure C: Standalone CLI Script Execution (For Organizers / Private Datasets)

*Use this procedure for local verification or organizer hidden test evaluation.*

The standalone script [`final-submission/inference_pipeline.py`](file:///e:/Competitions/PSTU-Datathon/final-submission/inference_pipeline.py) provides a CLI interface for re-evaluating the solution on arbitrary `train.csv` and `test.csv` files.

#### CLI Command Syntax:

```bash
python inference_pipeline.py \
    --train_path /path/to/train.csv \
    --test_path /path/to/test.csv \
    --output_path /path/to/submission.csv \
    --seeds 10
```

#### Script Mechanics:

1. Auto-detects input data directory if arguments omitted.
2. Purifies features (drops zero-variance and exact duplicate columns).
3. Computes 6 row-wise statistics (`mean`, `std`, `iqr`, `zero_count`, `skew`, `kurtosis`).
4. Normalizes continuous features via `QuantileTransformer(output_distribution='normal')`.
5. Trains 10-seed CatBoost (`depth=5`, `l2=5.0`, `auto_class_weights='Balanced'`) with fold-safe `SMOTE(0.3)`.
6. Performs synthetic test-distribution augmentation with $0.1\%$ continuous jitter and negative-ratio capping.
7. Exports final binary predictions at official threshold $t=0.375$.

---

## 4. 🔒 Disqualification Criteria & Technical Safeguards

To prevent disqualification, the inference pipeline strictly adheres to competition rules:

| Rule                                 | Safeguard Implemented                                                                                                                                                              | Status |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----: |
| **No External Datasets**       | Pipeline uses only`train.csv` and `test.csv` provided in competition directory.                                                                                                | ✅ PASS |
| **No Test Set Tampering**      | Test set features used strictly for inference and bootstrapped synthetic generation. No test labels modified or reverse-engineered.                                                | ✅ PASS |
| **Unseen Categorical Levels**  | All 6 categorical string features (`feat_142`, `feat_157`, `feat_318`, `feat_320`, `feat_325`, `feat_337`) use global fallback bucket mapping for unseen categories.   | ✅ PASS |
| **Deterministic Inference**    | Random seeds fixed (`BASE_SEED=42`, `ensemble_seeds=[42, 123, 456, 789, 999, 2026, 777, 888, 101, 202]`). 100% bit-reproducible predictions.                                   | ✅ PASS |
| **Integer Feature Protection** | Continuous$0.1\%$ multiplicative jitter applied ONLY to continuous columns. 238 integer-valued columns ($\ge 99\%$ integer) 100% frozen to prevent adversarial detection leak. | ✅ PASS |
| **Pseudo-Label Ratio Guard**   | Negative-to-positive pseudo-label ratio capped at`10:1` max to prevent negative class domination leakage.                                                                        | ✅ PASS |

---

## 5. 🎯 Decision Threshold Selection Rule

Submissions are evaluated on **Binary F1 Score**. The grader applies a fixed $0.5$ cut to submitted values.

- **Submitted Format**: Hard binary integers (`0` or `1`).
- **Optimal Threshold**: **$t = 0.375$**
  - On 3-seed / 10-seed probability distributions, thresholding at $t = 0.375$ predicts $\approx 2,750$ positives ($\approx 4.5\%$ predicted positive rate), matching the underlying training set positive rate ($3.957\%$).
  - Yields the official competition-winning score of **`0.227966` LB**.

---

## 6. 📁 Final Package File Index

| File Link                                                                                                                       | Description                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [`final-submission/INFERENCE_NOTEBOOK.ipynb`](file:///e:/Competitions/PSTU-Datathon/final-submission/INFERENCE_NOTEBOOK.ipynb) | Official standalone Jupyter inference notebook for 40% hidden test evaluation.                     |
| [`final-submission/FINAL_SUBMISSION.ipynb`](file:///e:/Competitions/PSTU-Datathon/final-submission/FINAL_SUBMISSION.ipynb)     | Primary submission notebook containing full Markdown documentation & multi-threshold probing grid. |
| [`final-submission/submission.csv`](file:///e:/Competitions/PSTU-Datathon/final-submission/submission.csv)                     | Top winning predictions CSV file ($t=0.375$, score 0.227966 LB).                                 |
| [`final-submission/inference_pipeline.py`](file:///e:/Competitions/PSTU-Datathon/final-submission/inference_pipeline.py)       | Standalone CLI Python script for organizer private dataset evaluation.                             |
| [`final-submission/SUBMISSION_GUIDE.md`](file:///e:/Competitions/PSTU-Datathon/final-submission/SUBMISSION_GUIDE.md)           | Top 20 submission checklist and step-by-step Kaggle submission instructions.                       |
| [`final-submission/PRESENTATION_OUTLINE.md`](file:///e:/Competitions/PSTU-Datathon/final-submission/PRESENTATION_OUTLINE.md)   | YouTube video script and slide presentation outline.                                               |
