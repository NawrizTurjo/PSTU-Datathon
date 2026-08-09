# Dataset Description

> Verbatim transcription of the competition **Data** and **Rules** tabs, reformatted as
> markdown. Wording is unchanged. See [`CLAUDE.md`](CLAUDE.md) for analysis and for findings
> that contradict the description (notably: 6 of the 350 "numerical" features are actually
> categorical strings).
>
> Source: <https://www.kaggle.com/competitions/pstu-data-thon-2026-vol-1/>

## 📊 Data

- **Train Set:** Historical account records containing 350 anonymized numerical features and a
  binary target variable.
- **Test Set:** Account records for which you must submit predictions. In the testing columns
  there have an extra columns named `"id"` that not present in training set.
- **Features:** `feat_1` through `feat_350` — pre-processed, anonymized numerical indicators.
  These features do not carry explicit semantic meaning and have been transformed to preserve
  confidentiality.
- **Target:** `TARGET`
  - `0` = Stable Account
  - `1` = At-Risk Account

> **For mark distribution please see the Rules section.**

---

# Competition Rules

## ⏱️ Competition Timeline

| Event | Date & Time (GMT+6) |
| :--- | :--- |
| Competition Starts | 9 August 2026, 6:00 PM |
| Final Submission Deadline | 13 August 2026, 6:00 PM |
| Private Leaderboard Reveal | 13 August 2026, 6:30 PM |
| Inference Notebook Submission Deadline | 13 August 2026, 11:59 PM |
| Final Results & Winner Announcement | 15 August 2026 |

## 📊 Evaluation Breakdown

| Component | Weight | Description |
| :--- | :---: | :--- |
| Public Leaderboard | 10% | Real-time score on 10% of the test data during the competition. |
| Private Leaderboard | 40% | Final score on 50% of the test data, revealed only after the deadline. |
| Hidden Test (Inference Notebook) | 40% | Post-competition evaluation on 40% of completely unseen data via your submitted inference notebook. |
| Presentation + Code + Submission | 10% | Quality of final presentation, code documentation and submission file format. |

- If two team score the same, then the nobility of solution will be consider.
- **Top 20 from private leaderboard must submit the code + inference notebook + presentation +
  pipeline of your works.**

## 1. Data & External Resources

| Rule | Status | Details |
| :--- | :---: | :--- |
| External Datasets | ❌ Prohibited | You may NOT use any external dataset for model training. Only the provided `train.csv` and official features are allowed. |
| Pre-trained Models | ⚠️ Allowed with disclosure | Publicly available pre-trained models (e.g., from HuggingFace, ImageNet weights) are permitted but must be declared in your write-up. |
| Data Augmentation | ✅ Allowed | Synthetic data generation, SMOTE, feature engineering, and augmentation on the training data only is permitted. |
| Test Data Modification | ❌ Strictly Prohibited | Any manual modification, label leakage, or tampering with the test set will result in immediate disqualification. |

## 2. Submission Requirements

| Requirement | Mandatory | Details |
| :--- | :---: | :--- |
| Prediction File (CSV) | ✅ Yes | `id,TARGET` format with binary scores. |
| Inference Notebook | ✅ Yes | Must be submitted within 6 hours after competition ends. The notebook must reproduce your predictions deterministically. |
| Code Repository | ✅ Yes | Full training and inference code must be shared via GitHub or Kaggle Notebook. |
| Presentation (PDF/PPT) | ✅ Yes | Brief explanation of approach, feature engineering, and model architecture. |

## 3. Leaderboard & Scoring

- **Public LB:** Updates in real-time on 10% of test data.
- **Private LB:** Frozen until competition ends; evaluated on 50% of test data.
- **Hidden Test:** A separate 40% holdout set will be evaluated using your inference notebook
  after the competition closes.
- **Tie-Breaker:** In case of a tie on model performance, the Presentation + Code Quality +
  Nobility of Solution score will be used as the tie-breaker.

## 4. Disqualification Criteria

You will be immediately disqualified if:

- You use external datasets not provided by the organizers.
- You modify, leak, or reverse-engineer labels from the test set.
- Your inference notebook fails to run or produces different results from your submission.
- You fail to submit the required notebook, code, or presentation by the deadline.
- You engage in team merging, account sharing, or any form of collusion.
- You try to generate target values with LLMs.

## 📤 Submission Format

Your CSV file must follow this exact format:

```csv
id,TARGET
2,0
5,0
6,0
7,0
```
