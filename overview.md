# Overview

> Verbatim transcription of the competition **Overview** tab, reformatted as markdown.
> Wording is unchanged — including the F1 / Macro F1 inconsistency between the *Evaluation*
> and *Submission File* sections. See [`CLAUDE.md`](CLAUDE.md) for analysis.
>
> Source: <https://www.kaggle.com/competitions/pstu-data-thon-2026-vol-1/>

In today's competitive financial services landscape, proactively identifying accounts at risk
of instability is essential for client retention and operational efficiency. A leading
multinational financial institution has compiled an extensive dataset of anonymized
account-level behavioral indicators. Your challenge is to build a robust predictive model that
accurately flags accounts likely to become unstable. You are provided with 350 numerical
features derived from transaction history, digital engagement, and account metadata — all
anonymized to protect client privacy.

## 🎯 Goal

For each account in the test set, predict the probability that the account will be flagged as
at-risk (`TARGET = 1`).

## 🚀 Getting Started

1. Download the dataset from the Data tab.
2. Explore the training data and build your model.
3. Submit your predictions on the Submit Predictions page.
4. Check your score on the Leaderboard!

Good luck and happy modeling!

## 📖 Description

- **Train Set:** Historical account records containing 350 anonymized numerical features and a
  binary target variable.
- **Test Set:** Account records for which you must submit binary predictions.
- **Features:** `feat_1` through `feat_350` — pre-processed, anonymized numerical indicators.
  These features do not carry explicit semantic meaning and have been transformed to preserve
  confidentiality.
- **Target:** `TARGET`
  - `0` = Stable Account
  - `1` = At-Risk Account

## ❓ Problem Statement

Financial institutions handle millions of accounts daily, and early detection of at-risk
accounts can significantly reduce churn and improve client satisfaction. However, the
underlying signals of account instability are often buried in complex, high-dimensional
behavioral data. This competition presents a binary classification problem where participants
must leverage 350 anonymized numerical features to predict whether an account will be flagged
as unstable. The features capture various aspects of account behavior, including transaction
patterns, engagement metrics, and account metadata — all stripped of identifiable information.

## 💡 Why This Matters

Accurate prediction of account instability allows institutions to:

- Intervene early with targeted support programs
- Optimize resource allocation for account management teams
- Reduce unexpected account closures and associated revenue loss

## 📄 Data Format

The training data is provided as a CSV file with the following structure:

*(No structure block was present on the competition page at the point this was captured.)*

## 📈 Evaluation

Submissions are evaluated using the **F1 Score**.

## 📤 Submission File

For each ID in the test set, you must predict a probability for the `TARGET` variable. But in
the traning set there is no ID column remember that. The file should contain a header and have
the following format:

```csv
id,TARGET
2,0
5,0
6,0
7,0
etc.
```

Submitted probabilities will be converted to binary predictions using a threshold of `0.5`
before computing the **Macro F1** score.

---

Best of luck, everyone! ❤️ Let's keep solving, learning, and growing together through this
Kaggle competition. Proudly representing Patuakhali Science and Technology University (PSTU). 💙🚀

**Sponsored By:** Poridhi.io
