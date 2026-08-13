# 📹 YouTube Presentation Outline & Slide Deck Script

**Competition:** PSTU DataThon 2026  
**Team Name:** NawrizTurjo  
**Winning Model:** `synthetic-fixissuesv2-winner` (LB `0.227966`)  
**Target Duration:** 5 - 7 minutes  

---

## 🎬 Slide-by-Slide Outline & Talking Points

### Slide 1: Title & Team Introduction (0:00 - 0:45)
* **Visuals:** Project Title, Team Name, Member Names, Competition Badge, Winning LB Score (`0.227966`).
* **Speaker Script:**
  > "Hello everyone! We are team NawrizTurjo, presenting our winning solution for the PSTU DataThon 2026. Our final model achieved a Public Leaderboard F1-score of 0.228. In this video, we'll walk you through our Exploratory Data Analysis, the progression from basic baselines to our winning architecture, and our novel Synthetic Test-Distribution Augmentation strategy."

---

### Slide 2: Problem Statement & EDA Insights (0:45 - 2:00)
* **Visuals:** Class Imbalance Bar Chart (96.2% Negative vs 3.8% Positive), Feature Variance plots, Table of Integer-like continuous columns.
* **Key Points:**
  1. **Extreme Imbalance:** 96:4 ratio meant standard default models predicted almost all zeros ($F1 \approx 0$).
  2. **Feature Quality:** 344 initial features contained zero-variance constants and duplicate columns with identical byte signatures.
  3. **Data Types:** Discovered many continuous columns were actually discrete integer counts. Preserving their integrity during augmentation was critical.

---

### Slide 3: Evolution of Strategy — Failures to Breakthroughs (2:00 - 3:30)
* **Visuals:** Timeline / Leaderboard Progression Chart.
  - *Phase 1 (Baselines):* Raw XGBoost/LightGBM $\rightarrow$ Score ~0.12 - 0.16. (Overfitting majority class).
  - *Phase 2 (Over-engineering):* Omega V7 (Polynomials + 2-Stage Pseudo-labeling) $\rightarrow$ Score 0.13 - 0.20. (Noise leakage).
  - *Phase 3 (Core Fix):* `FixIssuesV2` (CatBoost depth=5 + 6 Row Stats + QuantileTransformer + SMOTE 0.3) $\rightarrow$ Score **0.2257**.
  - *Phase 4 (Winner):* 10-Seed Ensemble + Calibrated 0.1% Synthetic Test Jitter $\rightarrow$ Score **0.2280**.

---

### Slide 4: The Winning Architecture & Pipeline (3:30 - 4:45)
* **Visuals:** Flowchart of the end-to-end pipeline (`FINAL_SUBMISSION.ipynb`).
* **Key Components:**
  1. **Feature Engineering:** 6 row-wise statistical meta-features (Mean, Std, IQR, Skew, Kurtosis, Zero Count).
  2. **Fold-Safe SMOTE:** Oversampling the minority class to 30% strictly *inside* 5-fold CV loops.
  3. **Calibrated Synthetic Test Jitter:** Resampling test rows with $0.1\%$ multiplicative noise on non-integer columns (verified by Adversarial Validation AUC of ~0.51).
  4. **Dynamic Threshold Probing:** Shifted decision threshold from $t=0.5$ down to $t=0.375$ to maximize F1-score on imbalanced predictions.

---

### Slide 5: Experimental Lessons & Ablation Analysis (4:45 - 5:45)
* **Visuals:** Comparison Table between Ensembles (Feature Forge 0.217) vs Single CatBoost (0.228).
* **Key Takeaway:**
  > "We tested multi-model soft-voting ensembles (CatBoost + LightGBM + XGBoost) and complex K-Means cluster distance features. Surprisingly, the single regularized CatBoost architecture with SMOTE out-performed complex ensembles. On noisy, highly imbalanced datasets, simplicity and strict fold-isolation prevent overfitting."

---

### Slide 6: Summary & Conclusion (5:45 - 6:30)
* **Visuals:** Summary Bullet Points, Code Repository Link, Thank You.
* **Speaker Script:**
  > "In conclusion, our key drivers for victory were strict fold-safe CV, calibrated test-distribution augmentation, and precise threshold tuning to $0.375$. All code is fully reproducible in our submitted Kaggle notebook. Thank you to the organizers and PSTU for hosting this great datathon!"

---

## 📽️ Video Recording Tips for Team
1. **Resolution:** 1080p (1920x1080) widescreen.
2. **Audio:** Clear microphone, minimal background noise.
3. **Screen Share:** Show slides and briefly navigate to the Kaggle notebook (`FINAL_SUBMISSION.ipynb`).
4. **Upload:** Unlisted or Public on YouTube, submit link in official submission form.
