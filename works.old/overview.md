# 🌊 Predictive Maintenance for Coastal Riverine Water Management Stations

## 🎯 Competition Goal

Build a machine learning model to predict the likelihood of critical failure in off-grid, solar-powered water management stations located in the remote riverine and coastal char (island) areas of Bangladesh. Your solution will help shift maintenance strategies from reactive to predictive, protecting vulnerable communities from saline water intrusion and ensuring uninterrupted access to fresh water.

## 📖 Background & Context

The southern and south-eastern coastal regions of Bangladesh (including Bhola, Patuakhali, Noakhali, and Barguna) are characterized by vast, complex river networks and fragile char (river island) ecosystems. In these remote areas, grid electricity is virtually non-existent.

To combat the severe threat of tidal saline water intrusion into agricultural lands and drinking water sources, the government and various NGOs have deployed off-grid, solar-powered water management stations. These stations operate submersible pumps to extract fresh groundwater and run sensors to monitor river salinity and sluice gate operations.

However, these stations operate in exceptionally harsh, humid, and saline environments. They are constantly exposed to:

- **Extreme salt-induced corrosion** of electrical components.
- **Heavy dust and mud accumulation** on solar panels during dry seasons.
- **Frequent voltage fluctuations and physical stress** from cyclonic weather and tidal surges.

Currently, maintenance in these remote riverine areas is strictly reactive. Technicians must travel by boat only after a station has completely broken down. This leads to prolonged water crises, crop destruction, and exorbitant emergency repair costs.

---

## ⚠️ The Problem Statement

Your mission is to solve this challenge by building a robust predictive maintenance model.

You are provided with historical operational data from hundreds of these remote riverine stations. Using a combination of real-time sensor readings, historical maintenance logs, and environmental metrics, you must predict whether a station will experience a critical failure in the upcoming 7 days.

Successfully identifying at-risk stations before they break down will allow authorities to dispatch boat-based technician teams proactively, order spare parts in advance, and prevent catastrophic saline water intrusion.

---

## 📊 The Data

The dataset provided is a comprehensive, tabular collection of features mimicking real-world IoT sensor logs from these remote stations. It includes:

- **Base Attributes:** Station installation age, distance from the main river estuary, and dependent farmer count.
- **Real-time Sensor Readings:** Battery voltage, water salinity (ppm), motor vibration levels, and solar panel surface temperature.
- **Operational Counts:** Number of dry-run events, voltage surges, and maintenance visits.
- **Financial Metrics:** Historical repair costs, community contributions, and government/NGO grants.

> [!NOTE]
> Due to the harsh, remote riverine environment, sensors occasionally fail to transmit data or experience communication dropouts. Instead of standard NaN or null values, the central logging server has marked these missing readings with a specific, highly anomalous numerical value that falls completely outside the physical range of any real-world sensor.
>
> Part of your challenge as a data scientist is to identify this hidden "ghost value" through Exploratory Data Analysis (EDA), treat it as missing data, and apply appropriate imputation techniques (or let tree-based models handle it natively) before building your predictive model.

---

## 💡 Why This Matters

A successful model will directly contribute to UN Sustainable Development Goal 6 (Clean Water and Sanitation) and SDG 13 (Climate Action). By optimizing maintenance schedules for these remote riverine stations, we can save millions of Taka in emergency repairs and, more importantly, protect the livelihoods and fresh water access of thousands of families in Bangladesh's most climate-vulnerable coastal char regions.

---

## 📈 Evaluation

Submissions are evaluated on a **Custom Weighted Composite Score**. Because real-world sensor failure data is highly imbalanced (the vast majority of stations operate normally), standard accuracy is a misleading metric. A model that simply predicts `0` (No Failure) for every station would achieve high accuracy but would be completely useless in preventing actual breakdowns. Therefore, your model will be scored on a combination of six distinct metrics, heavily penalizing both False Negatives (missing a failure) and False Positives (unnecessary maintenance alarms), while rewarding overall probabilistic discrimination.

### Evaluation Metrics

1. **F1-Score (Weight: 0.30)**  
   The harmonic mean of Precision and Recall. It is the primary metric for imbalanced datasets, ensuring a balance between finding all failures and not raising too many false alarms.
   $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

2. **ROC-AUC (Weight: 0.25)**  
   The Area Under the Receiver Operating Characteristic Curve. This evaluates your model's ability to rank a randomly chosen positive instance higher than a randomly chosen negative instance, regardless of the classification threshold.  
   *(Note: This is why your submission must contain probabilities, not just binary 0/1 labels).*

3. **Precision (Weight: 0.15)**  
   The proportion of predicted failures that were actual failures. High precision means your maintenance team won't waste resources on false alarms.
   $$Precision = \frac{TP}{TP + FP}$$

4. **Recall / Sensitivity (Weight: 0.15)**  
   The proportion of actual failures that were correctly identified by the model. High recall means the model rarely misses a critical breakdown.
   $$Recall = \frac{TP}{TP + FN}$$

5. **Specificity (Weight: 0.05)**  
   The proportion of actual normal operations that were correctly identified as normal.
   $$Specificity = \frac{TN}{TN + FP}$$

6. **Balanced Accuracy (Weight: 0.10)**  
   The arithmetic mean of Recall (Sensitivity) and Specificity. This metric inherently corrects for the severe class imbalance in the dataset.
   $$BalancedAccuracy = \frac{Recall + Specificity}{2}$$

### Final Scoring Formula

Your final leaderboard score will be calculated as the weighted sum of the six metrics above. The maximum possible score is **1.0000**.

$$FinalScore = (0.30 \times F1) + (0.25 \times ROCAUC) + (0.15 \times Precision) + (0.15 \times Recall) + (0.10 \times BalancedAccuracy) + (0.05 \times Specificity)$$

---

Best of luck, everyone! ❤️ Let’s keep solving, learning, and growing together through this Kaggle competition. Proudly representing Patuakhali Science and Technology University (PSTU). 💙🚀

**Sponsored By:** Poridhi.io
