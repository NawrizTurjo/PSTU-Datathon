# 🚀 PSTU DataThon: Alternative "Simpler & Faster" Machine Learning Paradigms

এই ফোল্ডারে (`late-submission/`) আমরা আলাদা আলাদা পদ্ধতির পাশাপাশি **সবগুলোকে একত্রিত করে একটি অল-ইন-ওয়ান মাস্টার নোটবুক** এবং **একটি আল্ট্রা-সিম্পল ৩০-লাইনের বেসলাইন নোটবুক** তৈরি করে দিয়েছি।

সবগুলো নোটবুকই **১০০% CPU-ফ্রেন্ডলি** এবং সাধারণ ল্যাপটপেই মাত্র **১৫ সেকেন্ড থেকে ৩ মিনিটের মধ্যে** রান হয়ে যায়।

---

## 📂 ফাইলগুলোর তালিকা ও বিবরণ

| ফাইল নেম | আর্কিটেকচার ও বিবরণ | CPU রান টাইম | মূল বৈশিষ্ট্য |
|---|---|---|---|
| **`00_ultra_simple_30_line_baseline.ipynb`** | **Ultra-Simple 30-Line Logistic Baseline** | **~১৫-২০ সেকেন্ড** | একদম পিওর ৩০ লাইনের কোড। নো ফিচার ইঞ্জিনিয়ারিং, নো SMOTE, ডিরেক্ট 10-Fold CV। |
| **`04_all_in_one_master_simple_pipeline.ipynb`** | 🏆 **Unified Master Consensus Pipeline (All-In-One)** | **~৩-৪ মিনিট** | ১, ২ ও ৩ নম্বর পাইপলাইন একসাথে রান করে তাদের **Rank-Averaging Consensus** ব্লেন্ড করে। |
| **`01_logistic_regression_regularized.ipynb`** | **10-Fold Multi-Seed Linear Ensemble (L1/L2/SGD/Ridge)** | ~১-২ মিনিট | ৩৪৪টি ফিচারের গ্লোবাল লিনিয়ার সিগন্যাল ($z = \sum w_i x_i$) ও আউটলায়ার ক্লিপিং। |
| **`02_balanced_bagging_ensemble.ipynb`** | **10-Fold Balanced Sub-Sampling Bagging (300 Sub-models)** | ~২-৩ মিনিট | ৩,০০৮ পজিটিভ + ৩,০০৮ নেগেটিভ স্যাম্পল নিয়ে ৩০০টি ব্যালেন্সড মডেলের এভারেজ। |
| **`03_linear_tree_stacking_ensemble.ipynb`** | **Two-Stage 10-Fold Stacking Meta-Learner** | ~৩-৪ মিনিট | ৫টি মডেলের প্রেডিকশন দিয়ে তৈরি মেটা-ম্যাট্রিক্সের ওপর লেভেল-১ মেটা-লার্নার। |

---

## 🌟 ১. অল-ইন-ওয়ান মাস্টার নোটবুক (`04_all_in_one_master_simple_pipeline.ipynb`)

আপনি যদি সবগুলো পদ্ধতির সেরা কম্বিনেশন একটিমাত্র ফাইলে রান করতে চান, তবে এটি আপনার জন্য সেরা:

1. **পাইপলাইন ১:** 10-Fold Regularized Linear Models (L2 + SGD ElasticNet).
2. **পাইপলাইন ২:** 10-Fold Balanced Bagging (200 Balanced Sub-models, No SMOTE).
3. **পাইপলাইন ৩:** 10-Fold Two-Stage Stacking Meta-Learner.
4. **মাস্টার কনসেনসাস ব্লেন্ডিং (Multi-Paradigm Rank Blending):**
   - ৩টি ভিন্ন ঘরানার মডেলের প্রবাবিলিটিকে পার্সেন্টাইল র্যাঙ্কে (Percentile Ranks) রূপান্তর করে তাদের এভারেজ নেওয়া হয়:
     $$\text{Consensus Rank} = \frac{\text{Rank}_{\text{Linear}} + \text{Rank}_{\text{Bagging}} + \text{Rank}_{\text{Stacking}}}{3}$$
   - র্যাঙ্ক ব্লেন্ডিং স্কেলের অমিল দূর করে এবং আনসিন টেস্ট সেটে সবচেয়ে শক্তিশালী জেনারেলাইজেশন দেয়।
5. **আউটপুট:** স্বয়ংক্রিয়ভাবে ৪টি আলাদা প্রেডিকশন ফাইল সেভ করে:
   - `submission_linear.csv`
   - `submission_bagging.csv`
   - `submission_stacking.csv`
   - `submission.csv` *(Master Consensus)*

---

## ⚡ ২. আল্ট্রা-সিম্পল ৩০-লাইন বেসলাইন (`00_ultra_simple_30_line_baseline.ipynb`)

* এটি সম্পূর্ণ ক্যাগল ডেটাসেটের জন্য **সবচেয়ে সরল কিন্তু শক্তিশালী বেসলাইন**।
* কোনো বাহ্যিক লাইব্রেরি বা জটিল প্রিপ্রসেসিং নেই—সরাসরি `StandardScaler` + `LogisticRegression(class_weight='balanced')` দিয়ে ১০-ফোল্ড ক্রস ভ্যালিডেশনে রান হয়।
* সময় নেয় মাত্র **১৫-২০ সেকেন্ড**!

---

## 🏃‍♂️ কীভাবে রান করবেন (How to Run)

ক্যাগলে বা লোকাল জুপিটারে:
* সবচেয়ে দ্রুত রেজাল্ট ও ব্লেন্ড পেতে সরাসরি **`04_all_in_one_master_simple_pipeline.ipynb`** ওপেন করে **`Run All`** দিন।
* এটি রান শেষে স্ক্রিনে ৩টি পাইপলাইন এবং মাস্টার ব্লেন্ডের **OOF AUC ও Peak F1-Score এর তুলনামূলক সামারি টেবিল** প্রিন্ট করবে এবং `submission.csv` তৈরি করে দেবে!
