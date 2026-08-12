# 📊 EDA Findings

| Metric | Value |
| --- | --- |
| **Train** | 76,020 rows × 350 feat + TARGET |
| **Test** | 60,654 rows × 350 feat + id |
| **Class 0 (Stable)** | 73,012 (96.04%) |
| **Class 1 (At-Risk)** | 3,008 (3.96%) |
| **Imbalance Ratio** | 24.27:1 🔴 |
| **Missing Values** | None (clean!) |
| **Infinite Values** | None |
| **Feature Types** | 344 numerical + 6 categorical (feat_142, feat_157, feat_318, feat_320, feat_325, feat_337) |
| **Zero-Variance** | 28 features → DROP |
| **Duplicate Pairs (r=1.0)** | 30+ detected → DROP one each |
| **PCA 80% variance** | ~45 components needed |
| **Train/Test shift** | Minimal — only 2/30 KS-test significant |

**Key Insight:** The anonymized features have very weak individual predictive power (max |corr| = 0.15). Winning requires non-linear tree ensembles, feature interactions, and row-wise statistical features that capture structural patterns.
