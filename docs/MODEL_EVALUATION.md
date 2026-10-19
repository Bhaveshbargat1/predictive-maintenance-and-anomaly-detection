# Model Evaluation & Trade-off Analysis (PMAD)

## 1. Experimental Methodology & Benchmark Setup
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset (10,000 industrial telemetry records)
- **Train/Test Partition**: Stratified 80/20 holdout split (8,000 training observations, 2,000 test observations)
- **Target Label**: `Machine failure` (binary 0 or 1)
- **Class Balance**: 
  - Normal: 96.61% (1,932 records in test partition)
  - Failure: 3.39% (68 failure events in test partition)
- **Data Leakage Prevention**: Identifiers (`UDI`, `Product ID`) and post-failure diagnostics (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) strictly excluded from input feature space.
- **Preprocessing Isolation**: `StandardScaler` fitted strictly on training partition and frozen before inference.

---

## 2. Quantitative Comparative Benchmark (Held-out Test Partition N=2,000)

| Metric | Baseline Logistic Regression | Balanced Random Forest | Performance Delta | Operational Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 84.95% | **98.75%** | +13.80% | Overall fleet classification fidelity |
| **Precision (PPV)** | 17.00% | **78.67%** | +61.67% | Purity of maintenance dispatch alarms |
| **Recall / Sensitivity** | 88.24% | **86.76%** | -1.48% | Interception rate of impending machine failures |
| **F1-Score** | 0.2850 | **0.8252** | +0.5402 | Balanced harmonic mean under 3.4% imbalance |
| **F2-Score ($\beta=2$)** | 0.4800 | **0.8501** | +0.3701 | Safety-prioritized metric weighting recall 2x over precision |
| **ROC-AUC** | 0.9406 | **0.9758** | +0.0352 | Global discriminative separation across thresholds |
| **PR-AUC (Avg Precision)** | 0.4486 | **0.8651** | +0.4165 | Key benchmark for extreme class imbalance |
| **Brier Score Loss** | 0.1036 | **0.0128** | -87.6% | Probabilistic calibration error (lower is better calibrated) |
| **Matthews Corr. Coeff. (MCC)**| 0.3473 | **0.8198** | +0.4725 | Full confusion matrix correlation |
| **Specificity (TNR)** | 84.83% | **99.17%** | +14.34% | Clear rate of healthy equipment |
| **False Positives (FP)** | 293 alarms | **16 alarms** | **18x reduction** | Precautionary inspections on healthy units |
| **False Negatives (FN)** | 8 missed | **9 missed** | +1 missed | Unintercepted catastrophic breakdowns |

---

## 3. Asymmetric Operational Cost & Financial Risk Optimization

In industrial environments (such as energetic slurry mixing and precision CNC turning at Solar Industries India Limited), false negatives and false positives have sharply asymmetric financial and safety costs:

$$\text{Total Cost}(T) = \text{FN}(T) \times C_{FN} + \text{FP}(T) \times C_{FP}$$

- **Cost of False Negative ($C_{FN}$)**: ₹2,50,000 - ₹10,00,000+ (unplanned catastrophic stall, destroyed carbide tooling, energetic slurry auto-ignition risk, hours of plant line stoppage).
- **Cost of False Positive ($C_{FP}$)**: ₹5,000 (15 minutes of technician sensor check, manual lubricant top-off).
- **Cost Ratio**: $C_{FN} / C_{FP} \ge 50:1$.

### Why Logistic Regression Fails in Practice Despite High Recall:
While Logistic Regression catches 60 out of 68 failures (88.2% recall), its low precision (17.0%) causes **293 false alarms** in just 2,000 records. In an operational plant, 293 false alarms create "alarm fatigue," causing technicians to ignore warnings. Random Forest reduces false alarms by **18-fold (only 16 FP)** while preserving 86.8% recall.

---

## 4. Feature Importance: Gini Impurity vs. Test-Set Permutation Importance

| Feature | Gini Impurity (Tree Splitting) | Permutation Importance (Test ROC-AUC Drop $\pm 1\sigma$) | Engineering Interpretation |
| :--- | :--- | :--- | :--- |
| **Tool wear [min]** | 0.1437 (Rank 4) | **0.0260 $\pm$ 0.0031 (Rank 1)** | Progressive wear governs true out-of-sample failure prediction |
| **Temp_Difference_K** | 0.0791 (Rank 7) | **0.0230 $\pm$ 0.0033 (Rank 2)** | Coolant dissipation collapse has highest generalization impact |
| **Rotational speed [rpm]**| 0.1553 (Rank 3) | **0.0203 $\pm$ 0.0022 (Rank 3)** | Spindle velocity excursions indicate motor stall conditions |
| **Overstrain_Index** | 0.1033 (Rank 6) | **0.0165 $\pm$ 0.0035 (Rank 4)** | Non-linear interaction between tool wear and torque load |
| **Power_W** | 0.1589 (Rank 2) | **0.0081 $\pm$ 0.0018 (Rank 5)** | Mechanical power envelope detects drive trips |
| **Torque [Nm]** | 0.1181 (Rank 5) | **0.0069 $\pm$ 0.0045 (Rank 6)** | Instantaneous cutting load on the workpiece |
| **Speed_Torque_Ratio** | **0.1691 (Rank 1)** | **0.0045 $\pm$ 0.0027 (Rank 7)** | Dominates early tree splits but correlated with Speed and Power |

> **Key Interview Takeaway**: Gini Importance measures training sample split purity, which can overvalue correlated engineered features (`Speed_Torque_Ratio`). Permutation Importance shuffles features on held-out test data, demonstrating that **Tool Wear** and **Temperature Difference** have the greatest impact on real generalization.

---

## 5. Unsupervised Anomaly Detection Alignment (Isolation Forest)

The Isolation Forest was fitted **strictly unsupervised** (contamination = 0.04) without access to failure labels. Post-hoc evaluation against ground truth reveals:
- **Alignment ROC-AUC**: **0.8691**
- **Alignment PR-AUC**: **0.2942**
- **Unsupervised Failure Recall**: **37.5%** (127 of 339 total failure events isolated purely by multi-dimensional density without labels)
- **Anomaly Precision**: **32.6%** (127 of 390 flagged points were true breakdown events)
- **Mean Anomaly Severity**: Normal = **21.36%**, Failed = **49.10%**

### Sensitivity by Failure Mode:
1. **Power Failure (PWF)**: **83.2% (79 / 95)** — Sudden torque/speed spikes create distinct out-of-distribution outliers.
2. **Overstrain Failure (OSF)**: **50.0% (49 / 98)** — Heavy mechanical load on worn tools breaks normal operating envelope.
3. **Heat Dissipation (HDF)**: **15.7% (18 / 115)** — Subtle temperature differential collapse overlaps normal density.
4. **Tool Wear Failure (TWF)**: **10.9% (5 / 46)** — Gradual linear degradation requires supervised tracking.
5. **Random Failure (RNF)**: **5.3% (1 / 19)** — Uniform noise points undetectable by density isolation alone.
