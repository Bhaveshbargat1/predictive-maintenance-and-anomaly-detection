# Predictive Maintenance & Anomaly Detection for Industrial Equipment (PMAD)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pytest](https://img.shields.io/badge/Testing-Pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org)
[![Industrial Focus](https://img.shields.io/badge/Domain-Heavy%20Manufacturing%20%26%20Solar%20Industries-darkred.svg)]()

> **B.Tech Artificial Intelligence & Data Science Engineering Capstone Project**  
> Tailored for High-Reliability Manufacturing and Energetic Materials Equipment (e.g., **Solar Industries India Limited**).

---

## 1. Executive Summary

In high-risk heavy manufacturing facilities—such as propellant mixers, continuous extruders, and precision CNC turning centers—unplanned equipment failure results in catastrophic downtime, scrap loss, and critical workplace safety risks. 

**PMAD (Predictive Maintenance & Anomaly Detection)** is an end-to-end industrial mechatronic analytics platform that pairs:
1. **Supervised Machine Failure Prediction**: Class-balanced Random Forest predicting binary machine failure with **0.975+ ROC-AUC** and **0.86+ PR-AUC**.
2. **Unsupervised Anomaly Detection**: Isolation Forest discovering multivariate sensor drifts and zero-day degradation without requiring labeled ground truth.
3. **Physics-Informed Prescriptive Guidance**: Thermodynamic and mechatronic domain rules that translate ML risk probabilities into immediate Standard Operating Procedures (SOPs).

---

## 2. Dataset & Zero-Leakage Protocol

The system utilizes the **UCI AI4I 2020 Predictive Maintenance Dataset** (10,000 synthetic observations reflecting real-world industrial milling machinery digital twins).

### ⚠️ Strict Data Leakage Prevention
The raw dataset contains five diagnostic failure mode indicators:
- `TWF` (Tool Wear Failure)
- `HDF` (Heat Dissipation Failure)
- `PWF` (Power Failure)
- `OSF` (Overstrain Failure)
- `RNF` (Random Failure)

In an operating plant, these labels are generated **after** an equipment teardown investigation—they do **not** exist prior to failure. Including them as inputs produces artificial 99.9% accuracy that fails in production. Our preprocessing pipeline:
- **Drops** `UDI` and `Product ID` (non-informative identifiers).
- **Drops** `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` (diagnostic leakage).
- **Trains strictly** on continuous sensor readings and physically derived mechatronic features.

---

## 3. Physics-Grounded Feature Engineering

Rather than feeding raw sensor readings blindly into machine learning, the system extracts mechatronic features derived from first principles:

| Feature Name | Mathematical Formulation | Physical Significance |
| :--- | :--- | :--- |
| **Temperature Difference ($\Delta T$)** | $T_{\text{process}} - T_{\text{air}}$ | Heat dissipation efficiency of coolant jacket and convection. |
| **Mechanical Power ($P$)** | $\tau \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$ | True mechanical spindle load (in Watts). Identifies drive stall and motor trips. |
| **Overstrain Fatigue Index** | $\text{Tool Wear [min]} \times \tau$ | Cumulative mechanical stress driving tool chipping and spindle fatigue. |
| **Speed-to-Torque Ratio** | $\frac{\text{RPM}}{\tau + 10^{-5}}$ | Spindle operating state metric across varying workpiece cut depths. |
| **Equipment Duty Grade** | One-hot encodings ($\text{Type}_H, \text{Type}_L, \text{Type}_M$) | Low (50%), Medium (30%), Heavy Duty (20%) machine tolerances. |

---

## 4. Modeling Strategy & Measured Empirical Benchmarks

### 4.1 Supervised Learning: Balanced Random Forest vs Logistic Regression
All models were evaluated on an identical **stratified 20% holdout test partition ($N=2,000$, 68 failure events, 1,932 normal operating records)** without data leakage.

| Evaluation Metric | Baseline Logistic Regression | Balanced Random Forest | Performance Delta | Operational Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 84.95% | **98.75%** | +13.80% | Overall fleet classification fidelity |
| **Precision (PPV)** | 17.00% | **78.67%** | +61.67% | Purity of maintenance dispatch alarms |
| **Recall / Sensitivity** | 88.24% | **86.76%** | -1.48% | Interception rate of impending machine failures |
| **F1-Score** | 0.2850 | **0.8252** | +0.5402 | Balanced harmonic mean under 3.4% imbalance |
| **F2-Score ($\beta=2$)** | 0.4800 | **0.8501** | +0.3701 | Safety-prioritized metric weighting recall 2x over precision |
| **ROC-AUC** | 0.9406 | **0.9758** | +0.0352 | Global discriminative separation across thresholds |
| **PR-AUC (Avg Precision)** | 0.4486 | **0.8651** | +0.4165 | Key benchmark for extreme class imbalance (baseline = 3.4%) |
| **Brier Score Loss** | 0.1036 | **0.0128** | -87.6% | Probabilistic calibration error (lower is better calibrated) |
| **Matthews Corr. Coeff. (MCC)**| 0.3473 | **0.8198** | +0.4725 | Full confusion matrix correlation |
| **Specificity (TNR)** | 84.83% | **99.17%** | +14.34% | Clear rate of healthy equipment |
| **False Positives (FP)** | 293 alarms | **16 alarms** | **18x reduction** | Precautionary inspections on healthy units |
| **False Negatives (FN)** | 8 missed | **9 missed** | +1 missed | Unintercepted catastrophic breakdowns |

### 4.2 Feature Importance: Tree Splitting vs Test Generalization Drop
| Mechatronic Feature | Gini Impurity (MDI) | Permutation Importance (Test ROC-AUC Drop $\pm 1\sigma$) | Physical Role |
| :--- | :--- | :--- | :--- |
| **Tool wear [min]** | 14.37% (Rank 4) | **0.0260 $\pm$ 0.0031 (Rank 1)** | Progressive wear governs true out-of-sample failure prediction |
| **Temp_Difference_K** | 7.91% (Rank 7) | **0.0230 $\pm$ 0.0033 (Rank 2)** | Coolant dissipation collapse has highest generalization impact |
| **Rotational speed [rpm]**| 15.53% (Rank 3) | **0.0203 $\pm$ 0.0022 (Rank 3)** | Spindle velocity excursions indicate motor stall conditions |
| **Overstrain_Index** | 10.33% (Rank 6) | **0.0165 $\pm$ 0.0035 (Rank 4)** | Non-linear interaction between tool wear and torque load |
| **Power_W** | 15.89% (Rank 2) | **0.0081 $\pm$ 0.0018 (Rank 5)** | Mechanical power envelope detects drive trips |
| **Torque [Nm]** | 11.81% (Rank 5) | **0.0069 $\pm$ 0.0045 (Rank 6)** | Instantaneous cutting load on the workpiece |
| **Speed_Torque_Ratio** | **16.91% (Rank 1)** | **0.0045 $\pm$ 0.0027 (Rank 7)** | Dominates early tree splits but correlated with Speed and Power |

### 4.3 Unsupervised Anomaly Detection Alignment (Isolation Forest)
The Isolation Forest was fitted **strictly unsupervised** (contamination = 0.04) without access to failure labels. Post-hoc evaluation demonstrates strong physical alignment:
- **Alignment ROC-AUC**: **0.8691** | **Alignment PR-AUC**: **0.2942**
- **Unsupervised Failure Recall**: **37.5%** (127 / 339 total breakdowns isolated purely by density)
- **Anomaly Precision**: **32.6%** (127 / 390 flagged points were true breakdown events)
- **Failure Mode Detection Sensitivity**:
  - **Power Failure (PWF)**: **83.2% (79/95)** — sudden torque/speed excursions form distinct low-density outliers.
  - **Overstrain (OSF)**: **50.0% (49/98)** — heavy mechanical torque on worn tooling breaches normal hulls.
  - **Heat Dissipation (HDF)**: **15.7% (18/115)** — subtle thermal differential drops require supervised classification.
  - **Tool Wear (TWF)**: **10.9% (5/46)** — gradual linear degradation follows normal cluster trajectories.
  - **Random Anomaly (RNF)**: **5.3% (1/19)** — uniform stochastic noise.

---

## 5. Web Application Features (`app.py`)

Run via Streamlit, the application features 9 operational views:

1. **Executive Overview**: Real-time fleet KPI metrics, failure prevalence breakdown, and one-click download for the executive plant audit report (`PMAD_Executive_Plant_Audit_Report.md`).
2. **Data Explorer**: Advanced multi-sensor telemetry range sliders (RPM, Tool Wear, Torque, $\Delta T$, Power), variant filters, summary statistics, and clean CSV export.
3. **Sensor Analytics**: Cross-sensor correlation matrix, theoretical power envelopes ($3.5$ kW and $9.0$ kW) with marginal boxplots, thermal dissipation distributions, and Kolmogorov-Smirnov separation stats.
4. **Anomaly Detection (Unsupervised)**: Isolation Forest 2D and 3D cluster visualizations, severity score histogram ($0-100\%$), post-hoc failure mode sensitivity bar chart, and filtered anomaly CSV export.
5. **Failure Prediction (Supervised)**:
   - Single-equipment diagnostic with physical boundary validation, model selector (Balanced Random Forest vs Logistic Regression), and risk gauge.
   - **Sensor Deviation Profile**: Normalized Z-score chart comparing current sensor values against healthy fleet median.
   - **Batch CSV Scoring Engine**: Automated feature extraction, physical sanity check, and scored CSV export.
6. **Maintenance Alerts & Prescriptive SOPs**: Live equipment alert feed with immediate Standard Operating Procedures (SOP-TL-01 Tool Replacement, SOP-TH-02 Coolant Flow Audit, SOP-DR-03 Inverter Trip Check) and exportable plant inspection audit report.
7. **Model Evaluation & Trade-offs**:
   - Comprehensive comparative performance matrix with 12 quantitative metrics.
   - Side-by-side Confusion Matrices.
   - Overlaid ROC and Precision-Recall curves.
   - **Interactive Decision Threshold & Financial Cost Simulator**: Sliders for Breakdown Cost ($C_{FN}$) and Inspection Cost ($C_{FP}$) finding the cost-minimizing operational threshold.
   - Feature Importance Analysis (Gini Impurity vs Permutation Importance with error bars).
8. **Simulated Live Monitoring (Clearly Labeled Demonstration)**: Real-time streaming IoT telemetry generator with dynamic live strip-charts and fault injection presets (PWF, HDF, TWF, OSF).
9. **Project Architecture & Defense Guide**: Technical defense questions, architecture diagram, and domain relevance to **Solar Industries India Limited**.

---

## 6. Directory Structure

```text
PMAD Project for solar/
├── .gitignore
├── requirements.txt
├── README.md
├── run_app.bat                      # 1-click Windows launcher for web app
├── run_tests.bat                    # 1-click Windows launcher for test suite
├── train_all.py                     # Master model training pipeline
├── app.py                           # Full-featured Streamlit interactive web dashboard
├── data/
│   ├── README.md                    # Data dictionary & leakage documentation
│   └── raw/
│       └── ai4i2020.csv             # Automated download / offline synthetic dataset
├── models/
│   ├── README.md                    # Model artifact and threshold documentation
│   ├── random_forest_model.joblib   # Trained balanced Random Forest classifier
│   ├── baseline_logistic_regression.joblib
│   ├── isolation_forest_model.joblib
│   ├── scaler.joblib
│   ├── metadata.json
│   └── evaluation_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # UCI download, zip extraction, offline generator
│   ├── preprocessing.py             # Feature engineering & stratified split
│   ├── models.py                    # Supervised training & evaluation metrics
│   ├── anomaly_detector.py          # Isolation Forest scoring & severity norm
│   └── recommender.py               # Prescriptive maintenance SOP engine
├── docs/
│   ├── PROJECT_OVERVIEW.md          # In-depth executive and engineering report
│   ├── MODEL_EVALUATION.md          # Quantitative benchmarks and cost analysis
│   └──  LIMITATIONS.md               # Technical limitations & edge roadmap       
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive Jupyter exploratory notebook
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_models.py
    ├── test_anomaly_detector.py
    └── test_recommender.py
```

---

