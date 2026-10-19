# Project Overview: Predictive Maintenance & Anomaly Detection (PMAD)

## 1. Executive Summary
The **Predictive Maintenance & Anomaly Detection (PMAD)** system is an industrial-grade mechatronic analytics platform designed for high-reliability manufacturing operations. The project specifically addresses heavy equipment degradation, spindle wear, thermal instability, and mechanical overstrain, matching the operational demands of manufacturing environments such as **Solar Industries India Limited** (propellant manufacturing, energetic chemical mixers, and precision CNC turning).

## 2. Problem Statement
In continuous industrial manufacturing:
- Unplanned machine downtime costs Indian manufacturing facilities crores of rupees in lost throughput.
- Traditional scheduled maintenance ("time-based") replaces serviceable components too early, wasting capital.
- Reactive maintenance ("run-to-failure") risks catastrophic tool breakage, workpiece damage, and severe workplace safety hazards.

**The Solution:** An integrated dual-layer ML architecture combining:
1. **Supervised Failure Prediction**: Class-balanced Random Forest predicting imminent failure probability.
2. **Unsupervised Anomaly Detection**: Isolation Forest detecting multidimensional sensor drifts and out-of-distribution states.
3. **Physics-Informed Prescriptive Guidance**: Thermodynamic and mechatronic threshold rules generating Standard Operating Procedures (SOPs).

## 3. Dataset Specification
- **Name**: UCI AI4I 2020 Predictive Maintenance Dataset.
- **Records**: 10,000 synthetic observations representing milling machinery digital twins.
- **Base Telemetry**:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
  - Product Quality Variant (`Type`: L, M, H)
- **Target**: `Machine failure` (binary 0 or 1, ~3.4% failure prevalence).

## 4. Key Differentiators & Engineering Rigor
- **Target Leakage Prevention**: Explicitly eliminated diagnostic fault flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the feature space.
- **Mechatronic Feature Engineering**: Added physical features including Mechanical Power ($P = \tau \cdot \omega$), Temperature Difference ($\Delta T = T_{process} - T_{air}$), and Overstrain Index ($\text{Wear} \times \tau$).
- **Asymmetric Cost Thresholding**: Shifted the operational decision threshold from 0.50 down to 0.35 to prioritize high recall (>85%) over precision.
