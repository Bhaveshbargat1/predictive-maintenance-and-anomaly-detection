# Dataset Documentation: AI4I 2020 Predictive Maintenance

## Source
- **Origin**: UCI Machine Learning Repository
- **Citation**: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications with a Digital Twin*. Advanced Intelligent Mechatronics (AIM), IEEE/ASME International Conference on.
- **Direct UCI URL**: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- **Fallback URL**: `https://raw.githubusercontent.com/ageron/handson-ml3/main/datasets/ai4i2020.csv` / Automated Local Synthetic Generator

## Overview
The dataset contains 10,000 synthetic observations reflecting real-world industrial milling/cutting machinery operating conditions. The data captures the digital twin telemetry of industrial operations, including high-risk operations comparable to heavy-duty chemical and defense manufacturing equipment (such as energetic material mixers, planetary extruders, and CNC turning centers at Solar Industries India Limited).

## Attribute Descriptions

| Column Name | Type | Unit | Description | Role in ML Modeling |
| :--- | :--- | :--- | :--- | :--- |
| `UDI` | Integer | - | Unique row index (1 to 10,000) | **Excluded** (Identifier) |
| `Product ID` | String | - | Variant letter (L/M/H) + serial number | **Excluded** (Identifier) |
| `Type` | Categorical | - | Quality variant: L (50%), M (30%), H (20%) | **Feature** (One-Hot Encoded) |
| `Air temperature [K]` | Float | Kelvin | Generated via random walk, normalized ~300 K | **Feature** |
| `Process temperature [K]` | Float | Kelvin | Heat generated during operation (+10 K offset) | **Feature** |
| `Rotational speed [rpm]` | Integer | RPM | Spindle speed regulated around 2860 W power | **Feature** |
| `Torque [Nm]` | Float | Nm | Spindle torque centered around 40 Nm | **Feature** |
| `Tool wear [min]` | Integer | Minutes | Total cutting time on tool tip | **Feature** |
| `Machine failure` | Binary | 0 or 1 | Overall failure status indicator | **Target Variable** |
| `TWF` | Binary | 0 or 1 | Tool Wear Failure mode indicator | **Excluded** (Prevents Target Leakage) |
| `HDF` | Binary | 0 or 1 | Heat Dissipation Failure mode indicator | **Excluded** (Prevents Target Leakage) |
| `PWF` | Binary | 0 or 1 | Power Failure mode indicator | **Excluded** (Prevents Target Leakage) |
| `OSF` | Binary | 0 or 1 | Overstrain Failure mode indicator | **Excluded** (Prevents Target Leakage) |
| `RNF` | Binary | 0 or 1 | Random Failure mode indicator | **Excluded** (Prevents Target Leakage) |

## Data Leakage Prevention Protocol
The dataset provides failure cause flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). If these columns are included as input features to predict `Machine failure`, the model suffers from **target leakage** because these causes directly define or coincide with the target in ground truth, yielding artificially inflated 99.9% accuracy that collapses in production.

Our preprocessing pipeline strictly enforces:
1. Removal of `UDI` and `Product ID`.
2. Removal of all failure-cause tags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
3. Pure physical sensor telemetry + engineered domain features only.
