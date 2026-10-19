# Model Artifacts & Evaluation Records

## Stored Artifacts
- `baseline_logistic_regression.joblib`: Standardized Logistic Regression model serving as interpretable baseline.
- `random_forest_model.joblib`: Production Random Forest Classifier trained with `class_weight='balanced'` and calibrated probabilities for severe imbalance (~3.39% positive failure rate).
- `isolation_forest_model.joblib`: Unsupervised Isolation Forest model detecting multidimensional sensor drifts and out-of-distribution operating states.
- `scaler.joblib`: Scikit-learn `StandardScaler` fitted only on training sensor distributions.
- `metadata.json`: Feature list, class distribution, training timestamp, and hyperparameters.
- `evaluation_metrics.json`: Detailed validation & test set performance scores (ROC-AUC, PR-AUC, Recall@Thresholds, Confusion Matrix).

## Industrial Operational Tradeoffs
In industrial equipment monitoring (e.g., explosive formulations, heavy propellants, high-precision machining):
- **False Negative (Missed Failure)**: Results in catastrophic downtime, tool breakage, or workplace safety incidents.
- **False Positive (Spurious Alarm)**: Causes a 15-minute inspection or tool swap.
- **Decision Thresholding**: The default 0.5 probability threshold yields lower recall (~0.70-0.78). Our system provides an adjustable operating decision threshold (defaulting to 0.30 - 0.40) to guarantee high recall (>0.85) without swamping maintenance personnel with false alarms.
