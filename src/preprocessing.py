"""
Data preprocessing and feature engineering module for AI4I 2020 dataset.
Enforces strict prevention of data leakage and prepares train/test splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Leakage columns that MUST NEVER be fed into the predictor
LEAKAGE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COLUMN = "Machine failure"

BASE_NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "Temp_Difference_K",
    "Power_W",
    "Overstrain_Index",
    "Speed_Torque_Ratio",
]

TYPE_CATEGORIES = ["H", "L", "M"]
TYPE_DUMMY_COLUMNS = [f"Type_{t}" for t in TYPE_CATEGORIES]

FINAL_FEATURE_COLUMNS = (
    BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES + TYPE_DUMMY_COLUMNS
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-grounded mechatronic engineered features:
    1. Temp_Difference_K: (Process temp - Air temp)
    2. Power_W: Mechanical power (Torque * Omega) in Watts
    3. Overstrain_Index: Tool wear * Torque
    4. Speed_Torque_Ratio: Rotational speed / Torque
    5. Type one-hot encoding (ensures Type_H, Type_L, Type_M are strictly present)
    """
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Temperature differential
    df["Temp_Difference_K"] = np.round(
        df["Process temperature [K]"] - df["Air temperature [K]"], 2
    )

    # Spindle mechanical power in Watts (Torque [Nm] * rad/s)
    # rad/s = RPM * 2 * pi / 60
    omega = df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
    df["Power_W"] = np.round(df["Torque [Nm]"] * omega, 2)

    # Tool wear * torque (directly causes Overstrain Failure OSF)
    df["Overstrain_Index"] = np.round(df["Tool wear [min]"] * df["Torque [Nm]"], 2)

    # Spindle operating ratio
    df["Speed_Torque_Ratio"] = np.round(
        df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1e-5), 2
    )

    # One-hot encode Type column
    if "Type" in df.columns:
        norm_type = df["Type"].astype(str).str.strip().str.upper()
        type_dummies = pd.get_dummies(norm_type, prefix="Type", dtype=int)
        for col in TYPE_DUMMY_COLUMNS:
            if col in type_dummies.columns:
                df[col] = type_dummies[col]
            else:
                df[col] = 0
    else:
        # Default fallback if Type missing
        for col in TYPE_DUMMY_COLUMNS:
            if col not in df.columns:
                df[col] = 0

    return df


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], StandardScaler]:
    """
    Cleans dataset, drops leakage identifiers/failure flags, creates engineered features,
    and returns stratified train/test splits along with a fitted StandardScaler.
    """
    df_clean = engineer_features(df)

    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    y = df_clean[TARGET_COLUMN].astype(int)

    # Ensure all final features are present
    for col in FINAL_FEATURE_COLUMNS:
        if col not in df_clean.columns:
            raise ValueError(f"Feature column '{col}' missing after feature engineering.")

    X = df_clean[FINAL_FEATURE_COLUMNS].copy()

    # Stratified train/test split to preserve ~3.4% failure prevalence
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on training numerical features only (prevent data snooping)
    num_cols = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    return X_train, X_test, y_train, y_test, FINAL_FEATURE_COLUMNS, scaler


def transform_single_sample(
    sample_data: dict,
    feature_columns: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Transforms a single user input dictionary into engineered feature row
    and optionally scaled numpy array for model inference.
    """
    df_sample = pd.DataFrame([sample_data])
    df_feat = engineer_features(df_sample)

    # Align columns
    for col in feature_columns:
        if col not in df_feat.columns:
            df_feat[col] = 0

    X_sample = df_feat[feature_columns].copy().fillna(0.0)

    if scaler is not None:
        num_cols = [c for c in feature_columns if not c.startswith("Type_")]
        X_scaled = X_sample.copy()
        X_scaled[num_cols] = scaler.transform(X_sample[num_cols])
        return X_sample, X_scaled.values
    else:
        return X_sample, X_sample.values


PHYSICAL_BOUNDS = {
    "Air temperature [K]": {"min": 280.0, "max": 335.0, "nominal": 300.0, "unit": "K"},
    "Process temperature [K]": {"min": 285.0, "max": 345.0, "nominal": 310.0, "unit": "K"},
    "Rotational speed [rpm]": {"min": 500, "max": 4000, "nominal": 1500, "unit": "RPM"},
    "Torque [Nm]": {"min": 0.0, "max": 120.0, "nominal": 40.0, "unit": "Nm"},
    "Tool wear [min]": {"min": 0.0, "max": 350.0, "nominal": 100.0, "unit": "min"},
}


def validate_sensor_data(data: dict | pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates physical feasibility of sensor measurements.
    Returns (is_valid, list of warning messages).
    """
    messages = []
    is_valid = True

    if isinstance(data, dict):
        # 1. Missing key checks
        for feature, bounds in PHYSICAL_BOUNDS.items():
            if feature not in data or data[feature] is None or data[feature] == "":
                messages.append(f"Missing required sensor reading: '{feature}'.")
                is_valid = False
                continue

            try:
                val = float(data[feature])
                if np.isnan(val):
                    messages.append(f"Sensor '{feature}' contains NaN/null value.")
                    is_valid = False
                elif val < bounds["min"] or val > bounds["max"]:
                    messages.append(
                        f"Sensor '{feature}' value {val} {bounds['unit']} is outside plausible range "
                        f"[{bounds['min']}, {bounds['max']} {bounds['unit']}]."
                    )
                    is_valid = False
            except (ValueError, TypeError):
                messages.append(f"Sensor '{feature}' has non-numeric value: '{data[feature]}'.")
                is_valid = False

        # 2. Thermodynamic gradient check
        if "Process temperature [K]" in data and "Air temperature [K]" in data:
            try:
                p_temp = float(data["Process temperature [K]"])
                a_temp = float(data["Air temperature [K]"])
                if p_temp < a_temp - 2.0:
                    messages.append(
                        f"Thermodynamic warning: Process temp ({p_temp} K) is lower than ambient Air temp ({a_temp} K). "
                        "Check sensor calibration or coolant system integrity."
                    )
                    is_valid = False
            except (ValueError, TypeError):
                pass

        # 3. Product variant validation
        if "Type" in data and str(data["Type"]).strip().upper() not in TYPE_CATEGORIES:
            messages.append(f"Variant Type '{data['Type']}' unknown. Expected L, M, or H (defaulting to M).")

    elif isinstance(data, pd.DataFrame):
        missing_cols = [c for c in BASE_NUMERICAL_FEATURES if c not in data.columns]
        if missing_cols:
            messages.append(f"DataFrame missing required base sensor columns: {missing_cols}")
            is_valid = False

        for feature, bounds in PHYSICAL_BOUNDS.items():
            if feature in data.columns:
                col_num = pd.to_numeric(data[feature], errors="coerce")
                nan_count = int(col_num.isna().sum())
                if nan_count > 0:
                    messages.append(f"Column '{feature}' contains {nan_count} non-numeric/NaN values.")
                    is_valid = False
                outliers = int(((col_num < bounds["min"]) | (col_num > bounds["max"])).sum())
                if outliers > 0:
                    messages.append(f"Column '{feature}' contains {outliers} records outside [{bounds['min']}, {bounds['max']}].")
                    is_valid = False

    return is_valid, messages

