"""
Tests for preprocessing and feature engineering.
Verifies strictly that data leakage columns are excluded.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_ai4i_dataset
from src.preprocessing import (
    engineer_features,
    prepare_data_splits,
    transform_single_sample,
    validate_sensor_data,
    PHYSICAL_BOUNDS,
    LEAKAGE_COLUMNS,
    FINAL_FEATURE_COLUMNS,
)


def test_engineer_features():
    df = generate_synthetic_ai4i_dataset(n_samples=50, random_state=1)
    df_feat = engineer_features(df)

    assert "Temp_Difference_K" in df_feat.columns
    assert "Power_W" in df_feat.columns
    assert "Overstrain_Index" in df_feat.columns
    assert "Speed_Torque_Ratio" in df_feat.columns
    assert "Type_H" in df_feat.columns
    assert "Type_L" in df_feat.columns
    assert "Type_M" in df_feat.columns

    # Verify mathematical formula of Temp_Difference_K
    expected_diff = np.round(df["Process temperature [K]"] - df["Air temperature [K]"], 2)
    assert np.allclose(df_feat["Temp_Difference_K"], expected_diff)


def test_prepare_data_splits_leakage_exclusion():
    df = generate_synthetic_ai4i_dataset(n_samples=200, random_state=42)
    X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_data_splits(df)

    # NONE of the leakage columns should be in the features
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in X_train.columns, f"Leakage column {leak_col} detected in X_train!"
        assert leak_col not in X_test.columns, f"Leakage column {leak_col} detected in X_test!"
        assert leak_col not in feat_cols, f"Leakage column {leak_col} detected in feat_cols!"

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40
    assert scaler is not None


def test_transform_single_sample():
    sample = {
        "Type": "M",
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.8,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 120,
    }
    df_synth = generate_synthetic_ai4i_dataset(n_samples=100, random_state=42)
    _, _, _, _, feat_cols, scaler = prepare_data_splits(df_synth)

    df_out, scaled_arr = transform_single_sample(sample, feat_cols, scaler)
    assert df_out.shape == (1, len(feat_cols))
    assert scaled_arr.shape == (1, len(feat_cols))
    assert df_out["Type_M"].iloc[0] == 1
    assert df_out["Type_H"].iloc[0] == 0
    assert df_out["Type_L"].iloc[0] == 0


def test_engineer_features_whitespace_and_casing():
    dirty_df = pd.DataFrame([{
        " Air temperature [K] ": 298.5,
        "Process temperature [K]": 308.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50,
        "Type": " l ",  # Lowercase with whitespace
    }])
    df_feat = engineer_features(dirty_df)
    assert "Temp_Difference_K" in df_feat.columns
    assert df_feat["Type_L"].iloc[0] == 1
    assert df_feat["Type_M"].iloc[0] == 0
    assert df_feat["Type_H"].iloc[0] == 0


def test_speed_torque_ratio_numerical_stability():
    zero_torque_df = pd.DataFrame([{
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 0.0,  # Zero torque edge case
        "Tool wear [min]": 0,
        "Type": "M",
    }])
    df_feat = engineer_features(zero_torque_df)
    ratio = df_feat["Speed_Torque_Ratio"].iloc[0]
    assert np.isfinite(ratio)
    assert ratio > 0.0


def test_validate_sensor_data_valid():
    valid_sample = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60,
    }
    is_valid, warnings = validate_sensor_data(valid_sample)
    assert is_valid is True
    assert len(warnings) == 0


def test_validate_sensor_data_out_of_bounds():
    invalid_sample = {
        "Type": "M",
        "Air temperature [K]": 250.0,  # Below min (280)
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 4500,  # Above max (4000)
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60,
    }
    is_valid, warnings = validate_sensor_data(invalid_sample)
    assert is_valid is False
    assert len(warnings) >= 2


def test_validate_sensor_data_thermodynamic_violation():
    thermo_viol = {
        "Type": "M",
        "Air temperature [K]": 310.0,
        "Process temperature [K]": 300.0,  # 10 K lower than ambient, physically impossible without refrigeration
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50,
    }
    is_valid, warnings = validate_sensor_data(thermo_viol)
    assert is_valid is False
    assert any("Thermodynamic warning" in w for w in warnings)
