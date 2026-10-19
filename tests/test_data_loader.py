"""
Tests for data loader module and dataset integrity.
"""

import pytest
import pandas as pd
from src.data_loader import generate_synthetic_ai4i_dataset, EXPECTED_COLUMNS, load_dataset


def test_synthetic_data_generation_shape_and_columns():
    df = generate_synthetic_ai4i_dataset(n_samples=200, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column '{col}' missing from generated dataset"


def test_synthetic_data_physical_ranges():
    df = generate_synthetic_ai4i_dataset(n_samples=500, random_state=42)
    # Air temperature should be reasonable Kelvin (~290 to 310 K)
    assert df["Air temperature [K]"].min() >= 280.0
    assert df["Air temperature [K]"].max() <= 320.0

    # Process temp should be >= Air temp in general operation
    assert (df["Process temperature [K]"] >= df["Air temperature [K]"]).mean() > 0.95

    # Rotational speed > 0
    assert (df["Rotational speed [rpm]"] > 0).all()

    # Torque > 0
    assert (df["Torque [Nm]"] > 0).all()

    # Tool wear between 0 and 300
    assert df["Tool wear [min]"].min() >= 0
    assert df["Tool wear [min]"].max() <= 300

    # Binary failure targets
    assert set(df["Machine failure"].unique()).issubset({0, 1})


def test_load_dataset_fallback():
    # Calling load_dataset should return a valid dataframe without crashing
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Machine failure" in df.columns
