"""
Tests for unsupervised Isolation Forest anomaly detection.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_ai4i_dataset
from src.preprocessing import engineer_features
from src.anomaly_detector import (
    IndustrialAnomalyDetector,
    train_anomaly_detector,
    evaluate_anomaly_detector_alignment,
)


@pytest.fixture
def temp_models_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_anomaly_detector_lifecycle(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=250, random_state=42)
    df_feat = engineer_features(df)

    detector = train_anomaly_detector(df_feat, contamination=0.05, models_dir=temp_models_dir)
    assert detector.is_fitted

    # Score samples
    raw_scores, score_pct, is_anomaly = detector.score_samples(df_feat)
    assert len(raw_scores) == len(df_feat)
    assert len(score_pct) == len(df_feat)
    assert len(is_anomaly) == len(df_feat)

    # Check normalized percentage bounds [0, 100]
    assert np.all(score_pct >= 0.0)
    assert np.all(score_pct <= 100.0)
    assert set(is_anomaly).issubset({0, 1})

    # Attach to dataframe
    df_enriched = detector.attach_anomaly_scores_to_df(df_feat)
    assert "Anomaly_Raw_Score" in df_enriched.columns
    assert "Anomaly_Severity_Pct" in df_enriched.columns
    assert "Is_Anomaly" in df_enriched.columns
    assert "Anomaly_Status" in df_enriched.columns

    # Verify reloading from disk
    save_path = os.path.join(temp_models_dir, "isolation_forest_model.joblib")
    reloaded_detector = IndustrialAnomalyDetector.load(save_path)
    assert reloaded_detector.is_fitted

    _, reload_score_pct, _ = reloaded_detector.score_samples(df_feat)
    assert np.allclose(score_pct, reload_score_pct)


def test_anomaly_detector_raw_input_auto_feature_prep(temp_models_dir):
    df_raw = generate_synthetic_ai4i_dataset(n_samples=100, random_state=7)
    detector = train_anomaly_detector(df_raw, contamination=0.05, models_dir=temp_models_dir)

    # Pass raw dataframe without pre-computing engineered features
    raw_scores, score_pct, is_anomaly = detector.score_samples(df_raw)
    assert len(score_pct) == len(df_raw)
    assert np.all((score_pct >= 0.0) & (score_pct <= 100.0))


def test_anomaly_detector_export_anomalies(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=200, random_state=42)
    detector = train_anomaly_detector(df, contamination=0.10, models_dir=temp_models_dir)

    export_path = os.path.join(temp_models_dir, "exported_anomalies.csv")
    df_anom = detector.export_anomalies(df, filepath=export_path)

    assert os.path.exists(export_path)
    assert isinstance(df_anom, pd.DataFrame)
    assert (df_anom["Is_Anomaly"] == 1).all()
    assert len(df_anom) > 0


def test_anomaly_extreme_outlier_detection(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=300, random_state=42)
    detector = train_anomaly_detector(df, contamination=0.04, models_dir=temp_models_dir)

    # Normal nominal operating sample
    normal_sample = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30,
    }])
    _, norm_pct, _ = detector.score_samples(normal_sample)

    # Severe out-of-distribution physical point (5000 RPM, 150 Nm, extreme temperature)
    extreme_sample = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 350.0,
        "Process temperature [K]": 390.0,
        "Rotational speed [rpm]": 4800,
        "Torque [Nm]": 150.0,
        "Tool wear [min]": 280,
    }])
    _, extreme_pct, extreme_flag = detector.score_samples(extreme_sample)

    assert extreme_pct[0] > norm_pct[0]
    assert extreme_flag[0] == 1


def test_evaluate_anomaly_detector_alignment():
    df = generate_synthetic_ai4i_dataset(n_samples=200, random_state=42)
    detector = train_anomaly_detector(df, contamination=0.08)

    align_res = evaluate_anomaly_detector_alignment(detector, df, target_col="Machine failure")
    assert align_res["evaluated"] is True
    assert 0.0 <= align_res["roc_auc"] <= 1.0
    assert 0.0 <= align_res["pr_auc"] <= 1.0
    assert 0.0 <= align_res["recall_on_failures"] <= 1.0
    assert "mean_severity_normal" in align_res
    assert "mean_severity_failure" in align_res
    assert "failure_mode_detection" in align_res
