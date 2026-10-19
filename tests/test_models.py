"""
Tests for supervised model training, evaluation, and inference.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_ai4i_dataset
from src.preprocessing import prepare_data_splits
from src.models import (
    train_supervised_models,
    load_artifacts,
    predict_machine_failure,
)


@pytest.fixture
def temp_models_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_and_evaluate_supervised_models(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=300, random_state=42)
    X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_data_splits(df)

    res = train_supervised_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_columns=feat_cols,
        models_dir=temp_models_dir,
    )

    assert "rf_model" in res
    assert "lr_model" in res
    assert "evaluation_report" in res

    eval_rep = res["evaluation_report"]
    rf_metrics = eval_rep["random_forest"]
    assert 0.0 <= rf_metrics["roc_auc"] <= 1.0
    assert 0.0 <= rf_metrics["pr_auc"] <= 1.0
    assert 0.0 <= rf_metrics["accuracy"] <= 1.0
    assert "f2_score" in rf_metrics
    assert "brier_score" in rf_metrics
    assert "mcc" in rf_metrics
    assert "specificity" in rf_metrics
    assert "permutation_importances" in eval_rep
    assert "permutation_importances_std" in eval_rep

    lr_metrics = eval_rep["baseline_logistic_regression"]
    assert "f2_score" in lr_metrics
    assert "brier_score" in lr_metrics

    # Test artifact files existence
    assert os.path.exists(os.path.join(temp_models_dir, "random_forest_model.joblib"))
    assert os.path.exists(os.path.join(temp_models_dir, "baseline_logistic_regression.joblib"))
    assert os.path.exists(os.path.join(temp_models_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(temp_models_dir, "metadata.json"))
    assert os.path.exists(os.path.join(temp_models_dir, "evaluation_metrics.json"))

    # Test loading artifacts
    artifacts = load_artifacts(models_dir=temp_models_dir)
    assert "rf_model" in artifacts
    assert "scaler" in artifacts

    # Test prediction helper
    preds, probs = predict_machine_failure(X_test, artifacts["rf_model"], threshold=0.5)
    assert len(preds) == len(X_test)
    assert len(probs) == len(X_test)
    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_train_supervised_models_default_scaler(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=150, random_state=1)
    X_train, X_test, y_train, y_test, feat_cols, _ = prepare_data_splits(df)

    # Call with scaler=None and feature_columns=None
    res = train_supervised_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=None,
        feature_columns=None,
        models_dir=temp_models_dir,
    )
    assert "rf_model" in res
    assert "scaler" in res
    assert res["scaler"] is not None


def test_predict_machine_failure_column_reordering_and_lr(temp_models_dir):
    df = generate_synthetic_ai4i_dataset(n_samples=150, random_state=42)
    X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_data_splits(df)

    res = train_supervised_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_columns=feat_cols,
        models_dir=temp_models_dir,
    )

    rf_model = res["rf_model"]
    lr_model = res["lr_model"]

    # Original order
    preds_orig, probs_orig = predict_machine_failure(X_test, rf_model, threshold=0.35)

    # Permuted columns
    permuted_cols = list(reversed(X_test.columns))
    X_test_permuted = X_test[permuted_cols]
    preds_perm, probs_perm = predict_machine_failure(X_test_permuted, rf_model, threshold=0.35)

    assert np.allclose(probs_orig, probs_perm)
    assert np.array_equal(preds_orig, preds_perm)

    # Test LR prediction with scaler
    lr_preds, lr_probs = predict_machine_failure(X_test, lr_model, threshold=0.5, scaler=scaler)
    assert len(lr_preds) == len(X_test)
    assert len(lr_probs) == len(X_test)
    assert np.all((lr_probs >= 0.0) & (lr_probs <= 1.0))
