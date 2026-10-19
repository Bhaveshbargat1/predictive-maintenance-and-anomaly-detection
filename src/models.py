"""
Supervised Machine Failure Prediction Module.
Implements Logistic Regression baseline and Random Forest Classifier with class balancing,
evaluation metrics calculation, artifact persistence, and inference helpers.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def train_supervised_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: Optional[StandardScaler] = None,
    feature_columns: Optional[list] = None,
    models_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trains Logistic Regression baseline and Random Forest with class_weight='balanced'.
    Evaluates both on test data, extracts feature importances, and returns results dict.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    if feature_columns is None:
        feature_columns = list(X_train.columns)

    num_cols = [c for c in feature_columns if not c.startswith("Type_")]

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train[num_cols])

    # Scale continuous numerical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # 1. Baseline: Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr_model.fit(X_train_scaled, y_train)

    # 2. Production: Random Forest Classifier
    logger.info("Training Random Forest Classifier with balanced class weights...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Evaluation helper
    def compute_metrics(model, X_eval, y_eval):
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Precision-Recall & ROC curves
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        precision_arr, recall_arr, _ = precision_recall_curve(y_eval, y_prob)

        has_both_classes = len(np.unique(y_eval)) > 1
        roc_val = round(float(roc_auc_score(y_eval, y_prob)), 4) if has_both_classes else 0.5
        pr_val = round(float(average_precision_score(y_eval, y_prob)), 4) if has_both_classes else 0.0

        f2_val = round(float(fbeta_score(y_eval, y_pred, beta=2.0, zero_division=0)), 4)
        brier_val = round(float(brier_score_loss(y_eval, y_prob)), 4)
        mcc_val = round(float(matthews_corrcoef(y_eval, y_pred)), 4)
        specificity_val = round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0

        # Sample curves for fast rendering & JSON serialization (at most 80 points)
        step = max(1, len(fpr) // 80)
        fpr_sampled = [round(float(v), 4) for v in fpr[::step]]
        tpr_sampled = [round(float(v), 4) for v in tpr[::step]]
        if fpr_sampled[-1] != round(float(fpr[-1]), 4):
            fpr_sampled.append(round(float(fpr[-1]), 4))
            tpr_sampled.append(round(float(tpr[-1]), 4))

        step_pr = max(1, len(recall_arr) // 80)
        p_sampled = [round(float(v), 4) for v in precision_arr[::step_pr]]
        r_sampled = [round(float(v), 4) for v in recall_arr[::step_pr]]
        if r_sampled[-1] != round(float(recall_arr[-1]), 4):
            p_sampled.append(round(float(precision_arr[-1]), 4))
            r_sampled.append(round(float(recall_arr[-1]), 4))

        # Probability calibration curve
        prob_true, prob_pred = calibration_curve(y_eval, y_prob, n_bins=8, strategy="uniform")

        metrics = {
            "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
            "precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
            "f2_score": f2_val,
            "brier_score": brier_val,
            "mcc": mcc_val,
            "specificity": specificity_val,
            "roc_auc": roc_val,
            "pr_auc": pr_val,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "roc_curve": {
                "fpr": fpr_sampled,
                "tpr": tpr_sampled,
            },
            "pr_curve": {
                "precision": p_sampled,
                "recall": r_sampled,
            },
            "calibration_curve": {
                "prob_true": [round(float(v), 4) for v in prob_true.tolist()],
                "prob_pred": [round(float(v), 4) for v in prob_pred.tolist()],
            },
        }
        return metrics

    lr_metrics = compute_metrics(lr_model, X_test_scaled, y_test)
    rf_metrics = compute_metrics(rf_model, X_test, y_test)

    # Feature Importance (Gini Impurity)
    feature_importances = dict(
        sorted(
            zip(feature_columns, [round(float(v), 4) for v in rf_model.feature_importances_]),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # Permutation Importance on held-out test set
    logger.info("Computing test-set permutation feature importance...")
    perm_res = permutation_importance(
        rf_model, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
    )
    permutation_importances = dict(
        sorted(
            zip(feature_columns, [round(float(v), 4) for v in perm_res.importances_mean]),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    permutation_importances_std = dict(
        zip(feature_columns, [round(float(v), 4) for v in perm_res.importances_std])
    )

    # Save artifacts
    joblib.dump(lr_model, os.path.join(models_dir, "baseline_logistic_regression.joblib"))
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    evaluation_report = {
        "baseline_logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "feature_importances": feature_importances,
        "permutation_importances": permutation_importances,
        "permutation_importances_std": permutation_importances_std,
        "test_size": len(y_test),
        "test_failure_count": int(y_test.sum()),
        "test_normal_count": int((y_test == 0).sum()),
    }

    with open(os.path.join(models_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_report, f, indent=2)

    metadata = {
        "feature_columns": feature_columns,
        "numerical_columns": num_cols,
        "target": "Machine failure",
        "n_features": len(feature_columns),
        "models": ["baseline_logistic_regression", "random_forest"],
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Successfully trained supervised models and stored artifacts.")
    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "evaluation_report": evaluation_report,
    }


def load_artifacts(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads saved model weights, scaler, and metadata.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    rf_path = os.path.join(models_dir, "random_forest_model.joblib")
    lr_path = os.path.join(models_dir, "baseline_logistic_regression.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    meta_path = os.path.join(models_dir, "metadata.json")
    eval_path = os.path.join(models_dir, "evaluation_metrics.json")

    artifacts = {}
    if os.path.exists(rf_path):
        artifacts["rf_model"] = joblib.load(rf_path)
    if os.path.exists(lr_path):
        artifacts["lr_model"] = joblib.load(lr_path)
    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            artifacts["evaluation_report"] = json.load(f)

    return artifacts


def predict_machine_failure(
    X_input: Any,
    model: Any,
    threshold: float = 0.50,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts binary failure flag and raw failure risk probability.
    Supports RandomForestClassifier, LogisticRegression, or any calibrated estimator.
    Handles column ordering and optional feature scaling.
    """
    X_processed = X_input
    if isinstance(X_processed, pd.DataFrame):
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            if all(col in X_processed.columns for col in expected):
                X_processed = X_processed[expected]

        if scaler is not None and isinstance(model, LogisticRegression):
            num_cols = [c for c in X_processed.columns if not c.startswith("Type_")]
            X_scaled = X_processed.copy()
            X_scaled[num_cols] = scaler.transform(X_processed[num_cols].fillna(0.0))
            X_processed = X_scaled

    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
