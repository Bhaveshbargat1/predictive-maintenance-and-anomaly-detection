"""
End-to-end training and artifact generation pipeline.
Trains supervised failure prediction models and unsupervised anomaly detector.
Saves all model weights and evaluation metrics into models/.
"""

import os
import sys
import logging

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_dataset
from src.preprocessing import prepare_data_splits
from src.models import train_supervised_models
from src.anomaly_detector import train_anomaly_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def run_training_pipeline():
    logger.info("Step 1: Loading AI4I 2020 dataset...")
    df = load_dataset()
    logger.info(f"Loaded {len(df)} records. Failure count: {df['Machine failure'].sum()} ({df['Machine failure'].mean()*100:.2f}%)")

    logger.info("Step 2: Preprocessing and feature engineering...")
    X_train, X_test, y_train, y_test, feature_columns, scaler = prepare_data_splits(df)
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    logger.info("Step 3: Training Supervised Models (Logistic Regression & Balanced Random Forest)...")
    models_dir = os.path.join(BASE_DIR, "models")
    sup_results = train_supervised_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_columns=feature_columns,
        models_dir=models_dir,
    )

    rf_eval = sup_results["evaluation_report"]["random_forest"]
    lr_eval = sup_results["evaluation_report"]["baseline_logistic_regression"]
    logger.info(f"Random Forest - ROC-AUC: {rf_eval['roc_auc']:.4f}, PR-AUC: {rf_eval['pr_auc']:.4f}, Recall: {rf_eval['recall']:.4f}, F1: {rf_eval['f1_score']:.4f}")
    logger.info(f"Logistic Regression - ROC-AUC: {lr_eval['roc_auc']:.4f}, PR-AUC: {lr_eval['pr_auc']:.4f}, Recall: {lr_eval['recall']:.4f}, F1: {lr_eval['f1_score']:.4f}")

    logger.info("Step 4: Training Unsupervised Isolation Forest Anomaly Detector...")
    detector = train_anomaly_detector(X_train=X_train, contamination=0.04, models_dir=models_dir)
    logger.info(f"Isolation Forest trained successfully. Artifacts stored in {models_dir}")

    logger.info("Step 5: Evaluating Unsupervised Anomaly Alignment with Failures (Post-Hoc)...")
    from src.anomaly_detector import evaluate_anomaly_detector_alignment
    import json
    anomaly_eval = evaluate_anomaly_detector_alignment(detector, df, target_col="Machine failure")
    logger.info(
        f"Anomaly Alignment - ROC-AUC: {anomaly_eval['roc_auc']:.4f}, "
        f"PR-AUC: {anomaly_eval['pr_auc']:.4f}, "
        f"Recall on Failures: {anomaly_eval['recall_on_failures']*100:.1f}%, "
        f"Precision: {anomaly_eval['precision_on_anomalies']*100:.1f}%"
    )

    eval_metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, "r") as f:
            full_report = json.load(f)
        full_report["unsupervised_anomaly_evaluation"] = anomaly_eval
        with open(eval_metrics_path, "w") as f:
            json.dump(full_report, f, indent=2)

    return {
        "supervised": sup_results,
        "detector": detector,
        "anomaly_evaluation": anomaly_eval,
        "test_data": (X_test, y_test),
    }


if __name__ == "__main__":
    run_training_pipeline()
