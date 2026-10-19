"""
Unsupervised Anomaly Detection Module using Isolation Forest.
Detects out-of-distribution equipment states, subtle sensor drifts, and novel failure modes.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ANOMALY_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Difference_K",
    "Power_W",
    "Overstrain_Index",
    "Speed_Torque_Ratio",
]


class IndustrialAnomalyDetector:
    """
    Wrapper for Scikit-Learn IsolationForest configured for industrial telemetry.
    """

    def __init__(self, contamination: float = 0.04, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.min_score_ = -0.5
        self.max_score_ = 0.5

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required anomaly features are present, auto-engineering if needed."""
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X, columns=ANOMALY_FEATURE_COLUMNS)
        
        missing = [c for c in ANOMALY_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            from src.preprocessing import engineer_features
            X_prep = engineer_features(X)
        else:
            X_prep = X.copy()

        for col in ANOMALY_FEATURE_COLUMNS:
            if col not in X_prep.columns:
                X_prep[col] = 0.0

        return X_prep[ANOMALY_FEATURE_COLUMNS].copy()

    def fit(self, X: pd.DataFrame) -> "IndustrialAnomalyDetector":
        """Fits Isolation Forest on continuous sensor and engineered features."""
        features = self._prepare_features(X)
        self.model.fit(features)
        raw_scores = self.model.decision_function(features)
        self.min_score_ = float(np.min(raw_scores))
        self.max_score_ = float(np.max(raw_scores))
        self.is_fitted = True
        return self

    def score_samples(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes anomaly indicators:
        - raw_score: Scikit-learn decision_function output (lower is more anomalous)
        - anomaly_score_pct: Normalized [0, 100] scale where 100% is most anomalous
        - is_anomaly: 1 if flagged anomaly (-1 in raw IF), 0 if normal
        """
        features = self._prepare_features(X)
        raw_scores = self.model.decision_function(features)
        predictions = self.model.predict(features)  # -1 for anomaly, 1 for normal

        # Normalize score to 0 - 100% anomaly severity
        denom = (self.max_score_ - self.min_score_) if (self.max_score_ != self.min_score_) else 1.0
        normalized = 1.0 - ((raw_scores - self.min_score_) / denom)
        anomaly_score_pct = np.clip(normalized * 100.0, 0.0, 100.0)

        is_anomaly = np.where(predictions == -1, 1, 0)
        return raw_scores, np.round(anomaly_score_pct, 2), is_anomaly

    def attach_anomaly_scores_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a telemetry dataframe with anomaly detection columns.
        """
        df_out = df.copy()
        raw_scores, score_pct, is_anomaly = self.score_samples(df_out)
        df_out["Anomaly_Raw_Score"] = np.round(raw_scores, 4)
        df_out["Anomaly_Severity_Pct"] = score_pct
        df_out["Is_Anomaly"] = is_anomaly
        df_out["Anomaly_Status"] = np.where(is_anomaly == 1, "Anomaly Alert", "Normal Operating")
        return df_out

    def export_anomalies(self, df: pd.DataFrame, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Detects anomalies and returns a DataFrame containing only flagged anomaly records.
        Optionally saves to CSV file.
        """
        df_enriched = self.attach_anomaly_scores_to_df(df)
        flagged = df_enriched[df_enriched["Is_Anomaly"] == 1].copy()
        if filepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            flagged.to_csv(filepath, index=False)
        return flagged

    def save(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "IndustrialAnomalyDetector":
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return joblib.load(filepath)


def train_anomaly_detector(
    X_train: pd.DataFrame,
    contamination: float = 0.04,
    models_dir: Optional[str] = None,
) -> IndustrialAnomalyDetector:
    """Trains and persists the Isolation Forest anomaly detector."""
    detector = IndustrialAnomalyDetector(contamination=contamination, random_state=42)
    detector.fit(X_train)
    save_path = os.path.join(models_dir or MODELS_DIR, "isolation_forest_model.joblib")
    detector.save(save_path)
    return detector


def evaluate_anomaly_detector_alignment(
    detector: IndustrialAnomalyDetector,
    df_eval: pd.DataFrame,
    target_col: str = "Machine failure",
) -> dict:
    """
    Evaluates the unsupervised anomaly detector's alignment with downstream equipment failures.
    
    IMPORTANT: The Isolation Forest was trained strictly unsupervised without target labels.
    This post-hoc evaluation measures how effectively density-based anomaly scoring
    correlates with true industrial failures and specific mechanical failure modes.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
    )

    raw_scores, severity_pct, is_anomaly = detector.score_samples(df_eval)

    if target_col not in df_eval.columns:
        return {
            "evaluated": False,
            "message": f"Target column '{target_col}' not present for alignment evaluation.",
        }

    y_true = df_eval[target_col].astype(int).values
    cm = confusion_matrix(y_true, is_anomaly, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    has_both = len(np.unique(y_true)) > 1
    roc_val = round(float(roc_auc_score(y_true, severity_pct)), 4) if has_both else 0.5
    pr_val = round(float(average_precision_score(y_true, severity_pct)), 4) if has_both else 0.0

    # Score separation between normal and failure groups
    normal_mask = y_true == 0
    failure_mask = y_true == 1
    mean_sev_normal = round(float(np.mean(severity_pct[normal_mask])), 2) if np.any(normal_mask) else 0.0
    mean_sev_fail = round(float(np.mean(severity_pct[failure_mask])), 2) if np.any(failure_mask) else 0.0

    # Failure mode specific detection rates
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_detection = {}
    for mode in failure_modes:
        if mode in df_eval.columns:
            mode_mask = df_eval[mode].astype(int) == 1
            mode_total = int(mode_mask.sum())
            if mode_total > 0:
                mode_caught = int((is_anomaly[mode_mask] == 1).sum())
                mode_detection[mode] = {
                    "total_incidents": mode_total,
                    "flagged_as_anomaly": mode_caught,
                    "detection_rate_pct": round((mode_caught / mode_total) * 100.0, 1),
                }

    return {
        "evaluated": True,
        "sample_count": len(df_eval),
        "total_failures": int(y_true.sum()),
        "total_anomalies_flagged": int(is_anomaly.sum()),
        "roc_auc": roc_val,
        "pr_auc": pr_val,
        "recall_on_failures": round(float(recall_score(y_true, is_anomaly, zero_division=0)), 4),
        "precision_on_anomalies": round(float(precision_score(y_true, is_anomaly, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, is_anomaly, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "mean_severity_normal": mean_sev_normal,
        "mean_severity_failure": mean_sev_fail,
        "failure_mode_detection": mode_detection,
    }

