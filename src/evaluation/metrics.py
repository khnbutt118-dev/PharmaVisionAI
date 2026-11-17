"""Evaluation helpers for binary tablet classification."""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score
)

def evaluate_binary(y_true, y_pred):
    """Assumes 1=defective and 0=normal."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "confusion_matrix": cm.tolist(),
        "precision_defective": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity_defective": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if (tn + fp) else None,
        "false_pass_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "f1_defective": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["normal", "defective"], zero_division=0
        )
    }
