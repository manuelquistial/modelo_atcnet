"""Classification metrics and per-trial prediction tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from src.data_loader import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy, Cohen's kappa, and confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
        ),
    }


def classification_summary(metrics: dict) -> str:
    return (
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Cohen's kappa: {metrics['kappa']:.4f}"
    )


def prediction_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    session_ids: np.ndarray | None = None,
    trial_offset: int = 0,
) -> pd.DataFrame:
    """Build per-trial prediction DataFrame for audit and error analysis."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    n = len(y_true)
    if subject_ids is None:
        subject_ids = np.full(n, -1, dtype=int)
    if session_ids is None:
        session_ids = np.full(n, -1, dtype=object)

    rows = []
    for i in range(n):
        tl, pl = int(y_true[i]), int(y_pred[i])
        rows.append(
            {
                "trial_index": trial_offset + i,
                "subject": int(subject_ids[i]),
                "session": str(session_ids[i]),
                "true_label": tl,
                "true_class": CLASS_NAMES[tl],
                "pred_label": pl,
                "pred_class": CLASS_NAMES[pl],
                "is_correct": tl == pl,
                "prob_left_hand": float(y_prob[i, 0]),
                "prob_right_hand": float(y_prob[i, 1]),
                "prob_feet": float(y_prob[i, 2]),
                "prob_tongue": float(y_prob[i, 3]),
            }
        )
    return pd.DataFrame(rows)
