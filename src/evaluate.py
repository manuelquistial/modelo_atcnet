"""Model evaluation on held-out trials."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tensorflow.keras import Model

from src.metrics import compute_metrics, prediction_table


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_ids: np.ndarray | None = None,
    session_ids: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run inference and build metrics + per-trial prediction table.

    Returns
    -------
    metrics : dict with accuracy, kappa, confusion_matrix
    prediction_df : per-trial audit table
    y_prob : (n_trials, n_classes)
    y_pred : (n_trials,)
    """
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred)
    pred_df = prediction_table(
        y_test,
        y_prob,
        y_pred=y_pred,
        subject_ids=subject_ids,
        session_ids=session_ids,
    )
    return metrics, pred_df, y_prob, y_pred
