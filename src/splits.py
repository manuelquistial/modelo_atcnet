"""Train/validation splitting for model training."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified hold-out split from training data."""
    strat = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat,
    )


def loso_masks(
    subject_ids: np.ndarray, test_subject: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for LOSO: train on all except test_subject."""
    test_mask = subject_ids == test_subject
    train_mask = ~test_mask
    return train_mask, test_mask
