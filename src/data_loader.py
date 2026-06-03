"""PhysioNet MI data access for ATCNet (binary left/right hand)."""

from __future__ import annotations

import numpy as np

from src.config import CLASS_NAMES, N_CLASSES, N_EEG_CHANNELS, N_SAMPLES
from src.physionet.constants import SFREQ
from src.physionet.dataset import (
    get_model_dims,
    load_and_preprocess_cohort,
    load_holdout_data,
    load_loso_fold,
    load_physionet_cohort,
)

N_SUBJECTS = 109  # MOABB PhysioNetMI (subject 88 excluded at load time)

__all__ = [
    "CLASS_NAMES",
    "N_CLASSES",
    "N_EEG_CHANNELS",
    "N_SAMPLES",
    "N_SUBJECTS",
    "SFREQ",
    "get_model_dims",
    "load_and_preprocess_cohort",
    "load_holdout_data",
    "load_loso_fold",
    "load_physionet_cohort",
    "validate_shapes",
]


def validate_shapes(X: np.ndarray, y: np.ndarray, n_channels: int = N_EEG_CHANNELS) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be (trials, channels, samples), got {X.shape}")
    if X.shape[1] != n_channels:
        raise ValueError(f"Expected {n_channels} channels, got {X.shape[1]}")
    if len(y) != X.shape[0]:
        raise ValueError(f"len(y)={len(y)} != n_trials={X.shape[0]}")
    if y.max() >= N_CLASSES or y.min() < 0:
        raise ValueError(f"Labels must be in [0, {N_CLASSES - 1}], got {np.unique(y)}")
