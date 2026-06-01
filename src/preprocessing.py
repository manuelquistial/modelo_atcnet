"""Input shaping and channel-wise standardization (official repo default)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical as keras_to_categorical


def prepare_input(X: np.ndarray) -> np.ndarray:
    """
    Convert (n_trials, n_channels, n_samples) -> (n_trials, 1, n_channels, n_samples).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim==3, got shape {X.shape}")
    return X[:, np.newaxis, :, :]


def to_categorical(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    y = np.asarray(y).astype(int)
    return keras_to_categorical(y, num_classes=n_classes)


def standardize_channels(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel StandardScaler fit on train, apply to train and test.
    Matches official preprocess.standardize_data (EEG-ATCNet).
    X: (trials, 1, channels, samples).
    """
    X_train = np.asarray(X_train, dtype=np.float32).copy()
    X_test = np.asarray(X_test, dtype=np.float32).copy()
    n_ch = X_train.shape[2]
    for j in range(n_ch):
        scaler = StandardScaler()
        tr = X_train[:, 0, j, :]
        scaler.fit(tr)
        X_train[:, 0, j, :] = scaler.transform(tr)
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


def standardize_per_trial_channel(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Optional z-score per trial and channel (not used by default; paper uses raw EEG).
    X: (trials, 1, channels, samples) or (trials, channels, samples).
    """
    X = np.asarray(X, dtype=np.float32)
    squeeze = False
    if X.ndim == 4 and X.shape[1] == 1:
        squeeze = True
        X = X[:, 0, :, :]
    if X.ndim != 3:
        raise ValueError(f"Unsupported shape for standardization: {X.shape}")

    out = np.empty_like(X)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            seg = X[i, c, :]
            std = seg.std()
            out[i, c, :] = (seg - seg.mean()) / (std + eps)

    if squeeze:
        out = out[:, np.newaxis, :, :]
    return out
