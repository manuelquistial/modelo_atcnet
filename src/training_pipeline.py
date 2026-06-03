"""Shared train/eval steps (EEG-ATCNet get_data + main_TrainTest)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from tensorflow.keras import Model

from src.config import N_CLASSES, PAPER_PROFILE, SHUFFLE_DATA, TrainingProfile, USE_CHANNEL_STANDARDIZATION
from src.preprocessing import prepare_input, standardize_channels, to_categorical
from src.train_utils import compile_model, set_seed, train_model


def prepare_official_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """prepare_input → shuffle → StandardScaler → categorical. Returns X_test 4D + y_test int."""
    X_train = prepare_input(X_train)
    X_test = prepare_input(X_test)
    if SHUFFLE_DATA:
        X_train, y_train = sk_shuffle(X_train, y_train, random_state=42)
        X_test, y_test = sk_shuffle(X_test, y_test, random_state=42)
    if USE_CHANNEL_STANDARDIZATION:
        X_train, X_test = standardize_channels(X_train, X_test)
    return (
        X_train,
        to_categorical(y_train, n_classes=n_classes),
        X_test,
        to_categorical(y_test, n_classes=n_classes),
        y_test,
    )


def train_best_of_runs(
    build_model_fn: Callable[[], Model],
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_test: np.ndarray,
    y_test_cat: np.ndarray,
    evaluate_fn: Callable[[Model], dict],
    output_dir,
    name_prefix: str,
    profile: TrainingProfile = PAPER_PROFILE,
) -> tuple[Model, dict]:
    """Train n_train runs; validation_data = test; keep best test accuracy."""
    best_acc = -1.0
    best_model = None
    best_metrics = None

    for run in range(profile.n_train_runs):
        set_seed(run + 1)
        model = build_model_fn()
        compile_model(model)
        train_model(
            model,
            X_train,
            y_train_cat,
            X_test,
            y_test_cat,
            output_dir,
            f"{name_prefix}_run{run + 1}",
            profile=profile,
        )
        metrics = evaluate_fn(model)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return best_model, best_metrics
