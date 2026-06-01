"""Training pipeline aligned with EEG-ATCNet main_TrainTest.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks

from src.config import BATCH_SIZE, EARLY_STOP_PATIENCE, EPOCHS, LEARNING_RATE
from src.utils import ensure_dir


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(output_dir: str | Path, model_name: str) -> list[callbacks.Callback]:
    """Official: EarlyStopping + Checkpoint on val_accuracy; ReduceLROnPlateau on val_loss."""
    output_dir = ensure_dir(output_dir)
    checkpoint_path = Path(output_dir) / f"{model_name}_best.keras"
    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.90,
            patience=20,
            min_lr=0.0001,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_val: np.ndarray,
    y_val_cat: np.ndarray,
    output_dir: str | Path,
    model_name: str,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> tf.keras.callbacks.History:
    output_dir = ensure_dir(output_dir)
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=build_callbacks(output_dir, model_name),
        verbose=1,
    )
    save_history(history, Path(output_dir) / f"{model_name}_history.json")
    return history


def save_history(history: tf.keras.callbacks.History, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
