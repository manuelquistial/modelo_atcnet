#!/usr/bin/env python3
"""Verify ATCNet shapes for PhysioNet MI (64 ch, binary classification)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.atcnet import build_atcnet, convolutional_block
from src.config import N_CLASSES, N_EEG_CHANNELS, N_SAMPLES
from src.train_utils import compile_model, set_seed


def main() -> None:
    set_seed(42)
    n_ch, n_samples, n_classes = N_EEG_CHANNELS, N_SAMPLES, N_CLASSES
    print("PhysioNet MI — ATCNet flow (batch=B):")
    print(f"  Input:           (B, 1, {n_ch}, {n_samples})")
    print(f"  Classes:         {n_classes} (left_hand, right_hand)")
    print()

    X = np.random.randn(2, 1, n_ch, n_samples).astype(np.float32)
    model = build_atcnet(n_channels=n_ch, n_samples=n_samples, n_classes=n_classes)
    compile_model(model)
    model.summary()

    y = model.predict(X, verbose=0)
    assert y.shape == (2, n_classes), f"Expected (2, {n_classes}), got {y.shape}"
    print(f"\nOutput OK: {y.shape}")

    inp = model.input
    cv_model = __import__("tensorflow").keras.Model(inp, convolutional_block(inp, n_channels=n_ch))
    seq = cv_model.predict(X, verbose=0)
    print(f"CV block output: {seq.shape}")
    if seq.shape[1] >= 5:
        print(f"Tw = {seq.shape[1] - 5 + 1} for n_windows=5")


if __name__ == "__main__":
    main()
