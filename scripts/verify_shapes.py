#!/usr/bin/env python3
"""Verify ATCNet shapes (EEG-ATCNet architecture)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.atcnet import build_atcnet, convolutional_block
from src.train_utils import compile_model, set_seed


def main() -> None:
    set_seed(42)
    print("EEG-ATCNet aligned flow (batch=B):")
    print("  Input:           (B, 1, 22, 1125)")
    print("  After CV:        (B, 20, 32)")
    print("  Windows:         5 x (B, 16, 32) → TCN → Dense(4) → Average → softmax")
    print()

    X = np.random.randn(2, 1, 22, 1125).astype(np.float32)
    model = build_atcnet()
    compile_model(model)
    model.summary()

    y = model.predict(X, verbose=0)
    assert y.shape == (2, 4), f"Expected (2, 4), got {y.shape}"
    print(f"\nOutput OK: {y.shape}")

    inp = model.input
    cv_model = __import__("tensorflow").keras.Model(inp, convolutional_block(inp))
    seq = cv_model.predict(X, verbose=0)
    print(f"CV block output: {seq.shape} (expected ~ (2, 20, 32))")
    print(f"Tw = {seq.shape[1] - 5 + 1} for n_windows=5")


if __name__ == "__main__":
    main()
