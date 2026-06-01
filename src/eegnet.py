"""EEGNet baseline for comparison (Lawhern et al.)."""

from __future__ import annotations

from tensorflow.keras import Model, layers
from tensorflow.keras.constraints import max_norm


def build_eegnet(
    n_channels: int = 22,
    n_samples: int = 1125,
    n_classes: int = 4,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    kernel_length: int = 64,
    dropout_rate: float = 0.5,
) -> Model:
    """
    EEGNet-style architecture for input (batch, 1, channels, samples).

    Uses temporal Conv2D, depthwise spatial Conv2D, separable Conv2D,
    pooling, dropout, and softmax classifier.
    """
    inp = layers.Input(shape=(1, n_channels, n_samples), name="eeg_input")

    x = layers.Conv2D(
        F1,
        (1, kernel_length),
        padding="same",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        name="temporal_conv",
    )(inp)
    x = layers.BatchNormalization(name="bn1")(x)

    x = layers.DepthwiseConv2D(
        (n_channels, 1),
        depth_multiplier=D,
        use_bias=False,
        depthwise_constraint=max_norm(1.0),
        name="spatial_dwconv",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("elu", name="elu1")(x)
    x = layers.AveragePooling2D((1, 8), name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.SeparableConv2D(
        F2,
        (1, 16),
        padding="same",
        use_bias=False,
        name="separable_conv",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("elu", name="elu2")(x)
    x = layers.AveragePooling2D((1, 7), name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    x = layers.Flatten(name="flatten")(x)
    out = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        name="classifier",
    )(x)
    return Model(inp, out, name="EEGNet")
