"""TCN residual stack (EEG-ATCNet models.TCN_block_)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

from src.config import CONV_MAX_NORM, CONV_WEIGHT_DECAY


def tcn_block(
    x: tf.Tensor,
    input_dimension: int,
    depth: int = 2,
    kernel_size: int = 4,
    filters: int = 32,
    dropout_rate: float = 0.3,
    activation: str = "elu",
    weight_decay: float = CONV_WEIGHT_DECAY,
    max_norm_value: float = CONV_MAX_NORM,
    name: str = "tcn",
) -> tf.Tensor:
    """
    Temporal Convolutional Network; returns full sequence (batch, time, filters).
    Caller takes x[:, -1, :] as in official ATCNet_.
    """
    conv_kw = dict(
        kernel_regularizer=l2(weight_decay),
        kernel_constraint=max_norm(max_norm_value, axis=[0, 1]),
        padding="causal",
        kernel_initializer="he_uniform",
        activation="linear",
        use_bias=True,
    )

    block = layers.Conv1D(
        filters, kernel_size, dilation_rate=1, name=f"{name}_d0_c1", **conv_kw
    )(x)
    block = layers.BatchNormalization(name=f"{name}_d0_bn1")(block)
    block = layers.Activation(activation, name=f"{name}_d0_a1")(block)
    block = layers.Dropout(dropout_rate, name=f"{name}_d0_drop1")(block)

    block = layers.Conv1D(
        filters, kernel_size, dilation_rate=1, name=f"{name}_d0_c2", **conv_kw
    )(block)
    block = layers.BatchNormalization(name=f"{name}_d0_bn2")(block)
    block = layers.Activation(activation, name=f"{name}_d0_a2")(block)
    block = layers.Dropout(dropout_rate, name=f"{name}_d0_drop2")(block)

    if input_dimension != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            kernel_regularizer=l2(weight_decay),
            kernel_constraint=max_norm(max_norm_value, axis=[0, 1]),
            name=f"{name}_proj",
        )(x)
    else:
        shortcut = x

    out = layers.Activation(activation, name=f"{name}_d0_out")(
        layers.Add(name=f"{name}_d0_add")([block, shortcut])
    )

    for i in range(depth - 1):
        d = 2 ** (i + 1)
        block = layers.Conv1D(
            filters, kernel_size, dilation_rate=d, name=f"{name}_d{i+1}_c1", **conv_kw
        )(out)
        block = layers.BatchNormalization(name=f"{name}_d{i+1}_bn1")(block)
        block = layers.Activation(activation, name=f"{name}_d{i+1}_a1")(block)
        block = layers.Dropout(dropout_rate, name=f"{name}_d{i+1}_drop1")(block)

        block = layers.Conv1D(
            filters, kernel_size, dilation_rate=d, name=f"{name}_d{i+1}_c2", **conv_kw
        )(block)
        block = layers.BatchNormalization(name=f"{name}_d{i+1}_bn2")(block)
        block = layers.Activation(activation, name=f"{name}_d{i+1}_a2")(block)
        block = layers.Dropout(dropout_rate, name=f"{name}_d{i+1}_drop2")(block)

        out = layers.Activation(activation, name=f"{name}_d{i+1}_out")(
            layers.Add(name=f"{name}_d{i+1}_add")([block, out])
        )

    return out


def tcn_last_step(
    x: tf.Tensor,
    input_dimension: int,
    **kwargs,
) -> tf.Tensor:
    """TCN + last temporal index (batch, filters)."""
    seq = tcn_block(x, input_dimension=input_dimension, **kwargs)
    return seq[:, -1, :]
