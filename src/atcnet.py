"""
ATCNet aligned with Altaheri/EEG-ATCNet models.ATCNet_ and Conv_block_.
Input: (batch, 1, n_channels, n_samples) e.g. (B, 1, 22, 1125).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

from src.attention_models import mha_block
from src.config import (
    ATT_DROPOUT,
    ATT_HEADS,
    ATT_KEY_DIM,
    CONV_MAX_NORM,
    CONV_WEIGHT_DECAY,
    DENSE_WEIGHT_DECAY,
    EEGN_D,
    EEGN_DROPOUT,
    EEGN_F1,
    EEGN_KERNEL_SIZE,
    EEGN_POOL_SIZE,
    FUSE_MODE,
    N_CLASSES,
    N_EEG_CHANNELS,
    N_SAMPLES,
    N_WINDOWS,
    TCN_DEPTH,
    TCN_DROPOUT,
    TCN_FILTERS,
    TCN_KERNEL_SIZE,
)
from src.tcn import tcn_block


def convolutional_block(
    inputs: tf.Tensor,
    F1: int = EEGN_F1,
    D: int = EEGN_D,
    kernel_length: int = EEGN_KERNEL_SIZE,
    pool1: int = 8,
    pool2: int = EEGN_POOL_SIZE,
    dropout_rate: float = EEGN_DROPOUT,
    n_channels: int = 22,
    weight_decay: float = CONV_WEIGHT_DECAY,
    max_norm_value: float = CONV_MAX_NORM,
    name: str = "cv",
) -> tf.Tensor:
    """
    Conv_block_ (official EEG-ATCNet).

    Input (B, 1, C, T) → Permute like official (B, T, C, 1) → channels_last convs
    → (B, Tc, F2).
    """
    F2 = F1 * D
    conv2d_kw = dict(
        padding="same",
        use_bias=False,
        data_format="channels_last",
        kernel_regularizer=l2(weight_decay),
        kernel_constraint=max_norm(max_norm_value, axis=[0, 1, 2]),
    )

    # (B, 1, 22, 1125) → (B, 1125, 22, 1)  [official: Permute(3,2,1) on (1,C,T)]
    x = layers.Permute((3, 2, 1), name=f"{name}_to_cl")(inputs)

    x = layers.Conv2D(
        F1, (kernel_length, 1), name=f"{name}_c1", **conv2d_kw
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)

    x = layers.DepthwiseConv2D(
        (1, n_channels),
        depth_multiplier=D,
        use_bias=False,
        data_format="channels_last",
        depthwise_regularizer=l2(weight_decay),
        depthwise_constraint=max_norm(max_norm_value, axis=[0, 1, 2]),
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("elu", name=f"{name}_elu1")(x)
    x = layers.AveragePooling2D((pool1, 1), data_format="channels_last", name=f"{name}_pool1")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop1")(x)

    x = layers.Conv2D(F2, (16, 1), name=f"{name}_c2", **conv2d_kw)(x)
    x = layers.BatchNormalization(name=f"{name}_bn3")(x)
    x = layers.Activation("elu", name=f"{name}_elu2")(x)
    x = layers.AveragePooling2D((pool2, 1), data_format="channels_last", name=f"{name}_pool2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop2")(x)

    # (B, Tc, 1, F2) → (B, Tc, F2)  [official: Lambda x[:,:,-1,:]]
    return layers.Lambda(lambda t: t[:, :, -1, :], name=f"{name}_seq")(x)


def build_atcnet(
    n_channels: int = N_EEG_CHANNELS,
    n_samples: int = N_SAMPLES,
    n_classes: int = N_CLASSES,
    n_windows: int = N_WINDOWS,
    fuse: str = FUSE_MODE,
    use_attention: bool = True,
    use_sliding_window: bool = True,
    use_tcn: bool = True,
    variant: str = "full_atcnet",
    **kwargs,
) -> Model:
    """
    ATCNet_ with fuse='average' (official default): per-window Dense(4) → Average → softmax.
    """
    F1 = kwargs.get("F1", EEGN_F1)
    F2 = F1 * kwargs.get("D", EEGN_D)

    inp = layers.Input(shape=(1, n_channels, n_samples), name="eeg_input")
    seq = convolutional_block(inp, F1=F1, n_channels=n_channels, name="cv")

    if variant == "cv_only":
        flat = layers.Flatten()(seq)
        out = layers.Dense(
            n_classes,
            activation="softmax",
            kernel_regularizer=l2(DENSE_WEIGHT_DECAY),
            name="classifier",
        )(flat)
        return Model(inp, out, name="ATCNet_cv_only")

    if not use_sliding_window:
        h = seq
        if use_attention:
            h = mha_block(h, key_dim=ATT_KEY_DIM, num_heads=ATT_HEADS, dropout=ATT_DROPOUT)
        if use_tcn:
            h = tcn_block(h, input_dimension=F2, depth=TCN_DEPTH, kernel_size=TCN_KERNEL_SIZE,
                          filters=TCN_FILTERS, dropout_rate=TCN_DROPOUT)
            vec = h[:, -1, :]
        else:
            vec = layers.GlobalAveragePooling1D()(h)
        out = layers.Dense(
            n_classes, activation="softmax", kernel_regularizer=l2(DENSE_WEIGHT_DECAY)
        )(vec)
        return Model(inp, out, name=f"ATCNet_{variant}")

    Tc = int(seq.shape[1])
    Tw = Tc - n_windows + 1
    sw_out = []

    for i in range(n_windows):
        st = i
        win = layers.Lambda(
            lambda t, s=st, l=Tw: t[:, s : s + l, :],
            name=f"sw_slice_{i}",
        )(seq)

        if use_attention:
            win = mha_block(
                win, key_dim=ATT_KEY_DIM, num_heads=ATT_HEADS, dropout=ATT_DROPOUT
            )

        if use_tcn:
            tcn_seq = tcn_block(
                win,
                input_dimension=F2,
                depth=TCN_DEPTH,
                kernel_size=TCN_KERNEL_SIZE,
                filters=TCN_FILTERS,
                dropout_rate=TCN_DROPOUT,
                name=f"tcn_w{i}",
            )
            vec = layers.Lambda(lambda x: x[:, -1, :], name=f"tcn_last_{i}")(tcn_seq)
        else:
            vec = layers.GlobalAveragePooling1D(name=f"gap_{i}")(win)

        if fuse == "average":
            sw_out.append(
                layers.Dense(
                    n_classes,
                    kernel_regularizer=l2(DENSE_WEIGHT_DECAY),
                    name=f"dense_w{i}",
                )(vec)
            )
        else:
            sw_out.append(vec)

    if fuse == "average":
        merged = layers.Average(name="sw_average")(sw_out) if len(sw_out) > 1 else sw_out[0]
        out = layers.Activation("softmax", name="softmax")(merged)
    else:
        merged = layers.Concatenate(name="sw_concat")(sw_out) if len(sw_out) > 1 else sw_out[0]
        out = layers.Dense(
            n_classes,
            activation="softmax",
            kernel_regularizer=l2(DENSE_WEIGHT_DECAY),
            name="classifier",
        )(merged)

    return Model(inp, out, name=f"ATCNet_{variant}")
