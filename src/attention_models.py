"""Multi-head self-attention (EEG-ATCNet official attention_models.mha_block)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Add, LayerNormalization, MultiHeadAttention


def mha_block(
    input_feature: tf.Tensor,
    key_dim: int = 8,
    num_heads: int = 2,
    dropout: float = 0.5,
) -> tf.Tensor:
    """LayerNorm → MHA(self) → residual (official vanilla MHA)."""
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    x = MultiHeadAttention(
        key_dim=key_dim,
        num_heads=num_heads,
        dropout=dropout,
    )(x, x)
    return Add()([input_feature, x])
