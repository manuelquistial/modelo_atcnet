"""Ablation variants (Table II, Altaheri et al. 2023)."""

from __future__ import annotations

from tensorflow.keras import Model

from src.atcnet import build_atcnet
from src.config import N_CLASSES, N_EEG_CHANNELS, N_SAMPLES

ABLATION_VARIANTS = (
    "full_atcnet",
    "no_attention",
    "no_sliding_window",
    "no_tcn",
    "cv_only",
)


def build_ablation_model(variant: str, **kwargs) -> Model:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {ABLATION_VARIANTS}")

    base = dict(
        n_channels=kwargs.get("n_channels", N_EEG_CHANNELS),
        n_samples=kwargs.get("n_samples", N_SAMPLES),
        n_classes=kwargs.get("n_classes", N_CLASSES),
        variant=variant,
    )

    if variant == "full_atcnet":
        return build_atcnet(**base, use_attention=True, use_sliding_window=True, use_tcn=True)
    if variant == "no_attention":
        return build_atcnet(**base, use_attention=False, use_sliding_window=True, use_tcn=True)
    if variant == "no_sliding_window":
        return build_atcnet(**base, use_attention=True, use_sliding_window=False, use_tcn=True)
    if variant == "no_tcn":
        return build_atcnet(**base, use_attention=True, use_sliding_window=True, use_tcn=False)
    if variant == "cv_only":
        return build_atcnet(**base, variant="cv_only")
    raise ValueError(variant)
