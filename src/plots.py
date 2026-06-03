"""Figures: confusion matrices, training curves, ablation summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import CLASS_NAMES
from src.utils import ensure_dir


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
        fmt = ".2f"
        suffix = " (normalized)"
    else:
        cm_plot = np.asarray(cm, dtype=int)
        fmt = "d"
        suffix = ""

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + suffix)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(history_path: str | Path, save_path: str | Path) -> None:
    import json

    with open(history_path, encoding="utf-8") as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.get("loss", []), label="train")
    axes[0].plot(hist.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(hist.get("accuracy", []), label="train")
    axes[1].plot(hist.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ablation_summary(summary_df: pd.DataFrame, save_path: str | Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    acc_col = "accuracy_mean" if "accuracy_mean" in summary_df.columns else "accuracy"
    kappa_col = "kappa_mean" if "kappa_mean" in summary_df.columns else "kappa"
    acc_err = summary_df["accuracy_std"] if "accuracy_std" in summary_df.columns else None
    kappa_err = summary_df["kappa_std"] if "kappa_std" in summary_df.columns else None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df[acc_col], width, yerr=acc_err, label="Accuracy")
    ax.bar(x + width / 2, summary_df[kappa_col], width, yerr=kappa_err, label="Kappa")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["variant"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Ablation study")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
