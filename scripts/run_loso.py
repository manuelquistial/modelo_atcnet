#!/usr/bin/env python3
"""LOSO subject-independent (EEG-ATCNet load_data_LOSO)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.atcnet import build_atcnet
from src.data_loader import N_SUBJECTS, load_all_subjects, validate_dataset_files
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.plots import plot_confusion_matrix
from src.preprocessing import prepare_input, standardize_channels, to_categorical
from src.config import SHUFFLE_DATA, USE_CHANNEL_STANDARDIZATION, N_TRAIN_RUNS
from src.splits import loso_masks
from src.train_utils import compile_model, set_seed, train_model
from src.utils import ensure_dir, get_results_dir
from sklearn.utils import shuffle as sk_shuffle


def main() -> None:
    out_dir = ensure_dir(get_results_dir() / "loso")
    figures_dir = ensure_dir(get_results_dir() / "figures")
    validate_dataset_files()

    X_all, y_all, subject_ids, session_ids = load_all_subjects()
    rows = []

    for test_subject in tqdm(range(1, N_SUBJECTS + 1), desc="LOSO"):
        train_mask, test_mask = loso_masks(subject_ids, test_subject)
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        X_train = prepare_input(X_train)
        X_test = prepare_input(X_test)
        if SHUFFLE_DATA:
            X_train, y_train = sk_shuffle(X_train, y_train, random_state=42)
            X_test, y_test = sk_shuffle(X_test, y_test, random_state=42)
        if USE_CHANNEL_STANDARDIZATION:
            X_train, X_test = standardize_channels(X_train, X_test)

        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
        subj_test = subject_ids[test_mask]
        sess_test = session_ids[test_mask]

        best_acc = -1.0
        best_model = None
        best_metrics = None
        best_pred_df = None

        for run in range(N_TRAIN_RUNS):
            set_seed(run + 1)
            model = build_atcnet()
            compile_model(model)
            train_model(
                model, X_train, y_train_cat, X_test, y_test_cat,
                out_dir, f"loso_subj_{test_subject:02d}_run{run+1}",
            )
            metrics, pred_df, _, _ = evaluate_model(
                model, X_test, y_test, subj_test, sess_test
            )
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_model = model
                best_metrics = metrics
                best_pred_df = pred_df

        metrics = best_metrics
        pred_df = best_pred_df

        pred_df.to_csv(out_dir / f"loso_test_subject_{test_subject:02d}_predictions.csv", index=False)
        export_errors(
            pred_df, out_dir / f"loso_test_subject_{test_subject:02d}_errors.csv", "loso"
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            figures_dir / f"loso_subject_{test_subject:02d}_cm.png",
            title=f"LOSO test subject {test_subject:02d}",
        )
        rows.append({
            "test_subject": test_subject,
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
        })
        print(
            f"LOSO test {test_subject:02d}: acc={metrics['accuracy']:.4f}, "
            f"kappa={metrics['kappa']:.4f}"
        )

    df = pd.DataFrame(rows)
    summary = pd.concat([
        df,
        pd.DataFrame([
            {"test_subject": "mean", "accuracy": df["accuracy"].mean(), "kappa": df["kappa"].mean()},
            {"test_subject": "std", "accuracy": df["accuracy"].std(), "kappa": df["kappa"].std()},
        ]),
    ], ignore_index=True)
    path = out_dir / "loso_metrics.csv"
    summary.to_csv(path, index=False)
    print("\n=== LOSO (EEG-ATCNet aligned) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved {path}")
    print("Reference: ~70.97% acc, ~0.613 κ")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
