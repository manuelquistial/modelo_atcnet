#!/usr/bin/env python3
"""LOSO on PhysioNet MI — binary left vs right hand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.atcnet import build_atcnet
from src.config import PAPER_PROFILE, QUICK_PROFILE, SHUFFLE_DATA, TrainingProfile, USE_CHANNEL_STANDARDIZATION
from src.data_loader import load_and_preprocess_cohort
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.physionet.dataset import get_model_dims, loso_from_processed
from src.plots import plot_confusion_matrix
from src.preprocessing import prepare_input, standardize_channels, to_categorical
from src.train_utils import compile_model, set_seed, train_model
from src.utils import ensure_dir, get_results_dir
from sklearn.utils import shuffle as sk_shuffle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOSO PhysioNet MI (left vs right).")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--subjects", type=str, default=None, help="Limit cohort for smoke tests")
    return p.parse_args()


def parse_subjects(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main() -> None:
    args = parse_args()
    profile: TrainingProfile = QUICK_PROFILE if args.quick else PAPER_PROFILE
    subject_ids = parse_subjects(args.subjects)

    out_dir = ensure_dir(get_results_dir() / "loso")
    figures_dir = ensure_dir(get_results_dir() / "figures")

    processed, meta = load_and_preprocess_cohort(subject_ids=subject_ids, download=False)
    n_ch, n_samples, n_classes = get_model_dims(meta)
    test_subjects = sorted(processed.keys())

    build_fn = lambda: build_atcnet(
        n_channels=n_ch, n_samples=n_samples, n_classes=n_classes
    )

    rows = []
    for test_subject in tqdm(test_subjects, desc="LOSO"):
        X_train, y_train, X_test, y_test, _, g_test, _ = loso_from_processed(
            processed, test_subject, meta
        )

        X_train = prepare_input(X_train)
        X_test = prepare_input(X_test)
        if SHUFFLE_DATA:
            X_train, y_train = sk_shuffle(X_train, y_train, random_state=42)
            X_test, y_test = sk_shuffle(X_test, y_test, random_state=42)
        if USE_CHANNEL_STANDARDIZATION:
            X_train, X_test = standardize_channels(X_train, X_test)

        y_train_cat = to_categorical(y_train, n_classes=n_classes)
        y_test_cat = to_categorical(y_test, n_classes=n_classes)

        best_acc = -1.0
        best_metrics = None
        best_pred_df = None

        n_runs = profile.n_train_runs if not args.quick else profile.n_train_runs
        for run in range(n_runs):
            set_seed(run + 1)
            model = build_fn()
            compile_model(model)
            train_model(
                model, X_train, y_train_cat, X_test, y_test_cat,
                out_dir, f"loso_subj_{test_subject:03d}_run{run+1}",
                profile=profile,
            )
            metrics, pred_df, _, _ = evaluate_model(model, X_test, y_test, g_test)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics
                best_pred_df = pred_df

        metrics = best_metrics
        pred_df = best_pred_df
        pred_df.to_csv(out_dir / f"loso_test_subject_{test_subject:03d}_predictions.csv", index=False)
        export_errors(pred_df, out_dir / f"loso_test_subject_{test_subject:03d}_errors.csv", "loso")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            figures_dir / f"loso_subject_{test_subject:03d}_cm.png",
            title=f"LOSO test subject {test_subject:03d}",
        )
        rows.append({
            "test_subject": test_subject,
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
        })
        print(f"LOSO {test_subject:03d}: acc={metrics['accuracy']:.4f}, kappa={metrics['kappa']:.4f}")

    df = pd.DataFrame(rows)
    summary = pd.concat([
        df,
        pd.DataFrame([
            {"test_subject": "mean", "accuracy": df["accuracy"].mean(), "kappa": df["kappa"].mean()},
            {"test_subject": "std", "accuracy": df["accuracy"].std(), "kappa": df["kappa"].std()},
        ]),
    ], ignore_index=True)
    suffix = "_quick" if args.quick else ""
    path = out_dir / f"loso_metrics{suffix}.csv"
    summary.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
