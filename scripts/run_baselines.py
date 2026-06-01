#!/usr/bin/env python3
"""EEGNet baseline — same data/training protocol as EEG-ATCNet."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import N_SUBJECTS, load_subject_dependent_data, validate_dataset_files
from src.eegnet import build_eegnet
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.plots import plot_confusion_matrix
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def main() -> None:
    out_dir = ensure_dir(get_results_dir() / "baselines")
    figures_dir = ensure_dir(get_results_dir() / "figures")
    validate_dataset_files()
    rows = []

    for subject in tqdm(range(1, N_SUBJECTS + 1), desc="EEGNet"):
        X_train, y_train, X_test, y_test = load_subject_dependent_data(subject)
        X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
            X_train, y_train, X_test, y_test
        )
        subj_test = np.full(len(y_test_int), subject, dtype=int)
        sess_test = np.full(len(y_test_int), "E", dtype=object)

        model, metrics = train_best_of_runs(
            build_eegnet,
            X_tr,
            y_tr,
            X_te,
            y_te,
            lambda m: evaluate_model(m, X_te, y_test_int, subj_test, sess_test)[0],
            out_dir,
            f"eegnet_subject_{subject:02d}",
        )
        _, pred_df, _, _ = evaluate_model(model, X_te, y_test_int, subj_test, sess_test)
        pred_df.to_csv(out_dir / f"eegnet_subject_{subject:02d}_predictions.csv", index=False)
        export_errors(pred_df, out_dir / f"eegnet_subject_{subject:02d}_errors.csv", "eegnet")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            figures_dir / f"eegnet_subject_{subject:02d}_cm.png",
            title=f"EEGNet subject {subject:02d}",
        )
        rows.append({"subject": subject, "accuracy": metrics["accuracy"], "kappa": metrics["kappa"]})

    df = pd.DataFrame(rows)
    summary = pd.concat([
        df,
        pd.DataFrame([
            {"subject": "mean", "accuracy": df["accuracy"].mean(), "kappa": df["kappa"].mean()},
            {"subject": "std", "accuracy": df["accuracy"].std(), "kappa": df["kappa"].std()},
        ]),
    ], ignore_index=True)
    path = out_dir / "eegnet_subject_dependent_metrics.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
