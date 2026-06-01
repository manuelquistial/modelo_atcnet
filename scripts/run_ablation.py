#!/usr/bin/env python3
"""Ablation study (Table II) — EEG-ATCNet training protocol."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.ablation import ABLATION_VARIANTS, build_ablation_model
from src.data_loader import N_SUBJECTS, load_subject_dependent_data, validate_dataset_files
from src.evaluate import evaluate_model
from src.plots import plot_ablation_summary
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def main() -> None:
    out_dir = ensure_dir(get_results_dir() / "ablation")
    validate_dataset_files()
    raw_rows = []

    for variant in ABLATION_VARIANTS:
        for subject in tqdm(range(1, N_SUBJECTS + 1), desc=variant):
            X_train, y_train, X_test, y_test = load_subject_dependent_data(subject)
            X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
                X_train, y_train, X_test, y_test
            )

            build_fn = lambda v=variant: build_ablation_model(v)
            _, metrics = train_best_of_runs(
                build_fn,
                X_tr,
                y_tr,
                X_te,
                y_te,
                lambda m, xt=X_te, yt=y_test_int: evaluate_model(m, xt, yt)[0],
                out_dir,
                f"ablation_{variant}_subj_{subject:02d}",
            )
            raw_rows.append({
                "variant": variant,
                "subject": subject,
                "accuracy": metrics["accuracy"],
                "kappa": metrics["kappa"],
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_path = out_dir / "ablation_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = (
        raw_df.groupby("variant")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
        )
        .reset_index()
    )
    summary_path = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_ablation_summary(
        summary_df, ensure_dir(get_results_dir() / "figures") / "ablation_summary.png"
    )
    print(summary_df.to_string(index=False))
    print(f"\nSaved {raw_path} and {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
