#!/usr/bin/env python3
"""Subject-dependent: train session T, test session E (EEG-ATCNet protocol)."""

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
from src.config import PAPER_PROFILE, QUICK_PROFILE, TrainingProfile
from src.data_loader import N_SUBJECTS, load_subject_dependent_data, validate_dataset_files
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.plots import plot_confusion_matrix
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def parse_subjects(value: str | None) -> list[int]:
    if value is None:
        return list(range(1, N_SUBJECTS + 1))
    subjects: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            subjects.update(range(int(start_s), int(end_s) + 1))
        else:
            subjects.add(int(part))
    chosen = sorted(subjects)
    if not chosen or any(s < 1 or s > N_SUBJECTS for s in chosen):
        raise ValueError(f"Subjects must be in 1..{N_SUBJECTS}, got {chosen}")
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-dependent ATCNet (T train → E test).")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast run (~5 h for 9 subjects): 1 run, max 200 epochs, patience 50. "
        "Not comparable to paper numbers.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma list or ranges, e.g. '1,3,5' or '1-3' (default: all 9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile: TrainingProfile = QUICK_PROFILE if args.quick else PAPER_PROFILE
    subjects = parse_subjects(args.subjects)

    out_dir = ensure_dir(get_results_dir() / "subject_dependent")
    figures_dir = ensure_dir(get_results_dir() / "figures")
    validate_dataset_files(subjects=subjects)

    if args.quick:
        print(
            "QUICK mode: "
            f"{profile.n_train_runs} run(s), max {profile.epochs} epochs, "
            f"early-stop patience {profile.early_stop_patience}. "
            "Expect ~5 h for 9 subjects on GPU; accuracy below paper (~85%)."
        )

    rows = []
    for subject in tqdm(subjects, desc="Subjects"):
        X_train, y_train, X_test, y_test = load_subject_dependent_data(subject)
        X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
            X_train, y_train, X_test, y_test
        )

        subj_test = np.full(len(y_test_int), subject, dtype=int)
        sess_test = np.full(len(y_test_int), "E", dtype=object)

        model, metrics = train_best_of_runs(
            build_atcnet,
            X_tr,
            y_tr,
            X_te,
            y_te,
            lambda m: evaluate_model(m, X_te, y_test_int, subj_test, sess_test)[0],
            out_dir,
            f"subject_{subject:02d}",
            profile=profile,
        )

        _, pred_df, _, _ = evaluate_model(model, X_te, y_test_int, subj_test, sess_test)

        pred_df.to_csv(out_dir / f"subject_{subject:02d}_predictions.csv", index=False)
        export_errors(pred_df, out_dir / f"subject_{subject:02d}_errors.csv", "subject_dependent")

        cm = metrics["confusion_matrix"]
        plot_confusion_matrix(
            cm, figures_dir / f"subject_{subject:02d}_cm.png",
            title=f"Subject {subject:02d} (T→E)",
        )
        plot_confusion_matrix(
            cm, figures_dir / f"subject_{subject:02d}_cm_norm.png",
            normalize=True, title=f"Subject {subject:02d} (T→E)",
        )

        rows.append({"subject": subject, "accuracy": metrics["accuracy"], "kappa": metrics["kappa"]})
        print(f"Subject {subject:02d}: acc={metrics['accuracy']:.4f}, kappa={metrics['kappa']:.4f}")

    df = pd.DataFrame(rows)
    summary = pd.concat([
        df,
        pd.DataFrame([
            {"subject": "mean", "accuracy": df["accuracy"].mean(), "kappa": df["kappa"].mean()},
            {"subject": "std", "accuracy": df["accuracy"].std(), "kappa": df["kappa"].std()},
        ]),
    ], ignore_index=True)
    suffix = "_quick" if args.quick else ""
    path = out_dir / f"subject_dependent_metrics{suffix}.csv"
    summary.to_csv(path, index=False)
    print(f"\n=== Subject-dependent ({'quick' if args.quick else 'paper'}) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved {path}")
    if not args.quick:
        print("Reference: ~85.38% acc, ~0.81 κ (Altaheri et al., 2023)")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
