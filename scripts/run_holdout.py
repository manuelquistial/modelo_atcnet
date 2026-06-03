#!/usr/bin/env python3
"""Hold-out evaluation: train on dev subjects, test on held-out subjects (PhysioNet MI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.atcnet import build_atcnet
from src.config import PAPER_PROFILE, QUICK_PROFILE, TrainingProfile
from src.data_loader import load_holdout_data
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.physionet.dataset import get_model_dims
from src.plots import plot_confusion_matrix
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhysioNet MI hold-out (left vs right hand).")
    p.add_argument("--quick", action="store_true", help="Fast training profile (~few hours).")
    p.add_argument("--subjects", type=str, default=None, help="Limit cohort, e.g. '1-10'")
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

    out_dir = ensure_dir(get_results_dir() / "holdout")
    figures_dir = ensure_dir(get_results_dir() / "figures")

    X_train, y_train, X_test, y_test, g_train, g_test, meta = load_holdout_data(
        subject_ids=subject_ids,
    )
    n_ch, n_samples, n_classes = get_model_dims(meta)

    build_fn = lambda: build_atcnet(
        n_channels=n_ch, n_samples=n_samples, n_classes=n_classes
    )

    X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
        X_train, y_train, X_test, y_test, n_classes=n_classes
    )

    model, metrics = train_best_of_runs(
        build_fn,
        X_tr,
        y_tr,
        X_te,
        y_te,
        lambda m: evaluate_model(m, X_te, y_test_int, g_test)[0],
        out_dir,
        "holdout",
        profile=profile,
    )

    _, pred_df, _, _ = evaluate_model(model, X_te, y_test_int, g_test)
    pred_df.to_csv(out_dir / "holdout_predictions.csv", index=False)
    export_errors(pred_df, out_dir / "holdout_errors.csv", "holdout")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures_dir / "holdout_cm.png",
        title="Hold-out (left vs right)",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures_dir / "holdout_cm_norm.png",
        normalize=True,
        title="Hold-out (left vs right)",
    )

    row = {
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "n_train_trials": len(y_train),
        "n_test_trials": len(y_test),
        "n_dev_subjects": len(meta["dev_subject_ids"]),
        "n_test_subjects": len(meta["test_subject_ids"]),
    }
    suffix = "_quick" if args.quick else ""
    metrics_path = out_dir / f"holdout_metrics{suffix}.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    meta_path = out_dir / f"holdout_meta{suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Hold-out PhysioNet MI ({'quick' if args.quick else 'full'}) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}  Kappa: {metrics['kappa']:.4f}")
    print(f"Dev subjects: {len(meta['dev_subject_ids'])}  Test subjects: {len(meta['test_subject_ids'])}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
