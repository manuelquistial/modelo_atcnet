#!/usr/bin/env python3
"""EEGNet baseline on PhysioNet MI hold-out (left vs right)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd

from src.config import QUICK_PROFILE, PAPER_PROFILE
from src.data_loader import load_holdout_data
from src.eegnet import build_eegnet
from src.error_analysis import export_errors
from src.evaluate import evaluate_model
from src.physionet.dataset import get_model_dims
from src.plots import plot_confusion_matrix
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    profile = QUICK_PROFILE if args.quick else PAPER_PROFILE

    out_dir = ensure_dir(get_results_dir() / "baselines")
    figures_dir = ensure_dir(get_results_dir() / "figures")

    X_train, y_train, X_test, y_test, _, g_test, meta = load_holdout_data()
    n_ch, n_samples, n_classes = get_model_dims(meta)

    X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
        X_train, y_train, X_test, y_test, n_classes=n_classes
    )

    build_fn = lambda: build_eegnet(n_channels=n_ch, n_samples=n_samples, n_classes=n_classes)
    model, metrics = train_best_of_runs(
        build_fn,
        X_tr,
        y_tr,
        X_te,
        y_te,
        lambda m: evaluate_model(m, X_te, y_test_int, g_test)[0],
        out_dir,
        "eegnet_holdout",
        profile=profile,
    )

    _, pred_df, _, _ = evaluate_model(model, X_te, y_test_int, g_test)
    pred_df.to_csv(out_dir / "eegnet_holdout_predictions.csv", index=False)
    export_errors(pred_df, out_dir / "eegnet_holdout_errors.csv", "holdout")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        figures_dir / "eegnet_holdout_cm.png",
        title="EEGNet hold-out",
    )

    row = {"model": "eegnet", "accuracy": metrics["accuracy"], "kappa": metrics["kappa"]}
    path = out_dir / "eegnet_holdout_metrics.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"EEGNet hold-out: acc={metrics['accuracy']:.4f}, kappa={metrics['kappa']:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
