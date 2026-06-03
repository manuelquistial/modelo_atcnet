#!/usr/bin/env python3
"""Ablation study on PhysioNet MI hold-out split."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ablation import ABLATION_VARIANTS, build_ablation_model
from src.config import QUICK_PROFILE, PAPER_PROFILE
from src.data_loader import load_holdout_data
from src.evaluate import evaluate_model
from src.physionet.dataset import get_model_dims
from src.plots import plot_ablation_summary
from src.training_pipeline import prepare_official_splits, train_best_of_runs
from src.utils import ensure_dir, get_results_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    profile = QUICK_PROFILE if args.quick else PAPER_PROFILE

    out_dir = ensure_dir(get_results_dir() / "ablation")
    X_train, y_train, X_test, y_test, _, g_test, meta = load_holdout_data()
    n_ch, n_samples, n_classes = get_model_dims(meta)

    raw_rows = []
    for variant in ABLATION_VARIANTS:
        X_tr, y_tr, X_te, y_te, y_test_int = prepare_official_splits(
            X_train, y_train, X_test, y_test, n_classes=n_classes
        )
        build_fn = lambda v=variant: build_ablation_model(
            v, n_channels=n_ch, n_samples=n_samples, n_classes=n_classes
        )
        _, metrics = train_best_of_runs(
            build_fn,
            X_tr,
            y_tr,
            X_te,
            y_te,
            lambda m, xt=X_te, yt=y_test_int: evaluate_model(m, xt, yt)[0],
            out_dir,
            f"ablation_{variant}",
            profile=profile,
        )
        raw_rows.append({
            "variant": variant,
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
        })
        print(f"{variant}: acc={metrics['accuracy']:.4f}, kappa={metrics['kappa']:.4f}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out_dir / "ablation_raw_results.csv", index=False)
    summary = raw_df.copy()
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    plot_ablation_summary(summary, get_results_dir() / "figures" / "ablation_summary.png")
    print(f"\nSaved {out_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
