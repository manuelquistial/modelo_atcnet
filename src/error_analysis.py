"""Error extraction and summaries for prediction audits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_dir


def extract_errors(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Return only misclassified trials with error_type column."""
    err = prediction_df.loc[~prediction_df["is_correct"]].copy()
    err["error_type"] = (
        err["true_class"] + "*predicted_as*" + err["pred_class"]
    )
    return err


def summarize_errors_by_subject(errors_df: pd.DataFrame) -> pd.DataFrame:
    if errors_df.empty:
        return pd.DataFrame(columns=["subject", "n_errors", "error_rate"])
    g = errors_df.groupby("subject").size().reset_index(name="n_errors")
    return g.sort_values("subject")


def summarize_errors_by_class(errors_df: pd.DataFrame) -> pd.DataFrame:
    if errors_df.empty:
        return pd.DataFrame(columns=["true_class", "n_errors"])
    g = errors_df.groupby("true_class").size().reset_index(name="n_errors")
    return g.sort_values("n_errors", ascending=False)


def export_errors(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    evaluation_name: str = "",
) -> pd.DataFrame:
    """Write errors CSV and return error DataFrame."""
    errors = extract_errors(prediction_df)
    if evaluation_name:
        errors.insert(0, "evaluation", evaluation_name)
    ensure_dir(Path(output_path).parent)
    errors.to_csv(output_path, index=False)
    return errors
