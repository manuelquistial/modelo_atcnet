"""Preprocessing for PhysioNet MI (from modelo_deep_eeg workshop pipeline)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.physionet.constants import SFREQ
from src.physionet.loader import SubjectRecord


def crop_or_pad_trial_time(trial: np.ndarray, target_n_times: int) -> np.ndarray:
    """Center-crop trial (n_times, n_channels) to target length."""
    n_times, _ = trial.shape
    if n_times == target_n_times:
        return trial.astype(np.float32, copy=False)
    if n_times > target_n_times:
        start = (n_times - target_n_times) // 2
        return trial[start : start + target_n_times, :].astype(np.float32, copy=False)
    raise ValueError(f"Trial n_times={n_times} < target={target_n_times}.")


def harmonize_time(
    data_dict: dict[int, SubjectRecord],
    target_n_times: int | None = None,
) -> tuple[dict[int, SubjectRecord], int]:
    all_trials = [t for s in data_dict.values() for t in s["trials"]]
    if target_n_times is None:
        target_n_times = int(min(t.shape[0] for t in all_trials))

    out: dict[int, SubjectRecord] = {}
    for sid, sdata in data_dict.items():
        fixed = [crop_or_pad_trial_time(t, target_n_times) for t in sdata["trials"]]
        out[sid] = {**sdata, "trials": fixed, "n_times": target_n_times}
    return out, target_n_times


def _infer_uv_scale(data_dict: dict[int, SubjectRecord], max_trials: int = 500) -> bool:
    values = []
    for sdata in data_dict.values():
        for trial in sdata["trials"]:
            values.append(float(np.max(np.abs(trial))))
            if len(values) >= max_trials:
                break
        if len(values) >= max_trials:
            break
    p90 = float(np.percentile(values, 90))
    return p90 < 1.0


def remove_outliers(
    data_dict: dict[int, SubjectRecord],
    threshold_uv: float = 800.0,
) -> dict[int, SubjectRecord]:
    multiply = _infer_uv_scale(data_dict)
    cleaned: dict[int, SubjectRecord] = {}
    for sid, sdata in data_dict.items():
        keep_trials, keep_labels = [], []
        for i, trial in enumerate(sdata["trials"]):
            max_uv = np.max(np.abs(trial)) * (1e6 if multiply else 1.0)
            if max_uv <= threshold_uv:
                keep_trials.append(trial)
                keep_labels.append(sdata["labels"][i])
        cleaned[sid] = {
            **sdata,
            "trials": keep_trials,
            "labels": np.asarray(keep_labels, dtype=int),
            "n_epochs": len(keep_trials),
        }
    return cleaned


def apply_highpass(
    data_dict: dict[int, SubjectRecord],
    highpass_hz: float = 4.0,
    order: int = 4,
    sfreq: float = SFREQ,
) -> dict[int, SubjectRecord]:
    sos = butter(order, highpass_hz, btype="highpass", fs=sfreq, output="sos")
    filtered: dict[int, SubjectRecord] = {}
    for sid, sdata in data_dict.items():
        trials = [sosfiltfilt(sos, t, axis=0).astype(np.float32) for t in sdata["trials"]]
        filtered[sid] = {**sdata, "trials": trials}
    return filtered


def preprocess_cohort(
    data_dict: dict[int, SubjectRecord],
    *,
    highpass_hz: float = 4.0,
    outlier_uv: float = 800.0,
) -> tuple[dict[int, SubjectRecord], dict[str, Any]]:
    cleaned = remove_outliers(data_dict, threshold_uv=outlier_uv)
    filtered = apply_highpass(cleaned, highpass_hz=highpass_hz)
    harmonized, n_times = harmonize_time(filtered)
    meta = {"n_times": n_times, "n_subjects": len(harmonized)}
    return harmonized, meta


def subject_dict_to_arrays(
    data_dict: dict[int, SubjectRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, C, T), y in {0,1}, subject_ids."""
    from src.physionet.constants import event_label_to_class

    X_list, y_list, groups = [], [], []
    for sid, sdata in data_dict.items():
        for i, trial in enumerate(sdata["trials"]):
            X_list.append(trial.T.astype(np.float32))
            y_list.append(event_label_to_class(int(sdata["labels"][i])))
            groups.append(sid)
    return (
        np.stack(X_list, axis=0),
        np.asarray(y_list, dtype=int),
        np.asarray(groups, dtype=int),
    )
