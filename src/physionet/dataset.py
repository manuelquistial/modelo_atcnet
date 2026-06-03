"""High-level PhysioNet MI dataset API for ATCNet training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    HOLDOUT_RANDOM_STATE,
    HOLDOUT_TEST_SIZE,
    HIGHPASS_HZ,
    OUTLIER_UV,
)
from src.physionet.loader import SubjectRecord, load_physionet_cohort
from src.physionet.preprocess import preprocess_cohort, subject_dict_to_arrays
from src.physionet.splits import split_subjects_holdout, subset_by_subjects
from src.utils import get_mne_data_dir, save_json


def _merge_meta(*parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in parts:
        out.update(p)
    return out


def load_and_preprocess_cohort(
    data_dir: Path | None = None,
    subject_ids: list[int] | None = None,
    download: bool = True,
) -> tuple[dict[int, SubjectRecord], dict[str, Any]]:
    if data_dir is None:
        data_dir = get_mne_data_dir()
    subj_data, ch_names = load_physionet_cohort(data_dir, subject_ids, download=download)
    processed, meta = preprocess_cohort(
        subj_data,
        highpass_hz=HIGHPASS_HZ,
        outlier_uv=OUTLIER_UV,
    )
    meta = _merge_meta(
        meta,
        {
            "dataset": "physionet_mi",
            "n_channels": len(ch_names),
            "ch_names": ch_names,
            "n_subjects": len(processed),
            "classes": ["left_hand", "right_hand"],
        },
    )
    return processed, meta


def load_holdout_data(
    data_dir: Path | None = None,
    subject_ids: list[int] | None = None,
    test_size: float = HOLDOUT_TEST_SIZE,
    random_state: int = HOLDOUT_RANDOM_STATE,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Subject-level hold-out split (modelo_deep_eeg protocol).

    Returns X_train, y_train, X_test, y_test, groups_train, groups_test, meta.
    """
    processed, meta = load_and_preprocess_cohort(data_dir, subject_ids, download=download)
    dev_ids, test_ids = split_subjects_holdout(processed, test_size, random_state)

    dev_dict = subset_by_subjects(processed, dev_ids)
    test_dict = subset_by_subjects(processed, test_ids)

    X_train, y_train, groups_train = subject_dict_to_arrays(dev_dict)
    X_test, y_test, groups_test = subject_dict_to_arrays(test_dict)

    meta = _merge_meta(
        meta,
        {
            "split": "holdout",
            "test_size": test_size,
            "random_state": random_state,
            "dev_subject_ids": dev_ids.tolist(),
            "test_subject_ids": test_ids.tolist(),
            "n_train_trials": int(len(y_train)),
            "n_test_trials": int(len(y_test)),
        },
    )
    cache_meta_path = get_mne_data_dir().parent / "processed" / "physionet_holdout_meta.json"
    save_json(meta, cache_meta_path)
    return X_train, y_train, X_test, y_test, groups_train, groups_test, meta


def load_loso_fold(
    test_subject: int,
    data_dir: Path | None = None,
    subject_ids: list[int] | None = None,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """LOSO: train on all subjects except test_subject."""
    processed, meta = load_and_preprocess_cohort(data_dir, subject_ids, download=download)
    if test_subject not in processed:
        raise ValueError(f"Subject {test_subject} not in loaded cohort.")

    test_dict = {test_subject: processed[test_subject]}
    train_dict = {sid: processed[sid] for sid in processed if sid != test_subject}

    X_train, y_train, groups_train = subject_dict_to_arrays(train_dict)
    X_test, y_test, groups_test = subject_dict_to_arrays(test_dict)

    meta = _merge_meta(
        meta,
        {
            "split": "loso",
            "test_subject": test_subject,
            "n_train_trials": int(len(y_train)),
            "n_test_trials": int(len(y_test)),
        },
    )
    return X_train, y_train, X_test, y_test, groups_train, groups_test, meta


def loso_from_processed(
    processed: dict[int, SubjectRecord],
    test_subject: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """LOSO split from an already preprocessed cohort."""
    if test_subject not in processed:
        raise ValueError(f"Subject {test_subject} not in cohort.")
    train_dict = subset_by_subjects(processed, np.array([s for s in processed if s != test_subject]))
    test_dict = subset_by_subjects(processed, np.array([test_subject]))
    X_train, y_train, groups_train = subject_dict_to_arrays(train_dict)
    X_test, y_test, groups_test = subject_dict_to_arrays(test_dict)
    fold_meta = _merge_meta(
        meta,
        {"split": "loso", "test_subject": test_subject},
    )
    return X_train, y_train, X_test, y_test, groups_train, groups_test, fold_meta


def get_model_dims(meta: dict[str, Any]) -> tuple[int, int, int]:
    """Return (n_channels, n_samples, n_classes) for ATCNet."""
    return int(meta["n_channels"]), int(meta["n_times"]), 2
