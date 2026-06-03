"""MOABB PhysioNet MI loading (from modelo_deep_eeg)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from src.physionet.constants import BINARY_EVENTS, EXCLUDED_SUBJECT_IDS, LABEL_NAME_TO_ID

logger = logging.getLogger(__name__)

SubjectRecord = dict[str, Any]


def setup_mne_paths(data_dir: str | Path) -> Path:
    resolved = Path(data_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(resolved)
    os.environ["MOABB_DATA"] = str(resolved)
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(resolved)
    mne.set_config("MNE_DATA", str(resolved), set_env=True)
    mne.set_config("MOABB_DATA", str(resolved), set_env=True)
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(resolved), set_env=True)
    return resolved


def get_motor_imagery_paradigm() -> MotorImagery:
    return MotorImagery(events=list(BINARY_EVENTS), n_classes=2)


def get_channel_names(dataset: PhysionetMI, subject: int = 1) -> list[str]:
    data = dataset.get_data(subjects=[subject])
    subject_key = next(iter(data.keys()))
    session_key = next(iter(data[subject_key].keys()))
    run_key = next(iter(data[subject_key][session_key].keys()))
    raw = data[subject_key][session_key][run_key]
    return [ch for ch in raw.info["ch_names"] if ch.upper() != "STIM"]


def labels_to_workshop_int(labels_arr: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels_arr).astype(str)
    return np.array([LABEL_NAME_TO_ID[str(lbl)] for lbl in labels_arr], dtype=int)


def build_subject_data(
    dataset: PhysionetMI,
    paradigm: MotorImagery,
    subject_ids: list[int],
    ch_names: list[str],
) -> dict[int, SubjectRecord]:
    subj_data: dict[int, SubjectRecord] = {}

    for sid in subject_ids:
        logger.info("Loading subject %s...", sid)
        try:
            X_s, y_s, metadata_s = paradigm.get_data(dataset=dataset, subjects=[sid])
        except Exception as exc:
            logger.warning("Skipping subject %s: %s", sid, exc)
            continue

        if X_s.ndim != 3 or X_s.shape[0] == 0:
            logger.warning("Subject %s: no valid epochs.", sid)
            continue

        y_s = np.asarray(y_s).astype(str)
        valid_mask = np.isin(y_s, BINARY_EVENTS)
        X_s = X_s[valid_mask]
        y_s = y_s[valid_mask]

        if X_s.shape[0] == 0:
            logger.warning("Subject %s: no left/right epochs.", sid)
            continue

        n_epochs, n_ch, n_times = X_s.shape
        if n_ch != len(ch_names):
            raise ValueError(f"Subject {sid}: expected {len(ch_names)} channels, got {n_ch}.")

        trials = [X_s[i].T.astype(np.float32) for i in range(n_epochs)]
        labels = labels_to_workshop_int(y_s)

        subj_data[sid] = {
            "trials": trials,
            "labels": labels,
            "ch_names": list(ch_names),
            "n_epochs": n_epochs,
            "n_channels": n_ch,
            "n_times": n_times,
        }

    return subj_data


def resolve_subject_ids(
    dataset: PhysionetMI,
    subject_ids: list[int] | None = None,
) -> list[int]:
    if subject_ids is None:
        subject_ids = [int(s) for s in dataset.subject_list]
    else:
        subject_ids = [int(s) for s in subject_ids]
    subject_ids = [s for s in subject_ids if s not in EXCLUDED_SUBJECT_IDS]
    if not subject_ids:
        raise ValueError("No subjects left after excluding incompatible IDs.")
    return subject_ids


def download_subjects(
    dataset: PhysionetMI,
    subject_ids: list[int],
    data_dir: Path,
    *,
    max_retries: int = 8,
    base_delay_s: float = 5.0,
) -> None:
    pending = list(subject_ids)
    for attempt in range(1, max_retries + 1):
        failed: list[int] = []
        for sid in pending:
            try:
                dataset.download(
                    subject_list=[sid],
                    path=str(data_dir),
                    accept=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("Download failed for subject %s (attempt %d): %s", sid, attempt, exc)
                failed.append(sid)
        if not failed:
            return
        pending = failed
        time.sleep(base_delay_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"Could not download subjects: {pending}")


def load_physionet_cohort(
    data_dir: Path,
    subject_ids: list[int] | None = None,
    download: bool = True,
) -> tuple[dict[int, SubjectRecord], list[str]]:
    setup_mne_paths(data_dir)
    dataset = PhysionetMI()
    subject_ids = resolve_subject_ids(dataset, subject_ids)

    if download:
        logger.info("Downloading PhysioNet MI if missing (%d subjects)...", len(subject_ids))
        download_subjects(dataset, subject_ids, data_dir)

    paradigm = get_motor_imagery_paradigm()
    ch_names = get_channel_names(dataset, subject=subject_ids[0])
    subj_data = build_subject_data(dataset, paradigm, subject_ids, ch_names)
    if not subj_data:
        raise RuntimeError("No subject data loaded. Check download path and network.")
    logger.info("Loaded %d subjects.", len(subj_data))
    return subj_data, ch_names
