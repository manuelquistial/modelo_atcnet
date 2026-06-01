"""BCI Competition IV Dataset 2a — load from .mat (Altaheri EEG-ATCNet / MOABB BNCI2014_001)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io as sio

from src.config import EPOCH_TMAX_SEC, EPOCH_TMIN_SEC
from src.utils import get_raw_data_dir

CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]

N_SUBJECTS = 9
N_EEG_CHANNELS = 22
N_SAMPLES = 1125
N_TRIALS_PER_SESSION = 288
FS = 250
WINDOW_LENGTH = 7 * FS  # 1750 samples in continuous EEG per trial window (official)
T1_SAMPLE = int(1.5 * FS)  # 375 — start of 4.5 s MI segment
T2_SAMPLE = int(6.0 * FS)  # 1500 — end (exclusive slice t1:t2 → 1125 samples)

EPOCH_TMIN_DEFAULT = EPOCH_TMIN_SEC
EPOCH_TMAX_DEFAULT = EPOCH_TMAX_SEC
EPOCH_DURATION_SEC = EPOCH_TMAX_DEFAULT - EPOCH_TMIN_DEFAULT

SessionType = Literal["T", "E"]


def _subject_file(subject: int, session: SessionType, data_dir: Path) -> Path | None:
    """
    Resolve MAT path: flat A01T.mat or official layout s1/A01T.mat.
    """
    flat = data_dir / f"A{subject:02d}{session}.mat"
    if flat.exists():
        return flat
    nested = data_dir / f"s{subject}" / f"A{subject:02d}{session}.mat"
    if nested.exists():
        return nested
    return None


def validate_dataset_files(
    data_dir: Path | None = None,
    subjects: range | list[int] | None = None,
) -> list[Path]:
    """Check that all expected MAT files exist."""
    if data_dir is None:
        data_dir = get_raw_data_dir()
    if subjects is None:
        subjects = range(1, N_SUBJECTS + 1)

    missing = []
    found = []
    for s in subjects:
        for sess in ("T", "E"):
            p = _subject_file(s, sess, data_dir)  # type: ignore[arg-type]
            if p is None:
                missing.append(f"A{s:02d}{sess}.mat")
            else:
                found.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing BCI IV 2a MAT files in "
            f"{data_dir.resolve()}.\n"
            "Expected A01T.mat, A01E.mat, ... A09E.mat (flat or under s1/ ... s9/).\n"
            "Download with: python scripts/download_bci2a.py\n"
            f"Missing ({len(missing)}): {', '.join(sorted(missing)[:12])}"
            + (" ..." if len(missing) > 12 else "")
            + f"\nFound ({len(found)}): {', '.join(p.name for p in found[:6])}"
            + (" ..." if len(found) > 6 else "")
        )
    return found


def load_bci2a_mat(
    data_path: str | Path,
    subject: int,
    training: bool,
    all_trials: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load BCI IV-2a from .mat exactly as EEG-ATCNet preprocess.load_BCI2a_data.

    Parameters
    ----------
    data_path : directory containing AxxT.mat / AxxE.mat (or sN/ subfolders)
    subject : 1..9
    training : True → session T, False → session E
    all_trials : if False, skip trials marked with artifacts

    Returns
    -------
    X : (n_trials, 22, 1125)
    y : (n_trials,) in {0,1,2,3}
    """
    data_path = Path(data_path)
    session: SessionType = "T" if training else "E"
    mat_path = _subject_file(subject, session, data_path)
    if mat_path is None:
        raise FileNotFoundError(
            f"MAT not found for subject {subject} session {session} under {data_path.resolve()}"
        )

    n_tests = 6 * 48
    data_return = np.zeros((n_tests, N_EEG_CHANNELS, WINDOW_LENGTH), dtype=np.float64)
    class_return = np.zeros(n_tests, dtype=np.int64)

    # Same as EEG-ATCNet preprocess.load_BCI2a_data (default loadmat, no struct_as_record=False).
    mat = sio.loadmat(str(mat_path))
    if "data" not in mat:
        raise KeyError(f"'data' field not found in {mat_path.name}. Keys: {list(mat.keys())}")

    a_data = mat["data"]
    no_valid = 0

    for ii in range(a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = np.asarray(a_data3[0])
        a_trial = np.asarray(a_data3[1]).flatten()
        a_y = np.asarray(a_data3[2]).flatten()
        a_artifacts = np.asarray(a_data3[5]).flatten()

        for trial in range(a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            start = int(a_trial[trial])
            segment = a_X[start : start + WINDOW_LENGTH, :N_EEG_CHANNELS]
            if segment.shape[0] != WINDOW_LENGTH:
                warnings.warn(
                    f"{mat_path.name} run {ii} trial {trial}: "
                    f"expected {WINDOW_LENGTH} samples, got {segment.shape[0]}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            data_return[no_valid] = np.transpose(segment)
            class_return[no_valid] = int(a_y[trial])
            no_valid += 1

    data_return = data_return[:no_valid, :, T1_SAMPLE:T2_SAMPLE]
    class_return = (class_return[:no_valid] - 1).astype(int)

    if data_return.shape[2] != N_SAMPLES:
        raise ValueError(
            f"{mat_path.name}: expected {N_SAMPLES} samples after [{T1_SAMPLE}:{T2_SAMPLE}], "
            f"got {data_return.shape[2]}"
        )

    n_trials = data_return.shape[0]
    if n_trials != N_TRIALS_PER_SESSION:
        warnings.warn(
            f"{mat_path.name}: expected {N_TRIALS_PER_SESSION} trials, got {n_trials}.",
            UserWarning,
            stacklevel=2,
        )

    return data_return.astype(np.float32), class_return.astype(int)


def load_bci2a_subject(
    subject: int,
    session: SessionType,
    data_dir: Path | str | None = None,
    all_trials: bool = True,
    **_,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject/session (wrapper over load_bci2a_mat)."""
    if data_dir is None:
        data_dir = get_raw_data_dir()
    else:
        data_dir = Path(data_dir)
    training = session == "T"
    return load_bci2a_mat(data_dir, subject, training=training, all_trials=all_trials)


def load_subject_dependent_data(
    subject: int,
    data_dir: Path | str | None = None,
    all_trials: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train on T, test on E (official competition split)."""
    if data_dir is None:
        data_dir = get_raw_data_dir()
    X_train, y_train = load_bci2a_mat(data_dir, subject, training=True, all_trials=all_trials)
    X_test, y_test = load_bci2a_mat(data_dir, subject, training=False, all_trials=all_trials)
    return X_train, y_train, X_test, y_test


def load_all_subjects(
    data_dir: Path | str | None = None,
    sessions: tuple[SessionType, ...] = ("T", "E"),
    all_trials: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all subjects/sessions (LOSO pool)."""
    validate_dataset_files(data_dir=data_dir)
    if data_dir is None:
        data_dir = get_raw_data_dir()

    X_list, y_list, subj_list, sess_list = [], [], [], []
    for s in range(1, N_SUBJECTS + 1):
        for sess in sessions:
            X, y = load_bci2a_subject(s, sess, data_dir, all_trials=all_trials)
            X_list.append(X)
            y_list.append(y)
            subj_list.append(np.full(len(y), s, dtype=int))
            sess_list.append(np.full(len(y), sess, dtype=object))

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(subj_list, axis=0),
        np.concatenate(sess_list, axis=0),
    )


def validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be (trials, channels, samples), got {X.shape}")
    if X.shape[1] != N_EEG_CHANNELS:
        raise ValueError(f"Expected {N_EEG_CHANNELS} channels, got {X.shape[1]}")
    if X.shape[2] != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples, got {X.shape[2]}")
    if len(y) != X.shape[0]:
        raise ValueError(f"len(y)={len(y)} != n_trials={X.shape[0]}")
