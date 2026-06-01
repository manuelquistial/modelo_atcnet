"""BCI Competition IV Dataset 2a loading and epoch extraction via MNE."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import mne
import numpy as np

from src.config import EPOCH_TMAX_SEC, EPOCH_TMIN_SEC
from src.utils import get_raw_data_dir

# Event codes in GDF files (BCI IV 2a)
BCI2A_EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "both_feet": 771,
    "tongue": 772,
}

LABEL_MAP = {
    769: 0,
    770: 1,
    771: 2,
    772: 3,
}

CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]

N_SUBJECTS = 9
N_EEG_CHANNELS = 22
N_SAMPLES = 1125
N_TRIALS_PER_SESSION = 288
FS = 250
# Official repo: samples [1.5 s, 6.0 s] after cue → 4.5 s = 1125 samples @ 250 Hz
EPOCH_TMIN_DEFAULT = EPOCH_TMIN_SEC
EPOCH_TMAX_DEFAULT = EPOCH_TMAX_SEC
EPOCH_DURATION_SEC = EPOCH_TMAX_DEFAULT - EPOCH_TMIN_DEFAULT

SessionType = Literal["T", "E"]


def _subject_file(subject: int, session: SessionType, data_dir: Path | None = None) -> Path:
    if data_dir is None:
        data_dir = get_raw_data_dir()
    return data_dir / f"A{subject:02d}{session}.gdf"


def validate_dataset_files(
    data_dir: Path | None = None,
    subjects: range | list[int] | None = None,
) -> list[Path]:
    """Check that all expected GDF files exist; raise FileNotFoundError with clear message."""
    if data_dir is None:
        data_dir = get_raw_data_dir()
    if subjects is None:
        subjects = range(1, N_SUBJECTS + 1)

    missing = []
    found = []
    for s in subjects:
        for sess in ("T", "E"):
            p = _subject_file(s, sess, data_dir)  # type: ignore[arg-type]
            if not p.exists():
                missing.append(p.name)
            else:
                found.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing BCI IV 2a GDF files in "
            f"{data_dir.resolve()}.\n"
            f"Expected files like A01T.gdf, A01E.gdf, ... A09E.gdf.\n"
            f"Missing ({len(missing)}): {', '.join(sorted(missing)[:12])}"
            + (" ..." if len(missing) > 12 else "")
            + f"\nFound ({len(found)}): {', '.join(p.name for p in found[:6])}"
            + (" ..." if len(found) > 6 else "")
        )
    return found


def _pick_eeg_channels(raw: mne.io.BaseRaw, n_eeg: int = N_EEG_CHANNELS) -> list[str]:
    """Select first n_eeg EEG channels; exclude EOG if present in naming."""
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = [raw.ch_names[i] for i in picks]
    # Exclude obvious EOG channels by name
    eeg_names = [
        c
        for c in ch_names
        if "EOG" not in c.upper() and "EOG" not in c
    ]
    if len(eeg_names) < n_eeg:
        # Fall back to first n picks
        eeg_names = ch_names[:n_eeg]
    selected = eeg_names[:n_eeg]
    if len(selected) != n_eeg:
        warnings.warn(
            f"Expected {n_eeg} EEG channels, got {len(selected)}. "
            f"Channel names: {raw.ch_names}",
            UserWarning,
            stacklevel=2,
        )
    return selected


def _resolve_mi_events(raw: mne.io.BaseRaw) -> dict[str, int]:
    """
    Map class names to event ids present in annotations.
    Handles MNE remapping of GDF codes to internal ids.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if not event_id:
        raise RuntimeError("No events found in annotations.")

    # Build reverse lookup: id -> name
    id_to_name = {v: k for k, v in event_id.items()}

    resolved: dict[str, int] = {}
    for class_name, code in BCI2A_EVENT_ID.items():
        # Direct name match (e.g. '769' or 'left_hand' depending on file)
        candidates = [
            str(code),
            f"EEG-{code}",
            class_name,
            f"Event_{code}",
        ]
        found_id = None
        for name, eid in event_id.items():
            if name in candidates or str(code) in name:
                found_id = eid
                break
            # Some GDF files store numeric codes in annotation description
            try:
                if int(name) == code:
                    found_id = eid
                    break
            except (ValueError, TypeError):
                pass

        if found_id is None:
            # Search events array for raw code (before annotation mapping)
            unique = np.unique(events[:, 2]) if len(events) else []
            if code in unique:
                found_id = int(code)
            else:
                # Try matching by event id values that equal code
                for eid in event_id.values():
                    if eid == code:
                        found_id = eid
                        break

        if found_id is not None:
            resolved[class_name] = int(found_id)

    if len(resolved) < 4:
        available = {
            "event_id_map": event_id,
            "unique_event_codes": np.unique(events[:, 2]).tolist() if len(events) else [],
            "id_to_name": id_to_name,
        }
        raise RuntimeError(
            "Could not find all four MI classes (769, 770, 771, 772) in file "
            f"{raw.filenames[0] if raw.filenames else 'unknown'}.\n"
            f"Resolved: {resolved}\n"
            f"Available for debugging: {available}\n"
            "Check GDF annotations or update event resolution in data_loader.py."
        )

    return resolved


def _align_epoch_length(epochs_data: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Ensure exactly n_samples along last axis; crop if longer, error if shorter."""
    current = epochs_data.shape[-1]
    if current == n_samples:
        return epochs_data
    if current > n_samples:
        return epochs_data[..., :n_samples]
    raise ValueError(
        f"Epoch has {current} samples but {n_samples} required ({EPOCH_DURATION_SEC}s @ {FS} Hz). "
        "Increase tmax or check cue timing / resampling. "
        f"Got shape {epochs_data.shape}."
    )


def load_bci2a_subject(
    subject: int,
    session: SessionType,
    data_dir: Path | str | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    verbose: bool | str = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one subject/session GDF and return epochs.

    Returns
    -------
    X : (n_trials, 22, 1125)
    y : (n_trials,) labels in {0,1,2,3}
    """
    if tmin is None:
        tmin = EPOCH_TMIN_DEFAULT
    if tmax is None:
        tmax = EPOCH_TMAX_DEFAULT

    if data_dir is None:
        data_dir = get_raw_data_dir()
    else:
        data_dir = Path(data_dir)

    path = _subject_file(subject, session, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Place BCI IV 2a files in {data_dir.resolve()} "
            "(e.g. A01T.gdf, A01E.gdf, ...)."
        )

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=verbose)
    raw.rename_channels(
        {c: c.strip().replace(" ", "") for c in raw.ch_names},
        verbose=False,
    )

    eeg_chs = _pick_eeg_channels(raw, N_EEG_CHANNELS)
    if len(eeg_chs) != N_EEG_CHANNELS:
        warnings.warn(
            f"{path.name}: using {len(eeg_chs)} channels (expected {N_EEG_CHANNELS}).",
            UserWarning,
            stacklevel=2,
        )

    mi_event_id = _resolve_mi_events(raw)
    events, _ = mne.events_from_annotations(raw, verbose=False)

    # Remap event codes to 0..3 for epoching
    epoch_event_id = {name: mi_event_id[name] for name in BCI2A_EVENT_ID}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=epoch_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=eeg_chs,
        preload=True,
        verbose=verbose,
        reject_by_annotation=False,
    )

    X = epochs.get_data(copy=True)  # (n_trials, n_ch, n_times)
    y = epochs.events[:, 2].copy()

    # Map internal event ids back to class indices 0..3
    inv_map = {mi_event_id[k]: LABEL_MAP[BCI2A_EVENT_ID[k]] for k in BCI2A_EVENT_ID}
    y_mapped = np.array([inv_map.get(int(v), -1) for v in y], dtype=int)
    if (y_mapped < 0).any():
        raise RuntimeError(f"Unmapped labels in {path.name}: {np.unique(y)}")

    X = _align_epoch_length(X, N_SAMPLES)

    if X.shape[1] != N_EEG_CHANNELS:
        warnings.warn(
            f"{path.name}: X has {X.shape[1]} channels (expected {N_EEG_CHANNELS}).",
            UserWarning,
            stacklevel=2,
        )
    if X.shape[2] != N_SAMPLES:
        raise ValueError(f"{path.name}: expected {N_SAMPLES} samples, got {X.shape[2]}")

    n_trials = X.shape[0]
    if n_trials != N_TRIALS_PER_SESSION:
        warnings.warn(
            f"{path.name}: expected {N_TRIALS_PER_SESSION} trials, got {n_trials}.",
            UserWarning,
            stacklevel=2,
        )

    return X.astype(np.float32), y_mapped.astype(int)


def load_subject_dependent_data(
    subject: int,
    data_dir: Path | str | None = None,
    verbose: bool | str = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load T for train and E for test (subject-dependent protocol)."""
    X_train, y_train = load_bci2a_subject(subject, "T", data_dir, verbose=verbose)
    X_test, y_test = load_bci2a_subject(subject, "E", data_dir, verbose=verbose)
    return X_train, y_train, X_test, y_test


def load_all_subjects(
    data_dir: Path | str | None = None,
    sessions: tuple[SessionType, ...] = ("T", "E"),
    verbose: bool | str = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all subjects and sessions.

    Returns
    -------
    X : (n_total_trials, 22, 1125)
    y : (n_total_trials,)
    subject_ids : (n_total_trials,)
    session_ids : (n_total_trials,) 'T' or 'E'
    """
    validate_dataset_files(data_dir=data_dir)

    X_list, y_list, subj_list, sess_list = [], [], [], []
    for s in range(1, N_SUBJECTS + 1):
        for sess in sessions:
            X, y = load_bci2a_subject(s, sess, data_dir, verbose=verbose)
            X_list.append(X)
            y_list.append(y)
            subj_list.append(np.full(len(y), s, dtype=int))
            sess_list.append(np.full(len(y), sess, dtype=object))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subj_list, axis=0)
    session_ids = np.concatenate(sess_list, axis=0)
    return X, y, subject_ids, session_ids


def validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
    """Assert expected trial, channel, sample dimensions."""
    if X.ndim != 3:
        raise ValueError(f"X must be (trials, channels, samples), got {X.shape}")
    if X.shape[1] != N_EEG_CHANNELS:
        raise ValueError(f"Expected {N_EEG_CHANNELS} channels, got {X.shape[1]}")
    if X.shape[2] != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples, got {X.shape[2]}")
    if len(y) != X.shape[0]:
        raise ValueError(f"len(y)={len(y)} != n_trials={X.shape[0]}")
