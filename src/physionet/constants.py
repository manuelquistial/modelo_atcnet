"""PhysioNet MI constants (aligned with modelo_deep_eeg)."""

from __future__ import annotations

BINARY_EVENTS = ["left_hand", "right_hand"]
LABEL_NAME_TO_ID = {"left_hand": 1, "right_hand": 2}
LABEL_TO_CLASS = {"left_hand": 0, "right_hand": 1}
CLASS_NAMES = ["left_hand", "right_hand"]

SFREQ = 160
N_EEG_CHANNELS = 64
N_CLASSES = 2

# Subject 88 was recorded at 128 Hz (incompatible with the 160 Hz cohort).
EXCLUDED_SUBJECT_IDS = {88}


def event_label_to_class(label: str | int) -> int:
    if isinstance(label, str):
        return LABEL_TO_CLASS[label]
    if int(label) == LABEL_NAME_TO_ID["left_hand"]:
        return 0
    if int(label) == LABEL_NAME_TO_ID["right_hand"]:
        return 1
    raise ValueError(f"Unknown label: {label}")
