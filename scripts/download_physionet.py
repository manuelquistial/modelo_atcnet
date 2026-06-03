#!/usr/bin/env python3
"""Download PhysioNet MI (MOABB) for binary left/right hand classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_physionet_cohort, validate_shapes
from src.physionet.constants import event_label_to_class
from src.physionet.preprocess import preprocess_cohort, subject_dict_to_arrays
from src.physionet.splits import subset_by_subjects
from src.utils import get_mne_data_dir


def parse_subjects(value: str | None) -> list[int] | None:
    if value is None:
        return None
    subjects: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            subjects.update(range(int(a), int(b) + 1))
        else:
            subjects.add(int(part))
    return sorted(subjects)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet MI via MOABB.")
    parser.add_argument("--subjects", type=str, default=None, help="e.g. '1-5' or '1,3,7'")
    parser.add_argument("--yes", action="store_true", help="Accept MOABB license (no prompt)")
    args = parser.parse_args()

    dest = get_mne_data_dir()
    print(f"MNE/MOABB data dir: {dest.resolve()}")

    subject_ids = parse_subjects(args.subjects)
    subj_data, ch_names = load_physionet_cohort(dest, subject_ids=subject_ids, download=True)
    processed, meta = preprocess_cohort(subj_data)
    X, y, groups = subject_dict_to_arrays(processed)
    validate_shapes(X, y, n_channels=len(ch_names))

    print(f"OK: {meta['n_subjects']} subjects, {len(y)} trials")
    print(f"  X shape: {X.shape}  labels: {set(y.tolist())}  channels: {len(ch_names)}")
    print(f"  classes: left_hand=0, right_hand=1")


if __name__ == "__main__":
    main()
