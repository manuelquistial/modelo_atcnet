"""Subject-level splits for PhysioNet MI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.physionet.loader import SubjectRecord


def split_subjects_holdout(
    subj_data: dict[int, SubjectRecord],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_subjects = np.array(sorted(subj_data.keys()))
    major_labels = []
    for sid in all_subjects:
        labels = subj_data[sid]["labels"]
        major_labels.append(pd.Series(labels).mode().iloc[0])
    major_labels = np.asarray(major_labels)
    stratify = major_labels if len(np.unique(major_labels)) > 1 else None
    dev, test = train_test_split(
        all_subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return dev, test


def subset_by_subjects(
    data_dict: dict[int, SubjectRecord],
    subject_ids: np.ndarray,
) -> dict[int, SubjectRecord]:
    return {int(s): data_dict[int(s)] for s in subject_ids if int(s) in data_dict}
