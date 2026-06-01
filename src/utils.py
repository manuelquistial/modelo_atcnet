"""Project utilities: paths, directories, JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path


def get_project_root() -> Path:
    """Return the atcnet_implementation project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Default: data/raw/BCICIV_2a_mat. Override with env MODEL_ATCNET_DATA_DIR (Paperspace)."""
    import os

    override = os.environ.get("MODEL_ATCNET_DATA_DIR")
    if override:
        return Path(override)
    return get_data_dir() / "raw" / "BCICIV_2a_mat"


def get_results_dir() -> Path:
    return get_data_dir() / "results"


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
