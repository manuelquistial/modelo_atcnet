#!/usr/bin/env python3
"""
Download BCI Competition IV Dataset 2a (BNCI2014_001) via MOABB and copy .mat
files into data/raw/BCICIV_2a_mat/ (layout compatible with EEG-ATCNet).
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before any `import mne` / `import moabb` (MOABB uses MNE_DATA at import).
_MNE_CACHE = ROOT / "data" / "mne_cache"
_MNE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = str(_MNE_CACHE.resolve())

from src.utils import ensure_dir, get_raw_data_dir


def _find_moabb_mat_files() -> list[Path]:
    """Locate 001-2014/AxxT.mat under MNE/MOABB data directory."""
    search_roots = [
        Path(os.environ.get("MNE_DATA", ROOT / "data" / "mne_cache")),
        ROOT / "data" / "mne_cache",
        Path.home() / "mne_data",
    ]
    paths: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        paths.extend(root.glob("**/001-2014/A*T.mat"))
        paths.extend(root.glob("**/001-2014/A*E.mat"))
    # Deduplicate by filename (A01T.mat, ...)
    by_name: dict[str, Path] = {}
    for p in paths:
        by_name[p.name] = p
    return sorted(by_name.values(), key=lambda p: p.name)


def _ensure_mne_data_dir() -> Path:
    """Force a writable MNE_DATA for this project (MOABB reads os.environ['MNE_DATA'])."""
    import mne

    data_home = ROOT / "data" / "mne_cache"
    data_home.mkdir(parents=True, exist_ok=True)
    resolved = str(data_home.resolve())
    os.environ["MNE_DATA"] = resolved
    mne.set_config("MNE_DATA", resolved, set_env=True)
    return data_home


def download_with_moabb(accept: bool = True) -> None:
    from moabb.datasets import BNCI2014_001

    data_home = _ensure_mne_data_dir()
    print(f"MNE_DATA: {data_home}")
    print("Downloading BNCI2014_001 (BCI IV-2a) via MOABB...")
    dataset = BNCI2014_001()
    dataset.download(
        subject_list=dataset.subject_list,
        path=str(data_home),
        accept=accept,
        verbose=True,
    )


def copy_to_project(moabb_mats: list[Path], dest: Path) -> None:
    ensure_dir(dest)
    for src in sorted(moabb_mats):
        dst = dest / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  skip (exists): {dst.name}")
            continue
        shutil.copy2(src, dst)
        print(f"  copied: {src.name} -> {dest}/")


def main() -> None:
    _ensure_mne_data_dir()
    dest = get_raw_data_dir()
    print(f"Target directory: {dest.resolve()}")

    existing = list(dest.glob("A*T.mat")) + list(dest.glob("A*E.mat"))
    if len(existing) >= 18:
        print(f"Found {len(existing)} MAT files already. Use --force to re-download.")
        if "--force" not in sys.argv:
            validate()
            return

    accept = "--yes" in sys.argv or "-y" in sys.argv
    if not accept:
        print(
            "MOABB will download BNCI2014_001 (CC BY-ND 4.0). "
            "Re-run with --yes to accept the license and download."
        )
        if input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit(0)
        accept = True

    download_with_moabb(accept=accept)
    mats = _find_moabb_mat_files()
    if not mats:
        raise RuntimeError(
            "Download finished but no .mat files found. "
            "Check ~/mne_data or MOABB_DATASETS_PATH."
        )
    print(f"Found {len(mats)} MAT files from MOABB cache.")
    copy_to_project(mats, dest)
    validate()


def validate() -> None:
    from src.data_loader import N_SUBJECTS, load_bci2a_subject, validate_dataset_files

    validate_dataset_files()
    X, y = load_bci2a_subject(1, "T")
    print(f"OK: subject 01 session T -> X{X.shape}, y{y.shape}, labels={set(y.tolist())}")


if __name__ == "__main__":
    main()
