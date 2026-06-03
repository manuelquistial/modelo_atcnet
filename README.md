# ATCNet — PhysioNet MI (left vs right hand)

[![Repository](https://img.shields.io/badge/GitHub-manuelquistial%2Fmodelo__atcnet-181717?logo=github)](https://github.com/manuelquistial/modelo_atcnet)

ATCNet for **binary motor imagery** on **PhysioNet MI** (MOABB), aligned with [modelo_deep_eeg](https://github.com/manuelquistial/modelo_deep_eeg).

**Classes:** `left_hand` vs `right_hand`  
**Ejecución GPU:** [docs/PAPERSPACE.md](docs/PAPERSPACE.md)

## Dataset

| Property | Value |
|----------|--------|
| Source | [PhysioNet MI](https://physionet.org/content/eegmmidb/1.0.0/) via MOABB `PhysionetMI` |
| Subjects | ~108 (excludes subject 88 @ 128 Hz) |
| Channels | 64 EEG |
| Sampling rate | 160 Hz |
| Classes | 2 (left hand, right hand) |
| Preprocessing | Outlier removal, 4 Hz HPF, time harmonization (see `modelo_deep_eeg`) |

### Download

```bash
python scripts/download_physionet.py --subjects 1-5   # smoke test
python scripts/download_physionet.py                  # full cohort (~108 subjects)
```

Data stored under `data/mne/` (MOABB/MNE cache).

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Verify model shapes (no download required)

```bash
python scripts/verify_shapes.py
```

Input: `(B, 1, 64, T)` → binary softmax.

## Training

```bash
# Hold-out: 80% subjects train, 20% test (subject-level split)
python scripts/run_holdout.py
python scripts/run_holdout.py --quick --subjects 1-10

# LOSO (leave-one-subject-out)
python scripts/run_loso.py --quick --subjects 1-10

# Ablation + EEGNet baseline
python scripts/run_ablation.py --quick
python scripts/run_baselines.py --quick
```

## Outputs

| Path | Description |
|------|-------------|
| `data/results/holdout/holdout_metrics.csv` | Hold-out accuracy & kappa |
| `data/results/loso/loso_metrics.csv` | LOSO per-subject metrics |
| `data/results/figures/*.png` | Confusion matrices |

## Project layout

```
src/physionet/   — MOABB loader, preprocessing, splits (from modelo_deep_eeg)
src/atcnet.py    — ATCNet architecture
scripts/         — download, holdout, LOSO, ablation
data/mne/        — raw MOABB downloads (not in git)
```

## Reference

- Altaheri et al. (2023) — ATCNet architecture
- `modelo_deep_eeg` — PhysioNet MI loading & preprocessing protocol
