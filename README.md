# ATCNet — BCI Competition IV-2a Motor Imagery

[![Repository](https://img.shields.io/badge/GitHub-manuelquistial%2Fmodelo__atcnet-181717?logo=github)](https://github.com/manuelquistial/modelo_atcnet)

Reproducible implementation of **Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification** (Altaheri et al., 2023) on **BCI Competition IV Dataset 2a**.

**Repositorio:** `git@github.com:manuelquistial/modelo_atcnet.git`  
**Ejecución en GPU (Paperspace):** ver [docs/PAPERSPACE.md](docs/PAPERSPACE.md)

## Objective

End-to-end pipeline for EEG motor imagery (4 classes) with:

- Subject-dependent evaluation (train `AxxT`, test `AxxE`)
- Subject-independent **LOSO** (leave-one-subject-out)
- Ablation study (5 ATCNet variants)
- **EEGNet** baseline
- Per-trial predictions, error analysis, confusion matrices, Cohen's kappa

## Dataset

Download [BCI Competition IV Dataset 2a](http://www.bbci.de/competition/iv/) GDF files and place them in:

```
data/raw/BCICIV_2a_gdf/
├── A01T.gdf
├── A01E.gdf
├── ...
├── A09T.gdf
└── A09E.gdf
```

| Property | Value |
|----------|--------|
| Subjects | 9 |
| EEG channels | 22 |
| Sampling rate | 250 Hz |
| Classes | left hand, right hand, feet, tongue (769–772) |
| Trials / session | 288 |
| Model input | 22 × 1125 samples (cue +1.5 s … +6.0 s), shape `(batch, 1, 22, 1125)` |

**Aligned with [Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet):** per-channel `StandardScaler`, `fuse='average'`, test set as validation, 10 training runs, L2 + max-norm as in `models.py`.

## Installation

```bash
cd modelo_atcnet
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Settings: `src/config.py`.

## Verify model shapes (no data required)

```bash
python scripts/verify_shapes.py
```

Flow: `(B,1,22,1125) → CV (B,20,32) → 5×TCN → Dense(4) → Average → softmax`.

## Training & evaluation

From project root:

```bash
# Subject-dependent (paper ~85.38% acc, ~0.81 κ)
python scripts/run_subject_dependent.py

# LOSO subject-independent (paper ~70.97% acc, ~0.613 κ)
python scripts/run_loso.py

# Ablation (5 variants × 9 subjects)
python scripts/run_ablation.py

# EEGNet baseline (subject-dependent)
python scripts/run_baselines.py
```

**Note:** Full runs use up to 1000 epochs with early stopping (patience 300). Use GPU when possible; ablation trains 45 models.

## Output files

| Path | Description |
|------|-------------|
| `data/results/subject_dependent/subject_dependent_metrics.csv` | Per-subject + mean/std accuracy & kappa |
| `data/results/subject_dependent/subject_XX_predictions.csv` | Per-trial predictions |
| `data/results/subject_dependent/subject_XX_errors.csv` | Misclassified trials |
| `data/results/loso/loso_metrics.csv` | LOSO metrics |
| `data/results/loso/loso_test_subject_XX_predictions.csv` | LOSO predictions |
| `data/results/ablation/ablation_raw_results.csv` | All variant × subject runs |
| `data/results/ablation/ablation_summary.csv` | Mean/std per variant |
| `data/results/baselines/eegnet_subject_dependent_metrics.csv` | EEGNet comparison |
| `data/results/figures/*.png` | Confusion matrices, ablation plot |

## Reference results (paper)

| Protocol | Accuracy | Cohen's κ |
|----------|----------|-----------|
| Subject-dependent (T→E) | ~85.38% | ~0.81 |
| LOSO (subject-independent) | ~70.97% | ~0.613 |

Your numbers may differ due to epoch extraction (`tmin`/`tmax`), MNE event mapping, TensorFlow version, hardware, and random seed.

## Project layout

```
src/           — data loading, ATCNet, TCN, training, metrics, plots
scripts/       — CLI entry points
notebooks/     — dataset & shape checks
data/raw/      — GDF files (not in git)
data/results/  — metrics, models, figures
```

## Reproducibility

- Alineado con [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet): ver `docs/PAPER_AUDIT.md`
- Adam lr=0.001, batch 64, 1000 épocas, early stopping `val_accuracy`, 10 runs
- Configuración: `src/config.py`

## License

Research / educational use. Cite Altaheri et al. (2023) when using this code.
