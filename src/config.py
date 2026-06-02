"""
Settings aligned with Altaheri/EEG-ATCNet (github.com/Altaheri/EEG-ATCNet).
https://doi.org/10.1109/TII.2022.3197419
"""

from __future__ import annotations

from dataclasses import dataclass

# BCI IV-2a: preprocess.load_BCI2a_data uses t1=1.5*fs, t2=6*fs
EPOCH_TMIN_SEC = 1.5
EPOCH_TMAX_SEC = 6.0

# models.ATCNet_ / main_TrainTest.py
CONV_WEIGHT_DECAY = 0.009
DENSE_WEIGHT_DECAY = 0.5
CONV_MAX_NORM = 0.6

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000
EARLY_STOP_PATIENCE = 300
N_TRAIN_RUNS = 10

# ATCNet_ defaults
N_WINDOWS = 5
EEGN_F1 = 16
EEGN_D = 2
EEGN_KERNEL_SIZE = 64
EEGN_POOL_SIZE = 7
EEGN_DROPOUT = 0.3
TCN_DEPTH = 2
TCN_KERNEL_SIZE = 4
TCN_FILTERS = 32
TCN_DROPOUT = 0.3
ATT_HEADS = 2
ATT_KEY_DIM = 8
ATT_DROPOUT = 0.5
FUSE_MODE = "average"

USE_CHANNEL_STANDARDIZATION = True
SHUFFLE_DATA = True


# Training profiles (see scripts/run_subject_dependent.py --quick)


@dataclass(frozen=True)
class TrainingProfile:
    """Hyperparameters for train_model / train_best_of_runs."""

    n_train_runs: int
    epochs: int
    early_stop_patience: int
    lr_reduce_patience: int
    fit_verbose: int
    checkpoint_verbose: int


PAPER_PROFILE = TrainingProfile(
    n_train_runs=N_TRAIN_RUNS,
    epochs=EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
    lr_reduce_patience=20,
    fit_verbose=1,
    checkpoint_verbose=1,
)

# ~30–40 min/subject × 9 ≈ 5 h on a typical Paperspace GPU (not paper-grade).
QUICK_PROFILE = TrainingProfile(
    n_train_runs=1,
    epochs=200,
    early_stop_patience=50,
    lr_reduce_patience=10,
    fit_verbose=2,
    checkpoint_verbose=0,
)
