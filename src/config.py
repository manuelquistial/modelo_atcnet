"""
ATCNet settings — PhysioNet MI binary left/right (modelo_deep_eeg aligned).
"""

from __future__ import annotations

from dataclasses import dataclass

# PhysioNet MI (MOABB MotorImagery left_hand / right_hand)
N_CLASSES = 2
N_EEG_CHANNELS = 64
FS = 160
# Set after preprocessing (harmonized min n_times across cohort); placeholder for verify_shapes.
N_SAMPLES = 480

CLASS_NAMES = ["left_hand", "right_hand"]

# Hold-out split (subject-level, no trial leakage)
HOLDOUT_TEST_SIZE = 0.20
HOLDOUT_RANDOM_STATE = 42

# Preprocessing (modelo_deep_eeg default)
HIGHPASS_HZ = 4.0
OUTLIER_UV = 800.0

# ATCNet architecture (Altaheri/EEG-ATCNet)
CONV_WEIGHT_DECAY = 0.009
DENSE_WEIGHT_DECAY = 0.5
CONV_MAX_NORM = 0.6

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000
EARLY_STOP_PATIENCE = 300
N_TRAIN_RUNS = 10

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


@dataclass(frozen=True)
class TrainingProfile:
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

QUICK_PROFILE = TrainingProfile(
    n_train_runs=1,
    epochs=200,
    early_stop_patience=50,
    lr_reduce_patience=10,
    fit_verbose=2,
    checkpoint_verbose=0,
)
