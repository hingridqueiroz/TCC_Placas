"""Shared hyperparameters and constants for the TCC_Placas project."""

NUM_CLASSES = 43
DEFAULT_SEED = 42

# Feature Extraction
EPOCHS_FE = 40
LR_FE = 1e-3

# Fine-Tuning
EPOCHS_FT = 30
LR_FT_BACKBONE = 1e-5
LR_FT_CLASSIFIER = 1e-4
PATIENCE_FT = 5

# Data
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2