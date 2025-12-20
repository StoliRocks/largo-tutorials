"""
Configuration for Hard Drive Failure Prediction Tutorial

Centralized settings for data paths, model hyperparameters, and training config.
"""

from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent / "models"

# Data settings
BACKBLAZE_BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
# Using 2023 Q4 as default - recent, stable, good size
DEFAULT_DATASET = "data_Q4_2023.zip"

# SMART attributes most predictive of failure
# Based on Backblaze's own analysis and literature
KEY_SMART_ATTRS = [
    "smart_5_raw",    # Reallocated Sectors Count - BAD SECTORS
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_197_raw",  # Current Pending Sector Count
    "smart_198_raw",  # Offline Uncorrectable Sector Count
    "smart_9_raw",    # Power-On Hours
    "smart_194_raw",  # Temperature
    "smart_12_raw",   # Power Cycle Count
    "smart_4_raw",    # Start/Stop Count
    "smart_1_raw",    # Read Error Rate
    "smart_7_raw",    # Seek Error Rate
    "smart_10_raw",   # Spin Retry Count
]

# All SMART attributes for full model
ALL_SMART_ATTRS = [f"smart_{i}_raw" for i in range(256)]

# Feature engineering
SEQUENCE_LENGTH = 30  # Days of history for sequence models
PREDICTION_HORIZON = 7  # Predict failure within N days

# Class imbalance handling
# Failure rate is ~1-2%, so we need strategies
UNDERSAMPLE_RATIO = 0.1  # Keep 10% of healthy samples for quick experiments
USE_CLASS_WEIGHTS = True

# Model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 50,  # Approx ratio of neg/pos
    "random_state": 42,
}

LSTM_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True,
}

TRANSFORMER_PARAMS = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
}

# Training
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

# Evaluation
TEST_SIZE = 0.2
VAL_SIZE = 0.1
# Use time-based split (not random) for realistic evaluation
TIME_BASED_SPLIT = True
