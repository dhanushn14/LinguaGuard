"""
LinguaGuard-FL Configuration Module
Defines centralized parameters for multilingual datasets, model architecture,
federated learning rounds, and privacy defenses (Adaptive DP + Secure Aggregation).
Supports 4 Indic & Dravidian Languages: Kannada, Tamil (Tanglish), Malayalam, Hindi.
"""

import os
from pathlib import Path
import torch

# Fast offline loading from local cache (prevents connection retry hangs)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Hardware device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_NAME = "distilbert-base-multilingual-cased"
NUM_LABELS = 3
MAX_LENGTH = 128
MAX_SEQ_LENGTH = 128

LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral/Mixed",
    2: "Positive"
}

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 2
LOCAL_EPOCHS = 2
RANDOM_SEED = 42

# 4 Supported Regional Languages
LANGUAGES = ["kannada", "tamil", "malayalam", "hindi"]

# Federated Learning Settings
NUM_ROUNDS = 3
MIN_CLIENTS = 4
MIN_FIT_CLIENTS = 4
MIN_EVAL_CLIENTS = 4
MIN_AVAILABLE_CLIENTS = 4
FRACTION_FIT = 1.0
FRACTION_EVAL = 1.0

# Privacy Settings
# 1. Adaptive Differential Privacy
DP_ENABLED = True
DP_CLIP_VALUE = 1.0
DP_CLIP_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.001  # Gaussian noise sigma

# 2. Secure Aggregation (Simulation Masking)
SECURE_AGG_ENABLED = True
SECURE_AGG_SCALE = 0.0005
SECURE_AGG_MASK_SCALE = 0.0005

# Directories
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
