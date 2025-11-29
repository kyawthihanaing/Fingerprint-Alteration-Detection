"""
Configuration file for SOCOFing fingerprint recognition project.
Single source of truth for all constants and parameters.
"""

from pathlib import Path

# Get project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Random seed for reproducibility
SEED = 42

# Image preprocessing parameters
IMG_SIZE = 224                  # for EfficientNetB0
HOG_SIZE = 128                  # square resize for HOG/Gabor

# Directory paths (absolute) - Using proper Path objects
DATA_DIR = str(PROJECT_ROOT / "data" / "SOCOFing")
CACHE_DIR = str(PROJECT_ROOT / "features")
SPLIT_DIR = str(PROJECT_ROOT / "splits")
REPORTS_DIR = str(PROJECT_ROOT / "reports")

# Parallel processing
N_JOBS = -1                     # sklearn parallelism (-1 = use all available cores)

# Target column for classification
# TARGET_COL = "gender"         # Task 1: gender classification (76.7% acc, fairness-focused)
TARGET_COL = "is_altered"       # Task 2: Real vs Altered (target >90% acc, widely studied)
