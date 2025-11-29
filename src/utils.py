"""
Utility functions for the project.
Contains helper functions for file I/O, logging, seeds, and plotting.
"""

from __future__ import annotations
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from . import config


def set_seeds(seed: int = config.SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🌱 Seeds set to {seed}")


def ensure_dirs():
    """
    Create all required project directories if they don't exist.
    """
    directories = [
        config.CACHE_DIR,
        f"{config.CACHE_DIR}/effb0",
        f"{config.CACHE_DIR}/gabor",
        config.SPLIT_DIR,
        config.REPORTS_DIR,
        "models"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("📁 Project directories ensured")


def load_metadata() -> pd.DataFrame:
    """
    Load the metadata CSV file.
    
    Returns:
        DataFrame with metadata (path, subject_id, gender, hand, finger, subset)
        
    Raises:
        FileNotFoundError: If metadata.csv doesn't exist
    """
    meta_path = Path(config.REPORTS_DIR) / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}. "
            f"Run 'python -m src.prepare_metadata --csv data\\SOCOFing_Full_Organised.csv' first."
        )
    df = pd.read_csv(meta_path)
    # Ensure paths are strings
    df["path"] = df["path"].astype(str)
    return df


def write_json(path, obj):
    """
    Write object to JSON file.
    
    Args:
        path: Output file path
        obj: Object to serialize (dict, list, etc.)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    """
    Read JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Deserialized object
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_encode(series: pd.Series):
    """
    Encode labels to integers.
    
    Args:
        series: Pandas series with categorical labels
        
    Returns:
        Tuple of (encoded_array, class_list, mapping_dict)
    """
    classes = sorted(series.dropna().unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = series.map(mapping).to_numpy()
    return encoded, classes, mapping


# ---------- Plotting Helpers ----------
import matplotlib.pyplot as plt


def save_fig(path: str):
    """
    Save current matplotlib figure and close it.
    
    Args:
        path: Output path for the figure
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved figure: {path}")


if __name__ == "__main__":
    print("Utility functions module loaded")
    ensure_dirs()
    print("All directories created successfully!")
