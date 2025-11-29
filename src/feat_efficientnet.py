"""
EfficientNet feature extraction.
Uses pre-trained EfficientNetB0 for deep feature extraction with frozen weights.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
from . import config, utils, preprocess


def build_extractor():
    """
    Build EfficientNetB0 feature extractor (frozen, pre-trained on ImageNet).
    
    Returns:
        Keras Model that outputs 1280-D feature vectors
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    base.trainable = False  # Freeze all weights
    
    # Global average pooling to get fixed-size features
    x = layers.GlobalAveragePooling2D()(base.output)  # 1280-D
    
    model = Model(base.input, x, name="EfficientNetB0_Extractor")
    return model


def main(batch_size=64):
    """
    Extract and cache EfficientNet features for all images.
    
    Args:
        batch_size: Batch size for inference
    """
    utils.ensure_dirs()
    utils.set_seeds()
    
    # Load metadata
    df = utils.load_metadata()
    print(f"📊 Loading {len(df)} images for feature extraction...")
    
    # Build extractor
    print("🔨 Building EfficientNetB0 extractor...")
    model = build_extractor()
    print(f"   Model: {model.name}")
    print(f"   Output shape: {model.output_shape}")
    
    # Extract features in batches
    features_list = []
    image_paths = df.path.tolist()
    
    print(f"🚀 Extracting features (batch_size={batch_size})...")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess batch
        batch_images = []
        for path in batch_paths:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            preprocessed = preprocess.preprocess_for_effnet(gray)
            batch_images.append(preprocessed)
        
        batch_array = np.stack(batch_images, axis=0)
        batch_array = preprocess_input(batch_array)  # Official EfficientNet normalization
        
        # Extract features
        batch_features = model.predict(batch_array, verbose=0).astype("float32")
        features_list.append(batch_features)
        
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(image_paths):
            print(f"   Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Concatenate all features
    X = np.concatenate(features_list, axis=0)
    
    # Save features and index (index now includes is_altered column from metadata)
    np.save(f"{config.CACHE_DIR}/effb0/all.npy", X)
    df.to_csv(f"{config.CACHE_DIR}/effb0/index.csv", index=False)
    
    print(f"\n✅ EfficientNet features saved:")
    print(f"   Shape: {X.shape}")
    print(f"   Location: {config.CACHE_DIR}\\effb0\\all.npy")
    print(f"   Index: {config.CACHE_DIR}\\effb0\\index.csv")
    print(f"   Index includes: path, subject_id, gender, hand, finger, subset, is_altered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EfficientNet features")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()
    
    main(batch_size=args.batch)
