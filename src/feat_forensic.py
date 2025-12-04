"""
Forensic fingerprint feature extraction.
Extracts biological consistency features for fingerprint alteration detection.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage import morphology, measure
from . import config, utils, preprocess


def extract_biological_features(image):
    """
    Extract biological consistency features from fingerprint image.

    Args:
        image: Preprocessed fingerprint image

    Returns:
        Dictionary of biological features
    """
    features = {}

    # Basic image statistics
    features['mean_intensity'] = np.mean(image)
    features['std_intensity'] = np.std(image)
    features['median_intensity'] = np.median(image)

    # Ridge density estimation (simplified)
    # Apply threshold to get binary ridge map
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonize to get ridge structure
    skeleton = morphology.skeletonize(binary.astype(bool))

    # Count ridge pixels
    ridge_pixels = np.sum(skeleton)
    total_pixels = image.shape[0] * image.shape[1]
    features['ridge_density'] = ridge_pixels / total_pixels

    # Ridge thickness variation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(skeleton.astype(np.uint8), kernel, iterations=1)
    features['ridge_thickness_variation'] = np.std(dilated - eroded)

    # Local orientation consistency (simplified)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    orientation = np.arctan2(sobely, sobelx)
    features['orientation_consistency'] = np.std(orientation)

    # Minutiae-like features (simplified ridge endings and bifurcations)
    # This is a very basic approximation
    features['ridge_endings'] = np.sum(cv2.cornerHarris(skeleton.astype(np.float32), 2, 3, 0.04) > 0.01)
    features['ridge_bifurcations'] = np.sum(cv2.cornerHarris(skeleton.astype(np.float32), 2, 3, 0.04) > 0.02)

    # Texture homogeneity
    glcm = cv2.calcHist([image], [0], None, [256], [0, 256])
    features['texture_homogeneity'] = np.sum(glcm * glcm) / np.sum(glcm)**2

    # Frequency domain features
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # Dominant frequency
    features['dominant_frequency'] = np.argmax(np.sum(magnitude_spectrum, axis=0))

    # Spectral centroid
    freq_bins = np.arange(magnitude_spectrum.shape[1])
    features['spectral_centroid'] = np.sum(freq_bins * np.sum(magnitude_spectrum, axis=0)) / np.sum(magnitude_spectrum)

    return features


def main():
    """Extract forensic features from all fingerprint images."""
    print("📁 Project directories ensured")
    print("🔬 Extracting FORENSIC features from all images...")

    # Load metadata
    df = pd.read_csv(f"{config.REPORTS_DIR}/metadata.csv")

    features_list = []
    feature_names = None

    for idx, row in df.iterrows():
        image_path = Path(row['path'])

        # Load and preprocess image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        # Apply raw preprocessing (no binarization to preserve biological features)
        processed = preprocess.preprocess_for_texture(image)

        # Extract biological features
        features = extract_biological_features(processed)

        if feature_names is None:
            feature_names = list(features.keys())

        # Convert to array
        feature_vector = np.array([features[name] for name in feature_names])
        features_list.append(feature_vector)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"   Processed {idx + 1}/{len(df)} images")

    # Stack into array
    X = np.vstack(features_list)

    # Save forensic features
    Path(f"{config.CACHE_DIR}/forensic").mkdir(parents=True, exist_ok=True)
    np.save(f"{config.CACHE_DIR}/forensic/forensic_features.npy", X)
    df.to_csv(f"{config.CACHE_DIR}/forensic/index.csv", index=False)

    print("\n✅ Forensic features saved:")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(feature_names)} biological consistency metrics")
    print(f"   Feature names: {', '.join(feature_names)}")
    print(f"   Location: {config.CACHE_DIR}\\forensic\\forensic_features.npy")
    print(f"   Index: {config.CACHE_DIR}\\forensic\\index.csv")


if __name__ == "__main__":
    main()