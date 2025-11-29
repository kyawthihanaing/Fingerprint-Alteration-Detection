"""
Enhanced Gabor filter feature extraction.
Implements comprehensive texture-based features using proper Gabor filter bank.
Addresses the severe limitations in the original implementation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.feature_extraction import image
from . import config, utils, preprocess


def enhanced_gabor_kernels():
    """
    Create a comprehensive bank of Gabor kernels for fingerprint ridge analysis.
    
    Returns:
        List of 40 Gabor kernels (5 frequencies × 8 orientations)
        Standard configuration for fingerprint texture analysis
    """
    # Optimized parameters for fingerprint ridge patterns
    frequencies = [0.05, 0.1, 0.15, 0.2, 0.25]  # 5 frequencies (more comprehensive)
    orientations = [i * np.pi / 8 for i in range(8)]  # 8 orientations (0° to 157.5°)
    
    kernels = []
    for freq in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),  # Larger kernel for better frequency response
                sigma=5.0,       # Optimized for fingerprint ridges
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.7,       # Better aspect ratio for ridges
                psi=0,
                ktype=cv2.CV_32F
            )
            kernels.append(kernel)
    
    return kernels  # 40 filters (5×8)


def enhanced_gabor_features(image, kernels):
    """
    Extract comprehensive Gabor feature vector from an image.
    Computes multiple texture statistics per kernel for richer representation.
    
    Args:
        image: Grayscale image
        kernels: List of Gabor kernels
        
    Returns:
        80-D feature vector (40 kernels × 2 statistics: mean + std)
    """
    features = []
    
    for kernel in kernels:
        # Apply Gabor filter
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        
        # Compute amplitude (magnitude of response)
        amplitude = np.abs(filtered)
        
        # Extract comprehensive statistics
        mean_amp = float(amplitude.mean())
        std_amp = float(amplitude.std())
        
        features.extend([mean_amp, std_amp])
    
    return np.array(features, dtype="float32")  # 80-D


def gabor_texture_statistics(image, kernels):
    """
    Extract advanced texture statistics using Gabor filter responses.
    Adds texture measures beyond simple mean/std.
    
    Args:
        image: Grayscale image  
        kernels: List of Gabor kernels
        
    Returns:
        Additional texture features (energy, entropy, contrast)
    """
    # Compute responses for all kernels
    responses = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        amplitude = np.abs(filtered)
        responses.append(amplitude)
    
    # Stack all responses
    response_stack = np.stack(responses, axis=-1)  # H×W×40
    
    # Global texture statistics across all responses
    energy = float(np.sum(response_stack ** 2))
    entropy = float(-np.sum(response_stack * np.log(response_stack + 1e-8)))
    contrast = float(np.var(response_stack))
    
    # Maximum response across orientations (for each frequency)
    max_responses = []
    for freq_idx in range(5):  # 5 frequencies
        freq_responses = response_stack[:, :, freq_idx*8:(freq_idx+1)*8]
        max_resp = np.max(freq_responses, axis=-1)
        max_responses.append(max_resp.mean())
    
    # Directional coherence (consistency across orientations)
    coherence_scores = []
    for freq_idx in range(5):
        freq_responses = response_stack[:, :, freq_idx*8:(freq_idx+1)*8]
        # Compute coefficient of variation across orientations
        mean_resp = np.mean(freq_responses, axis=-1)
        std_resp = np.std(freq_responses, axis=-1)
        coherence = np.mean(std_resp / (mean_resp + 1e-8))
        coherence_scores.append(coherence)
    
    additional_features = [energy, entropy, contrast] + max_responses + coherence_scores
    return np.array(additional_features, dtype="float32")  # 13-D


def main():
    """
    Extract and cache enhanced Gabor features for all images.
    """
    utils.ensure_dirs()
    
    # Load metadata
    df = utils.load_metadata()
    print(f"📊 Extracting ENHANCED Gabor features from {len(df)} images...")
    
    # Create enhanced Gabor kernel bank
    kernels = enhanced_gabor_kernels()
    print(f"🔨 Created {len(kernels)} enhanced Gabor kernels (5 freq × 8 orient)")
    
    # Extract features
    features_list = []
    
    for idx, image_path in enumerate(df.path):
        # Load and preprocess
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        preprocessed = preprocess.preprocess_for_texture(gray)
        
        # Extract enhanced Gabor features
        basic_features = enhanced_gabor_features(preprocessed, kernels)  # 80-D
        texture_stats = gabor_texture_statistics(preprocessed, kernels)  # 13-D
        
        # Combine all features
        combined_features = np.concatenate([basic_features, texture_stats])  # 93-D
        features_list.append(combined_features)
        
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"   Processed {idx + 1}/{len(df)} images")
    
    # Stack into array
    X = np.vstack(features_list)
    
    # Save enhanced features
    np.save(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy", X)
    df.to_csv(f"{config.CACHE_DIR}/gabor/index_enhanced.csv", index=False)
    
    print(f"\n✅ Enhanced Gabor features saved:")
    print(f"   Shape: {X.shape}")
    print(f"   Feature breakdown:")
    print(f"     - Basic Gabor (40 filters × 2 stats): 80 features")
    print(f"     - Advanced texture statistics: 13 features")
    print(f"     - Total: 93 features")
    print(f"   Location: {config.CACHE_DIR}\\gabor\\gabor_enhanced.npy")
    print(f"   Index: {config.CACHE_DIR}\\gabor\\index_enhanced.csv")


if __name__ == "__main__":
    main()