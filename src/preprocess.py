"""
Image preprocessing utilities.
Handles image loading, resizing, enhancement, and visualization.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from . import config, utils


def _clahe(gray_image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def preprocess_for_effnet(gray_image):
    """
    Preprocess image for EfficientNet (deep features).
    Applies CLAHE, bilateral filtering, and converts to 3-channel RGB.
    NOTE: Returns 0..255 range - preprocess_input will normalize later.
    
    Args:
        gray_image: Grayscale image (or BGR, will be converted)
        
    Returns:
        3-channel float32 image in [0, 255] range, shape (224, 224, 3)
    """
    # Convert to grayscale if needed
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    enhanced = _clahe(gray_image)
    
    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Resize to EfficientNet input size
    resized = cv2.resize(denoised, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Convert to 3-channel (keep 0..255 range for preprocess_input)
    rgb = np.stack([resized, resized, resized], axis=-1).astype("float32")
    
    return rgb


def preprocess_for_texture(gray_image):
    """
    Research-Aligned Preprocessing: Raw CLAHE (No Binarization).
    Preserves micro-texture for Gabor filters.
    """
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE (Contrast Enhancement) - ClipLimit=2.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # 2. NO Thresholding/Morphology (Removed to save micro-texture)
    
    # 3. Resize directly
    resized = cv2.resize(enhanced, (config.HOG_SIZE, config.HOG_SIZE), interpolation=cv2.INTER_AREA)
    
    return resized


# ---------- Visualization ----------
def preview_preprocessing(n=8, seed=config.SEED):
    """
    Create a comprehensive visualization grid showing detailed preprocessing steps.
    
    Args:
        n: Number of random samples to visualize
        seed: Random seed for sample selection
    """
    import random
    utils.ensure_dirs()
    
    df = utils.load_metadata()
    random.seed(seed)
    idxs = random.sample(range(len(df)), k=min(n, len(df)))
    
    # Create figure with 6 columns: Raw, CLAHE, Bilateral, EfficientNet Final, Gabor, Texture Final
    fig, axes = plt.subplots(len(idxs), 6, figsize=(20, 3 * len(idxs)))
    
    # Handle single row case
    if len(idxs) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, df_idx in enumerate(idxs):
        image_path = df.path.iloc[df_idx]
        
        # Extract metadata from path
        filename = Path(image_path).name
        subset = df.subset.iloc[df_idx]
        
        # Load raw image
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Step-by-step EfficientNet preprocessing
        clahe_applied = _clahe(raw)
        bilateral_applied = cv2.bilateralFilter(clahe_applied, 9, 75, 75)
        eff_final = cv2.resize(bilateral_applied, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Step-by-step Texture preprocessing
        tex_resized = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA)
        tex_normalized = (tex_resized.astype("float32") / 255.0)
        
        # Gabor filter visualization (apply one filter for demo - matches enhanced implementation)
        ksize = 31
        sigma = 5.0  # Enhanced parameter
        theta = 0  # 0 degrees for horizontal (1 of 8 orientations)
        lambd = 1.0 / 0.1  # Enhanced frequency parameter (freq=0.1)
        gamma = 0.7  # Enhanced aspect ratio
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(tex_normalized, cv2.CV_32F, gabor_kernel)
        gabor_vis = np.abs(gabor_response)
        
        # Column 1: Raw Image
        ax = axes[row_idx, 0]
        ax.imshow(raw, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"1. Raw Image\n{raw.shape[0]}×{raw.shape[1]}", 
                    fontsize=10, fontweight='bold')
        ax.axis("off")
        # Add stats box
        mean_val = raw.mean()
        std_val = raw.std()
        ax.text(0.02, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Column 2: CLAHE Applied
        ax = axes[row_idx, 1]
        ax.imshow(clahe_applied, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"2. CLAHE\nContrast Enhanced", 
                    fontsize=10, fontweight='bold', color='darkgreen')
        ax.axis("off")
        mean_val = clahe_applied.mean()
        std_val = clahe_applied.std()
        ax.text(0.02, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Column 3: Bilateral Filter Applied
        ax = axes[row_idx, 2]
        ax.imshow(bilateral_applied, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"3. Bilateral Filter\nDenoised", 
                    fontsize=10, fontweight='bold', color='darkblue')
        ax.axis("off")
        mean_val = bilateral_applied.mean()
        std_val = bilateral_applied.std()
        ax.text(0.02, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Column 4: EfficientNet Final (resized to 224x224)
        ax = axes[row_idx, 3]
        ax.imshow(eff_final, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"4. EfficientNet Ready\n{eff_final.shape[0]}×{eff_final.shape[1]}", 
                    fontsize=10, fontweight='bold', color='darkred')
        ax.axis("off")
        ax.text(0.02, 0.98, f'Stream A\nDeep Learning', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Column 5: Gabor Response (sample)
        ax = axes[row_idx, 4]
        ax.imshow(gabor_vis, cmap="gray")
        ax.set_title(f"5. Enhanced Gabor\nθ=0° (1/8), f=0.1 (1/5)", 
                    fontsize=10, fontweight='bold', color='darkorange')
        ax.axis("off")
        ax.text(0.02, 0.98, f'Texture\nAnalysis', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
        
        # Column 6: Texture Final (resized to 128x128)
        ax = axes[row_idx, 5]
        ax.imshow(tex_normalized * 255, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"6. Texture Ready\n{tex_normalized.shape[0]}×{tex_normalized.shape[1]}", 
                    fontsize=10, fontweight='bold', color='purple')
        ax.axis("off")
        ax.text(0.02, 0.98, f'Stream B\nGabor (93-D)', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
        
        # Add row label with file info
        axes[row_idx, 0].text(-0.15, 0.5, f'{subset}\n{filename[:20]}...', 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=9, verticalalignment='center', 
                             rotation=90, fontweight='bold')
    
    # Add processing pipeline description
    pipeline_text = """
    Stream A (EfficientNet): Raw → CLAHE (contrast) → Bilateral (denoise) → Resize (224×224) → 3-channel RGB → 1280-D → PCA(384) → RF
    Stream B (Gabor): Raw → CLAHE → Threshold → Morphology → Resize (128×128) → 40 Gabor filters (5×8) → 93-D features → RF
    FUSION: Concatenate Stream A (1280-D) + Stream B (93-D) → 1373-D → PCA(384) → Random Forest
    """
    
    fig.text(0.5, 0.02, pipeline_text.strip(), ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    plt.suptitle("Comprehensive Preprocessing Pipeline - Step-by-Step Visualization", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    
    utils.save_fig(f"{config.REPORTS_DIR}/viz_preprocessing_grid.png")
    print(f"🖼️  Preprocessing visualization saved")


if __name__ == "__main__":
    print("Running preprocessing visualization...")
    preview_preprocessing(n=8)
    print("✅ Done!")
