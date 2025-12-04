"""
Comprehensive EDA and Feature Visualization (Assignment Version).
Includes 3D t-SNE, Forensic Feature Analysis, and Model Comparison.
Clean version without Deep Fusion/Stacking for assignment submission.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import glob
import re
import joblib
import os
import random
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import shannon_entropy

try:
    from . import config, utils, preprocess
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src import config, utils, preprocess


# --- Helper utilities ---
def _sample_metadata(n=1, target=None, seed=config.SEED):
    """Return n metadata rows optionally filtered by class."""
    df = utils.load_metadata()
    if target is not None:
        df = df[df[config.TARGET_COL] == target]
    if df.empty:
        raise ValueError("No samples found for the specified criteria")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx]


def _load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def subset_label(row):
    cls = "Altered" if getattr(row, config.TARGET_COL) == 1 else "Real"
    subset = getattr(row, 'subset', 'Unknown')
    return f"{cls} | {subset}"


def _compute_pattern_maps(image):
    """Compute intermediate representations for pattern-recognition plots."""
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    thresh = threshold_otsu(enhanced)
    binary = enhanced > thresh
    skeleton = skeletonize(binary)
    skel_u8 = skeleton.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endings = (skel_u8 == 1) & (neighbor_count == 2)
    bifurcations = (skel_u8 == 1) & (neighbor_count >= 4)

    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=5)
    orientation = (np.degrees(np.arctan2(sobely, sobelx)) + 180) % 180

    # Simple Gabor bank (4 orientations for speed)
    gabor_responses = []
    for theta in np.linspace(0, np.pi, 4, endpoint=False):
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 8.0, 0.6, 0, ktype=cv2.CV_32F)
        response = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
        gabor_responses.append(np.abs(response))

    return {
        "raw": resized,
        "enhanced": enhanced,
        "binary": binary,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "orientation": orientation,
        "gabor_bank": gabor_responses,
    }


# --- EXISTING EDA FUNCTIONS ---
def viz_preprocessing():
    """Show preprocessing pipeline visualization."""
    print("\n" + "="*60)
    print("GENERATING PREPROCESSING PREVIEW")
    print("="*60)
    preprocess.preview_preprocessing(n=6)
    print(f"✅ Preprocessing preview saved to {config.REPORTS_DIR}/viz_preprocessing_grid.png")


def viz_preprocessing_metrics(n_samples: int = 150):
    """Quantify how preprocessing stages affect intensity statistics."""
    print("\n" + "="*60)
    print("ANALYZING PREPROCESSING METRICS")
    print("="*60)

    df = utils.load_metadata()
    if df.empty:
        print("❌ Metadata unavailable.")
        return

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)

    records = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i in idx:
        img = _load_image(df.path.iloc[i])
        clahe_img = clahe.apply(img)
        bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)
        eff = cv2.resize(bilateral, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        tex = preprocess.preprocess_for_texture(img)

        steps = {
            "Raw": img,
            "CLAHE": clahe_img,
            "Bilateral": bilateral,
            "EfficientNet": eff,
            "Texture": tex,
        }

        for step_name, data in steps.items():
            arr = data.astype(np.float32)
            records.append({
                "Step": step_name,
                "Mean": float(arr.mean()),
                "Contrast": float(arr.std()),
                "Entropy": float(shannon_entropy(arr.astype(np.uint8))),
            })

    metrics_df = pd.DataFrame(records)
    summary = metrics_df.groupby("Step").mean().reindex(["Raw", "CLAHE", "Bilateral", "EfficientNet", "Texture"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    summaries = ["Mean", "Contrast", "Entropy"]
    colors = ["#2E86C1", "#27AE60", "#8E44AD"]
    for ax, metric, color in zip(axes, summaries, colors):
        ax.plot(summary.index, summary[metric], marker='o', color=color, linewidth=2)
        ax.set_title(f"{metric} Across Pipeline", fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=25)

    plt.suptitle("Quantitative Impact of Preprocessing Stages", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    utils.save_fig(f"{config.REPORTS_DIR}/viz_preprocessing_metrics.png")

def viz_class_balance():
    print("\n" + "="*60); print("GENERATING CLASS BALANCE CHART"); print("="*60)
    df = utils.load_metadata()
    class_counts = df[config.TARGET_COL].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_counts.index, class_counts.values, color=["steelblue", "coral"], alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}\n({height/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.title(f"Class Distribution: {config.TARGET_COL.capitalize()}", fontsize=15, fontweight='bold')
    plt.xticks([0, 1], ['Real (0)', 'Altered (1)'], fontsize=11)
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/class_balance.png")

def viz_subset_distribution():
    print("\n" + "="*60); print("GENERATING SUBSET DISTRIBUTION CHART"); print("="*60)
    df = utils.load_metadata()
    subset_order = ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
    subset_counts = df["subset"].value_counts().reindex(subset_order)
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'lightgreen', 'orange', 'crimson']
    bars = plt.bar(range(len(subset_counts)), subset_counts.values, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}\n({height/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(range(len(subset_counts)), subset_counts.index, rotation=30, ha='right', fontsize=11)
    plt.title("Dataset Subset Distribution", fontsize=15, fontweight='bold')
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/subset_distribution.png")

def viz_comprehensive_eda():
    """Generate comprehensive EDA with multiple panels (Metadata Focus)."""
    print("\n" + "="*60); print("GENERATING METADATA DASHBOARD"); print("="*60)
    df = utils.load_metadata()
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    
    # 1. Gender
    ax1 = fig.add_subplot(gs[0, 0])
    df['gender'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'pink'], alpha=0.8, edgecolor='black')
    ax1.set_title("Gender Distribution")
    
    # 2. Hand
    ax2 = fig.add_subplot(gs[0, 1])
    df['hand'].value_counts().plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
    ax2.set_title("Hand Distribution")
    
    # 3. Subject Counts
    ax3 = fig.add_subplot(gs[0, 2])
    df.groupby('subject_id').size().hist(bins=30, ax=ax3, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax3.set_title("Samples per Subject")
    
    # 4. Alteration Types
    ax4 = fig.add_subplot(gs[1, 0])
    altered = df[df['is_altered'] == 1]
    if not altered.empty:
        types = altered['path'].str.extract(r'_(CR|Obl|Zcut)\.BMP')[0].value_counts()
        types.plot(kind='bar', ax=ax4, color=['gold', 'lightblue', 'lightcoral'], alpha=0.8, edgecolor='black')
    ax4.set_title("Alteration Types")
    
    # 5. Gender x Subset
    ax5 = fig.add_subplot(gs[1, 1])
    pd.crosstab(df['subset'], df['gender']).plot(kind='bar', ax=ax5, color=['skyblue', 'pink'], alpha=0.8, edgecolor='black')
    ax5.set_title("Gender per Subset")
    
    # 6. Hand x Target
    ax6 = fig.add_subplot(gs[1, 2])
    pd.crosstab(df['hand'], df['is_altered']).plot(kind='bar', ax=ax6, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax6.set_title("Hand vs Target")
    
    fig.suptitle("Comprehensive Dataset Metadata Analysis", fontsize=16, fontweight='bold')
    utils.save_fig(f"{config.REPORTS_DIR}/comprehensive_eda.png")


def viz_pattern_recognition():
    """Create pattern-recognition diagnostics for real vs altered samples."""
    print("\n" + "="*60)
    print("GENERATING PATTERN RECOGNITION PANELS")
    print("="*60)

    try:
        sample_rows = pd.concat([
            _sample_metadata(1, target=0, seed=config.SEED),
            _sample_metadata(1, target=1, seed=config.SEED + 1)
        ], ignore_index=True)
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    titles = ["Raw", "Binary Ridge Map", "Skeleton + Minutiae", "Orientation Field"]

    for row_idx, row in enumerate(sample_rows.itertuples(index=False)):
        image = _load_image(row.path)
        patterns = _compute_pattern_maps(image)

        # Raw
        axes[row_idx, 0].imshow(patterns["raw"], cmap='gray')
        axes[row_idx, 0].set_title(f"{titles[0]}\n{subset_label(row)}", fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Binary
        axes[row_idx, 1].imshow(patterns["binary"], cmap='gray')
        axes[row_idx, 1].set_title(titles[1], fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Skeleton overlay
        axes[row_idx, 2].imshow(patterns["raw"], cmap='gray')
        axes[row_idx, 2].contour(patterns["skeleton"], colors='red', linewidths=0.7)
        axes[row_idx, 2].scatter(*np.where(patterns["endings"])[::-1], s=5, c='cyan', label='Endings')
        axes[row_idx, 2].scatter(*np.where(patterns["bifurcations"])[::-1], s=5, c='yellow', label='Bifurcations')
        axes[row_idx, 2].set_title(titles[2], fontweight='bold')
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].legend(loc='lower right', fontsize=8)

        # Orientation
        ori_img = axes[row_idx, 3].imshow(patterns["orientation"], cmap='hsv', vmin=0, vmax=180)
        axes[row_idx, 3].set_title(titles[3], fontweight='bold')
        axes[row_idx, 3].axis('off')
        fig.colorbar(ori_img, ax=axes[row_idx, 3], fraction=0.046, pad=0.04, label='Degrees')

    plt.suptitle("Pattern Recognition Diagnostics (Real vs Altered)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    utils.save_fig(f"{config.REPORTS_DIR}/viz_pattern_recognition.png")


# --- FEATURE ANALYSIS FUNCTIONS ---

def viz_forensic_feature_distributions():
    """Visualize the distributions of the new Forensic Features (Stream C)."""
    print("\n" + "="*60)
    print("GENERATING FORENSIC FEATURE DISTRIBUTIONS")
    print("="*60)
    
    try:
        X_forensic = np.load(f"{config.CACHE_DIR}/forensic/forensic_features.npy")
        df = pd.read_csv(f"{config.CACHE_DIR}/forensic/index.csv")
    except FileNotFoundError:
        print("❌ Forensic features not found. Run 'python -m src.feat_forensic' first.")
        return

    feature_names = [
        "Mean Intensity", "Std Intensity", "Median Intensity", "Ridge Density", 
        "Ridge Thickness Variation", "Orientation Consistency", "Ridge Endings", 
        "Ridge Bifurcations", "Texture Homogeneity", "Dominant Frequency", "Spectral Centroid"
    ]
    
    cols = feature_names[:X_forensic.shape[1]]
    plot_df = pd.DataFrame(X_forensic, columns=cols)
    plot_df['Class'] = df[config.TARGET_COL].map({0: 'Real', 1: 'Altered'})
    
    features_to_plot = cols
    n_cols = 4
    n_rows = int(np.ceil(len(features_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        ax = axes[i]
        sns.violinplot(data=plot_df, x='Class', y=col, ax=ax, hue='Class', palette=['steelblue', 'coral'], inner='quartile', legend=False)
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Value")
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Forensic Feature Discrimination: Real vs Altered", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    utils.save_fig(f"{config.REPORTS_DIR}/viz_forensic_features.png")


def viz_feature_space_projection_3d():
    """Visualize 3D t-SNE projection of the Fused Features."""
    print("\n" + "="*60)
    print("GENERATING 3D FEATURE SPACE PROJECTION (t-SNE)")
    print("="*60)
    
    try:
        X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
        idx = np.random.choice(len(X_eff), 2000, replace=False)
        df = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv").iloc[idx]
        labels = df[config.TARGET_COL].map({0: 'Real', 1: 'Altered'}).values
        subsets = df['subset'].values
        
        print("   Computing PCA (50)...")
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_eff[idx])
        
        print("   Computing 3D t-SNE...")
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        
        fig = plt.figure(figsize=(18, 8))
        
        ax1 = fig.add_subplot(121, projection='3d')
        colors = {'Real': 'steelblue', 'Altered': 'coral'}
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax1.scatter(X_tsne[mask,0], X_tsne[mask,1], X_tsne[mask,2], c=colors[lbl], label=lbl, alpha=0.6, s=20)
        ax1.set_title("3D Feature Space: Real vs Altered", fontweight='bold')
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
        
        ax2 = fig.add_subplot(122, projection='3d')
        subset_colors = {'Real': 'steelblue', 'Altered-Easy': 'lightgreen', 'Altered-Medium': 'orange', 'Altered-Hard': 'crimson'}
        for sub in subset_colors.keys():
            mask = subsets == sub
            if mask.sum() > 0:
                ax2.scatter(X_tsne[mask,0], X_tsne[mask,1], X_tsne[mask,2], c=subset_colors[sub], label=sub, alpha=0.6, s=20)
        ax2.set_title("3D Feature Space: Difficulty Subsets", fontweight='bold')
        ax2.legend()
        ax2.view_init(elev=20, azim=45)
        
        plt.suptitle("3D t-SNE Projection of EfficientNet Features", fontsize=16, fontweight='bold')
        utils.save_fig(f"{config.REPORTS_DIR}/viz_feature_tsne_3d.png")
        
    except Exception as e:
        print(f"⚠️ Skipped t-SNE viz: {e}")


def viz_dimensionality_suite(max_samples: int = 1200):
    """Generate side-by-side PCA, LDA, and t-SNE plots."""
    print("\n" + "="*60)
    print("GENERATING DIMENSIONALITY ANALYSIS (PCA/LDA/t-SNE)")
    print("="*60)

    feat_path = Path(config.CACHE_DIR) / "effb0" / "all.npy"
    idx_path = Path(config.CACHE_DIR) / "effb0" / "index.csv"
    if not feat_path.exists() or not idx_path.exists():
        print("❌ EfficientNet feature cache missing. Run feature extraction first.")
        return

    X = np.load(feat_path)
    df_idx = pd.read_csv(idx_path)
    n = min(max_samples, len(X))
    rng = np.random.default_rng(config.SEED)
    sel = rng.choice(len(X), size=n, replace=False)
    X_sel = X[sel]
    df_sel = df_idx.iloc[sel].reset_index(drop=True)
    y = df_sel[config.TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = {0: '#1f77b4', 1: '#d62728'}

    # PCA
    pca = PCA(n_components=2, random_state=config.SEED)
    X_pca = pca.fit_transform(X_scaled)
    for cls in np.unique(y):
        mask = y == cls
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.6, c=palette[cls], label='Altered' if cls else 'Real')
    axes[0].set_title(f"PCA (var={pca.explained_variance_ratio_.sum():.2f})", fontweight='bold')
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].grid(True, alpha=0.3)

    # LDA
    try:
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X_scaled, y)
        axes[1].scatter(X_lda[y==0], np.zeros_like(X_lda[y==0]), c=palette[0], label='Real', alpha=0.6, s=15)
        axes[1].scatter(X_lda[y==1], np.ones_like(X_lda[y==1]), c=palette[1], label='Altered', alpha=0.6, s=15)
        axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(['Real', 'Altered'])
        axes[1].set_title("LDA Discriminant", fontweight='bold')
        axes[1].set_xlabel('Component 1'); axes[1].grid(True, axis='x', alpha=0.3)
    except Exception as exc:
        axes[1].text(0.5, 0.5, f"LDA failed: {exc}", ha='center', va='center')

    # t-SNE
    pca_50 = PCA(n_components=min(50, X_scaled.shape[1]), random_state=config.SEED)
    X_pca50 = pca_50.fit_transform(X_scaled)
    perplexity = max(5, min(35, n // 5))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=config.SEED)
    X_tsne = tsne.fit_transform(X_pca50)
    for cls in np.unique(y):
        mask = y == cls
        axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=palette[cls], s=12, alpha=0.6)
    axes[2].set_title(f"t-SNE (perplexity={perplexity})", fontweight='bold')
    axes[2].set_xlabel('Dim 1'); axes[2].set_ylabel('Dim 2'); axes[2].grid(True, alpha=0.3)

    handles = [Line2D([0], [0], marker='o', color='w', label='Real', markerfacecolor=palette[0], markersize=8),
               Line2D([0], [0], marker='o', color='w', label='Altered', markerfacecolor=palette[1], markersize=8)]
    axes[0].legend(handles=handles, loc='best')

    plt.suptitle("Dimensionality Analysis: PCA vs LDA vs t-SNE", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    utils.save_fig(f"{config.REPORTS_DIR}/viz_dimensionality_suite.png")


def viz_model_comparison():
    """Generate model comparison chart (Assignment Version - A, B, FUSION, TRIPLE only)."""
    print("\n" + "="*60)
    print("GENERATING MODEL COMPARISON CHART")
    print("="*60)
    
    reports_dir = Path(config.REPORTS_DIR).resolve()
    metrics_files = list(reports_dir.glob("metrics_*.txt"))
    
    data = []
    
    # RESTRICTED MAP - Only assignment streams
    name_map = {
        "A": "Stream A (Deep Features)",
        "B": "Stream B (Texture Features)",
        "FUSION": "FUSION (Deep + Texture)",
        "TRIPLE_FUSION": "TRIPLE (Deep + Texture + Forensic)"
    }
    
    for fpath in metrics_files:
        name = fpath.stem.replace("metrics_", "")
        if name not in name_map: 
            continue  # Skip Stacking/Deep Fusion
        
        with open(fpath, 'r') as f: 
            content = f.read()
        display_name = name_map.get(name, name)
        
        acc = re.search(r"(?:Overall|Test) Accuracy:\s*([\d\.]+)", content)
        f1 = re.search(r"Macro F1-Score:\s*([\d\.]+)", content)
        
        if acc and f1:
            data.append({'Model': display_name, 'Accuracy': float(acc.group(1)), 'Macro F1': float(f1.group(1))})

    if not data:
        print("❌ No metrics found for assignment streams.")
        return

    df_res = pd.DataFrame(data).set_index('Model')
    
    # Sort in logical order
    desired_order = [name_map[k] for k in ["A", "B", "FUSION", "TRIPLE_FUSION"] if name_map[k] in df_res.index]
    df_res = df_res.reindex(desired_order)
    
    ax = df_res.plot(kind='bar', figsize=(12, 6), width=0.8, colormap='viridis', edgecolor='black')
    plt.title("Model Performance Comparison", fontsize=16, fontweight='bold')
    plt.ylabel("Score (0-1)", fontsize=12)
    plt.ylim(0.5, 1.05)
    plt.xticks(rotation=30, ha='right', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
        
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/viz_model_comparison_assignment.png")


def main():
    utils.ensure_dirs()
    
    # Standard EDA
    viz_preprocessing()
    viz_preprocessing_metrics()
    viz_class_balance()
    viz_subset_distribution()
    viz_comprehensive_eda()
    viz_pattern_recognition()
    
    # Feature Analysis
    viz_forensic_feature_distributions()
    viz_feature_space_projection_3d()
    viz_dimensionality_suite()
    
    # Model Comparison (Assignment Version - No Deep Fusion/Stacking)
    viz_model_comparison()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED (Assignment Version)")
    print("="*60)


if __name__ == "__main__":
    main()
