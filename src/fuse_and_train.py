"""
Feature fusion and model training.
Implements Stream A (EfficientNet), Stream B (Gabor), and FUSION strategies.
Includes PCA visualization and Random Forest feature importance plots.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from . import config, utils


def _select(df, X, subject_ids):
    """Select samples for specific subjects."""
    mask = df.subject_id.isin(subject_ids).to_numpy()
    return X[mask], df[mask]


def _pca_plot(pca, name):
    """Plot cumulative explained variance for PCA."""
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(7, 4))
    plt.plot(np.cumsum(var), linewidth=2)
    plt.xlabel("Number of Components", fontsize=12)
    plt.ylabel("Cumulative Explained Variance", fontsize=12)
    plt.title(f"PCA Variance Explained – {name}", fontsize=14)
    plt.grid(alpha=0.3)
    utils.save_fig(f"{config.REPORTS_DIR}/pca_variance_{name}.png")


def stream_A(pca_dim):
    """
    Train Stream A: EfficientNet features only.
    Uses PCA for dimensionality reduction + Random Forest classifier.
    """
    print("\n" + "="*60)
    print("STREAM A: EfficientNet Features")
    print("="*60)
    
    # Load data (index.csv now includes is_altered column)
    df_eff = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
    X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
    print(f"Loaded EfficientNet features: {X_eff.shape}")
    
    # Load splits
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    
    # Select train/val data
    X_train, df_train = _select(df_eff, X_eff, train_ids)
    X_val, df_val = _select(df_eff, X_eff, val_ids)
    
    # Encode labels
    y_train, classes, _ = utils.label_encode(df_train[config.TARGET_COL])
    y_val, *_ = utils.label_encode(df_val[config.TARGET_COL])
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Classes: {classes}")
    
    # Standardize
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # PCA
    print(f"Applying PCA (n_components={pca_dim})...")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=config.SEED)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    _pca_plot(pca, f"A_{pca_dim}")
    
    # Hyperparameter grid for Random Forest (Stream A)
    from sklearn.model_selection import GridSearchCV
    print("Training Random Forest with GridSearchCV...")
    rf_grid = {
        "n_estimators": [400, 600, 800],
        "max_depth": [32, 48, 64],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [{0: 2.5, 1: 1.0}],
        "random_state": [config.SEED]
    }
    rf_base = RandomForestClassifier(n_jobs=config.N_JOBS)
    grid = GridSearchCV(rf_base, rf_grid, scoring="f1_macro", cv=3, n_jobs=config.N_JOBS, verbose=1)
    grid.fit(X_train_pca, y_train)
    rf = grid.best_estimator_
    print(f"Best RF params: {grid.best_params_}")
    # Evaluate
    y_pred = rf.predict(X_val_pca)
    f1 = f1_score(y_val, y_pred, average="macro")
    print(f"Validation Macro-F1: {f1:.4f}")
    # Fixed threshold for Stream A
    threshold = 0.775
    # Save model with dynamic filename
    model_path = f"models/effb0_pca{pca_dim}_rf.joblib"
    joblib.dump({
        "model": rf,
        "scaler": scaler,
        "pca": pca,
        "classes": classes,
        "stream": "A",
        "threshold": threshold
    }, model_path)
    print(f"✅ Model saved: {model_path}")


def stream_B():
    """
    Train Stream B: Enhanced Gabor texture features (93-D).
    Uses Random Forest with feature scaling and threshold tuning.
    """
    print("\n" + "="*60)
    print("STREAM B: Enhanced Gabor Texture Features (93-D)")
    print("="*60)
    
    # Try to load enhanced features first, fallback to original
    try:
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index_enhanced.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy")
        print(f"✅ Loaded ENHANCED Gabor features: {X_gabor.shape}")
        enhanced = True
    except FileNotFoundError:
        print("⚠️  Enhanced features not found, using original Gabor features...")
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
        print(f"Loaded original Gabor features: {X_gabor.shape}")
        enhanced = False
    
    # Load splits
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    
    # Select train/val data
    X_train, df_train = _select(df_gabor, X_gabor, train_ids)
    X_val, df_val = _select(df_gabor, X_gabor, val_ids)
    
    # Encode labels
    y_train, classes, _ = utils.label_encode(df_train[config.TARGET_COL])
    y_val, *_ = utils.label_encode(df_val[config.TARGET_COL])
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Classes: {classes}")
    
    # Feature scaling (important for Random Forest with high-dimensional features)
    print("Applying feature scaling...")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Hyperparameter grid for Random Forest (Stream B)
    from sklearn.model_selection import GridSearchCV
    print("Training Random Forest with GridSearchCV...")
    rf_grid = {
        "n_estimators": [300, 400, 600],
        "max_depth": [16, 24, 32],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [{0: 2.0, 1: 1.0}],
        "random_state": [config.SEED]
    }
    rf_base = RandomForestClassifier(n_jobs=config.N_JOBS)
    grid = GridSearchCV(rf_base, rf_grid, scoring="f1_macro", cv=3, n_jobs=config.N_JOBS, verbose=1)
    grid.fit(X_train_scaled, y_train)
    rf = grid.best_estimator_
    print(f"Best RF params: {grid.best_params_}")
    # Evaluate
    y_pred = rf.predict(X_val_scaled)
    f1 = f1_score(y_val, y_pred, average="macro")
    print(f"Validation Macro-F1: {f1:.4f}")
    # Fixed threshold for Stream B
    threshold = 0.725
    if enhanced and X_gabor.shape[1] >= 20:
        print("Plotting feature importances...")
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(top_indices)), importances[top_indices], color="forestgreen", alpha=0.7)
        
        # Add numeric labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)
        
        # Create meaningful feature names
        feature_names = []
        for idx in top_indices:
            if idx < 80:  # Basic Gabor features (40 filters × 2 stats)
                filter_idx = idx // 2
                stat_type = "Mean" if idx % 2 == 0 else "Std"
                freq_idx = filter_idx // 8
                orient_idx = filter_idx % 8
                feature_names.append(f"G{freq_idx}{orient_idx}_{stat_type}")
            else:  # Advanced texture statistics (13 features)
                advanced_idx = idx - 80
                if advanced_idx < 3:
                    feature_names.append(["Energy", "Entropy", "Contrast"][advanced_idx])
                elif advanced_idx < 8:
                    feature_names.append(f"MaxResp_F{advanced_idx-3}")
                else:
                    feature_names.append(f"Coherence_F{advanced_idx-8}")
        
        plt.xlabel("Feature Index", fontsize=12)
        plt.ylabel("Importance", fontsize=12)
        plt.title("Enhanced Gabor Features - Random Forest Importances (Top 20)", fontsize=14)
        plt.xticks(range(len(top_indices)), feature_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        utils.save_fig(f"{config.REPORTS_DIR}/rf_feature_importance_gabor_enhanced.png")
    
    # Save model
    suffix = "_enhanced" if enhanced else ""
    model_path = f"models/gabor{suffix}_rf.joblib"
    joblib.dump({
        "model": rf,
        "scaler": scaler,
        "classes": classes,
        "stream": "B",
        "enhanced": enhanced,
        "threshold": threshold
    }, model_path)
    print(f"✅ Model saved: {model_path}")


def stream_FUSION(pca_dim):
    """
    Train FUSION: Concatenated EfficientNet + Gabor features.
    Uses PCA for dimensionality reduction + Random Forest classifier.
    Also plots feature importances.
    """
    print("\n" + "="*60)
    print("STREAM FUSION: EfficientNet + Gabor")
    print("="*60)
    
    # Load data (index.csv now includes is_altered column)
    df_eff = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
    X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
    
    # Try to load enhanced Gabor features first, fallback to original
    try:
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index_enhanced.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy")
        print(f"✅ Using ENHANCED Gabor features: {X_gabor.shape}")
        enhanced = True
    except FileNotFoundError:
        print("⚠️  Enhanced features not found, using original Gabor features...")
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
        enhanced = False
    
    # Ensure alignment
    assert (df_eff["path"] == df_gabor["path"]).all(), "Feature indices must align!"
    
    # Use df_eff as the index (already has is_altered column)
    df = df_eff
    
    # Concatenate features
    X = np.concatenate([X_eff, X_gabor], axis=1)
    print(f"Fused features shape: {X.shape}")
    
    # Load splits
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    
    # Create masks
    mask_train = df.subject_id.isin(train_ids).to_numpy()
    mask_val = df.subject_id.isin(val_ids).to_numpy()
    
    # Encode all labels
    y_all, classes, _ = utils.label_encode(df[config.TARGET_COL])
    
    print(f"Train samples: {mask_train.sum()}, Val samples: {mask_val.sum()}")
    print(f"Classes: {classes}")
    
    # Standardize
    scaler = StandardScaler().fit(X[mask_train])
    X_scaled = scaler.transform(X)
    
    # PCA
    print(f"Applying PCA (n_components={pca_dim})...")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=config.SEED)
    pca.fit(X_scaled[mask_train])
    X_pca = pca.transform(X_scaled)
    
    _pca_plot(pca, f"FUSION_{pca_dim}")
    
    # Hyperparameter grid for Random Forest (FUSION)
    from sklearn.model_selection import GridSearchCV
    print("Training Random Forest with GridSearchCV...")
    rf_grid = {
        "n_estimators": [400, 600, 800],
        "max_depth": [32, 48, 64],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [{0: 2.5, 1: 1.0}],
        "random_state": [config.SEED]
    }
    rf_base = RandomForestClassifier(n_jobs=config.N_JOBS)
    grid = GridSearchCV(rf_base, rf_grid, scoring="f1_macro", cv=3, n_jobs=config.N_JOBS, verbose=1)
    grid.fit(X_pca[mask_train], y_all[mask_train])
    rf = grid.best_estimator_
    print(f"Best RF params: {grid.best_params_}")
    # Evaluate
    y_pred = rf.predict(X_pca[mask_val])
    f1 = f1_score(y_all[mask_val], y_pred, average="macro")
    print(f"Validation Macro-F1: {f1:.4f}")
    # Fixed threshold for FUSION
    threshold = 0.775
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(top_indices)), importances[top_indices], color="teal", alpha=0.7)
    
    # Add numeric labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)
    
    plt.xlabel("PCA Component Index", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("Random Forest Feature Importances (Top 20)", fontsize=14)
    plt.xticks(range(len(top_indices)), [f'PC{top_indices[i]}' for i in range(len(top_indices))], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/rf_feature_importance_fusion.png")
    
    # Save model with dynamic filename
    suffix = "_enhanced" if enhanced else ""
    model_path = f"models/fusion{suffix}_pca{pca_dim}_rf.joblib"
    joblib.dump({
        "model": rf,
        "scaler": scaler,
        "pca": pca,
        "classes": classes,
        "stream": "FUSION",
        "enhanced": enhanced,
        "threshold": threshold
    }, model_path)
    print(f"✅ Model saved: {model_path}")


def main(stream, pca_dim):
    """Main training function."""
    utils.ensure_dirs()
    utils.set_seeds()
    
    if stream == "A":
        stream_A(pca_dim)
    elif stream == "B":
        stream_B()
    elif stream == "FUSION":
        stream_FUSION(pca_dim)
    else:
        raise ValueError(f"Unknown stream: {stream}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with feature fusion")
    parser.add_argument("--stream", choices=["A", "B", "FUSION"], required=True,
                        help="Stream to train: A (EfficientNet), B (Gabor), or FUSION")
    parser.add_argument("--pca", type=int, default=256,
                        help="Number of PCA components (for A and FUSION)")
    args = parser.parse_args()
    
    main(args.stream, args.pca)
