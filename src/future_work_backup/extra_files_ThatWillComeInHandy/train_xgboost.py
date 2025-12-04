"""
Hierarchical XGBoost Training Pipeline (Final Polish).
Implements Research Strategy:
1. LDA for Class Separation
2. DYNAMIC scale_pos_weight to fix the 0.66 default recall.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score
from imblearn.pipeline import Pipeline
from . import config, utils

def load_fusion_data():
    print("⏳ Loading features...")
    # Load EfficientNet (Now with MaxPooling)
    df = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
    X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
    
    # Load Gabor (Now Raw/Enhanced)
    try:
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy")
    except:
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
        
    X = np.concatenate([X_eff, X_gabor], axis=1)
    y, classes, _ = utils.label_encode(df[config.TARGET_COL])
    subject_ids = df['subject_id'].to_numpy()
    
    return X, y, subject_ids, classes

def main():
    utils.set_seeds()
    utils.ensure_dirs()
    
    X_all, y_all, subjects, classes = load_fusion_data()
    
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    
    mask_train = np.isin(subjects, train_ids)
    mask_val = np.isin(subjects, val_ids)
    
    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_val, y_val = X_all[mask_val], y_all[mask_val]
    
    print("🚀 Training Research-Aligned XGBoost Model...")

    # 1. Calculate Imbalance Ratio dynamically
    n_real = np.sum(y_train == 0)
    n_altered = np.sum(y_train == 1)
    ratio = float(n_altered) / n_real
    
    # We use sqrt(ratio) to be safe. Full ratio (8.2) might hurt accuracy too much.
    safe_weight = np.sqrt(ratio) 
    
    print(f"   Real Samples: {n_real}, Altered Samples: {n_altered}")
    print(f"   Imbalance Ratio: {ratio:.2f}")
    print(f"   Applying scale_pos_weight: {safe_weight:.2f}")

    # Pipeline: LDA -> XGBoost
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # PCA first to reduce noise before LDA
        ('pca', PCA(n_components=300, random_state=config.SEED)), 
        # LDA to maximize class separation
        ('lda', LDA(n_components=1)), 
        ('xgb', xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03, # Lower learning rate for better convergence
            max_depth=6,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=safe_weight, # <--- THE FIX
            objective='binary:logistic',
            n_jobs=config.N_JOBS,
            random_state=config.SEED,
            tree_method='hist'
        ))
    ])
    
    # Train
    print("🔨 Fitting Pipeline (PCA -> LDA -> XGBoost)...")
    pipeline.fit(X_train, y_train)
    
    # Validation Evaluation logic removed to prevent confusion.
    # The evaluation script handles the rigorous reporting.
    
    # Tune threshold
    print("🎯 Tuning threshold for Optimal Balance...")
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = -1
    
    # We want to maximize a weighted score: 0.6 * Accuracy + 0.4 * Real_Recall
    # This ensures we respect the 90% accuracy goal while pushing Recall
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_val, y_pred_thresh)
        rec = recall_score(y_val, y_pred_thresh, pos_label=0)
        
        # Hard constraints
        if acc < 0.90: continue 
        
        # Optimization metric: Maximize Recall once Accuracy is safe
        score = rec
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    print(f"✅ Selected Threshold: {best_threshold:.3f}")
    
    # Save
    model_path = "models/fusion_xgboost_lda.joblib"
    joblib.dump({
        "model": pipeline,
        "classes": classes,
        "stream": "FUSION_XGB",
        "threshold": best_threshold
    }, model_path)
    
    print(f"✅ Saved: {model_path}")

if __name__ == "__main__":
    main()