"""
Feature fusion and model training (Assignment Version).
Implements PCA -> LDA -> Weighted XGBoost with OPTIMIZED Hyperparameter Tuning.
Supports 4 Streams: A, B, FUSION, TRIPLE_FUSION only.
Clean version without Stacking for assignment submission.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import argparse
import os
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from . import config, utils


def _select(df, X, subject_ids):
    mask = df.subject_id.isin(subject_ids).to_numpy()
    return X[mask], df[mask]


def train_xgboost_pipeline(X_train, y_train, X_val, y_val, classes, stream_name, pca_components):
    """Generic trainer for XGBoost streams."""
    print(f"\n🚀 Training XGBoost Pipeline for {stream_name}...")
    
    n_real = np.sum(y_train == 0)
    n_altered = np.sum(y_train == 1)
    ratio = float(n_altered) / n_real
    
    print(f"   Imbalance Ratio: {ratio:.2f}. Tuning scale_pos_weight...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_components, random_state=config.SEED)),
        ('lda', LDA(n_components=1)),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            n_jobs=config.N_JOBS,
            random_state=config.SEED,
            tree_method='hist'
        ))
    ])

    # Tuned Grid for BALANCE
    param_dist = {
        'xgb__n_estimators': [300, 500, 800],
        'xgb__learning_rate': [0.03, 0.05, 0.1],
        'xgb__max_depth': [4, 6],
        'xgb__scale_pos_weight': [1.0, 2.0, 3.0, 4.0],
        'xgb__subsample': [0.8],
        'xgb__colsample_bytree': [0.8]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=12, 
        scoring='f1_macro', 
        cv=cv, 
        n_jobs=1,  # CRITICAL FIX FOR WINDOWS [WinError 1450]
        verbose=1,
        random_state=config.SEED
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    print(f"\n✅ Best CV F1-Macro: {search.best_score_:.4f}")
    print("🏆 Best Hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"   {param}: {value}")

    print("\n🎯 Tuning Decision Threshold for Precision/Recall Balance...")
    y_proba_val = best_model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = -1
    val_metrics = {}
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_thresh = (y_proba_val >= threshold).astype(int)
        
        acc = accuracy_score(y_val, y_pred_thresh)
        
        # Manual stats for Real (0)
        tp = np.sum((y_pred_thresh == 0) & (y_val == 0))
        fp = np.sum((y_pred_thresh == 0) & (y_val == 1))
        fn = np.sum((y_pred_thresh == 1) & (y_val == 0))
        
        rec = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        
        if acc < 0.90: continue 
        
        # Score: Harmonic Mean of Prec and Rec
        f1_real = 2 * (prec * rec) / (prec + rec + 1e-6)
        
        if f1_real > best_score:
            best_score = f1_real
            best_threshold = threshold
            val_metrics = {"accuracy": acc, "recall": rec, "precision": prec}
            
    print(f"✅ Selected Threshold: {best_threshold:.3f}")
    print(f"   Val Real Recall:    {val_metrics.get('recall',0):.4f}")
    print(f"   Val Real Precision: {val_metrics.get('precision',0):.4f}")

    model_path = f"models/{stream_name.lower()}_xgboost.joblib"
    joblib.dump({
        "model": best_model,
        "classes": classes,
        "stream": stream_name,
        "threshold": best_threshold,
        "best_params": search.best_params_,
        "val_metrics": val_metrics
    }, model_path)
    print(f"💾 Saved: {model_path}")


def main(stream):
    utils.ensure_dirs()
    utils.set_seeds()

    df_eff = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
    X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    
    if stream in ["B", "FUSION", "TRIPLE_FUSION"]:
        try:
            X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy")
            df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index_enhanced.csv")
        except:
            X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
            df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index.csv")

    if stream == "TRIPLE_FUSION":
        try:
            X_forensic = np.load(f"{config.CACHE_DIR}/forensic/forensic_features.npy")
        except FileNotFoundError:
            print("❌ Error: Run 'python -m src.feat_forensic' first!")
            return

    # Select Features
    if stream == "A":
        X, df, pca = X_eff, df_eff, 250
    elif stream == "B":
        X, df, pca = X_gabor, df_gabor, 80
    elif stream == "FUSION":
        X = np.concatenate([X_eff, X_gabor], axis=1)
        df, pca = df_eff, 300
    elif stream == "TRIPLE_FUSION":
        X = np.concatenate([X_eff, X_gabor, X_forensic], axis=1)
        df, pca = df_eff, 300
        
    X_train, df_train = _select(df, X, train_ids)
    X_val, df_val = _select(df, X, val_ids)
    
    y_train, classes, _ = utils.label_encode(df_train[config.TARGET_COL])
    y_val, *_ = utils.label_encode(df_val[config.TARGET_COL])
    
    train_xgboost_pipeline(X_train, y_train, X_val, y_val, classes, stream, pca)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RESTRICTED CHOICES for Assignment (No Stacking)
    parser.add_argument("--stream", choices=["A", "B", "FUSION", "TRIPLE_FUSION"], required=True)
    args = parser.parse_args()
    main(args.stream)
