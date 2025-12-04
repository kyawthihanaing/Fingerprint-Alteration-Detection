"""
Enhanced Stream B training with improved Gabor features and Random Forest classifier.
Addresses the performance issues in the original Stream B implementation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from . import config, utils


def _select(df, X, subject_ids):
    """Select samples belonging to given subjects."""
    mask = df.subject_id.isin(subject_ids)
    return X[mask], df[mask]


def stream_B_enhanced():
    """
    Train Enhanced Stream B: Improved Gabor texture features.
    Uses Random Forest classifier with comprehensive 93-D feature set.
    """
    print("\n" + "="*60)
    print("STREAM B ENHANCED: Improved Gabor Texture Features")
    print("="*60)
    
    try:
        # Load enhanced data
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index_enhanced.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor_enhanced.npy") 
        print(f"Loaded enhanced Gabor features: {X_gabor.shape}")
    except FileNotFoundError:
        print("❌ Enhanced Gabor features not found!")
        print("   Please run: python -m src.feat_gabor_enhanced first")
        return
    
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
    print(f"Feature dimensionality: {X_train.shape[1]}")
    
    # Feature scaling (important for Random Forest with diverse feature scales)
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Random Forest (much better than Naive Bayes for this problem)
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=500,     # Sufficient trees for stability
        max_depth=15,         # Prevent overfitting with limited depth
        min_samples_split=10, # Conservative splitting
        min_samples_leaf=5,   # Conservative leaf size
        random_state=config.SEED,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_val_scaled)
    f1 = f1_score(y_val, y_pred, average="macro")
    
    print(f"Validation Macro-F1: {f1:.4f}")
    
    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Feature importance analysis
    feature_names = []
    # Basic Gabor features (40 filters × 2 stats)
    freqs = [0.05, 0.1, 0.15, 0.2, 0.25]
    orients = [f"{i*22.5:.1f}°" for i in range(8)]
    for f_idx, freq in enumerate(freqs):
        for o_idx, orient in enumerate(orients):
            feature_names.extend([
                f"Gabor_f{freq}_o{orient}_mean",
                f"Gabor_f{freq}_o{orient}_std"
            ])
    
    # Advanced texture statistics (13 features)
    texture_names = ['energy', 'entropy', 'contrast'] + \
                   [f'max_resp_f{freq}' for freq in freqs] + \
                   [f'coherence_f{freq}' for freq in freqs]
    feature_names.extend(texture_names)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Print top 15 most important features
    print(f"\nTop 15 Most Important Features:")
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance_pairs[:15]):
        print(f"{i+1:2d}. {name:<30} {importance:.4f}")
    
    # Save enhanced model
    model_path = "models/gabor_enhanced_rf.joblib"
    joblib.dump({
        "model": rf,
        "scaler": scaler,
        "classes": classes,
        "stream": "B_Enhanced",
        "feature_names": feature_names,
        "n_features": X_train.shape[1]
    }, model_path)
    print(f"\n✅ Enhanced model saved: {model_path}")
    
    return f1


def compare_stream_B_versions():
    """
    Compare original vs enhanced Stream B performance.
    """
    print("\n" + "="*60)
    print("STREAM B COMPARISON: Original vs Enhanced")
    print("="*60)
    
    # Try to load both models and compare
    try:
        # Original model
        original_model = joblib.load("models/gabor_nb.joblib")
        print("✅ Original Stream B (Naive Bayes) model found")
        
        # Enhanced model  
        enhanced_model = joblib.load("models/gabor_enhanced_rf.joblib")
        print("✅ Enhanced Stream B (Random Forest) model found")
        
        print(f"\nModel Comparison:")
        print(f"Original:  {original_model['stream']}, Naive Bayes, ? features")
        print(f"Enhanced:  {enhanced_model['stream']}, Random Forest, {enhanced_model['n_features']} features")
        
        # Load test data for both and compare
        # This would require running evaluation on both models
        print("\n💡 Run evaluation on both models to see performance difference:")
        print("   python -m src.evaluate --stream B --model original")
        print("   python -m src.evaluate --stream B --model enhanced")
        
    except FileNotFoundError as e:
        print(f"❌ Model comparison not possible: {e}")


if __name__ == "__main__":
    # Train enhanced Stream B
    enhanced_f1 = stream_B_enhanced()
    
    # Compare versions
    compare_stream_B_versions()