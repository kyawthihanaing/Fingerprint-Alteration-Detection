"""
Model evaluation and performance metrics (Consolidated).
Handles Single Streams, Feature Fusion, Late Fusion (Stacking), and End-to-End Deep Learning.
Preserves all visualizations (Bootstrap, ROC, PR, Confusion Matrix).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import cv2
from . import config, utils, preprocess

# --- HELPER FUNCTIONS ---
def _compute_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return np.array(specificity)

def _compute_bootstrap_ci(y_true, y_pred, subjects, n_bootstrap=1000, confidence_level=0.95, seed=42):
    np.random.seed(seed)
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    acc_boots, f1_boots = [], []
    for i in range(n_bootstrap):
        boot_subjects = np.random.choice(unique_subjects, size=n_subjects, replace=True)
        boot_indices = []
        for subj in boot_subjects:
            indices = np.where(subjects == subj)[0]
            boot_indices.extend(indices)
        boot_indices = np.array(boot_indices)
        y_true_boot = y_true[boot_indices]
        y_pred_boot = y_pred[boot_indices]
        acc_boots.append(accuracy_score(y_true_boot, y_pred_boot))
        f1_boots.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
    
    alpha = 1 - confidence_level
    acc_ci = np.percentile(acc_boots, [(alpha/2)*100, (1-alpha/2)*100])
    f1_ci = np.percentile(f1_boots, [(alpha/2)*100, (1-alpha/2)*100])
    
    return {
        'accuracy': {'ci_lower': acc_ci[0], 'ci_upper': acc_ci[1], 'samples': np.array(acc_boots)},
        'macro_f1': {'ci_lower': f1_ci[0], 'ci_upper': f1_ci[1], 'samples': np.array(f1_boots)},
        'n_bootstrap': n_bootstrap, 'n_subjects': n_subjects
    }

def _plot_bootstrap_distributions(bootstrap_ci, stream_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(bootstrap_ci['accuracy']['samples'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_ci['accuracy']['samples'].mean(), color='darkred', linestyle='--', linewidth=2.5)
    ax.set_title(f'Bootstrap: Accuracy\nStream {stream_name}', fontsize=14, fontweight='bold')
    ax = axes[1]
    ax.hist(bootstrap_ci['macro_f1']['samples'], bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_ci['macro_f1']['samples'].mean(), color='darkred', linestyle='--', linewidth=2.5)
    ax.set_title(f'Bootstrap: Macro-F1\nStream {stream_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/bootstrap_ci_{stream_name}.png")

def evaluate_model(model_data, X, y_true, df, stream_name, y_proba_external=None):
    print("\n" + "="*60)
    print(f"EVALUATING STREAM {stream_name}")
    print("="*60)
    
    model = model_data["model"]
    classes = model_data["classes"]
    thr = model_data.get("threshold", None)
    
    # Prediction Logic
    if y_proba_external is not None:
        y_proba = y_proba_external
        y_pred = (y_proba >= (thr if thr else 0.5)).astype(int) 
    else:
        if "scaler" in model_data and not isinstance(model["model"] if isinstance(model, dict) else model, Pipeline):
            X = model_data["scaler"].transform(X)
        if "pca" in model_data and not isinstance(model["model"] if isinstance(model, dict) else model, Pipeline):
            X = model_data["pca"].transform(X)
            
        if hasattr(model, "predict_proba") and len(classes) == 2:
            y_proba = model.predict_proba(X)[:,1]
            if thr is not None:
                print(f"   Using tuned threshold: {thr:.3f}")
                y_pred = (y_proba >= thr).astype(int)
            else:
                y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X)
            y_proba = None

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    spec = _compute_specificity(y_true, y_pred, labels=range(len(classes))).mean()
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    
    roc_auc_val = None
    if y_proba is not None:
        roc_auc_val = auc(*roc_curve(y_true, y_proba)[:2])
        print(f"ROC-AUC: {roc_auc_val:.4f}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0))

    # Save Metrics
    report_path = f"{config.REPORTS_DIR}/metrics_{stream_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"STREAM {stream_name} Evaluation Metrics\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n")
        if roc_auc_val: f.write(f"ROC-AUC: {roc_auc_val:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0))
    print(f"✅ Metrics saved: {report_path}")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(c) for c in classes], yticklabels=[str(c) for c in classes])
    plt.title(f"Confusion Matrix – Stream {stream_name}")
    utils.save_fig(f"{config.REPORTS_DIR}/confusion_matrix_{stream_name}.png")

    # 2. Subset Accuracy (With Labels)
    def get_difficulty_level(path_str):
        if pd.isna(path_str): return "Unknown"
        s = str(path_str)
        if "Real" in s and "Altered" not in s: return "Real"
        if "Altered-Easy" in s: return "Altered-Easy"
        if "Altered-Medium" in s: return "Altered-Medium"
        if "Altered-Hard" in s: return "Altered-Hard"
        return "Other"
    
    df_copy = df.copy()
    df_copy["difficulty"] = df_copy["path"].apply(get_difficulty_level)
    subsets = df_copy["difficulty"].unique()
    subset_accs = [(s, accuracy_score(y_true[df_copy["difficulty"]==s], y_pred[df_copy["difficulty"]==s])) 
                   for s in subsets if (df_copy["difficulty"]==s).sum() > 0]
    
    if subset_accs:
        names, accs = zip(*subset_accs)
        plt.figure(figsize=(8, 4))
        bars = plt.bar(names, accs, color="steelblue")
        
        # Add Labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
        plt.title(f"Per-Subset Accuracy – Stream {stream_name}")
        plt.ylim(0, 1.1) # Extra space for labels
        utils.save_fig(f"{config.REPORTS_DIR}/subset_accuracy_{stream_name}.png")

    # 3. ROC & PR Curves (For ANY stream with probabilities)
    if y_proba is not None:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_val:.3f})", linewidth=2)
        plt.plot([0,1], [0,1], "k--")
        plt.title(f"ROC Curve – Stream {stream_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        utils.save_fig(f"{config.REPORTS_DIR}/roc_curve_{stream_name}.png")
        
        # Precision-Recall
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_curve, prec_curve, label=f"PR (AP={pr_auc_val:.3f})", linewidth=2, color='green')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve – Stream {stream_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        utils.save_fig(f"{config.REPORTS_DIR}/pr_curve_{stream_name}.png")
        print(f"📊 Saved PR Curve: {config.REPORTS_DIR}/pr_curve_{stream_name}.png")

    # 4. Bootstrap
    print("🔄 Computing bootstrap CIs...")
    subjects = df["subject_id"].to_numpy()
    bootstrap_ci = _compute_bootstrap_ci(y_true, y_pred, subjects, n_bootstrap=1000)
    _plot_bootstrap_distributions(bootstrap_ci, stream_name)
    print("✅ Bootstrap plots saved.")


def load_features(stream):
    """Helper to load features for any stream."""
    path_eff = f"{config.CACHE_DIR}/effb0"
    path_gab = f"{config.CACHE_DIR}/gabor"
    path_for = f"{config.CACHE_DIR}/forensic"

    if stream == "A":
        return np.load(f"{path_eff}/all.npy")
    elif stream == "B":
        try: return np.load(f"{path_gab}/gabor_enhanced.npy")
        except: return np.load(f"{path_gab}/gabor.npy")
    elif stream == "FUSION":
        X_eff = np.load(f"{path_eff}/all.npy")
        try: X_gab = np.load(f"{path_gab}/gabor_enhanced.npy")
        except: X_gab = np.load(f"{path_gab}/gabor.npy")
        return np.concatenate([X_eff, X_gab], axis=1)
    elif stream == "TRIPLE_FUSION":
        X_eff = np.load(f"{path_eff}/all.npy")
        try: X_gab = np.load(f"{path_gab}/gabor_enhanced.npy")
        except: X_gab = np.load(f"{path_gab}/gabor.npy")
        X_for = np.load(f"{path_for}/forensic_features.npy")
        return np.concatenate([X_eff, X_gab, X_for], axis=1)
    return None

def evaluate_deep_fusion(test_ids):
    print("\n" + "="*60)
    print("EVALUATING STREAM DEEP_FUSION")
    print("="*60)
    
    model_path = "models/deep_fusion_best.keras"
    if not Path(model_path).exists():
        print("❌ Model not found. Run src.train_deep_fusion first.")
        return

    print(f"Loading Keras model: {model_path}")
    model = load_model(model_path)
    
    # Load metadata
    df = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv") 
    df_test = df[df.subject_id.isin(test_ids)].reset_index(drop=True)
    
    print(f"Test samples: {len(df_test)}")
    
    # Generate predictions
    gen = FusionDataGenerator(df_test, batch_size=32, shuffle=False)
    y_true = []
    y_pred_proba = []
    
    for i in range(len(gen)):
        X_batch, y_batch = gen[i]
        preds = model.predict_on_batch(X_batch)
        y_true.extend(y_batch)
        y_pred_proba.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Create fake model_data dict to reuse evaluate_model logic
    fake_model_data = {
        "model": None, 
        "classes": ["Real", "Altered"],
        "threshold": 0.5
    }
    
    evaluate_model(fake_model_data, None, y_true, df_test, "DEEP_FUSION", y_proba_external=y_pred_proba)

# --- FusionDataGenerator must be available for evaluate_deep_fusion ---
class FusionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, shuffle=False):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        X_eff, X_tex, y = [], [], []
        for _, row in batch_df.iterrows():
            img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
            eff_img = preprocess.preprocess_for_effnet(img)
            eff_img = preprocess_input(eff_img)
            tex_img = preprocess.preprocess_for_texture(img)
            tex_img = tex_img / 255.0
            tex_img = np.expand_dims(tex_img, axis=-1)
            X_eff.append(eff_img); X_tex.append(tex_img); y.append(row[config.TARGET_COL])
        return ({"Input_EffNet": np.array(X_eff), "Input_Texture": np.array(X_tex)}, np.array(y))

def main(stream, pca_dim=None):
    utils.ensure_dirs()
    test_ids = utils.read_json(f"{config.SPLIT_DIR}/test_subjects.json")
    
    if stream == "DEEP_FUSION":
        evaluate_deep_fusion(test_ids)
        return
    
    if stream == "STACKING":
        model_path = "models/stacking_xgboost.joblib"
        print(f"Loading Stacking Model: {model_path}")
        model_data = joblib.load(model_path)
        base_models = model_data["base_models"]
        meta_model = model_data["model"]
        
        df = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
        mask_test = df.subject_id.isin(test_ids).to_numpy()
        df_test = df[mask_test].reset_index(drop=True)
        
        classes = model_data["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_test = df_test[config.TARGET_COL].map(class_to_idx).to_numpy()
        
        print("🤖 Generating Meta-Features for Stacking Evaluation...")
        P_test = pd.DataFrame()
        for name, path in base_models.items():
            print(f"   Getting predictions from Stream {name}...")
            X_stream = load_features(name)
            base_model_data = joblib.load(path)
            model = base_model_data["model"]
            probs = model.predict_proba(X_stream[mask_test])[:, 1]
            P_test[name] = probs
            
        y_proba_final = meta_model.predict_proba(P_test)[:, 1]
        evaluate_model(model_data, None, y_test, df_test, stream, y_proba_external=y_proba_final)
        return

    # Standard Streams
    xgb_path = f"models/{stream.lower()}_xgboost.joblib"
    if not Path(xgb_path).exists():
        print(f"❌ Model not found: {xgb_path}")
        return
        
    print(f"Loading model: {xgb_path}")
    model_data = joblib.load(xgb_path)
    
    path_eff = f"{config.CACHE_DIR}/effb0"
    path_gab = f"{config.CACHE_DIR}/gabor"
    path_for = f"{config.CACHE_DIR}/forensic"
    
    df_eff = pd.read_csv(f"{path_eff}/index.csv")
    X_eff = np.load(f"{path_eff}/all.npy")
    
    if stream == "A":
        X, df = X_eff, df_eff
    elif stream == "B":
        try:
            X = np.load(f"{path_gab}/gabor_enhanced.npy")
            df = pd.read_csv(f"{path_gab}/index_enhanced.csv")
        except:
            X = np.load(f"{path_gab}/gabor.npy")
            df = pd.read_csv(f"{path_gab}/index.csv")
    elif stream == "FUSION":
        try: Xg = np.load(f"{path_gab}/gabor_enhanced.npy")
        except: Xg = np.load(f"{path_gab}/gabor.npy")
        X = np.concatenate([X_eff, Xg], axis=1)
        df = df_eff
    elif stream == "TRIPLE_FUSION":
        try: Xg = np.load(f"{path_gab}/gabor_enhanced.npy")
        except: Xg = np.load(f"{path_gab}/gabor.npy")
        Xf = np.load(f"{path_for}/forensic_features.npy")
        X = np.concatenate([X_eff, Xg, Xf], axis=1)
        df = df_eff

    mask_test = df.subject_id.isin(test_ids).to_numpy()
    X_test = X[mask_test]
    df_test = df[mask_test].reset_index(drop=True)
    
    classes = model_data["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = df_test[config.TARGET_COL].map(class_to_idx).to_numpy()
    
    if isinstance(model_data["model"], Pipeline):
        if "scaler" in model_data: del model_data["scaler"]
        if "pca" in model_data: del model_data["pca"]

    evaluate_model(model_data, X_test, y_test, df_test, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", choices=["A", "B", "FUSION", "TRIPLE_FUSION", "STACKING", "DEEP_FUSION"], required=True)
    parser.add_argument("--pca", type=int, default=None)
    args = parser.parse_args()
    main(args.stream, args.pca)