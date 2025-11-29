"""
Model evaluation and performance metrics.
Computes accuracy, precision, recall, specificity, F1-score, and confusion matrix.
Generates per-subset accuracy bars and ROC/PR curves for binary tasks.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from . import config, utils


def _compute_specificity(y_true, y_pred, labels):
    """Compute specificity for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return np.array(specificity)


def _compute_bootstrap_ci(y_true, y_pred, subjects, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Compute bootstrap confidence intervals by resampling subjects.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        subjects: Subject IDs for each sample
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level (default 0.95)
        seed: Random seed
    
    Returns:
        dict with accuracy and macro-F1 statistics
    """
    np.random.seed(seed)
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    acc_boots = []
    f1_boots = []
    
    for i in range(n_bootstrap):
        # Resample subjects with replacement
        boot_subjects = np.random.choice(unique_subjects, size=n_subjects, replace=True)
        
        # Get all samples for selected subjects
        boot_indices = []
        for subj in boot_subjects:
            indices = np.where(subjects == subj)[0]
            boot_indices.extend(indices)
        
        boot_indices = np.array(boot_indices)
        
        # Compute metrics on bootstrap sample
        y_true_boot = y_true[boot_indices]
        y_pred_boot = y_pred[boot_indices]
        
        acc = accuracy_score(y_true_boot, y_pred_boot)
        f1 = f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        
        acc_boots.append(acc)
        f1_boots.append(f1)
    
    acc_boots = np.array(acc_boots)
    f1_boots = np.array(f1_boots)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    acc_ci = np.percentile(acc_boots, [lower_percentile, upper_percentile])
    f1_ci = np.percentile(f1_boots, [lower_percentile, upper_percentile])
    
    return {
        'accuracy': {
            'ci_lower': acc_ci[0],
            'ci_upper': acc_ci[1],
            'samples': acc_boots
        },
        'macro_f1': {
            'ci_lower': f1_ci[0],
            'ci_upper': f1_ci[1],
            'samples': f1_boots
        },
        'n_bootstrap': n_bootstrap,
        'n_subjects': n_subjects
    }


def _plot_bootstrap_distributions(bootstrap_ci, stream_name):
    """Plot bootstrap distributions with confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax = axes[0]
    ax.hist(bootstrap_ci['accuracy']['samples'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_ci['accuracy']['samples'].mean(), color='darkred', linestyle='--', 
               linewidth=2.5, label='Mean')
    ax.axvline(bootstrap_ci['accuracy']['ci_lower'], color='orange', linestyle='--', 
               linewidth=2, label='95% CI')
    ax.axvline(bootstrap_ci['accuracy']['ci_upper'], color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'Subject-Bootstrap: Accuracy\nStream {stream_name} (n=1000 resamples)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add CI text box
    ci_text = f"95% CI:\n[{bootstrap_ci['accuracy']['ci_lower']:.4f}, {bootstrap_ci['accuracy']['ci_upper']:.4f}]"
    ax.text(0.02, 0.98, ci_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Macro-F1
    ax = axes[1]
    ax.hist(bootstrap_ci['macro_f1']['samples'], bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.axvline(bootstrap_ci['macro_f1']['samples'].mean(), color='darkred', linestyle='--', 
               linewidth=2.5, label='Mean')
    ax.axvline(bootstrap_ci['macro_f1']['ci_lower'], color='orange', linestyle='--', 
               linewidth=2, label='95% CI')
    ax.axvline(bootstrap_ci['macro_f1']['ci_upper'], color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel('Macro-F1', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(f'Subject-Bootstrap: Macro-F1\nStream {stream_name} (n=1000 resamples)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add CI text box
    ci_text = f"95% CI:\n[{bootstrap_ci['macro_f1']['ci_lower']:.4f}, {bootstrap_ci['macro_f1']['ci_upper']:.4f}]"
    ax.text(0.02, 0.98, ci_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/bootstrap_ci_{stream_name}.png")
    print(f"📊 Bootstrap distribution plot saved: {config.REPORTS_DIR}/bootstrap_ci_{stream_name}.png")


def evaluate_model(model_data, X, y_true, df, stream_name):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_data (dict): Loaded model dictionary
        X (np.ndarray): Feature matrix
        y_true (np.ndarray): True labels
        df (pd.DataFrame): Metadata for subset analysis
        stream_name (str): Name of the stream (A, B, or FUSION)
    """
    print("\n" + "="*60)
    print(f"EVALUATING STREAM {stream_name}")
    print("="*60)
    
    # Apply preprocessing pipeline
    if "scaler" in model_data:
        X = model_data["scaler"].transform(X)
    if "pca" in model_data:
        X = model_data["pca"].transform(X)
    
    # Predict with threshold if available
    model = model_data["model"]
    classes = model_data["classes"]
    thr = model_data.get("threshold", None)
    
    if thr is not None and hasattr(model, "predict_proba") and len(classes) == 2:
        print(f"   Using tuned threshold: {thr:.3f}")
        y_proba = model.predict_proba(X)[:,1]
        y_pred = (y_proba >= thr).astype(int)
    else:
        y_pred = model.predict(X)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    spec = _compute_specificity(y_true, y_pred, labels=range(len(classes)))
    spec_macro = spec.mean()
    
    # Compute ROC-AUC and PR-AUC for binary classification
    roc_auc_val = None
    pr_auc_val = None
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        pr_auc_val = average_precision_score(y_true, y_proba)
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print(f"Macro Specificity: {spec_macro:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    if roc_auc_val is not None:
        print(f"ROC-AUC: {roc_auc_val:.4f}")
        print(f"PR-AUC (Average Precision): {pr_auc_val:.4f}")
    
    # Extract difficulty from paths for per-subset breakdown
    def get_difficulty_level(path_str):
        """Extract difficulty level from file path."""
        if pd.isna(path_str):
            return "Unknown"
        path_str = str(path_str)
        if "Real" in path_str and "Altered" not in path_str:
            return "Real"
        elif "Altered-Easy" in path_str:
            return "Altered-Easy"
        elif "Altered-Medium" in path_str:
            return "Altered-Medium"
        elif "Altered-Hard" in path_str:
            return "Altered-Hard"
        return "Other"
    
    df_copy = df.copy()
    df_copy["difficulty"] = df_copy["path"].apply(get_difficulty_level)
    
    # Compute per-difficulty accuracy
    difficulty_metrics = {}
    for difficulty in sorted(df_copy["difficulty"].unique()):
        mask = (df_copy["difficulty"] == difficulty).to_numpy()
        if mask.sum() > 0:
            acc_diff = accuracy_score(y_true[mask], y_pred[mask])
            n_samples = mask.sum()
            difficulty_metrics[difficulty] = {"accuracy": acc_diff, "n_samples": n_samples}
    
    # Save metrics to file
    report_path = f"{config.REPORTS_DIR}/metrics_{stream_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"STREAM {stream_name} Evaluation Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {prec:.4f}\n")
        f.write(f"Macro Recall: {rec:.4f}\n")
        f.write(f"Macro Specificity: {spec_macro:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n")
        if roc_auc_val is not None:
            f.write(f"ROC-AUC: {roc_auc_val:.4f}\n")
            f.write(f"PR-AUC (Average Precision): {pr_auc_val:.4f}\n")
        f.write("\n")
        
        # Add calibration note
        if thr is not None:
            f.write("Threshold Calibration:\n")
            f.write(f"  Threshold: {thr:.3f}\n")
            f.write(f"  Selection: Optimized on validation set to maximize macro-F1,\n")
            f.write(f"             then frozen and applied to test set (no test leakage).\n")
            f.write("\n")
        
        # Add per-difficulty breakdown
        if difficulty_metrics:
            f.write("Per-Difficulty Accuracy (Robustness Analysis):\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Difficulty':<20} {'Accuracy':>10} {'Samples':>10}\n")
            f.write("-" * 50 + "\n")
            for difficulty in sorted(difficulty_metrics.keys()):
                metrics = difficulty_metrics[difficulty]
                f.write(f"{difficulty:<20} {metrics['accuracy']:>10.4f} {metrics['n_samples']:>10}\n")
            f.write("\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0))
    print(f"✅ Metrics saved: {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[str(c) for c in classes], yticklabels=[str(c) for c in classes])
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix – Stream {stream_name}", fontsize=14)
    utils.save_fig(f"{config.REPORTS_DIR}/confusion_matrix_{stream_name}.png")
    
    # Per-subset accuracy
    subsets = df["subset"].unique()
    subset_accs = []
    for subset in subsets:
        mask = (df["subset"] == subset).to_numpy()
        if mask.sum() > 0:
            acc_subset = accuracy_score(y_true[mask], y_pred[mask])
            subset_accs.append((subset, acc_subset))
    
    if subset_accs:
        subset_names, accs = zip(*subset_accs)
        plt.figure(figsize=(8, 4))
        bars = plt.bar(subset_names, accs, color="steelblue")
        
        # Add numeric labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel("Subset", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title(f"Per-Subset Accuracy – Stream {stream_name}", fontsize=14)
        plt.ylim([0, 1.05])  # Slightly higher to accommodate labels
        plt.grid(axis="y", alpha=0.3)
        utils.save_fig(f"{config.REPORTS_DIR}/subset_accuracy_{stream_name}.png")
    
    # ROC and PR curves (only for binary classification)
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curve – Stream {stream_name}", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        utils.save_fig(f"{config.REPORTS_DIR}/roc_curve_{stream_name}.png")
        
        # PR curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(rec_curve, prec_curve, linewidth=2, label=f"PR (AP={ap:.3f})")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(f"Precision-Recall Curve – Stream {stream_name}", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        utils.save_fig(f"{config.REPORTS_DIR}/pr_curve_{stream_name}.png")
    
    # Subject-level bootstrap confidence intervals
    print("\n🔄 Computing subject-level bootstrap confidence intervals (1000 resamples)...")
    subjects = df["subject_id"].to_numpy()
    bootstrap_ci = _compute_bootstrap_ci(y_true, y_pred, subjects, n_bootstrap=1000, seed=42)
    
    # Print bootstrap results
    print(f"\nBootstrap 95% Confidence Intervals:")
    print(f"  Accuracy:  [{bootstrap_ci['accuracy']['ci_lower']:.4f}, {bootstrap_ci['accuracy']['ci_upper']:.4f}]")
    print(f"  Macro-F1:  [{bootstrap_ci['macro_f1']['ci_lower']:.4f}, {bootstrap_ci['macro_f1']['ci_upper']:.4f}]")
    
    # Save bootstrap results to file
    bootstrap_path = f"{config.REPORTS_DIR}/bootstrap_ci_{stream_name}.txt"
    with open(bootstrap_path, "w") as f:
        f.write(f"Subject-Level Bootstrap Confidence Intervals\n")
        f.write(f"Stream: {stream_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Bootstrap Settings:\n")
        f.write(f"  Number of resamples: {bootstrap_ci['n_bootstrap']}\n")
        f.write(f"  Resampling unit: Subjects (preserves subject-level independence)\n")
        f.write(f"  Confidence level: 95%\n")
        f.write(f"  Number of test subjects: {bootstrap_ci['n_subjects']}\n\n")
        f.write(f"Accuracy:\n")
        f.write(f"  Point estimate: {acc:.4f}\n")
        f.write(f"  95% CI: [{bootstrap_ci['accuracy']['ci_lower']:.4f}, {bootstrap_ci['accuracy']['ci_upper']:.4f}]\n")
        f.write(f"  CI Width: {bootstrap_ci['accuracy']['ci_upper'] - bootstrap_ci['accuracy']['ci_lower']:.4f}\n\n")
        f.write(f"Macro-F1:\n")
        f.write(f"  Point estimate: {f1:.4f}\n")
        f.write(f"  95% CI: [{bootstrap_ci['macro_f1']['ci_lower']:.4f}, {bootstrap_ci['macro_f1']['ci_upper']:.4f}]\n")
        f.write(f"  CI Width: {bootstrap_ci['macro_f1']['ci_upper'] - bootstrap_ci['macro_f1']['ci_lower']:.4f}\n")
    print(f"✅ Bootstrap CIs saved: {bootstrap_path}")
    
    # Plot bootstrap distributions
    _plot_bootstrap_distributions(bootstrap_ci, stream_name)
    
    print(f"✅ All visualizations saved to {config.REPORTS_DIR}/")


def main(stream, pca_dim=None):
    """Main evaluation function."""
    utils.ensure_dirs()
    
    # Load test split
    test_ids = utils.read_json(f"{config.SPLIT_DIR}/test_subjects.json")
    
    # Determine model path (search for existing models if pca_dim not specified)
    if stream == "A":
        if pca_dim is None:
            # Try to find the latest model
            from pathlib import Path
            models_dir = Path("models")
            matches = list(models_dir.glob("effb0_pca*_rf.joblib"))
            if matches:
                model_path = str(max(matches, key=lambda p: p.stat().st_mtime))
            else:
                model_path = "models/effb0_pca256_rf.joblib"  # fallback
        else:
            model_path = f"models/effb0_pca{pca_dim}_rf.joblib"
        # Load EfficientNet features (index.csv now includes is_altered column)
        df = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
        X = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
    elif stream == "B":
        model_path = "models/gabor_nb.joblib"
        # Load Gabor features (index.csv now includes is_altered column)
        df = pd.read_csv(f"{config.CACHE_DIR}/gabor/index.csv")
        X = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
    elif stream == "FUSION":
        if pca_dim is None:
            # Try to find the latest model
            from pathlib import Path
            models_dir = Path("models")
            matches = list(models_dir.glob("fusion_pca*_rf.joblib"))
            if matches:
                model_path = str(max(matches, key=lambda p: p.stat().st_mtime))
            else:
                model_path = "models/fusion_pca256_rf.joblib"  # fallback
        else:
            model_path = f"models/fusion_pca{pca_dim}_rf.joblib"
        # Load both features (index.csv now includes is_altered column)
        df_eff = pd.read_csv(f"{config.CACHE_DIR}/effb0/index.csv")
        X_eff = np.load(f"{config.CACHE_DIR}/effb0/all.npy")
        df_gabor = pd.read_csv(f"{config.CACHE_DIR}/gabor/index.csv")
        X_gabor = np.load(f"{config.CACHE_DIR}/gabor/gabor.npy")
        X = np.concatenate([X_eff, X_gabor], axis=1)
        df = df_eff  # Use df_eff as the index (already has is_altered column)
    else:
        raise ValueError(f"Unknown stream: {stream}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model_data = joblib.load(model_path)
    
    # Select test data
    mask_test = df.subject_id.isin(test_ids).to_numpy()
    X_test = X[mask_test]
    df_test = df[mask_test].reset_index(drop=True)
    
    # Encode labels using the model's class order (critical alignment!)
    classes = model_data["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = df_test[config.TARGET_COL].map(class_to_idx).to_numpy()
    
    print(f"Test samples: {len(X_test)}")
    print(f"Classes (model order): {classes}")
    
    # Evaluate
    evaluate_model(model_data, X_test, y_test, df_test, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--stream", choices=["A", "B", "FUSION"], required=True,
                        help="Stream to evaluate: A (EfficientNet), B (Gabor), or FUSION")
    parser.add_argument("--pca", type=int, default=None,
                        help="PCA dimension to evaluate (auto-detects latest if not specified)")
    args = parser.parse_args()
    
    main(args.stream, args.pca)
