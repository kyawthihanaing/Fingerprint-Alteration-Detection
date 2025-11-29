"""
Create subject-wise train/validation/test splits.
Ensures no subject appears in multiple splits to prevent data leakage.
Also creates cross-validation folds for hyperparameter tuning.
"""

from __future__ import annotations
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from . import config, utils


def main():
    """
    Create and save subject-wise splits (70% train, 15% val, 15% test).
    Also creates 5-fold CV splits for the training set.
    """
    utils.ensure_dirs()
    
    # Load metadata
    df = utils.load_metadata()
    print(f"📊 Loaded {len(df)} samples from {df['subject_id'].nunique()} subjects")
    
    # Get unique subjects with their MAJORITY target label
    # (Important: For Real vs Altered, a subject appears in both classes)
    subj_df = df.groupby("subject_id")[config.TARGET_COL].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
    X_subjects = subj_df["subject_id"].to_numpy()
    y_subjects = subj_df[config.TARGET_COL].to_numpy()
    
    print(f"🎯 Target column: {config.TARGET_COL}")
    print(f"   Classes: {sorted(subj_df[config.TARGET_COL].unique())}")
    
    # First split: separate test set (15%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config.SEED)
    trainval_idx, test_idx = next(sss.split(X_subjects, y_subjects))
    
    X_trainval = X_subjects[trainval_idx]
    y_trainval = y_subjects[trainval_idx]
    X_test = X_subjects[test_idx]
    
    # Second split: separate validation from train (15% of total = ~17.65% of trainval)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=config.SEED)
    train_idx, val_idx = next(sss2.split(X_trainval, y_trainval))
    
    train_ids = X_trainval[train_idx].tolist()
    val_ids = X_trainval[val_idx].tolist()
    test_ids = X_test.tolist()
    
    # Save splits
    utils.write_json(f"{config.SPLIT_DIR}/train_subjects.json", train_ids)
    utils.write_json(f"{config.SPLIT_DIR}/val_subjects.json", val_ids)
    utils.write_json(f"{config.SPLIT_DIR}/test_subjects.json", test_ids)
    
    print(f"\n✅ Subject-wise splits saved:")
    print(f"   Train: {len(train_ids)} subjects ({len(train_ids)/len(X_subjects)*100:.1f}%)")
    print(f"   Val:   {len(val_ids)} subjects ({len(val_ids)/len(X_subjects)*100:.1f}%)")
    print(f"   Test:  {len(test_ids)} subjects ({len(test_ids)/len(X_subjects)*100:.1f}%)")
    
    # Create 5-fold cross-validation splits for hyperparameter tuning
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    train_df = subj_df[subj_df.subject_id.isin(train_ids)]
    
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df.subject_id, train_df[config.TARGET_COL])):
        fold = {
            "train": train_df.subject_id.iloc[tr_idx].tolist(),
            "val": train_df.subject_id.iloc[va_idx].tolist()
        }
        folds.append(fold)
        print(f"   Fold {fold_idx+1}: {len(fold['train'])} train, {len(fold['val'])} val subjects")
    
    utils.write_json(f"{config.SPLIT_DIR}/cv_folds.json", folds)
    print(f"\n✅ 5-fold CV splits saved to {config.SPLIT_DIR}/cv_folds.json")
    
    # Calculate sample counts
    train_samples = df[df.subject_id.isin(train_ids)].shape[0]
    val_samples = df[df.subject_id.isin(val_ids)].shape[0]
    test_samples = df[df.subject_id.isin(test_ids)].shape[0]
    
    print(f"\n📊 Sample counts:")
    print(f"   Train: {train_samples} samples")
    print(f"   Val:   {val_samples} samples")
    print(f"   Test:  {test_samples} samples")
    print(f"   Total: {train_samples + val_samples + test_samples} samples")


if __name__ == "__main__":
    main()
