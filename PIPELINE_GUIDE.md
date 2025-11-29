# 🚀 Complete Pipeline Execution Guide

This guide provides step-by-step instructions to run the entire SOCOFing gender recognition pipeline.

## ✅ Prerequisites

1. **Virtual environment activated**:
   ```powershell
   cd "C:\All Assignments\socofing-pr-py311"
   .\Scripts\Activate.ps1
   ```

2. **Dataset verified**: Ensure SOCOFing dataset is in `data/SOCOFing/`

## 📋 Pipeline Steps

### Step 1: Generate Metadata Index
```powershell
python -m src.prepare_metadata
```
**Expected output**: `reports/metadata.csv` with 55,270 rows  
**Time**: ~30 seconds

### Step 2: Create Data Splits
```powershell
python -m src.make_splits
```
**Expected output**: `splits/train_subjects.json`, `val_subjects.json`, `test_subjects.json`, `cv_folds.json`  
**Time**: ~5 seconds

### Step 3: Quick Visualizations (Optional)
```powershell
python -m src.viz_quick_checks
```
**Expected output**: 
- `reports/preprocessing_preview.png`
- `reports/class_balance.png`
- `reports/subset_distribution.png`

**Time**: ~1 minute

### Step 4: Extract EfficientNet Features
```powershell
python -m src.feat_efficientnet
```
**Expected output**: `features/effb0/all.npy` (55270, 1280), `features/effb0/index.csv`  
**Time**: ~10-15 minutes (depending on GPU/CPU)

### Step 5: Extract Gabor Features
```powershell
python -m src.feat_gabor
```
**Expected output**: `features/gabor/gabor.npy` (55270, 12), `features/gabor/index.csv`  
**Time**: ~5-10 minutes

### Step 6: Train Models

#### Stream A: EfficientNet Only
```powershell
python -m src.fuse_and_train --stream A --pca 256
```
**Expected output**: 
- `models/effb0_pca256_rf.joblib`
- `reports/pca_variance_A_256.png`

**Time**: ~2-3 minutes

#### Stream B: Gabor Only
```powershell
python -m src.fuse_and_train --stream B
```
**Expected output**: `models/gabor_nb.joblib`  
**Time**: <1 minute

#### Stream FUSION: Combined Features
```powershell
python -m src.fuse_and_train --stream FUSION --pca 256
```
**Expected output**: 
- `models/fusion_pca256_rf.joblib`
- `reports/pca_variance_FUSION_256.png`
- `reports/rf_feature_importance_fusion.png`

**Time**: ~3-4 minutes

### Step 7: Evaluate Models

#### Evaluate Stream A
```powershell
python -m src.evaluate --stream A
```
**Expected output**:
- `reports/metrics_A.txt`
- `reports/confusion_matrix_A.png`
- `reports/subset_accuracy_A.png`
- `reports/roc_curve_A.png` (binary only)
- `reports/pr_curve_A.png` (binary only)

**Time**: ~30 seconds

#### Evaluate Stream B
```powershell
python -m src.evaluate --stream B
```
**Expected output**:
- `reports/metrics_B.txt`
- `reports/confusion_matrix_B.png`
- `reports/subset_accuracy_B.png`
- `reports/roc_curve_B.png` (binary only)
- `reports/pr_curve_B.png` (binary only)

**Time**: ~20 seconds

#### Evaluate Stream FUSION
```powershell
python -m src.evaluate --stream FUSION
```
**Expected output**:
- `reports/metrics_FUSION.txt`
- `reports/confusion_matrix_FUSION.png`
- `reports/subset_accuracy_FUSION.png`
- `reports/roc_curve_FUSION.png` (binary only)
- `reports/pr_curve_FUSION.png` (binary only)

**Time**: ~30 seconds

## 📊 Expected Results

### Stream A: EfficientNet
- **Validation F1**: ~0.85-0.90
- **Test Accuracy**: ~87-92%

### Stream B: Gabor
- **Validation F1**: ~0.75-0.80
- **Test Accuracy**: ~77-82%

### Stream FUSION: Combined
- **Validation F1**: ~0.88-0.93
- **Test Accuracy**: ~90-95%
- **Best overall performance**

## 🎯 All-in-One Script

Run the complete pipeline with a single command:

```powershell
# Generate metadata
python -m src.prepare_metadata

# Create splits
python -m src.make_splits

# Quick checks
python -m src.viz_quick_checks

# Extract features
python -m src.feat_efficientnet
python -m src.feat_gabor

# Train all models
python -m src.fuse_and_train --stream A --pca 256
python -m src.fuse_and_train --stream B
python -m src.fuse_and_train --stream FUSION --pca 256

# Evaluate all models
python -m src.evaluate --stream A
python -m src.evaluate --stream B
python -m src.evaluate --stream FUSION

Write-Host "✅ Pipeline completed! Check reports/ for results."
```

## 📂 Output Files Summary

After running the complete pipeline, you'll have:

### `splits/` (4 files)
- train_subjects.json
- val_subjects.json
- test_subjects.json
- cv_folds.json

### `features/` (4 files)
- effb0/all.npy
- effb0/index.csv
- gabor/gabor.npy
- gabor/index.csv

### `models/` (3 files)
- effb0_pca256_rf.joblib
- gabor_nb.joblib
- fusion_pca256_rf.joblib

### `reports/` (20+ files)
- metadata.csv
- preprocessing_preview.png
- class_balance.png
- subset_distribution.png
- pca_variance_A_256.png
- pca_variance_FUSION_256.png
- rf_feature_importance_fusion.png
- confusion_matrix_A.png
- confusion_matrix_B.png
- confusion_matrix_FUSION.png
- subset_accuracy_A.png
- subset_accuracy_B.png
- subset_accuracy_FUSION.png
- roc_curve_A.png
- roc_curve_B.png
- roc_curve_FUSION.png
- pr_curve_A.png
- pr_curve_B.png
- pr_curve_FUSION.png
- metrics_A.txt
- metrics_B.txt
- metrics_FUSION.txt

## 🔧 Customization Options

### Change Classification Target
Edit `src/config.py`:
```python
TARGET_COL = "hand"    # or "finger" instead of "gender"
```

### Adjust PCA Dimensions
```powershell
python -m src.fuse_and_train --stream A --pca 128   # Less dimensions
python -m src.fuse_and_train --stream A --pca 512   # More dimensions
```

### Modify Random Forest Parameters
Edit `src/fuse_and_train.py` (line ~92 or ~220):
```python
rf = RandomForestClassifier(
    n_estimators=500,      # Increase trees
    max_depth=128,         # Deeper trees
    min_samples_split=5,   # More regularization
    ...
)
```

## 🐛 Common Issues

### "No module named 'tensorflow'"
**Solution**: Activate virtual environment first
```powershell
.\Scripts\Activate.ps1
```

### "FileNotFoundError: metadata.csv"
**Solution**: Run prepare_metadata.py first
```powershell
python -m src.prepare_metadata
```

### Out of Memory
**Solution**: Reduce batch size in `src/feat_efficientnet.py`
```python
batch_size = 16  # Change from 32 to 16
```

### TensorFlow Warnings
**Status**: Safe to ignore. Known Keras 3 compatibility warnings.

## ⏱️ Total Pipeline Time

- **Fast setup** (features already extracted): ~10 minutes
- **Complete pipeline** (from scratch): ~30-40 minutes

## 📈 Monitoring Progress

All scripts print detailed progress:
- ✅ Checkmarks indicate completed steps
- Progress bars show feature extraction status
- Validation metrics printed during training
- File paths shown when outputs are saved

## 🎉 Success Indicators

Pipeline completed successfully when you see:
1. All expected files in `reports/`, `models/`, `features/`, `splits/`
2. No errors in terminal output
3. Confusion matrices show reasonable classification
4. Test accuracy > 85% for FUSION stream

---

**Questions?** Check inline documentation in each source file or review the main README.md
