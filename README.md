# SOCOFing Fingerprint Recognition Project

This project implements a fingerprint recognition system using the SOCOFing dataset, combining deep learning (EfficientNet) and traditional texture features (Gabor filters).

## 📁 Project Structure

```
socofing-pr/
  data/               # Place the SOCOFing dataset here
  splits/             # Saved subject-wise splits (json/csv)
  features/           # Cached features (npy)
  models/             # Trained models (joblib)
  reports/            # Figures, tables, and manuscript
  src/
    config.py                 # Configuration and constants
    prepare_metadata.py       # Dataset metadata preparation
    make_splits.py            # Subject-wise train/val/test splits
    preprocess.py             # Image preprocessing utilities
    feat_efficientnet.py      # EfficientNet feature extraction
    feat_gabor.py             # Gabor filter feature extraction
    fuse_and_train.py         # Feature fusion and model training
    evaluate.py               # Model evaluation and metrics
    utils.py                  # Helper utilities
  env.yml             # Conda environment specification
  README.md           # This file
```

## 🛠️ Setup

### 1. Create the Conda environment

```bash
conda env create -f env.yml
conda activate socofing
```

### 2. Download the Dataset

Download the SOCOFing dataset and place it in the `data/` directory:
```
data/SOCOFing/
  Real/
    ...fingerprint images...
  Altered/
    ...altered fingerprint images...
```

## 🚀 Usage

### 1. Prepare Metadata
```bash
python src/prepare_metadata.py
```

### 2. Create Splits
```bash
python src/make_splits.py
```

### 3. Extract Features
```bash
python src/feat_efficientnet.py
python src/feat_gabor.py
```

### 4. Train Model
```bash
python src/fuse_and_train.py
```

### 5. Evaluate
```bash
python src/evaluate.py
```

## 📊 Configuration

All constants and parameters are defined in `src/config.py`:
- `SEED = 42` - Random seed for reproducibility
- `IMG_SIZE = 224` - Image size for EfficientNetB0
- `HOG_SIZE = 128` - Square resize for HOG/Gabor
- `N_JOBS = -1` - Parallel processing (uses all cores)

## 📦 Dependencies

See `env.yml` for pinned versions:
- Python 3.10
- NumPy 1.26.4
- Pandas 2.2.2
- scikit-learn 1.5.2
- OpenCV 4.10.0.84
- scikit-image 0.24.0
- TensorFlow 2.15.0
- EfficientNet 1.1.1
- Joblib 1.4.2
- Matplotlib 3.9.2
- Optuna 3.6.1

## 📝 Notes

- The project uses subject-wise splits to prevent data leakage
- Features are cached to speed up repeated experiments
- Models are saved in joblib format for easy loading

## 🎯 Project Timeline

Estimated setup time: 1 hour
