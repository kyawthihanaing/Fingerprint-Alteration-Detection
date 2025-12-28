# 🚀 Fingerprint Alteration Detection - Quick Start Guide

## 📋 Overview

This web platform detects altered fingerprints using deep learning and pattern analysis, providing comprehensive forensic visualization across 5 analysis tabs.

### **serving.py** - Backend API Server
- **FastAPI backend** exposing fingerprint analysis pipeline
- Processes uploaded fingerprint images through complete ML pipeline
- Generates 6 preprocessing visualizations (grayscale, enhanced, normalized, etc.)
- Extracts features using **EfficientNetB0** (1280D) + **Gabor filters** (8 orientations)
- Runs 4 classification models: Stream A, Stream B, FUSION, and TRIPLE_FUSION
- Computes forensic metrics (ridge flow entropy, minutiae density, orientation consistency)
- Performs dimensionality reduction (PCA, t-SNE, LDA) for visualization
- Returns JSON response with predictions, confidence scores, and all visualizations

### **dashboard.py** - Streamlit Web Interface
- **5-tab interactive dashboard** matching original Tkinter GUI functionality
- **Tab 1 - Preprocessing Pipeline**: 6 preprocessing steps in 3-column grid
- **Tab 2 - Feature Extraction**: Gabor filter bank (4-column compact layout) + statistics
- **Tab 3 - Pattern Recognition**: 4 pattern visualizations + dynamic gauge meter
- **Tab 4 - Forensic Analysis**: Biological consistency metrics visualization
- **Tab 5 - Dimensionality Analysis**: PCA/t-SNE/LDA scatter plots
- Real-time communication with backend API
- Optimized image layouts for better UX

---

## 🔧 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/kyawthihanaing/Fingerprint-Alteration-Detection.git
cd Fingerprint-Alteration-Detection
```

### 2️⃣ Create Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
```

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `fastapi==0.126.0` - Backend API framework
- `uvicorn==0.38.1` - ASGI server
- `streamlit==1.52.2` - Frontend dashboard
- `tensorflow==2.20.0` - Deep learning (EfficientNetB0)
- `xgboost==3.1` - Classification models
- `scikit-learn==1.7.2` - ML utilities
- `scikit-image==0.25.2` - Image processing
- `plotly==6.5.0` - Interactive visualizations

---

## ▶️ Running the Application

### Step 1: Start Backend Server (Port 8000)
Open a terminal window and run:

```powershell
# Windows
cd c:\path\to\Fingerprint-Alteration-Detection
.\venv\Scripts\activate
set TF_CPP_MIN_LOG_LEVEL=3
python serving.py
```

```bash
# Linux/Mac
cd /path/to/Fingerprint-Alteration-Detection
source venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3
python serving.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://localhost:8000
INFO:     Application startup complete.
```

### Step 2: Start Dashboard (Port 8501)
Open a **new terminal window** and run:

```powershell
# Windows
cd c:\path\to\Fingerprint-Alteration-Detection
.\venv\Scripts\activate
streamlit run dashboard.py --server.port 8501
```

```bash
# Linux/Mac
cd /path/to/Fingerprint-Alteration-Detection
source venv/bin/activate
streamlit run dashboard.py --server.port 8501
```

**Expected output:**
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Step 3: Access Dashboard
Open your browser and navigate to: **http://localhost:8501**

---

## 🖼️ Using the Dashboard

1. **Upload Image**: Click "Browse files" and select a fingerprint image (PNG, JPG, BMP)
2. **Select Model Stream**: Choose from 4 classification models
   - **Stream A**: EfficientNetB0 features only
   - **Stream B**: Gabor texture features only
   - **FUSION**: Combined A + B features
   - **TRIPLE_FUSION**: A + B + Forensic metrics (recommended)
3. **Click "🔍 Analyze Fingerprint"**: Processing takes 10-15 seconds
4. **Review Results**: Explore 5 tabs with comprehensive analysis
   - Risk score and confidence displayed at top
   - Gauge meter shows classification confidence
   - All visualizations rendered in optimized layouts

---

## 📊 Analysis Tabs Breakdown

| Tab | Content | Layout |
|-----|---------|--------|
| **📊 Preprocessing Pipeline** | 6 preprocessing steps | 3 columns × 250px |
| **🌊 Feature Extraction** | Gabor bank + statistics | 4 columns × 120px |
| **🔍 Pattern Recognition** | 4 patterns + gauge meter | 2 columns × 200px |
| **🧬 Forensic Analysis** | Biological metrics chart | Full width |
| **📈 Dimensionality Analysis** | PCA/t-SNE/LDA plots | 3 columns responsive |

---

## 🛑 Troubleshooting

### Backend not starting?
- **Port 8000 already in use**: Kill existing process
  ```powershell
  # Windows
  Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | Stop-Process -Force
  ```
- **TensorFlow warnings**: Set `TF_CPP_MIN_LOG_LEVEL=3` to suppress

### Dashboard shows "Connection Error"?
- Ensure backend is running on http://localhost:8000
- Check firewall settings
- Verify `BACKEND_URL` in dashboard.py line 24

### Models not found?
- Ensure `.joblib` files exist in `models/` directory:
  - `a_xgboost.joblib`
  - `b_xgboost.joblib`
  - `fusion_xgboost.joblib`
  - `triple_fusion_xgboost.joblib`

### Images too large/small?
- Adjust `width` parameter in `display_image_grid()` calls
- Current settings: Preprocessing (250px), Gabor (120px), Patterns (200px)

---

## 📞 Support

For issues or questions:
- **GitHub Issues**: https://github.com/kyawthihanaing/Fingerprint-Alteration-Detection/issues
- **Repository**: https://github.com/kyawthihanaing/Fingerprint-Alteration-Detection

---

## ✅ Quick Command Reference

```powershell
# Start both servers (Windows)
# Terminal 1 - Backend
.\venv\Scripts\activate; python serving.py

# Terminal 2 - Dashboard
.\venv\Scripts\activate; streamlit run dashboard.py --server.port 8501

# Stop servers
Ctrl + C in each terminal
```

**Happy analyzing! 🔬**
