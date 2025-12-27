"""
FastAPI Backend for Fingerprint Alteration Detection
Exposes inference endpoint with complete visualization pipeline
Matches functionality of Tkinter GUI (gui_app.py)
"""

import os
import io
import base64
import logging
from typing import Optional, Dict, List
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# Import feature extraction modules
from src import preprocess
from src import feat_gabor
from src import feat_forensic
from src import feat_efficientnet
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI App Instance
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fingerprint Alteration Detection API",
    description="Microservice for forensic fingerprint analysis with deep+texture+forensic fusion",
    version="3.0.0"
)

# CORS middleware for cross-origin requests (enables Streamlit dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "A": "models/a_xgboost.joblib",
    "B": "models/b_xgboost.joblib",
    "FUSION": "models/fusion_xgboost.joblib",
    "TRIPLE_FUSION": "models/triple_fusion_xgboost.joblib"
}

STREAM_LABELS = {
    "A": "Stream A (EfficientNet Deep Features)",
    "B": "Stream B (Gabor Texture Features)",
    "FUSION": "Dual Fusion (Deep + Texture)",
    "TRIPLE_FUSION": "Triple Fusion (Deep + Texture + Forensic)"
}

# Cache for models and EfficientNet extractor
_model_cache: Dict[str, dict] = {}
_effnet_extractor = None

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def numpy_to_base64(img: np.ndarray, target_size: tuple = None) -> str:
    """Convert NumPy array to Base64 string."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3 and img.dtype == np.uint8:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(img.astype(np.uint8))
    if target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def load_effnet_extractor():
    """Load EfficientNet feature extractor (lazy loaded)."""
    global _effnet_extractor
    if _effnet_extractor is None:
        logger.info("Loading EfficientNet extractor...")
        _effnet_extractor = feat_efficientnet.build_extractor()
        logger.info("EfficientNet extractor loaded")
    return _effnet_extractor

# ─────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────
def load_model(stream: str) -> dict:
    """Load model pipeline for specified stream (lazy loaded with caching)."""
    if stream not in MODEL_PATHS:
        raise ValueError(f"Invalid stream: {stream}. Choose from {list(MODEL_PATHS.keys())}")
    
    if stream in _model_cache:
        logger.info(f"Using cached model for {stream}")
        return _model_cache[stream]
    
    model_path = MODEL_PATHS[stream]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    
    logger.info(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
    
    _model_cache[stream] = model_data
    logger.info(f"Model {stream} loaded successfully")
    return model_data

# ─────────────────────────────────────────────────────────────
# Visualization Generation Functions
# ─────────────────────────────────────────────────────────────
def generate_preprocessing_images(img: np.ndarray) -> List[Dict[str, str]]:
    """Generate 6 preprocessing pipeline visualizations (200x200 each)."""
    results = []
    
    # 1. Raw image
    raw_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "1. Raw Input",
        "image": numpy_to_base64(raw_display, (200, 200))
    })
    
    # 2. CLAHE Enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    clahe_display = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "2. CLAHE Enhanced",
        "image": numpy_to_base64(clahe_display, (200, 200))
    })
    
    # 3. Bilateral Filter
    bilateral = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)
    bilateral_display = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "3. Bilateral Filter",
        "image": numpy_to_base64(bilateral_display, (200, 200))
    })
    
    # 4. EfficientNet Ready
    eff_img = preprocess.preprocess_for_effnet(img)
    results.append({
        "title": "4. EfficientNet Ready",
        "image": numpy_to_base64(eff_img, (200, 200))
    })
    
    # 5. Gabor Ready
    tex_img = preprocess.preprocess_for_texture(img)
    tex_display = cv2.cvtColor(tex_img, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "5. Gabor Ready",
        "image": numpy_to_base64(tex_display, (200, 200))
    })
    
    # 6. Orientation Field (grayscale like gui_app.py)
    sobelx = cv2.Sobel(tex_img, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(tex_img, cv2.CV_32F, 0, 1, ksize=5)
    orientation = np.arctan2(sobely, sobelx)
    orient_vis = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    orient_display = cv2.cvtColor(orient_vis, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "6. Orientation Field",
        "image": numpy_to_base64(orient_display, (200, 200))
    })
    
    return results

def generate_gabor_bank(img: np.ndarray) -> List[Dict[str, str]]:
    """Generate 8 Gabor filter responses at different orientations (130x130 each)."""
    tex_img = preprocess.preprocess_for_texture(img)
    kernels = feat_gabor.enhanced_gabor_kernels()
    
    results = []
    orientations = np.linspace(0, 157.5, 8)
    
    for i, angle in enumerate(orientations):
        # Get kernel for this orientation (kernels are organized by frequency then orientation)
        kernel_idx = i  # Use first frequency, vary orientation
        if kernel_idx < len(kernels):
            response = cv2.filter2D(tex_img, cv2.CV_32F, kernels[kernel_idx])
            # Normalize and convert to grayscale (like gui_app.py)
            vis = np.abs(response)
            vis = np.clip(vis * 255 / (vis.max() + 1e-8), 0, 255).astype(np.uint8)
            vis_display = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            
            results.append({
                "title": f"{angle:.1f}°",
                "image": numpy_to_base64(vis_display, (130, 130))
            })
    
    return results

def generate_pattern_images(img: np.ndarray) -> List[Dict[str, str]]:
    """Generate 4 pattern analysis visualizations (160x160 each)."""
    tex_img = preprocess.preprocess_for_texture(img)
    
    results = []
    
    # 1. Binary threshold
    _, binary = cv2.threshold(tex_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "Binary Map",
        "image": numpy_to_base64(binary_display, (160, 160))
    })
    
    # 2. Skeleton
    from skimage.morphology import skeletonize
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    skeleton_display = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "Skeleton",
        "image": numpy_to_base64(skeleton_display, (160, 160))
    })
    
    # 3. Corner detection (grayscale like gui_app.py)
    corners = cv2.cornerHarris(tex_img.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    corners_vis = np.zeros_like(tex_img)
    corners_vis[corners > 0.01 * corners.max()] = 255
    corners_display = cv2.cvtColor(corners_vis, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "Corner Detection",
        "image": numpy_to_base64(corners_display, (160, 160))
    })
    
    # 4. Orientation field (grayscale like gui_app.py)
    sobelx = cv2.Sobel(tex_img, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(tex_img, cv2.CV_32F, 0, 1, ksize=5)
    orientation = np.arctan2(sobely, sobelx)
    orient_vis = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    orient_display = cv2.cvtColor(orient_vis, cv2.COLOR_GRAY2BGR)
    results.append({
        "title": "Orientation Field",
        "image": numpy_to_base64(orient_display, (160, 160))
    })
    
    return results

def compute_dimensionality_reduction(eff_feat: np.ndarray) -> Dict:
    """Compute PCA, t-SNE, LDA projections with context dataset."""
    try:
        # Load context features
        eff_features_path = "features/effb0/all.npy"
        eff_index_path = "features/effb0/index.csv"
        
        if not os.path.exists(eff_features_path) or not os.path.exists(eff_index_path):
            return {"error": "Context features not found. Run feature extraction first."}
        
        # Load dataset features (sample for speed)
        all_features = np.load(eff_features_path)
        index_df = pd.read_csv(eff_index_path)
        
        n_samples = min(300, len(all_features))
        np.random.seed(42)
        sample_idx = np.random.choice(len(all_features), n_samples, replace=False)
        
        X_context = all_features[sample_idx]
        y_context = index_df.iloc[sample_idx]['is_altered'].values
        
        X_current = eff_feat.reshape(1, -1)
        X_all = np.vstack([X_context, X_current])
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_all)
        
        pca_data = {
            "context": X_pca[:-1].tolist(),
            "current": X_pca[-1].tolist(),
            "labels": y_context.tolist(),
            "variance": pca.explained_variance_ratio_.tolist()
        }
        
        # t-SNE (minimum allowed iterations is 250)
        pca_50 = PCA(n_components=min(50, X_all.shape[1]), random_state=42)
        X_pca50 = pca_50.fit_transform(X_all)
        
        perplexity = min(30, max(5, n_samples // 5))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                   max_iter=250, learning_rate='auto', init='pca')  # Minimum allowed is 250
        X_tsne = tsne.fit_transform(X_pca50)
        
        tsne_data = {
            "context": X_tsne[:-1].tolist(),
            "current": X_tsne[-1].tolist(),
            "labels": y_context.tolist(),
            "perplexity": int(perplexity)  # Convert to Python int
        }
        
        # LDA
        lda = LDA(n_components=1)
        X_lda_context = lda.fit_transform(X_context, y_context)
        X_lda_current = lda.transform(X_current)
        
        lda_data = {
            "context": X_lda_context.flatten().tolist(),
            "current": float(X_lda_current[0, 0]),  # Convert numpy to Python float
            "labels": y_context.tolist()
        }
        
        return {
            "pca": pca_data,
            "tsne": tsne_data,
            "lda": lda_data
        }
    
    except Exception as e:
        logger.error(f"Dimensionality reduction error: {str(e)}")
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Response model for /predict endpoint."""
    prediction: str
    risk_score: float
    confidence: float
    threshold: float
    stream: str
    
    # Feature statistics
    feature_stats: Dict
    
    # Forensic metrics
    forensic_dict: Dict
    
    # Visualization data
    preprocessing_images: List[Dict[str, str]]
    gabor_images: List[Dict[str, str]]
    pattern_images: List[Dict[str, str]]
    
    # Dimensionality reduction
    dimred_data: Dict
    
    # System trace
    system_trace: str

# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": "fingerprint-api",
        "version": "3.0.0",
        "models_loaded": list(_model_cache.keys()),
        "effnet_loaded": _effnet_extractor is not None
    }

@app.get("/test-error")
async def test_error():
    """Test error handling."""
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Test error: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}\n\nTraceback:\n{error_trace}")

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test file upload."""
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        return {"filename": file.filename, "size": len(contents)}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Upload test error: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}\n\nTraceback:\n{error_trace}")

# ─────────────────────────────────────────────────────────────
# Inference Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/predict")  # Removed response_model temporarily for debugging
async def predict_image(
    file: UploadFile = File(...),
    stream: str = "TRIPLE_FUSION"
):
    """
    Predict whether fingerprint is altered or real.
    Returns complete visualization pipeline data.
    
    Args:
        file: Uploaded fingerprint image
        stream: Model stream (A, B, FUSION, TRIPLE_FUSION)
    
    Returns:
        PredictionResponse with classification result and all visualizations
    """
    try:
        logger.info(f"=== PREDICT ENDPOINT CALLED === File: {file.filename if file else 'None'}, Stream: {stream}")
        logger.info("Step 1: Loading model...")
        # Load model
        model_data = load_model(stream)
        model = model_data["model"]
        threshold = model_data.get("threshold", 0.5)
        logger.info(f"Step 1 complete: Model loaded, threshold={threshold}")
        
        logger.info("Step 2: Reading image...")
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        logger.info(f"Step 2 complete: Image shape={img.shape if img is not None else 'None'}")
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.info(f"Processing image: {file.filename}, Shape: {img.shape}, Stream: {stream}")
        
        logger.info("Step 3: Generating preprocessing images...")
        # ─── Generate Preprocessing Visualizations ───
        preprocessing_images = generate_preprocessing_images(img)
        logger.info(f"Step 3 complete: {len(preprocessing_images)} images generated")
        
        logger.info("Step 4: Extract features...")
        # ─── Feature Extraction ───
        
        # Stream A: EfficientNet features (1280-D)
        eff_img = preprocess.preprocess_for_effnet(img)
        eff_img = effnet_preprocess(eff_img)
        eff_img = np.expand_dims(eff_img, axis=0)
        
        eff_extractor = load_effnet_extractor()
        eff_feat = eff_extractor.predict(eff_img, verbose=0)[0]
        
        # Stream B: Gabor features (93-D)
        tex_img = preprocess.preprocess_for_texture(img)
        kernels = feat_gabor.enhanced_gabor_kernels()
        basic_feat = feat_gabor.enhanced_gabor_features(tex_img, kernels)  # 80-D
        stats_feat = feat_gabor.gabor_texture_statistics(tex_img, kernels)  # 13-D
        tex_feat = np.concatenate([basic_feat, stats_feat])  # 93-D
        logger.info(f"Step 4 complete: eff_feat={eff_feat.shape}, tex_feat={tex_feat.shape}")
        
        logger.info("Step 5: Generate Gabor bank and pattern images...")
        # ─── Generate Gabor Bank Visualizations ───
        gabor_images = generate_gabor_bank(img)
        
        # ─── Generate Pattern Visualizations ───
        pattern_images = generate_pattern_images(img)
        logger.info(f"Step 5 complete: gabor={len(gabor_images)}, patterns={len(pattern_images)}")
        
        logger.info("Step 6: Extract forensic features...")
        # Forensic Features (11-D)
        forensic_dict_raw = feat_forensic.extract_biological_features(tex_img)
        # Convert all numpy types to native Python types for JSON serialization
        forensic_dict = {k: float(v) if hasattr(v, 'item') else float(v) for k, v in forensic_dict_raw.items()}
        forensic_keys = [
            "mean_intensity", "std_intensity", "median_intensity", "ridge_density",
            "ridge_thickness_variation", "orientation_consistency", "ridge_endings",
            "ridge_bifurcations", "texture_homogeneity", "dominant_frequency", "spectral_centroid"
        ]
        for_feat = np.array([forensic_dict.get(k, 0.0) for k in forensic_keys])
        logger.info(f"Step 6 complete: for_feat={for_feat.shape}")
        
        logger.info("Step 7: Construct feature vector and predict...")
        # Construct feature vector based on stream
        if stream == "A":
            X = eff_feat.reshape(1, -1)
        elif stream == "B":
            X = tex_feat.reshape(1, -1)
        elif stream == "FUSION":
            X = np.concatenate([eff_feat, tex_feat]).reshape(1, -1)
        elif stream == "TRIPLE_FUSION":
            X = np.concatenate([eff_feat, tex_feat, for_feat]).reshape(1, -1)
        
        # ─── Inference ───
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
        else:
            proba = float(model.predict(X)[0])
        
        is_altered = proba >= threshold
        prediction = "ALTERED" if is_altered else "REAL"
        confidence = proba if is_altered else (1 - proba)
        logger.info(f"Step 7 complete: prediction={prediction}, proba={proba:.4f}, confidence={confidence:.4f}")
        
        logger.info("Step 8: Calculate feature statistics...")
        # ─── Feature Statistics ───
        feature_stats = {
            "efficientnet": {
                "dimensions": len(eff_feat),
                "mean": float(eff_feat.mean()),
                "std": float(eff_feat.std()),
                "min": float(eff_feat.min()),
                "max": float(eff_feat.max()),
                "sample": eff_feat[:20].tolist()  # First 20 for visualization
            },
            "gabor": {
                "dimensions": len(tex_feat),
                "mean": float(tex_feat.mean()),
                "std": float(tex_feat.std()),
                "min": float(tex_feat.min()),
                "max": float(tex_feat.max())
            },
            "total_dimensions": X.shape[1]
        }
        logger.info(f"Step 8 complete: feature_stats calculated")
        
        logger.info("Step 9: Compute dimensionality reduction...")
        # ─── Dimensionality Reduction ───
        dimred_data = compute_dimensionality_reduction(eff_feat)
        logger.info(f"Step 9 complete: dimred computed")
        
        logger.info("Step 10: Build response trace...")
        # ─── Build Response ───
        trace = (
            f"📥 Image received: {file.filename}\n"
            f"🔍 Stream: {STREAM_LABELS[stream]}\n"
            f"📐 Features extracted: {X.shape[1]}D\n"
            f"   • EfficientNet: {len(eff_feat)}D (μ={eff_feat.mean():.3f}, σ={eff_feat.std():.3f})\n"
            f"   • Gabor Texture: {len(tex_feat)}D (μ={tex_feat.mean():.3f}, σ={tex_feat.std():.3f})\n"
        )
        
        if stream == "TRIPLE_FUSION":
            trace += f"   • Forensic: {len(for_feat)}D\n"
        
        trace += (
            f"🎯 Altered probability: {proba:.3f}\n"
            f"⚖️ Decision threshold: {threshold:.3f}\n"
            f"✅ Classification: {prediction} (confidence: {confidence:.1%})\n"
            f"📊 Preprocessing images: {len(preprocessing_images)}\n"
            f"🌊 Gabor bank images: {len(gabor_images)}\n"
            f"🔍 Pattern images: {len(pattern_images)}\n"
            f"📈 Dimensionality analysis: {'✓' if 'error' not in dimred_data else '✗'}"
        )
        
        logger.info(f"Prediction complete: {prediction} ({confidence:.1%})")
        
        return {
            "prediction": prediction,
            "risk_score": float(proba),
            "confidence": float(confidence),
            "threshold": float(threshold),
            "stream": stream,
            "feature_stats": feature_stats,
            "forensic_dict": forensic_dict,
            "preprocessing_images": preprocessing_images,
            "gabor_images": gabor_images,
            "pattern_images": pattern_images,
            "dimred_data": dimred_data,
            "system_trace": trace
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Prediction error: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}\n\nTraceback:\n{error_trace}")

# ─────────────────────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
