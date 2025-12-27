"""
Streamlit Dashboard for Fingerprint Alteration Detection
Complete migration of Tkinter GUI (gui_app.py) to web architecture
Matches functional depth, layout, and information density exactly
"""

import streamlit as st
import requests
import io
import base64
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

STREAM_LABELS = {
    "A": "Stream A (EfficientNet Deep Features)",
    "B": "Stream B (Gabor Texture Features)",
    "FUSION": "Dual Fusion (Deep + Texture)",
    "TRIPLE_FUSION": "Triple Fusion (Deep + Texture + Forensic)"
}

COLORS = {
    "bg_main": "#ECEFF1",
    "bg_sidebar": "#263238",
    "accent": "#3498DB",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "text": "#2C3E50",
    "text_light": "#7F8C8D"
}

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fingerprint Alteration Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better tab visibility and contrast
st.markdown("""
<style>
    /* Ultra-specific selectors with maximum priority using testid */
    [data-testid="stTabs"] {
        background-color: #1a1a1a !important;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #1a1a1a !important;
        padding: 10px 20px !important;
    }
    
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        background: #2a2a2a !important;
        color: #FFD700 !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        font-family: 'Arial Black', Arial, sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #FF4500 0%, #FF8C00 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(255, 69, 0, 0.5) !important;
    }
    
    [data-testid="stTabs"] button[data-baseweb="tab"] > div,
    [data-testid="stTabs"] button[data-baseweb="tab"] > div > div,
    [data-testid="stTabs"] button[data-baseweb="tab"] p {
        color: inherit !important;
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
    .main { background-color: #ECEFF1; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #BDC3C7;
        border-radius: 4px 4px 0px 0px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def base64_to_image(b64_string: str) -> Image.Image:
    """Convert Base64 string to PIL Image."""
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))

def display_image_grid(images: List[Dict], cols: int, width: int = 200):
    """Display images in a grid layout."""
    for i in range(0, len(images), cols):
        row_images = images[i:i+cols]
        columns = st.columns(cols)
        for col, img_data in zip(columns, row_images):
            with col:
                st.markdown(f"**{img_data['title']}**", unsafe_allow_html=False)
                img = base64_to_image(img_data['image'])
                # Use width parameter if provided, otherwise use container width with cap
                if width:
                    st.image(img, width=width)
                else:
                    st.image(img, use_container_width=True)

def create_gauge_chart(value: float, threshold: float, title: str = "Classification Confidence"):
    """Create a dynamic gauge chart with clear needle indicator showing actual confidence."""
    # Determine if altered or real based on risk score comparison to threshold
    is_altered = value >= threshold
    # Confidence is the certainty of the prediction
    confidence = value if is_altered else (1 - value)
    prediction_label = "🚨 ALTERED" if is_altered else "✅ REAL"
    
    # Create color scale based on confidence level
    if confidence >= 0.85:
        bar_color = "#27AE60"  # Dark green for very high confidence
    elif confidence >= 0.7:
        bar_color = "#2ECC71"  # Light green for high confidence
    elif confidence >= 0.5:
        bar_color = "#F39C12"  # Orange for medium confidence
    else:
        bar_color = "#E74C3C"  # Red for low confidence
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,  # This is what moves the needle
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{title}</b><br><span style='font-size:16px; color:#E74C3C'>{prediction_label}</span>", 
            'font': {'size': 24, 'color': '#1a1a1a', 'family': 'Arial Black'}
        },
        number={
            'suffix': '', 
            'font': {
                'size': 64, 
                'color': bar_color,
                'family': 'Arial Black'
            },
            'valueformat': '.1%'
        },
        gauge={
            'axis': {
                'range': [0, 1], 
                'tickwidth': 3, 
                'tickcolor': "#2C3E50", 
                'tickmode': 'array',
                'tickvals': [0, 0.25, 0.5, 0.75, 1.0],
                'ticktext': ['0%', '25%', '50%', '75%', '100%'],
                'tickfont': {'size': 16, 'family': 'Arial Black', 'color': '#1a1a1a'}
            },
            'bar': {
                'color': bar_color,
                'thickness': 1.0,  # Make the needle/bar very prominent
                'line': {'color': "#000000", 'width': 3}  # Bold black outline
            },
            'bgcolor': "white",
            'borderwidth': 5,
            'bordercolor': "#2C3E50",
            'steps': [
                {'range': [0, 0.5], 'color': '#ffcccc'},     # Light red - low confidence
                {'range': [0.5, 0.7], 'color': '#ffffcc'},   # Light yellow - medium
                {'range': [0.7, 0.85], 'color': '#ccffcc'},  # Light green - high
                {'range': [0.85, 1], 'color': '#99ff99'}     # Bright green - very high
            ],
            'threshold': {
                'line': {'color': "rgba(200, 200, 200, 0.5)", 'width': 2},  # Make threshold barely visible
                'thickness': 0.5,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(
        height=450,
        margin=dict(l=50, r=50, t=130, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#1a1a1a", 'family': "Arial", 'size': 14}
    )
    
    return fig

def create_feature_bar_chart(feature_sample: List[float], title: str = "EfficientNet Feature Sample"):
    """Create bar chart for feature vector visualization."""
    colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in feature_sample]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(feature_sample))),
            y=feature_sample,
            marker_color=colors,
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature Index",
        yaxis_title="Value",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_forensic_bar_chart(forensic_dict: Dict):
    """Create horizontal bar chart for forensic features."""
    # Define features with normalization ranges
    feature_config = [
        ('ridge_density', 'Ridge Density', 0, 0.15),
        ('orientation_consistency', 'Orientation Var.', 0, 2.0),
        ('texture_homogeneity', 'Texture Homog.', 0, 5e-6),
        ('ridge_thickness_variation', 'Thickness Var.', 0, 0.5),
        ('mean_intensity', 'Mean Intensity', 0, 255),
    ]
    
    labels = []
    normalized_values = []
    raw_values = []
    
    for key, label, min_val, max_val in feature_config:
        raw = forensic_dict.get(key, 0)
        raw_values.append(raw)
        labels.append(label)
        
        # Normalize to 0-1 range
        norm = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
        norm = max(0, min(1, norm))
        normalized_values.append(norm)
    
    fig = go.Figure(go.Bar(
        x=normalized_values,
        y=labels,
        orientation='h',
        marker=dict(
            color=normalized_values,
            colorscale='Blues',
            line=dict(color='#333333', width=0.5)
        ),
        text=[f"{raw:.4f}" if raw < 1 else f"{raw:.1f}" for raw in raw_values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Normalized: %{x:.2f}<br>Raw: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Forensic Feature Profile",
        xaxis_title="Normalized Scale (0-1)",
        height=350,
        margin=dict(l=150, r=100, t=60, b=40),
        xaxis=dict(range=[0, 1.4]),
        yaxis=dict(autorange="reversed")
    )
    
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    return fig

def create_pca_plot(pca_data: Dict):
    """Create PCA scatter plot."""
    if not pca_data:
        return None
    
    context = np.array(pca_data['context'])
    current = pca_data['current']
    labels = pca_data['labels']
    variance = pca_data['variance']
    
    # Separate by class
    altered_mask = np.array(labels) == 1
    real_mask = np.array(labels) == 0
    
    fig = go.Figure()
    
    # Plot context points
    fig.add_trace(go.Scatter(
        x=context[altered_mask, 0],
        y=context[altered_mask, 1],
        mode='markers',
        name='Altered',
        marker=dict(color='#E74C3C', size=6, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=context[real_mask, 0],
        y=context[real_mask, 1],
        mode='markers',
        name='Real',
        marker=dict(color='#27AE60', size=6, opacity=0.5)
    ))
    
    # Plot current sample
    fig.add_trace(go.Scatter(
        x=[current[0]],
        y=[current[1]],
        mode='markers',
        name='Current',
        marker=dict(color='#F39C12', size=20, symbol='star', line=dict(color='black', width=1.5))
    ))
    
    fig.update_layout(
        title=f"PCA Projection",
        xaxis_title=f"PC1 ({variance[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({variance[1]*100:.1f}%)",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest'
    )
    
    return fig

def create_tsne_plot(tsne_data: Dict):
    """Create t-SNE scatter plot."""
    if not tsne_data:
        return None
    
    context = np.array(tsne_data['context'])
    current = tsne_data['current']
    labels = tsne_data['labels']
    perplexity = tsne_data.get('perplexity', 30)
    
    altered_mask = np.array(labels) == 1
    real_mask = np.array(labels) == 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=context[altered_mask, 0],
        y=context[altered_mask, 1],
        mode='markers',
        name='Altered',
        marker=dict(color='#E74C3C', size=6, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=context[real_mask, 0],
        y=context[real_mask, 1],
        mode='markers',
        name='Real',
        marker=dict(color='#27AE60', size=6, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=[current[0]],
        y=[current[1]],
        mode='markers',
        name='Current',
        marker=dict(color='#F39C12', size=20, symbol='star', line=dict(color='black', width=1.5))
    ))
    
    fig.update_layout(
        title=f"t-SNE Embedding (perplexity={perplexity})",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest'
    )
    
    return fig

def create_lda_plot(lda_data: Dict):
    """Create LDA strip plot."""
    if not lda_data:
        return None
    
    context = lda_data['context']
    current = lda_data['current']
    labels = lda_data['labels']
    
    altered_mask = np.array(labels) == 1
    real_mask = np.array(labels) == 0
    
    fig = go.Figure()
    
    # Add jittered points for better visualization
    np.random.seed(42)
    altered_y = np.random.uniform(-0.3, 0.3, np.sum(altered_mask)) + 1
    real_y = np.random.uniform(-0.3, 0.3, np.sum(real_mask))
    
    fig.add_trace(go.Scatter(
        x=np.array(context)[altered_mask],
        y=altered_y,
        mode='markers',
        name='Altered',
        marker=dict(color='#E74C3C', size=6, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=np.array(context)[real_mask],
        y=real_y,
        mode='markers',
        name='Real',
        marker=dict(color='#27AE60', size=6, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=[current],
        y=[0.5],
        mode='markers',
        name='Current',
        marker=dict(color='#F39C12', size=20, symbol='star', line=dict(color='black', width=1.5))
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="purple", line_width=2, opacity=0.7)
    
    fig.update_layout(
        title="LDA Projection",
        xaxis_title="LDA Discriminant",
        yaxis_title="Class",
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Real', 'Altered']),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest'
    )
    
    return fig

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Biometric Security")
    st.markdown("---")
    
    # API Health Check
    api_status = check_api_health()
    status_color = "🟢" if api_status else "🔴"
    status_text = "Online" if api_status else "Offline"
    st.markdown(f"**API Status:** {status_color} {status_text}")
    
    if not api_status:
        st.error("⚠️ Backend API is not available. Please start the server:\n```\npython serving.py\n```")
    
    st.markdown("---")
    
    # Stream Selection
    st.subheader("🔧 Model Configuration")
    selected_stream = st.radio(
        "Select Feature Stream:",
        options=list(STREAM_LABELS.keys()),
        format_func=lambda x: STREAM_LABELS[x],
        index=3  # Default to TRIPLE_FUSION
    )
    
    st.markdown("---")
    
    # File Upload
    st.subheader("📤 Upload Fingerprint")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Upload a fingerprint image for analysis"
    )
    
    analyze_button = st.button("▶ RUN ANALYSIS", use_container_width=True, type="primary", disabled=not api_status)
    
    st.markdown("---")
    st.caption(f"**Version:** 3.0.0 | Platform Engineer Demo")

# ─────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────
st.title("🔍 Fingerprint Alteration Detection Platform")
st.markdown("**AI-powered forensic analysis with multi-stream feature fusion**")

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# Handle analysis
if analyze_button and uploaded_file is not None:
    with st.spinner('🔄 Processing fingerprint...'):
        try:
            # Send request to API
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {'stream': selected_stream}
            
            response = requests.post(f"{API_URL}/predict", files=files, params=params, timeout=60)
            
            if response.status_code == 200:
                st.session_state.result = response.json()
                st.success("✅ Analysis complete!")
            else:
                # Try to parse error as JSON, fall back to text if it fails
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = f"Status {response.status_code}: {response.text[:200]}"
                st.error(f"❌ Error: {error_detail}")
                st.session_state.result = None
        
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. The analysis is taking too long. Please try again.")
            st.session_state.result = None
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Please ensure the server is running on port 8000.")
            st.session_state.result = None
        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")
            st.session_state.result = None

# Display results
if st.session_state.result:
    result = st.session_state.result
    
    # ─── Result Banner ───
    prediction = result['prediction']
    confidence = result['confidence']
    risk_score = result['risk_score']
    threshold = result['threshold']
    
    is_altered = prediction == "ALTERED"
    banner_color = "#E74C3C" if is_altered else "#27AE60"
    icon = "⚠️" if is_altered else "✅"
    
    st.markdown(f"""
    <div style="background-color: {banner_color}; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h1 style="color: white; text-align: center; margin: 0;">{icon} {prediction}</h1>
        <p style="color: white; text-align: center; font-size: 18px; margin: 10px 0 0 0;">
            Confidence: {confidence:.1%} | Threshold: {threshold:.2f} | Stream: {selected_stream}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ─── Metrics Row ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Score", f"{risk_score:.3f}")
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        st.metric("Threshold", f"{threshold:.2f}")
    with col4:
        st.metric("Features", f"{result['feature_stats']['total_dimensions']}D")
    
    st.markdown("---")
    
    # ─── Tab Structure (Matching Tkinter GUI) ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Preprocessing Pipeline",
        "🌊 Feature Extraction",
        "🔍 Pattern Recognition",
        "🧬 Forensic Analysis",
        "📈 Dimensionality Analysis"
    ])
    
    # ─────────────────────────────────────────────────────────
    # TAB 1: Preprocessing Pipeline
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Image Preprocessing Pipeline")
        st.markdown("**Visualization of preprocessing steps applied to fingerprint image**")
        
        preprocessing_images = result.get('preprocessing_images', [])
        if preprocessing_images:
            display_image_grid(preprocessing_images, cols=3, width=250)
        else:
            st.warning("No preprocessing images available")
    
    # ─────────────────────────────────────────────────────────
    # TAB 2: Feature Extraction
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Extraction Visualization")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("**Gabor Filter Bank (8 Orientations)**")
            gabor_images = result.get('gabor_images', [])
            if gabor_images:
                display_image_grid(gabor_images, cols=4, width=120)
            else:
                st.warning("No Gabor images available")
        
        with col_right:
            st.markdown("**Feature Vector Statistics**")
            feature_stats = result.get('feature_stats', {})
            
            if feature_stats:
                eff_stats = feature_stats.get('efficientnet', {})
                gabor_stats = feature_stats.get('gabor', {})
                
                st.markdown(f"""
                **EfficientNet Features:**
                - Dimensions: {eff_stats.get('dimensions', 0)}D
                - Mean: {eff_stats.get('mean', 0):.3f}
                - Std: {eff_stats.get('std', 0):.3f}
                
                **Gabor Texture Features:**
                - Dimensions: {gabor_stats.get('dimensions', 0)}D
                - Mean: {gabor_stats.get('mean', 0):.3f}
                - Std: {gabor_stats.get('std', 0):.3f}
                
                **Total Dimensions:** {feature_stats.get('total_dimensions', 0)}D
                """)
                
                # Feature sample bar chart
                feature_sample = eff_stats.get('sample', [])
                if feature_sample:
                    fig = create_feature_bar_chart(feature_sample)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No feature statistics available")
    
    # ─────────────────────────────────────────────────────────
    # TAB 3: Pattern Recognition
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Pattern Analysis & Decision Confidence")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("**Pattern Visualizations**")
            pattern_images = result.get('pattern_images', [])
            if pattern_images:
                display_image_grid(pattern_images, cols=2, width=200)
            else:
                st.warning("No pattern images available")
        
        with col_right:
            st.markdown("**Classification Decision**")
            
            # Decision info
            st.markdown(f"""
            **Classification:** {prediction}
            - **Confidence:** {confidence:.1%}
            - **Threshold:** {threshold:.2f}
            - **Stream:** {selected_stream}
            """)
            
            # Gauge chart
            fig = create_gauge_chart(risk_score, threshold)
            st.plotly_chart(fig, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────
    # TAB 4: Forensic Analysis
    # ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Biological Consistency Analysis")
        
        if selected_stream == "TRIPLE_FUSION":
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("**Forensic Metrics**")
                forensic_dict = result.get('forensic_dict', {})
                
                if forensic_dict:
                    st.markdown(f"""
                    **Biological Consistency Metrics:**
                    
                    - **Ridge Density:** {forensic_dict.get('ridge_density', 0):.6f}
                    - **Orientation Consistency:** {forensic_dict.get('orientation_consistency', 0):.4f}
                    - **Ridge Thickness Var:** {forensic_dict.get('ridge_thickness_variation', 0):.4f}
                    - **Texture Homogeneity:** {forensic_dict.get('texture_homogeneity', 0):.8f}
                    - **Mean Intensity:** {forensic_dict.get('mean_intensity', 0):.2f}
                    - **Std Intensity:** {forensic_dict.get('std_intensity', 0):.2f}
                    - **Dominant Frequency:** {forensic_dict.get('dominant_frequency', 0):.1f}
                    - **Spectral Centroid:** {forensic_dict.get('spectral_centroid', 0):.2f}
                    - **Ridge Endings:** {forensic_dict.get('ridge_endings', 0):.0f}
                    - **Ridge Bifurcations:** {forensic_dict.get('ridge_bifurcations', 0):.0f}
                    """)
                else:
                    st.warning("No forensic metrics available")
            
            with col_right:
                st.markdown("**Forensic Feature Profile**")
                if forensic_dict:
                    fig = create_forensic_bar_chart(forensic_dict)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Forensic features not used in Stream {selected_stream}. Switch to **TRIPLE FUSION** to enable biological consistency analysis.")
    
    # ─────────────────────────────────────────────────────────
    # TAB 5: Dimensionality Analysis
    # ─────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Dimensionality Reduction Analysis")
        st.markdown("**Current fingerprint shown in context of training dataset**")
        
        dimred_data = result.get('dimred_data', {})
        
        if 'error' in dimred_data:
            st.error(f"❌ {dimred_data['error']}")
        elif dimred_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pca_data = dimred_data.get('pca')
                if pca_data:
                    fig = create_pca_plot(pca_data)
                    st.plotly_chart(fig, use_container_width=True)
                    variance_total = sum(pca_data['variance']) * 100
                    st.caption(f"Variance explained: {variance_total:.1f}%")
            
            with col2:
                tsne_data = dimred_data.get('tsne')
                if tsne_data:
                    fig = create_tsne_plot(tsne_data)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Non-linear embedding")
            
            with col3:
                lda_data = dimred_data.get('lda')
                if lda_data:
                    fig = create_lda_plot(lda_data)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Linear discriminant projection")
            
            st.info("★ = Current fingerprint | 🔴 = Altered samples | 🟢 = Real samples")
        else:
            st.warning("No dimensionality reduction data available")
    
    # ─── System Trace ───
    with st.expander("🔍 System Trace", expanded=False):
        st.code(result.get('system_trace', 'No trace available'), language='text')

else:
    # No results yet - show welcome message
    st.info("""
    👋 **Welcome to the Fingerprint Alteration Detection Platform**
    
    To get started:
    1. Select a feature stream from the sidebar (default: TRIPLE FUSION)
    2. Upload a fingerprint image
    3. Click "▶ RUN ANALYSIS"
    
    The system will perform:
    - **Preprocessing Pipeline**: CLAHE, Bilateral Filter, Orientation Field
    - **Feature Extraction**: EfficientNet (1280D) + Gabor (93D) + Forensic (11D)
    - **Pattern Recognition**: Binary Map, Skeleton, Corner Detection
    - **Forensic Analysis**: Ridge density, texture homogeneity, biological metrics
    - **Dimensionality Analysis**: PCA, t-SNE, LDA projections
    """)
    
    # Example workflow (text-based)
    st.markdown("---")
    st.subheader("📚 Example Analysis Workflow")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 1️⃣ Upload
        - Select fingerprint image
        - Supported formats: BMP, PNG, JPG
        - Size: Any (will be preprocessed)
        """)
    with col2:
        st.markdown("""
        ### 2️⃣ Process
        - Feature extraction (1384D)
        - Deep + Texture + Forensic
        - XGBoost classification
        """)
    with col3:
        st.markdown("""
        ### 3️⃣ Analyze
        - View 5 tabs of results
        - Interactive visualizations
        - Forensic metrics
        """)
