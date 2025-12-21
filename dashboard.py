"""
dashboard.py - Biometric Security Platform Frontend

Client-side dashboard that consumes the FastAPI backend.
Demonstrates full-stack capability: API + UI = Complete Platform.
"""

import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go
import time

# --- Configuration ---
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Supernal AI - Biometric Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Platform" Aesthetic ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0068c9;
        color: white;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444e;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: System Status ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fingerprint.png", width=64)
    st.title("Biometric\nSecurity Platform")
    st.markdown("---")
    
    st.subheader("System Health")
    status_placeholder = st.empty()
    
    # Check Backend Connection
    try:
        health = requests.get(f"{API_URL}/", timeout=2).json()
        status_placeholder.success(f"● Backend Online")
        st.json(health['system'], expanded=False)
    except:
        status_placeholder.error("● Backend Offline")
        st.warning("⚠️ Ensure 'serving.py' is running!")
        st.stop()

    st.markdown("---")
    st.info("Upload a fingerprint BMP/PNG to analyze for alterations.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Fingerprint Image", type=['bmp', 'png', 'jpg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Input Tensor: {uploaded_file.name}", use_container_width=True)
        
        if st.button("🚀 Run Inference Pipeline"):
            with st.spinner("Processing through Triple-Fusion Network..."):
                # Prepare payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
                
                try:
                    start_time = time.time()
                    response = requests.post(f"{API_URL}/predict", files=files)
                    latency = (time.time() - start_time) * 1000
                    result = response.json()
                    
                    # Store result in session state to persist data during reruns
                    st.session_state['last_result'] = result
                    st.session_state['latency'] = latency
                except Exception as e:
                    st.error(f"Inference Failed: {e}")

with col2:
    st.subheader("2. Platform Analysis")
    
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        lat = st.session_state['latency']
        
        # --- Metrics Row ---
        m1, m2, m3 = st.columns(3)
        
        is_altered = res['prediction'] == "ALTERED"
        risk_score = res['risk_score']
        
        with m1:
            st.metric("Prediction", res['prediction'], 
                      delta="High Risk" if is_altered else "Verified", 
                      delta_color="inverse")
        with m2:
            st.metric("Risk Score", f"{risk_score:.2%}")
        with m3:
            st.metric("Inference Latency", f"{lat:.0f}ms")
            
        st.markdown("---")
        
        # --- Gauge Chart (The Visual "Wow") ---
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Alteration Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#ff4b4b" if is_altered else "#00c0f2"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [50, 100], 'color': "rgba(255, 0, 0, 0.1)"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # --- Backend Trace (The "Platform Engineer" Flex) ---
        with st.expander("View Microservice Trace Logs", expanded=True):
            st.markdown("**Raw JSON response from `POST /predict`:**")
            st.json(res['system_trace'])
            
    else:
        # Placeholder Empty State
        st.info("Waiting for inference request...")
        st.markdown("""
        <div style="text-align: center; color: gray; padding: 50px; border: 2px dashed #444;">
            <h3>Awaiting Input Signal</h3>
            <p>Upload an image to trigger the backend pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
