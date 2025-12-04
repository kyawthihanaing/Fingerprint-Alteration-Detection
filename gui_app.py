"""
Fingerprint Liveness Detection System - Professional GUI Application
Final Version for Assignment Submission
Supports: Stream A, B, FUSION, TRIPLE_FUSION with Forensic Analysis
Enhanced with Feature Extraction & Pattern Recognition Visualizations
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
import joblib
import cv2
import os
import threading
import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from src import preprocess, feat_efficientnet, feat_gabor, feat_forensic, config

# --- Configuration & Styling ---
COLORS = {
    "bg_main": "#F0F2F5",       # Modern light gray
    "bg_sidebar": "#FFFFFF",    # White card background
    "header": "#2C3E50",        # Dark Blue
    "accent": "#3498DB",        # Bright Blue
    "success": "#27AE60",       # Green
    "danger": "#E74C3C",        # Red
    "text": "#2C3E50",          
    "text_light": "#7F8C8D",
    "highlight": "#E8F6F3"      # Light cyan for active areas
}

MODEL_PATHS = {
    "A": "models/a_xgboost.joblib",
    "B": "models/b_xgboost.joblib",
    "FUSION": "models/fusion_xgboost.joblib",
    "TRIPLE_FUSION": "models/triple_fusion_xgboost.joblib"
}

STREAM_LABELS = {
    "A": "Stream A (Deep Features)",
    "B": "Stream B (Texture Features)",
    "FUSION": "FUSION (Deep + Texture)",
    "TRIPLE_FUSION": "TRIPLE FUSION (Deep + Texture + Forensic)"
}

# Larger preview size
PREVIEW_SIZE = (150, 150)


class ModelHandler:
    """Handles lazy loading of models and extractors."""
    def __init__(self):
        self.effnet_extractor = None
        self.loaded_models = {}

    def get_effnet(self):
        if self.effnet_extractor is None:
            print("⏳ Initializing EfficientNet Extractor...")
            self.effnet_extractor = feat_efficientnet.build_extractor()
        return self.effnet_extractor

    def get_model(self, stream):
        path = MODEL_PATHS[stream]
        if stream not in self.loaded_models:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file missing: {path}\nPlease run: python -m src.fuse_and_train --stream {stream}")
            print(f"Loading model for {stream}...")
            self.loaded_models[stream] = joblib.load(path)
        return self.loaded_models[stream]


class FingerprintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Liveness Detection System")
        self.root.geometry("1600x950")
        self.root.configure(bg=COLORS["bg_main"])
        
        self.handler = ModelHandler()
        self.image_path = None
        self.stream = tk.StringVar(value="TRIPLE_FUSION")
        self.result_var = tk.StringVar(value="System Ready")
        self.confidence_var = tk.StringVar(value="")
        self.forensic_info_var = tk.StringVar(value="")
        
        self.preprocessing_images = [] 
        self.preprocessing_labels = []
        self.feature_images = []
        self.feature_labels = []
        self.pattern_images = []
        self.pattern_labels = []
        self.history_data = []
        
        # Store extracted features for visualization
        self.last_eff_feat = None
        self.last_tex_feat = None
        self.last_forensic_dict = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- Header ---
        header = tk.Frame(self.root, bg=COLORS["header"], height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="🛡️ Advanced Fingerprint Liveness Detection", 
                 font=("Segoe UI", 22, "bold"), fg="white", bg=COLORS["header"]).pack(side="left", padx=30)
        tk.Label(header, text="Master's Project", 
                 font=("Segoe UI", 11), fg="#BDC3C7", bg=COLORS["header"]).pack(side="right", padx=30)

        # --- Main Content Area ---
        main = tk.Frame(self.root, bg=COLORS["bg_main"])
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # LEFT PANEL (Input & Controls) - Narrower
        left_panel = tk.Frame(main, bg=COLORS["bg_main"], width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # 1. Image Selection Card
        img_card = tk.LabelFrame(left_panel, text=" Input Source ", bg=COLORS["bg_sidebar"], 
                                 font=("Segoe UI", 10, "bold"), fg=COLORS["text"], bd=0, relief="flat")
        img_card.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)
        
        self.img_preview = tk.Label(img_card, bg="#ECF0F1", text="No Image Selected", 
                                    fg=COLORS["text_light"], height=10, font=("Segoe UI", 9))
        self.img_preview.pack(fill="x", pady=8, padx=8)
        
        tk.Button(img_card, text="📂 Select Fingerprint Image", command=self.select_image,
                  bg=COLORS["accent"], fg="white", font=("Segoe UI", 9, "bold"), 
                  relief="flat", cursor="hand2", pady=4).pack(fill="x", padx=8)

        # 2. Model Selection Card
        model_card = tk.LabelFrame(left_panel, text=" Architecture ", bg=COLORS["bg_sidebar"], 
                                   font=("Segoe UI", 10, "bold"), fg=COLORS["text"], bd=0, relief="flat")
        model_card.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)
        
        for s in ["A", "B", "FUSION", "TRIPLE_FUSION"]:
            lbl = STREAM_LABELS[s]
            rb = tk.Radiobutton(model_card, text=lbl, variable=self.stream, value=s, 
                                bg=COLORS["bg_sidebar"], activebackground=COLORS["bg_sidebar"],
                                font=("Segoe UI", 9), fg=COLORS["text"], selectcolor=COLORS["highlight"])
            rb.pack(anchor="w", padx=8, pady=1)

        # 3. Action Card
        action_card = tk.Frame(left_panel, bg=COLORS["bg_sidebar"], bd=0)
        action_card.pack(fill="x", ipadx=8, ipady=8)
        
        self.btn_predict = tk.Button(action_card, text="▶ RUN ANALYSIS", command=self.run_prediction_thread,
                                     bg=COLORS["success"], fg="white", font=("Segoe UI", 11, "bold"), 
                                     relief="flat", cursor="hand2", pady=8)
        self.btn_predict.pack(fill="x", padx=8, pady=4)
        
        self.progress = ttk.Progressbar(action_card, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=4)

        tk.Button(action_card, text="📊 View Metrics", command=self.show_metrics,
                  bg="#95A5A6", fg="white", font=("Segoe UI", 9), relief="flat").pack(fill="x", padx=8, pady=(4,0))

        # 4. Result Banner (Compact)
        self.res_banner = tk.Frame(left_panel, bg=COLORS["header"], pady=12)
        self.res_banner.pack(fill="x", pady=(10, 10))
        
        self.lbl_res = tk.Label(self.res_banner, textvariable=self.result_var, 
                                font=("Segoe UI", 16, "bold"), bg=COLORS["header"], fg="white")
        self.lbl_res.pack()
        self.lbl_conf = tk.Label(self.res_banner, textvariable=self.confidence_var, 
                                 font=("Segoe UI", 10), bg=COLORS["header"], fg="white")
        self.lbl_conf.pack()

        # 5. History Card
        hist_card = tk.LabelFrame(left_panel, text=" Recent Scans ", bg=COLORS["bg_sidebar"], 
                                  font=("Segoe UI", 10, "bold"), fg=COLORS["text"], bd=0)
        hist_card.pack(fill="both", expand=True, pady=(0, 0), ipadx=8, ipady=8)
        
        self.history_list = tk.Listbox(hist_card, font=("Consolas", 8), bg="#ECF0F1", bd=0, height=6)
        self.history_list.pack(fill="both", expand=True, padx=8, pady=4)

        # RIGHT PANEL (Visualizations with Tabs)
        right_panel = tk.Frame(main, bg=COLORS["bg_main"])
        right_panel.pack(side="right", fill="both", expand=True)

        # Create Notebook (Tabs)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Preprocessing Pipeline
        self.tab_preprocess = tk.Frame(self.notebook, bg=COLORS["bg_sidebar"])
        self.notebook.add(self.tab_preprocess, text="  🔧 Preprocessing Pipeline  ")
        self.setup_preprocessing_tab()

        # Tab 2: Feature Extraction Visualization
        self.tab_features = tk.Frame(self.notebook, bg=COLORS["bg_sidebar"])
        self.notebook.add(self.tab_features, text="  📊 Feature Extraction  ")
        self.setup_features_tab()

        # Tab 3: Pattern Recognition Visualization
        self.tab_patterns = tk.Frame(self.notebook, bg=COLORS["bg_sidebar"])
        self.notebook.add(self.tab_patterns, text="  🔍 Pattern Recognition  ")
        self.setup_patterns_tab()

        # Tab 4: Forensic Analysis
        self.tab_forensic = tk.Frame(self.notebook, bg=COLORS["bg_sidebar"])
        self.notebook.add(self.tab_forensic, text="  🧬 Forensic Analysis  ")
        self.setup_forensic_tab()

        # Tab 5: Dimensionality Analysis (PCA, t-SNE, LDA)
        self.tab_dimred = tk.Frame(self.notebook, bg=COLORS["bg_sidebar"])
        self.notebook.add(self.tab_dimred, text="  📉 Dimensionality Analysis  ")
        self.setup_dimred_tab()

    def setup_preprocessing_tab(self):
        """Setup the preprocessing pipeline visualization tab with larger images."""
        tk.Label(self.tab_preprocess, text="Image Preprocessing Pipeline", 
                 font=("Segoe UI", 14, "bold"), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text"]).pack(pady=(15, 10))
        
        # Grid for preprocessing steps - 2 rows x 3 columns for larger images
        self.pipe_grid = tk.Frame(self.tab_preprocess, bg=COLORS["bg_sidebar"])
        self.pipe_grid.pack(expand=True, pady=10)
        
        step_names = [
            ("Raw Input", "Original fingerprint image"),
            ("CLAHE Enhanced", "Contrast Limited AHE"),
            ("Bilateral Filter", "Edge-preserving denoise"),
            ("EfficientNet Input", "224×224 RGB ready"),
            ("Gabor Response", "Texture filter output"),
            ("Ridge Orientation", "Gradient magnitude")
        ]
        
        for i, (name, desc) in enumerate(step_names):
            row, col = i // 3, i % 3
            
            frame = tk.Frame(self.pipe_grid, bg="white", bd=2, relief="groove")
            frame.grid(row=row*2, column=col, padx=20, pady=12)
            
            # Fixed pixel size container frame (200x200)
            img_container = tk.Frame(frame, width=200, height=200, bg="#ECF0F1")
            img_container.pack_propagate(False)  # Prevent resizing
            img_container.pack(padx=4, pady=4)
            
            l = tk.Label(img_container, bg="#ECF0F1")
            l.pack(fill="both", expand=True)
            self.preprocessing_labels.append(l)
            
            # Title and description
            tk.Label(self.pipe_grid, text=name, font=("Segoe UI", 10, "bold"), 
                     bg=COLORS["bg_sidebar"], fg=COLORS["text"]).grid(row=row*2+1, column=col, pady=(2, 0))

    def setup_features_tab(self):
        """Setup feature extraction visualization tab."""
        tk.Label(self.tab_features, text="Feature Extraction Visualization", 
                 font=("Segoe UI", 14, "bold"), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text"]).pack(pady=(15, 5))
        
        # Main container
        feat_container = tk.Frame(self.tab_features, bg=COLORS["bg_sidebar"])
        feat_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left: Gabor Filter Bank Response
        left_frame = tk.LabelFrame(feat_container, text=" Gabor Filter Bank (8 Orientations) ", 
                                   bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.gabor_grid = tk.Frame(left_frame, bg=COLORS["bg_sidebar"])
        self.gabor_grid.pack(expand=True, pady=10)
        
        # 2x4 grid for 8 Gabor orientations
        for i in range(8):
            row, col = i // 4, i % 4
            f = tk.Frame(self.gabor_grid, bg="white", bd=1, relief="solid")
            f.grid(row=row*2, column=col, padx=8, pady=8)
            
            # Fixed pixel size container frame (130x130)
            img_container = tk.Frame(f, width=130, height=130, bg="#ECF0F1")
            img_container.pack_propagate(False)  # Prevent resizing
            img_container.pack(padx=3, pady=3)
            
            l = tk.Label(img_container, bg="#ECF0F1")
            l.pack(fill="both", expand=True)
            self.feature_labels.append(l)
            angle = i * 22.5
            tk.Label(self.gabor_grid, text=f"{angle:.1f}°", font=("Segoe UI", 9), 
                     bg=COLORS["bg_sidebar"]).grid(row=row*2+1, column=col)
        
        # Right: Feature Vector Visualization
        right_frame = tk.LabelFrame(feat_container, text=" Feature Vector Summary ", 
                                    bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Canvas for matplotlib figure
        self.feat_fig_frame = tk.Frame(right_frame, bg=COLORS["bg_sidebar"])
        self.feat_fig_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.feat_info_var = tk.StringVar(value="Run analysis to see feature vector statistics.")
        self.feat_info_label = tk.Label(right_frame, textvariable=self.feat_info_var, 
                                        font=("Consolas", 9), bg="#F8F9F9", justify="left",
                                        anchor="w", padx=10, pady=8)
        self.feat_info_label.pack(fill="x", padx=10, pady=(0, 10))

    def setup_patterns_tab(self):
        """Setup pattern recognition visualization tab."""
        tk.Label(self.tab_patterns, text="Pattern Recognition Analysis", 
                 font=("Segoe UI", 14, "bold"), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text"]).pack(pady=(15, 5))
        
        # Main container
        pattern_container = tk.Frame(self.tab_patterns, bg=COLORS["bg_sidebar"])
        pattern_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left: Ridge Pattern Analysis
        left_frame = tk.LabelFrame(pattern_container, text=" Ridge Pattern Detection ", 
                                   bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.pattern_grid = tk.Frame(left_frame, bg=COLORS["bg_sidebar"])
        self.pattern_grid.pack(expand=True, pady=10)
        
        pattern_names = ["Binary Ridge Map", "Skeleton", "Ridge Endings", "Orientation Field"]
        for i, name in enumerate(pattern_names):
            row, col = i // 2, i % 2
            f = tk.Frame(self.pattern_grid, bg="white", bd=1, relief="solid")
            f.grid(row=row*2, column=col, padx=15, pady=10)
            
            # Fixed pixel size container frame (160x160)
            img_container = tk.Frame(f, width=160, height=160, bg="#ECF0F1")
            img_container.pack_propagate(False)  # Prevent resizing
            img_container.pack(padx=3, pady=3)
            
            l = tk.Label(img_container, bg="#ECF0F1")
            l.pack(fill="both", expand=True)
            self.pattern_labels.append(l)
            tk.Label(self.pattern_grid, text=name, font=("Segoe UI", 10, "bold"), 
                     bg=COLORS["bg_sidebar"]).grid(row=row*2+1, column=col)
        
        # Right: Decision Confidence Visualization
        right_frame = tk.LabelFrame(pattern_container, text=" Classification Decision ", 
                                    bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Canvas for decision visualization
        self.decision_fig_frame = tk.Frame(right_frame, bg=COLORS["bg_sidebar"])
        self.decision_fig_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.decision_info_var = tk.StringVar(value="Run analysis to see classification decision.")
        self.decision_info_label = tk.Label(right_frame, textvariable=self.decision_info_var, 
                                            font=("Consolas", 9), bg="#F8F9F9", justify="left",
                                            anchor="w", padx=10, pady=8)
        self.decision_info_label.pack(fill="x", padx=10, pady=(0, 10))

    def setup_forensic_tab(self):
        """Setup forensic analysis tab."""
        tk.Label(self.tab_forensic, text="Forensic Biological Consistency Analysis", 
                 font=("Segoe UI", 14, "bold"), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text"]).pack(pady=(15, 10))
        
        # Main container
        forensic_container = tk.Frame(self.tab_forensic, bg=COLORS["bg_sidebar"])
        forensic_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left: Forensic metrics display
        left_frame = tk.LabelFrame(forensic_container, text=" Extracted Biological Features ", 
                                   bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.lbl_forensic = tk.Label(left_frame, textvariable=self.forensic_info_var, 
                                     font=("Consolas", 11), bg="#F8F9F9", fg="#2C3E50", justify="left",
                                     relief="sunken", bd=1, padx=15, pady=15, anchor="nw")
        self.lbl_forensic.pack(fill="both", expand=True, padx=15, pady=15)
        self.forensic_info_var.set("Select TRIPLE FUSION and run analysis\nto see forensic biological features.")
        
        # Right: Forensic chart
        right_frame = tk.LabelFrame(forensic_container, text=" Feature Comparison Chart ", 
                                    bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.forensic_fig_frame = tk.Frame(right_frame, bg=COLORS["bg_sidebar"])
        self.forensic_fig_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_dimred_tab(self):
        """Setup dimensionality analysis visualization tab (PCA, t-SNE, LDA)."""
        tk.Label(self.tab_dimred, text="Dimensionality Analysis", 
                 font=("Segoe UI", 14, "bold"), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text"]).pack(pady=(15, 5))
        
        tk.Label(self.tab_dimred, text="Visualize how features separate Real vs Altered samples in reduced dimensions", 
                 font=("Segoe UI", 10), bg=COLORS["bg_sidebar"], 
                 fg=COLORS["text_light"]).pack(pady=(0, 10))
        
        # Main container with 3 plots: PCA, t-SNE, LDA
        dimred_container = tk.Frame(self.tab_dimred, bg=COLORS["bg_sidebar"])
        dimred_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configure grid for 3 equal columns
        dimred_container.columnconfigure(0, weight=1)
        dimred_container.columnconfigure(1, weight=1)
        dimred_container.columnconfigure(2, weight=1)
        dimred_container.rowconfigure(0, weight=1)
        
        # Left: PCA Plot
        pca_frame = tk.LabelFrame(dimred_container, text=" PCA (2D Projection) ", 
                                  bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        pca_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.pca_fig_frame = tk.Frame(pca_frame, bg=COLORS["bg_sidebar"])
        self.pca_fig_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.pca_info = tk.Label(pca_frame, text="Variance explained: --", 
                                 font=("Segoe UI", 9), bg=COLORS["bg_sidebar"], fg=COLORS["text_light"])
        self.pca_info.pack(pady=(0, 5))
        
        # Middle: t-SNE Plot
        tsne_frame = tk.LabelFrame(dimred_container, text=" t-SNE (Manifold Learning) ", 
                                   bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        tsne_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.tsne_fig_frame = tk.Frame(tsne_frame, bg=COLORS["bg_sidebar"])
        self.tsne_fig_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tsne_info = tk.Label(tsne_frame, text="Perplexity: 30 | Non-linear embedding", 
                                  font=("Segoe UI", 9), bg=COLORS["bg_sidebar"], fg=COLORS["text_light"])
        self.tsne_info.pack(pady=(0, 5))
        
        # Right: LDA Plot
        lda_frame = tk.LabelFrame(dimred_container, text=" LDA (Supervised) ", 
                                  bg=COLORS["bg_sidebar"], font=("Segoe UI", 10, "bold"))
        lda_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        self.lda_fig_frame = tk.Frame(lda_frame, bg=COLORS["bg_sidebar"])
        self.lda_fig_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.lda_info = tk.Label(lda_frame, text="Class separation via Fisher criterion", 
                                 font=("Segoe UI", 9), bg=COLORS["bg_sidebar"], fg=COLORS["text_light"])
        self.lda_info.pack(pady=(0, 5))
        
        # Control panel at bottom
        control_frame = tk.Frame(self.tab_dimred, bg=COLORS["bg_sidebar"])
        control_frame.pack(fill="x", padx=20, pady=(5, 15))
        
        self.dimred_status = tk.Label(control_frame, 
                                      text="Run analysis to generate dimensionality reduction plots with sample context", 
                                      font=("Segoe UI", 9), bg=COLORS["bg_sidebar"], fg=COLORS["text_light"])
        self.dimred_status.pack(side="left", padx=10)
        
        # Legend
        legend_frame = tk.Frame(control_frame, bg=COLORS["bg_sidebar"])
        legend_frame.pack(side="right", padx=10)
        
        tk.Label(legend_frame, text="●", font=("Segoe UI", 14), fg="#E74C3C", 
                 bg=COLORS["bg_sidebar"]).pack(side="left")
        tk.Label(legend_frame, text="Altered ", font=("Segoe UI", 9), 
                 bg=COLORS["bg_sidebar"]).pack(side="left", padx=(0, 15))
        tk.Label(legend_frame, text="●", font=("Segoe UI", 14), fg="#27AE60", 
                 bg=COLORS["bg_sidebar"]).pack(side="left")
        tk.Label(legend_frame, text="Real ", font=("Segoe UI", 9), 
                 bg=COLORS["bg_sidebar"]).pack(side="left", padx=(0, 15))
        tk.Label(legend_frame, text="★", font=("Segoe UI", 14), fg="#F39C12", 
                 bg=COLORS["bg_sidebar"]).pack(side="left")
        tk.Label(legend_frame, text="Current Sample", font=("Segoe UI", 9), 
                 bg=COLORS["bg_sidebar"]).pack(side="left")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Fingerprint Image",
            filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg;*.tif;*.tiff")]
        )
        if path:
            self.image_path = path
            # Display preview
            img = Image.open(path)
            img = ImageOps.contain(img, (280, 220), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img)
            self.img_preview.config(image=self.img_tk, height=0)
            
            # Reset results
            self.result_var.set("Ready to Analyze")
            self.confidence_var.set("")
            self.forensic_info_var.set("Select TRIPLE FUSION and run analysis\nto see forensic biological features.")
            self.res_banner.config(bg=COLORS["header"])
            self.lbl_res.config(bg=COLORS["header"])
            self.lbl_conf.config(bg=COLORS["header"])
            
            # Generate all visualizations
            self.generate_previews()
            self.generate_pattern_previews()

    def generate_previews(self):
        """Generate preprocessing pipeline visualizations with larger images."""
        if not self.image_path: 
            return
        self.preprocessing_images = []
        
        raw = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return
        
        steps = []
        
        # 1. Raw
        steps.append(raw)
        
        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(raw)
        steps.append(enhanced)
        
        # 3. Bilateral Denoised
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        steps.append(denoised)
        
        # 4. EfficientNet input (224×224)
        eff = cv2.resize(denoised, (224, 224), interpolation=cv2.INTER_AREA)
        steps.append(eff)
        
        # 5. Gabor Response (sample filter)
        tex = preprocess.preprocess_for_texture(raw)  # 128×128
        kernel = cv2.getGaborKernel((31, 31), 5.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_resp = cv2.filter2D(tex, cv2.CV_32F, kernel)
        gabor_vis = np.clip(np.abs(gabor_resp) * 255 / (np.abs(gabor_resp).max() + 1e-8), 0, 255).astype(np.uint8)
        steps.append(gabor_vis)
        
        # 6. Orientation field visualization
        sobelx = cv2.Sobel(raw, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(raw, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orient_vis = np.clip(magnitude * 255 / (magnitude.max() + 1e-8), 0, 255).astype(np.uint8)
        steps.append(orient_vis)
        
        # Larger preview size
        for i, img_data in enumerate(steps):
            pil = Image.fromarray(img_data)
            pil = ImageOps.contain(pil, (200, 200), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.preprocessing_images.append(tk_img)
            if i < len(self.preprocessing_labels):
                self.preprocessing_labels[i].config(image=tk_img)
        
        # Generate Gabor filter bank visualization
        self.generate_gabor_bank_preview(tex)

    def generate_gabor_bank_preview(self, tex_img):
        """Generate visualization of Gabor filter responses at different orientations."""
        self.feature_images = []
        
        orientations = [i * np.pi / 8 for i in range(8)]  # 8 orientations
        
        for i, theta in enumerate(orientations):
            kernel = cv2.getGaborKernel((31, 31), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(tex_img, cv2.CV_32F, kernel)
            
            # Normalize to 0-255
            vis = np.abs(filtered)
            vis = np.clip(vis * 255 / (vis.max() + 1e-8), 0, 255).astype(np.uint8)
            
            pil = Image.fromarray(vis)
            pil = ImageOps.contain(pil, (130, 130), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.feature_images.append(tk_img)
            
            if i < len(self.feature_labels):
                self.feature_labels[i].config(image=tk_img)

    def generate_pattern_previews(self):
        """Generate pattern recognition visualizations."""
        if not self.image_path:
            return
        
        self.pattern_images = []
        raw = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return
        
        tex = preprocess.preprocess_for_texture(raw)
        
        # 1. Binary Ridge Map (Otsu threshold)
        _, binary = cv2.threshold(tex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Skeleton (morphological thinning approximation)
        kernel = np.ones((3, 3), np.uint8)
        skeleton = cv2.erode(binary, kernel, iterations=2)
        skeleton = cv2.dilate(skeleton, kernel, iterations=1)
        
        # 3. Ridge Endings (corner detection as proxy)
        corners = cv2.cornerHarris(tex.astype(np.float32), 2, 3, 0.04)
        corners_vis = np.zeros_like(tex)
        corners_vis[corners > 0.01 * corners.max()] = 255
        
        # 4. Orientation Field
        sobelx = cv2.Sobel(tex, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(tex, cv2.CV_32F, 0, 1, ksize=5)
        orientation = np.arctan2(sobely, sobelx)
        orient_vis = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        
        patterns = [binary, skeleton, corners_vis, orient_vis]
        
        for i, img_data in enumerate(patterns):
            pil = Image.fromarray(img_data)
            pil = ImageOps.contain(pil, (160, 160), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.pattern_images.append(tk_img)
            
            if i < len(self.pattern_labels):
                self.pattern_labels[i].config(image=tk_img)

    def run_prediction_thread(self):
        if not self.image_path:
            messagebox.showwarning("Input Missing", "Please select an image first.")
            return
        self.btn_predict.config(state="disabled", text="Processing...")
        self.progress.start(10)
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        try:
            stream = self.stream.get()
            
            # Load Model (Pipeline-based)
            model_data = self.handler.get_model(stream)
            model = model_data["model"]  # This is a Pipeline
            threshold = model_data.get("threshold", 0.5)
            
            # Load image
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not read image file")
            
            # --- Feature Extraction ---
            
            # Stream A: EfficientNet features (1280-D)
            eff_img = preprocess.preprocess_for_effnet(img)
            eff_img = feat_efficientnet.preprocess_input(eff_img)
            eff_img = np.expand_dims(eff_img, axis=0)
            eff_extractor = self.handler.get_effnet()
            eff_feat = eff_extractor.predict(eff_img, verbose=0)[0]
            self.last_eff_feat = eff_feat.copy()
            
            # Stream B: Gabor features (93-D)
            tex_img = preprocess.preprocess_for_texture(img)
            kernels = feat_gabor.enhanced_gabor_kernels()
            basic_feat = feat_gabor.enhanced_gabor_features(tex_img, kernels)  # 80-D
            stats_feat = feat_gabor.gabor_texture_statistics(tex_img, kernels)  # 13-D
            tex_feat = np.concatenate([basic_feat, stats_feat])  # 93-D
            self.last_tex_feat = tex_feat.copy()
            
            # Forensic Features (11-D)
            forensic_dict = feat_forensic.extract_biological_features(tex_img)
            self.last_forensic_dict = forensic_dict.copy()
            
            # Ensure correct feature order (must match training)
            forensic_keys = [
                "mean_intensity", "std_intensity", "median_intensity", "ridge_density", 
                "ridge_thickness_variation", "orientation_consistency", "ridge_endings",
                "ridge_bifurcations", "texture_homogeneity", "dominant_frequency", "spectral_centroid"
            ]
            for_feat = np.array([forensic_dict.get(k, 0.0) for k in forensic_keys])
            
            # Construct feature vector based on stream
            if stream == "A":
                X = eff_feat.reshape(1, -1)
            elif stream == "B":
                X = tex_feat.reshape(1, -1)
            elif stream == "FUSION":
                X = np.concatenate([eff_feat, tex_feat]).reshape(1, -1)
            elif stream == "TRIPLE_FUSION":
                X = np.concatenate([eff_feat, tex_feat, for_feat]).reshape(1, -1)
            
            # Predict using Pipeline (handles scaling/PCA internally)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0, 1]
            else:
                proba = float(model.predict(X)[0])
            
            # Update UI in main thread
            self.root.after(0, lambda: self.update_result(proba, threshold, stream, forensic_dict))
            self.root.after(0, lambda: self.update_feature_visualizations(stream, proba, threshold))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        finally:
            self.root.after(0, lambda: [
                self.btn_predict.config(state="normal", text="▶ RUN ANALYSIS"), 
                self.progress.stop()
            ])

    def update_result(self, proba, threshold, stream, forensic_data):
        """Update result display with prediction outcome."""
        is_altered = proba >= threshold
        res_text = "⚠️ ALTERED" if is_altered else "✅ REAL"
        color = COLORS["danger"] if is_altered else COLORS["success"]
        
        # Show probability of the classified class
        if is_altered:
            conf_text = f"Altered Prob: {proba:.1%} | Thr: {threshold:.2f}"
        else:
            conf_text = f"Real Prob: {1-proba:.1%} | Thr: {threshold:.2f}"
        
        self.result_var.set(res_text)
        self.confidence_var.set(conf_text)
        self.res_banner.config(bg=color)
        self.lbl_res.config(bg=color)
        self.lbl_conf.config(bg=color, fg="white")
        
        # Update Forensic Panel
        if stream == "TRIPLE_FUSION":
            info = (
                f"Biological Consistency Metrics:\n"
                f"{'─'*40}\n"
                f"• Ridge Density:           {forensic_data.get('ridge_density', 0):.6f}\n"
                f"• Orientation Consistency: {forensic_data.get('orientation_consistency', 0):.4f}\n"
                f"• Ridge Thickness Var:     {forensic_data.get('ridge_thickness_variation', 0):.4f}\n"
                f"• Texture Homogeneity:     {forensic_data.get('texture_homogeneity', 0):.8f}\n"
                f"• Mean Intensity:          {forensic_data.get('mean_intensity', 0):.2f}\n"
                f"• Std Intensity:           {forensic_data.get('std_intensity', 0):.2f}\n"
                f"• Dominant Frequency:      {forensic_data.get('dominant_frequency', 0):.1f}\n"
                f"• Spectral Centroid:       {forensic_data.get('spectral_centroid', 0):.2f}\n"
                f"• Ridge Endings:           {forensic_data.get('ridge_endings', 0):.0f}\n"
                f"• Ridge Bifurcations:      {forensic_data.get('ridge_bifurcations', 0):.0f}"
            )
            self.forensic_info_var.set(info)
        else:
            self.forensic_info_var.set(
                f"Forensic features not used in Stream {stream}.\n"
                f"Switch to TRIPLE FUSION to enable\n"
                f"biological consistency analysis."
            )
        
        # Update History
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        fname = os.path.basename(self.image_path)
        icon = "🔴" if is_altered else "🟢"
        entry = f"{timestamp} | {icon} {fname[:12]}.. | {proba:.1%}"
        self.history_list.insert(0, entry)

    def update_feature_visualizations(self, stream, proba, threshold):
        """Update feature extraction and pattern recognition visualizations."""
        # Update Feature Vector Info
        if self.last_eff_feat is not None and self.last_tex_feat is not None:
            eff_stats = f"EfficientNet: {len(self.last_eff_feat)}D | μ={self.last_eff_feat.mean():.3f} | σ={self.last_eff_feat.std():.3f}"
            tex_stats = f"Gabor:        {len(self.last_tex_feat)}D | μ={self.last_tex_feat.mean():.3f} | σ={self.last_tex_feat.std():.3f}"
            
            total_dim = len(self.last_eff_feat) + len(self.last_tex_feat)
            if stream == "TRIPLE_FUSION":
                total_dim += 11
            
            self.feat_info_var.set(f"{eff_stats}\n{tex_stats}\nTotal Dimensions: {total_dim}")
        
        # Update Decision Info
        is_altered = proba >= threshold
        decision = "ALTERED" if is_altered else "REAL"
        confidence = proba if is_altered else (1 - proba)
        
        self.decision_info_var.set(
            f"Classification: {decision}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Threshold: {threshold:.2f}\n"
            f"Stream: {stream}"
        )
        
        # Create feature distribution chart
        self.create_feature_chart()
        
        # Create decision gauge chart
        self.create_decision_chart(proba, threshold)
        
        # Create forensic radar chart if TRIPLE_FUSION
        if stream == "TRIPLE_FUSION" and self.last_forensic_dict:
            self.create_forensic_chart()
        
        # Create dimensionality analysis plots
        self.create_dimred_plots()

    def create_feature_chart(self):
        """Create a bar chart showing feature statistics."""
        # Clear previous
        for widget in self.feat_fig_frame.winfo_children():
            widget.destroy()
        
        if self.last_eff_feat is None:
            return
        
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        
        # Sample of EfficientNet features (first 20)
        sample = self.last_eff_feat[:20]
        colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in sample]
        bars = ax.bar(range(len(sample)), sample, color=colors, alpha=0.7)
        ax.set_xlabel("Feature Index (sample)", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_title("EfficientNet Feature Sample", fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=5, rotation=90)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.feat_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def create_decision_chart(self, proba, threshold):
        """Create a gauge-style chart showing classification decision."""
        # Clear previous
        for widget in self.decision_fig_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        
        # Horizontal bar showing probability
        ax.barh(['Altered\nProbability'], [proba], color='#E74C3C', alpha=0.7, height=0.4)
        ax.barh(['Real\nProbability'], [1-proba], color='#27AE60', alpha=0.7, height=0.4)
        
        ax.axvline(x=threshold, color='#2C3E50', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=9)
        ax.set_title("Classification Confidence", fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.tick_params(labelsize=8)
        
        # Add value labels
        ax.text(proba/2, 0, f'{proba:.1%}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text((1-proba)/2, 1, f'{1-proba:.1%}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.decision_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def create_forensic_chart(self):
        """Create a comprehensive visualization for forensic features."""
        # Clear previous
        for widget in self.forensic_fig_frame.winfo_children():
            widget.destroy()
        
        if not self.last_forensic_dict:
            return
        
        # Create figure with subplots - horizontal bar chart for better readability
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=85)
        fig.patch.set_facecolor('#FFFFFF')
        
        # Define features with their expected ranges for proper scaling
        feature_config = [
            ('ridge_density', 'Ridge Density', 0, 0.15, '{:.4f}'),
            ('orientation_consistency', 'Orientation Var.', 0, 2.0, '{:.3f}'),
            ('texture_homogeneity', 'Texture Homog.', 0, 5e-6, '{:.2e}'),
            ('ridge_thickness_variation', 'Thickness Var.', 0, 0.5, '{:.3f}'),
            ('mean_intensity', 'Mean Intensity', 0, 255, '{:.1f}'),
        ]
        
        # Extract values and normalize to 0-1 for visualization
        labels = []
        normalized_values = []
        raw_values = []
        display_texts = []
        
        for key, label, min_val, max_val, fmt in feature_config:
            raw = self.last_forensic_dict.get(key, 0)
            raw_values.append(raw)
            labels.append(label)
            
            # Normalize to 0-1 range based on expected min/max
            norm = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
            norm = max(0, min(1, norm))  # Clamp to [0, 1]
            normalized_values.append(norm)
            
            # Format display text
            display_texts.append(fmt.format(raw))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))  # Blue gradient
        
        bars = ax.barh(y_pos, normalized_values, color=colors, edgecolor='#333333', linewidth=0.5, height=0.6)
        
        # Add value labels - always position outside the bar on the right
        for i, (bar, txt, norm_val) in enumerate(zip(bars, display_texts, normalized_values)):
            # Always place text to the right of the bar
            ax.text(norm_val + 0.03, bar.get_y() + bar.get_height()/2, txt,
                   ha='left', va='center', fontsize=9, fontweight='bold', color='#2C3E50')
        
        # Customize axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1.35)  # Extended to make room for labels
        ax.set_xlabel('Normalized Scale (0-1)', fontsize=8)
        ax.set_title('Forensic Feature Profile', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7)
        
        # Add gridlines
        ax.xaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add reference lines
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Invert y-axis so first feature is at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.forensic_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def create_dimred_plots(self):
        """Create PCA, t-SNE, and LDA plots showing current sample in context."""
        # Clear previous plots
        for frame in [self.pca_fig_frame, self.tsne_fig_frame, self.lda_fig_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        if self.last_eff_feat is None:
            self.dimred_status.config(text="No features extracted yet. Run analysis first.")
            return
        
        self.dimred_status.config(text="Generating dimensionality analysis plots...")
        self.root.update()
        
        try:
            # Load pre-computed features for context (sample from training data)
            eff_features_path = "features/effb0/all.npy"
            eff_index_path = "features/effb0/index.csv"
            
            if not os.path.exists(eff_features_path) or not os.path.exists(eff_index_path):
                self.dimred_status.config(text="Feature files not found. Run feature extraction first.")
                return
            
            # Load dataset features (use a sample for speed)
            import pandas as pd
            all_features = np.load(eff_features_path)
            index_df = pd.read_csv(eff_index_path)
            
            # Sample up to 300 points for visualization speed
            n_samples = min(300, len(all_features))
            np.random.seed(42)
            sample_idx = np.random.choice(len(all_features), n_samples, replace=False)
            
            X_context = all_features[sample_idx]
            # Use 'is_altered' column: 0=Real, 1=Altered
            y_context = index_df.iloc[sample_idx]['is_altered'].values
            
            # Current sample feature
            X_current = self.last_eff_feat.reshape(1, -1)
            
            # Combine for fitting
            X_all = np.vstack([X_context, X_current])
            
            # --- PCA ---
            self.dimred_status.config(text="Computing PCA projection...")
            self.root.update()
            
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_all)
            
            fig_pca = plt.figure(figsize=(3.2, 2.5), dpi=90)
            ax_pca = fig_pca.add_subplot(111)
            fig_pca.patch.set_facecolor('#FFFFFF')
            
            # Plot context points: is_altered=0 (Real=Green), is_altered=1 (Altered=Red)
            for is_altered, color in [(1, '#E74C3C'), (0, '#27AE60')]:
                mask = y_context == is_altered
                ax_pca.scatter(X_pca[:-1][mask, 0], X_pca[:-1][mask, 1], 
                              c=color, alpha=0.5, s=15, edgecolors='white', linewidth=0.2)
            
            # Plot current sample (star marker)
            ax_pca.scatter(X_pca[-1, 0], X_pca[-1, 1], c='#F39C12', marker='*', 
                          s=250, edgecolors='black', linewidth=1.5, zorder=5)
            
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8)
            ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8)
            ax_pca.set_title("PCA Projection", fontsize=9, fontweight='bold')
            ax_pca.tick_params(labelsize=6)
            ax_pca.grid(True, alpha=0.3)
            fig_pca.tight_layout()
            
            canvas_pca = FigureCanvasTkAgg(fig_pca, master=self.pca_fig_frame)
            canvas_pca.draw()
            canvas_pca.get_tk_widget().pack(fill="both", expand=True)
            self.root.update()
            plt.close(fig_pca)
            
            total_var = sum(pca.explained_variance_ratio_[:2]) * 100
            self.pca_info.config(text=f"Variance explained: {total_var:.1f}%")
            
            # --- t-SNE ---
            self.dimred_status.config(text="Computing t-SNE embedding (this may take a moment)...")
            self.root.update()
            
            # Use PCA features for faster t-SNE
            pca_50 = PCA(n_components=min(50, X_all.shape[1]), random_state=42)
            X_pca50 = pca_50.fit_transform(X_all)
            
            perplexity = min(30, max(5, n_samples // 5))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                       max_iter=300, learning_rate='auto', init='pca')
            X_tsne = tsne.fit_transform(X_pca50)
            
            fig_tsne = plt.figure(figsize=(3.2, 2.5), dpi=90)
            ax_tsne = fig_tsne.add_subplot(111)
            fig_tsne.patch.set_facecolor('#FFFFFF')
            
            # is_altered=1 (Altered=Red), is_altered=0 (Real=Green)
            for is_altered, color in [(1, '#E74C3C'), (0, '#27AE60')]:
                mask = y_context == is_altered
                ax_tsne.scatter(X_tsne[:-1][mask, 0], X_tsne[:-1][mask, 1], 
                               c=color, alpha=0.5, s=15, edgecolors='white', linewidth=0.2)
            
            ax_tsne.scatter(X_tsne[-1, 0], X_tsne[-1, 1], c='#F39C12', marker='*', 
                           s=250, edgecolors='black', linewidth=1.5, zorder=5)
            
            ax_tsne.set_xlabel("t-SNE 1", fontsize=8)
            ax_tsne.set_ylabel("t-SNE 2", fontsize=8)
            ax_tsne.set_title("t-SNE Embedding", fontsize=9, fontweight='bold')
            ax_tsne.tick_params(labelsize=6)
            ax_tsne.grid(True, alpha=0.3)
            fig_tsne.tight_layout()
            
            canvas_tsne = FigureCanvasTkAgg(fig_tsne, master=self.tsne_fig_frame)
            canvas_tsne.draw()
            canvas_tsne.get_tk_widget().pack(fill="both", expand=True)
            self.root.update()
            plt.close(fig_tsne)
            
            self.tsne_info.config(text=f"Perplexity: {perplexity} | Non-linear")
            
            # --- LDA ---
            self.dimred_status.config(text="Computing LDA projection...")
            self.root.update()
            
            lda = LDA(n_components=1)
            X_lda_context = lda.fit_transform(X_context, y_context)
            X_lda_current = lda.transform(X_current)
            
            fig_lda = plt.figure(figsize=(3.2, 2.5), dpi=90)
            ax_lda = fig_lda.add_subplot(111)
            fig_lda.patch.set_facecolor('#FFFFFF')
            
            # For 1D LDA, create a strip plot with jitter
            np.random.seed(42)
            # is_altered=1 (Altered=Red, y=1), is_altered=0 (Real=Green, y=0)
            for is_altered, color in [(1, '#E74C3C'), (0, '#27AE60')]:
                mask = y_context == is_altered
                y_jitter = np.random.uniform(-0.3, 0.3, mask.sum())
                ax_lda.scatter(X_lda_context[mask, 0], y_jitter + is_altered, 
                              c=color, alpha=0.5, s=15, edgecolors='white', linewidth=0.2)
            
            # Current sample
            ax_lda.scatter(X_lda_current[0, 0], 0.5, c='#F39C12', marker='*', 
                          s=250, edgecolors='black', linewidth=1.5, zorder=5)
            
            # Add vertical line at 0
            ax_lda.axvline(x=0, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_lda.set_xlabel("LDA Discriminant", fontsize=8)
            ax_lda.set_ylabel("Class", fontsize=8)
            ax_lda.set_yticks([0, 1])
            ax_lda.set_yticklabels(['Real', 'Altered'], fontsize=7)  # 0=Real, 1=Altered
            ax_lda.set_title("LDA Projection", fontsize=9, fontweight='bold')
            ax_lda.tick_params(labelsize=6)
            ax_lda.grid(True, alpha=0.3, axis='x')
            fig_lda.tight_layout()
            
            canvas_lda = FigureCanvasTkAgg(fig_lda, master=self.lda_fig_frame)
            canvas_lda.draw()
            canvas_lda.get_tk_widget().pack(fill="both", expand=True)
            self.root.update()
            plt.close(fig_lda)
            
            self.dimred_status.config(text=f"✓ Analysis complete ({n_samples} samples). ★ = Current fingerprint")
            
        except Exception as e:
            self.dimred_status.config(text=f"Error: {str(e)[:60]}")
            import traceback
            traceback.print_exc()

    def show_metrics(self):
        """Display metrics report for selected stream."""
        stream = self.stream.get()
        path = f"reports/metrics_{stream}.txt"
        
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            
            # Create new window
            top = tk.Toplevel(self.root)
            top.title(f"Performance Metrics - Stream {stream}")
            top.geometry("650x550")
            top.configure(bg=COLORS["bg_main"])
            
            # Header
            tk.Label(top, text=f"📊 {STREAM_LABELS[stream]}", 
                     font=("Segoe UI", 14, "bold"), bg=COLORS["bg_main"], 
                     fg=COLORS["text"]).pack(pady=12)
            
            # Text display with scrollbar
            frame = tk.Frame(top, bg=COLORS["bg_main"])
            frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            
            text = tk.Text(frame, font=("Consolas", 10), padx=12, pady=12, 
                          wrap="word", yscrollcommand=scrollbar.set,
                          bg="white", fg=COLORS["text"])
            text.pack(fill="both", expand=True)
            scrollbar.config(command=text.yview)
            
            text.insert("1.0", content)
            text.config(state="disabled")
            
            # Close button
            tk.Button(top, text="Close", command=top.destroy,
                     bg=COLORS["accent"], fg="white", font=("Segoe UI", 9),
                     relief="flat", padx=15, pady=4).pack(pady=8)
        else:
            messagebox.showinfo(
                "Metrics Not Found", 
                f"Metrics file not found: {path}\n\n"
                f"Please run evaluation first:\n"
                f"python -m src.evaluate --stream {stream}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    
    # DPI awareness for Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = FingerprintGUI(root)
    root.mainloop()
