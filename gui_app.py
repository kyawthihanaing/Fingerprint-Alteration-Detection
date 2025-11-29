import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import joblib
import cv2
import os
import threading
import datetime

# Import your local modules
from src import preprocess, feat_efficientnet, feat_gabor, config

# --- Configuration & Styling ---
COLORS = {
    "bg_main": "#F4F6F9",       
    "bg_dark": "#2C3E50",       
    "accent": "#3498DB",        
    "success": "#27AE60",       
    "danger": "#E74C3C",        
    "text": "#2C3E50",          
    "text_light": "#7F8C8D",    
    "card": "#FFFFFF",
    "highlight": "#FFF3CD"      
}

MODEL_PATHS = {
    "A": "models/effb0_pca384_rf.joblib",
    "B": "models/gabor_enhanced_rf.joblib",
    "FUSION": "models/fusion_enhanced_pca384_rf.joblib"
}

STREAM_LABELS = {
    "A": "Stream A: EfficientNet (Deep)",
    "B": "Stream B: Gabor (Texture)",
    "FUSION": "Fusion (Best Performance)"
}

class ModelHandler:
    """Handles loading models once and caching them to improve performance."""
    def __init__(self):
        self.extractor = None
        self.loaded_models = {}

    def get_extractor(self):
        """Lazy load the EfficientNet extractor."""
        if self.extractor is None:
            print("⏳ Initializing EfficientNet Extractor (One-time setup)...")
            self.extractor = feat_efficientnet.build_extractor()
        return self.extractor

    def get_model_data(self, stream):
        """Load and return the sklearn model/pca/scaler."""
        path = MODEL_PATHS[stream]
        if stream not in self.loaded_models:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            print(f"Loading model for {stream}...")
            self.loaded_models[stream] = joblib.load(path)
        return self.loaded_models[stream]

class FingerprintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Alteration Detection System v2.0")
        self.root.geometry("1350x900")
        self.root.configure(bg=COLORS["bg_main"])
        
        # Logic Handler
        self.handler = ModelHandler()

        # State Variables
        self.image_path = None
        self.stream = tk.StringVar(value="FUSION")
        self.result_var = tk.StringVar(value="Ready")
        self.confidence_var = tk.StringVar(value="")
        self.history_data = [] # Stores recent scans
        
        self.preprocessing_images = [] 
        self.preprocessing_labels = []

        self.setup_ui()

    def _clahe(self, gray_image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def setup_ui(self):
        # 1. Header
        header = tk.Frame(self.root, bg=COLORS["bg_dark"], height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="🛡️ Fingerprint Alteration Detection System", 
                 font=("Segoe UI", 22, "bold"), fg="white", bg=COLORS["bg_dark"]).pack(side="left", padx=30)
        
        # 2. Main Layout (3 Columns: Left=Input, Middle=Pipeline/Result, Right=History/Controls)
        main_container = tk.Frame(self.root, bg=COLORS["bg_main"])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # --- LEFT COLUMN: Input & Preview ---
        left_col = tk.Frame(main_container, bg=COLORS["bg_main"], width=300)
        left_col.pack(side="left", fill="y", padx=(0, 20))

        # Input Card
        input_card = tk.Frame(left_col, bg=COLORS["card"], relief="flat", bd=1, padx=15, pady=15)
        input_card.pack(fill="x")
        
        tk.Label(input_card, text="1. Select Input", font=("Segoe UI", 12, "bold"), 
                 bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")
        
        self.img_label = tk.Label(input_card, bg="#ECF0F1", text="No Image", fg=COLORS["text_light"], height=10)
        self.img_label.pack(pady=15, fill="x")

        tk.Button(input_card, text="📂 Browse Image...", command=self.select_image,
                  font=("Segoe UI", 10), bg=COLORS["accent"], fg="white", 
                  relief="flat", pady=5, cursor="hand2").pack(fill="x")

        # Config Card
        config_card = tk.Frame(left_col, bg=COLORS["card"], relief="flat", bd=1, padx=15, pady=15)
        config_card.pack(fill="x", pady=20)

        tk.Label(config_card, text="2. Select Stream", font=("Segoe UI", 12, "bold"), 
                 bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w", pady=(0,10))
        
        for stream in ["A", "B", "FUSION"]:
            tk.Radiobutton(config_card, text=STREAM_LABELS[stream], variable=self.stream, 
                           value=stream, bg=COLORS["card"], activebackground=COLORS["card"],
                           font=("Segoe UI", 9)).pack(anchor="w", pady=2)

        # --- MIDDLE COLUMN: Visualization & Results ---
        mid_col = tk.Frame(main_container, bg=COLORS["bg_main"])
        mid_col.pack(side="left", fill="both", expand=True)

        # Result Banner (Dynamic Color)
        self.result_frame = tk.Frame(mid_col, bg=COLORS["bg_dark"], pady=15)
        self.result_frame.pack(fill="x", pady=(0, 20))
        
        self.lbl_result = tk.Label(self.result_frame, textvariable=self.result_var, 
                                   font=("Segoe UI", 20, "bold"), bg=COLORS["bg_dark"], fg="white")
        self.lbl_result.pack()
        self.lbl_conf = tk.Label(self.result_frame, textvariable=self.confidence_var, 
                                 font=("Segoe UI", 12), bg=COLORS["bg_dark"], fg="white")
        self.lbl_conf.pack()

        # Pipeline
        pipeline_frame = tk.LabelFrame(mid_col, text=" Preprocessing Pipeline ", font=("Segoe UI", 11, "bold"),
                                       bg=COLORS["bg_main"], fg=COLORS["text"], bd=1, relief="solid")
        pipeline_frame.pack(fill="both", expand=True)
        
        self.pipeline_grid = tk.Frame(pipeline_frame, bg=COLORS["bg_main"])
        self.pipeline_grid.pack(expand=True, pady=20)

        step_titles = [
            "Raw Input", "CLAHE", "Denoised",
            "EffNet Ready\n(224x224)", "Gabor\nResponse", "Texture Ready\n(128x128)"
        ]

        for i in range(6):
            f = tk.Frame(self.pipeline_grid, bg=COLORS["card"], bd=1, relief="solid")
            f.grid(row=0, column=i, padx=5, pady=5)
            lbl = tk.Label(f, bg="#ECF0F1", width=16, height=8)
            lbl.pack(padx=2, pady=2)
            self.preprocessing_labels.append(lbl)
            tk.Label(self.pipeline_grid, text=step_titles[i], font=("Segoe UI", 8, "bold"), 
                     bg=COLORS["bg_main"], fg=COLORS["text"]).grid(row=1, column=i, pady=(5, 0))

        # --- RIGHT COLUMN: Actions & History ---
        right_col = tk.Frame(main_container, bg=COLORS["bg_main"], width=250)
        right_col.pack(side="right", fill="y", padx=(20, 0))

        # Actions
        action_card = tk.Frame(right_col, bg=COLORS["card"], padx=15, pady=15)
        action_card.pack(fill="x")
        
        tk.Label(action_card, text="3. Actions", font=("Segoe UI", 12, "bold"), bg=COLORS["card"]).pack(anchor="w")

        self.btn_run = tk.Button(action_card, text="▶ Run Analysis", command=self.run_prediction_thread,
                                 font=("Segoe UI", 11, "bold"), bg=COLORS["success"], fg="white", 
                                 relief="flat", pady=8, cursor="hand2")
        self.btn_run.pack(fill="x", pady=10)
        
        # Progress Bar
        self.progress = ttk.Progressbar(action_card, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))

        tk.Button(action_card, text="📊 View Metrics Dashboard", command=self.show_metrics_dashboard,
                  font=("Segoe UI", 10), bg="#95A5A6", fg="white", 
                  relief="flat", pady=5, cursor="hand2").pack(fill="x")

        # History
        hist_card = tk.Frame(right_col, bg=COLORS["card"], padx=15, pady=15, height=300)
        hist_card.pack(fill="x", pady=20)
        tk.Label(hist_card, text="Recent Scans", font=("Segoe UI", 12, "bold"), bg=COLORS["card"]).pack(anchor="w", pady=(0,5))
        
        self.history_list = tk.Listbox(hist_card, font=("Consolas", 9), bg="#ECF0F1", bd=0, height=15)
        self.history_list.pack(fill="both")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")])
        if path:
            self.image_path = path
            img = Image.open(path)
            img = ImageOps.contain(img, (260, 200), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.img_tk, height=0)
            self.generate_preprocessing_steps()
            
            # Reset
            self.result_var.set("Ready to Analyze")
            self.confidence_var.set("")
            self.result_frame.config(bg=COLORS["bg_dark"])
            self.lbl_result.config(bg=COLORS["bg_dark"], fg="white")
            self.lbl_conf.config(bg=COLORS["bg_dark"], fg="white")

    def generate_preprocessing_steps(self):
        if not self.image_path: return
        self.preprocessing_images = [] 
        raw = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        
        # --- Preprocessing Pipeline ---
        steps = []
        steps.append(raw) # 1. Raw
        clahe_applied = self._clahe(raw)
        steps.append(clahe_applied) # 2. CLAHE
        bilateral = cv2.bilateralFilter(clahe_applied, 9, 75, 75)
        steps.append(bilateral) # 3. Denoised
        eff_final = cv2.resize(bilateral, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        steps.append(eff_final) # 4. EffNet Ready
        
        # Gabor Vis
        tex_resized = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA)
        tex_norm = (tex_resized.astype("float32") / 255.0)
        g_kernel = cv2.getGaborKernel((31, 31), 5.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        g_resp = cv2.filter2D(tex_norm, cv2.CV_32F, g_kernel)
        g_vis = ((np.abs(g_resp) / np.abs(g_resp).max()) * 255).astype(np.uint8)
        steps.append(g_vis) # 5. Gabor
        
        tex_ready = preprocess.preprocess_for_texture(raw)
        steps.append(tex_ready) # 6. Texture Ready

        for i, img_data in enumerate(steps):
            pil = Image.fromarray(img_data)
            pil = ImageOps.contain(pil, (110, 110), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.preprocessing_images.append(tk_img)
            self.preprocessing_labels[i].config(image=tk_img, width=0, height=0)

    def run_prediction_thread(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return
        
        # UI State updates
        self.btn_run.config(state="disabled", text="Processing...")
        self.progress.start(10) # Start indeterminate progress bar
        
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        try:
            stream = self.stream.get()
            model_data = self.handler.get_model_data(stream)
            threshold = model_data.get("threshold", 0.5)
            
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            
            # Feature Extraction Logic
            if stream == "A" or stream == "FUSION":
                # Only load extractor if needed
                extractor = self.handler.get_extractor()
                eff_img = preprocess.preprocess_for_effnet(img)
                eff_input = feat_efficientnet.preprocess_input(eff_img) # Keras norm
                eff_input = np.expand_dims(eff_input, axis=0)
                eff_feat = extractor.predict(eff_input, verbose=0)[0]

            if stream == "B" or stream == "FUSION":
                tex_img = preprocess.preprocess_for_texture(img)
                kernels = feat_gabor.enhanced_gabor_kernels()
                basic_feat = feat_gabor.enhanced_gabor_features(tex_img, kernels)
                adv_feat = feat_gabor.gabor_texture_statistics(tex_img, kernels)
                tex_feat = np.concatenate([basic_feat, adv_feat])

            # Combine Features
            if stream == "A":
                X = eff_feat.reshape(1, -1)
            elif stream == "B":
                X = tex_feat.reshape(1, -1)
            elif stream == "FUSION":
                X = np.concatenate([eff_feat, tex_feat]).reshape(1, -1)

            # Transform & Predict
            X_scaled = model_data["scaler"].transform(X)
            if "pca" in model_data:
                X_final = model_data["pca"].transform(X_scaled)
            else:
                X_final = X_scaled
                
            proba = model_data["model"].predict_proba(X_final)[0, 1]
            
            # UI Updates must happen in main thread
            self.root.after(0, lambda: self.update_results(proba, threshold))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        finally:
            self.root.after(0, self.reset_ui_state)

    def update_results(self, proba, threshold):
        is_altered = proba >= threshold
        text = "⚠️ ALTERED DETECTED" if is_altered else "✅ REAL FINGERPRINT"
        color = COLORS["danger"] if is_altered else COLORS["success"]
        
        self.result_var.set(text)
        self.confidence_var.set(f"Probability: {proba:.2%} (Threshold: {threshold})")
        
        self.result_frame.config(bg=color)
        self.lbl_result.config(bg=color)
        self.lbl_conf.config(bg=color)
        
        # Update History
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        fname = os.path.basename(self.image_path)
        icon = "🔴" if is_altered else "🟢"
        entry = f"{timestamp} | {icon} {fname[:10]}... | {proba:.2f}"
        self.history_list.insert(0, entry)

    def reset_ui_state(self):
        self.btn_run.config(state="normal", text="▶ Run Analysis")
        self.progress.stop()

    def show_metrics_dashboard(self):
        stream = self.stream.get()
        dashboard = tk.Toplevel(self.root)
        dashboard.title(f"Metrics Dashboard - {STREAM_LABELS[stream]}")
        dashboard.geometry("900x700")
        
        notebook = ttk.Notebook(dashboard)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        tab_report = tk.Frame(notebook)
        tab_cm = tk.Frame(notebook)
        tab_roc = tk.Frame(notebook)
        tab_feat = tk.Frame(notebook)

        notebook.add(tab_report, text=" 📄 Text Report ")
        notebook.add(tab_cm, text=" 🟦 Confusion Matrix ")
        notebook.add(tab_roc, text=" 📈 ROC / PR Curves ")
        if stream != "A": # Stream A doesn't have feature importance plot in evaluate.py usually unless modified
             notebook.add(tab_feat, text=" 📊 Feature Importance ")

        # 1. Load Text Report
        metrics_file = f"reports/metrics_{stream}.txt"
        text_area = tk.Text(tab_report, font=("Consolas", 11), padx=20, pady=20)
        text_area.pack(fill="both", expand=True)
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f: text_area.insert("1.0", f.read())
        else:
            text_area.insert("1.0", "Report file not found. Run evaluation first.")
        text_area.config(state="disabled")

        # 2. Helper to load images
        def display_image(parent, filename):
            path = f"reports/{filename}"
            if os.path.exists(path):
                img = Image.open(path)
                # Resize to fit window roughly
                img = ImageOps.contain(img, (800, 600), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(parent, image=tk_img, bg="white")
                lbl.image = tk_img # Keep reference
                lbl.pack(expand=True)
            else:
                tk.Label(parent, text=f"Image not found:\n{path}", font=("Segoe UI", 14)).pack(expand=True)

        # 3. Load Images into Tabs
        display_image(tab_cm, f"confusion_matrix_{stream}.png")
        display_image(tab_roc, f"roc_curve_{stream}.png")
        
        if stream == "B":
            display_image(tab_feat, "rf_feature_importance_gabor_enhanced.png")
        elif stream == "FUSION":
            display_image(tab_feat, "rf_feature_importance_fusion.png")

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    app = FingerprintGUI(root)
    root.mainloop()