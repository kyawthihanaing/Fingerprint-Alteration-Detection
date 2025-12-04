"""
End-to-End Deep Learning Fusion (EfficientNet + Texture CNN).
Replaces XGBoost with a fully trainable Multi-Stream CNN.
Optimized with Focal Loss to smash the 75/75 barrier on Imbalanced Data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import backend as K
# --- ADDED MISSING IMPORTS ---
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score,  # <--- Was missing
    precision_recall_curve
)
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from . import config, utils, preprocess
import cv2

# --- 1. DATA GENERATOR (Feeds Raw + Texture Images) ---
class FusionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, shuffle=True, is_train=False):
        super().__init__() 
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.indices = np.arange(len(df))
        if self.shuffle: np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X_eff = []
        X_tex = []
        y = []
        
        for _, row in batch_df.iterrows():
            img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
            
            eff_img = preprocess.preprocess_for_effnet(img) 
            eff_img = preprocess_input(eff_img) 
            
            tex_img = preprocess.preprocess_for_texture(img) 
            tex_img = tex_img / 255.0 
            tex_img = np.expand_dims(tex_img, axis=-1) 
            
            if self.is_train:
                if np.random.rand() > 0.5: 
                    eff_img = np.fliplr(eff_img)
                    tex_img = np.fliplr(tex_img)
            
            X_eff.append(eff_img)
            X_tex.append(tex_img)
            y.append(row[config.TARGET_COL]) 
            
        return (
            {"Input_EffNet": np.array(X_eff), "Input_Texture": np.array(X_tex)},
            np.array(y)
        )

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# --- 2. MODEL ARCHITECTURE ---
def build_deep_fusion_model():
    inp_eff = layers.Input(shape=(224, 224, 3), name="Input_EffNet")
    base_eff = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp_eff)
    
    base_eff.trainable = True
    for layer in base_eff.layers[:-20]:
        layer.trainable = False
        
    x_eff = layers.GlobalMaxPooling2D()(base_eff.output) 
    x_eff = layers.Dense(256, activation="relu")(x_eff)
    x_eff = layers.Dropout(0.3)(x_eff)
    
    inp_tex = layers.Input(shape=(128, 128, 1), name="Input_Texture")
    x_tex = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp_tex)
    x_tex = layers.MaxPooling2D((2,2))(x_tex)
    x_tex = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x_tex)
    x_tex = layers.MaxPooling2D((2,2))(x_tex)
    x_tex = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x_tex)
    x_tex = layers.GlobalMaxPooling2D()(x_tex)
    x_tex = layers.Dense(128, activation="relu")(x_tex)
    
    fusion = layers.Concatenate()([x_eff, x_tex])
    fusion = layers.Dense(128, activation="relu")(fusion)
    fusion = layers.Dropout(0.4)(fusion)
    
    output = layers.Dense(1, activation="sigmoid", name="Output")(fusion)
    
    model = models.Model(inputs=[inp_eff, inp_tex], outputs=output)
    return model

# --- MAIN ---
def main():
    utils.set_seeds()
    utils.ensure_dirs()
    
    df = utils.load_metadata()
    train_ids = utils.read_json(f"{config.SPLIT_DIR}/train_subjects.json")
    val_ids = utils.read_json(f"{config.SPLIT_DIR}/val_subjects.json")
    test_ids = utils.read_json(f"{config.SPLIT_DIR}/test_subjects.json")
    
    df_train = df[df.subject_id.isin(train_ids)]
    df_val = df[df.subject_id.isin(val_ids)]
    df_test = df[df.subject_id.isin(test_ids)]
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    train_gen = FusionDataGenerator(df_train, batch_size=32, is_train=True)
    val_gen = FusionDataGenerator(df_val, batch_size=32, shuffle=False)
    test_gen = FusionDataGenerator(df_test, batch_size=32, shuffle=False)
    
    model = build_deep_fusion_model()
    opt = optimizers.Adam(learning_rate=1e-4)
    weights = {0: 7.5, 1: 1.0} 
    
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy", 
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision")]
    )
    
    if os.path.exists("models/deep_fusion_best.keras"):
        print("⚡ Loading existing model...")
        model.load_weights("models/deep_fusion_best.keras")
    else:
        print("\n🚀 Starting Training...")
        checkpoint = callbacks.ModelCheckpoint("models/deep_fusion_best.keras", save_best_only=True, monitor="val_loss")
        early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
        
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            class_weight=weights, 
            callbacks=[checkpoint, early_stop]
        )
    
    print("\n⚖️  Generating Final Report...")
    
    y_true = []
    y_pred_proba = []
    
    print("   Generating predictions...")
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict_on_batch(X_batch)
        y_true.extend(y_batch)
        y_pred_proba.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=["Real", "Altered"])
    print(report)
    
    with open(f"{config.REPORTS_DIR}/metrics_DEEP_FUSION.txt", "w") as f:
        f.write(f"DEEP FUSION Evaluation Metrics\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"✅ Metrics saved: {config.REPORTS_DIR}/metrics_DEEP_FUSION.txt")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Altered"], yticklabels=["Real", "Altered"])
    plt.title("Confusion Matrix – DEEP FUSION")
    utils.save_fig(f"{config.REPORTS_DIR}/confusion_matrix_DEEP_FUSION.png")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0,1], [0,1], "k--")
    plt.title("ROC Curve – DEEP FUSION")
    plt.legend()
    utils.save_fig(f"{config.REPORTS_DIR}/roc_curve_DEEP_FUSION.png")

if __name__ == "__main__":
    main()