import os
import sys

from LULC_digital_twin.config import CFG
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from LULC_digital_twin.config import CFG

layers = tf.keras.layers
models = tf.keras.models


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(CFG.SEED)

# -----------------------------
# WorldCover code -> name
# -----------------------------
WC_NAME = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    70: "Snow / ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss / lichen",
}

# -----------------------------
# Load + clean
# -----------------------------
df = pd.read_csv(CFG.TRAIN_CSV)
if CFG.DEBUG:
    print("Loaded:", df.shape, "| path:", CFG.TRAIN_CSV)

# Drop non-feature columns if present
drop_cols = [c for c in df.columns if any(tok in c for tok in CFG.DROP_COLS_CONTAINING)]
if drop_cols and CFG.DEBUG:
    print("Dropping cols:", drop_cols)
df = df.drop(columns=drop_cols, errors="ignore")

df = df.dropna()

y_raw = df[CFG.LABEL_COL].astype(int).values
X = df.drop(columns=[CFG.LABEL_COL]).values

# Map WorldCover codes (10..100) -> contiguous IDs (0..K-1)
unique_codes = sorted(np.unique(y_raw).tolist())
code_to_id = {code: i for i, code in enumerate(unique_codes)}
id_to_code = {i: code for code, i in code_to_id.items()}

y = np.array([code_to_id[c] for c in y_raw], dtype=int)
num_classes = len(unique_codes)

class_names = [WC_NAME.get(id_to_code[i], str(id_to_code[i])) for i in range(num_classes)]

if CFG.DEBUG:
    print("Classes:", num_classes, unique_codes)
    print("Feature columns:", df.drop(columns=[CFG.LABEL_COL]).columns.tolist())


# -----------------------------
# Split + scale
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=CFG.TEST_SIZE,
    random_state=CFG.SEED,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


# -----------------------------
# Model (baseline MLP)
# -----------------------------
model = models.Sequential(
    [
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train_oh,
    validation_split=CFG.VAL_SPLIT,
    epochs=CFG.EPOCHS,
    batch_size=CFG.BATCH_SIZE,
    verbose=1,
)

# -----------------------------
# Outputs folder
# -----------------------------
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)


# -----------------------------
# Evaluation
# -----------------------------
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix (class IDs):")
print(cm)

print("\nClassification report (named classes):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))

if CFG.DEBUG:
    print("\nClass ID ↔ WorldCover code mapping:")
    for i in range(num_classes):
        print(f"  {i} -> {id_to_code[i]} ({class_names[i]})")


# -----------------------------
# Plots
# -----------------------------
tick_marks = np.arange(len(class_names))

# Confusion matrix (counts)
plt.figure(figsize=(8, 7))
plt.imshow(cm)
plt.title("Baseline ANN Confusion Matrix")
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(CFG.RESULTS_DIR, "confusion_matrix_baseline.png"), dpi=300)
plt.show()

# Confusion matrix (row-normalised)
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

plt.figure(figsize=(8, 7))
plt.imshow(cm_norm, vmin=0, vmax=1)
plt.title("Baseline ANN Confusion Matrix (Normalised)")
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(CFG.RESULTS_DIR, "confusion_matrix_baseline_normalised.png"), dpi=300)
plt.show()

# Per-class precision/recall/F1
report = classification_report(
    y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
)
prec = [report[name]["precision"] for name in class_names]
rec = [report[name]["recall"] for name in class_names]
f1 = [report[name]["f1-score"] for name in class_names]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, prec, width, label="Precision")
plt.bar(x, rec, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-score")
plt.xticks(x, class_names, rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Baseline ANN Metrics by Class")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG.RESULTS_DIR, "class_metrics_baseline.png"), dpi=300)
plt.show()

# Training curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline ANN Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG.RESULTS_DIR, "training_loss_baseline.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline ANN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG.RESULTS_DIR, "training_accuracy_baseline.png"), dpi=300)
plt.show()


# -----------------------------
# Save artefacts
# -----------------------------
model.save(os.path.join(CFG.RESULTS_DIR, CFG.MODEL_FILENAME))
pd.Series(unique_codes).to_csv(os.path.join(CFG.RESULTS_DIR, CFG.CODES_FILENAME), index=False)

if CFG.DEBUG:
    print("\nSaved model ->", os.path.join(CFG.RESULTS_DIR, CFG.MODEL_FILENAME))
    print("Saved codes ->", os.path.join(CFG.RESULTS_DIR, CFG.CODES_FILENAME))
