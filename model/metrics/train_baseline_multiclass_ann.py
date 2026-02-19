import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # adjust if needed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

# -----------------------------
# Config
# -----------------------------
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "EastAnglia2020_WorldCover_Samples.csv")
LABEL_COL = "label"

# Keep it simple: use all numeric feature columns except label
DROP_COLS_CONTAINING = [".geo", "system:index"]  # GEE often includes these

EPOCHS = 20
BATCH_SIZE = 128
SEED = 42

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)

# Drop non-feature columns if present
for c in list(df.columns):
    if any(tok in c for tok in DROP_COLS_CONTAINING):
        df = df.drop(columns=[c])

# Remove rows with missing values
df = df.dropna()

# Separate X/y
y_raw = df[LABEL_COL].astype(int).values
X = df.drop(columns=[LABEL_COL]).values

# Map WorldCover codes (10..100) to contiguous class IDs (0..K-1)
unique_codes = sorted(np.unique(y_raw).tolist())
code_to_id = {code: i for i, code in enumerate(unique_codes)}
id_to_code = {i: code for code, i in code_to_id.items()}

# Map WorldCover codes to readable names
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

y = np.array([code_to_id[c] for c in y_raw], dtype=int)
num_classes = len(unique_codes)
class_names = [WC_NAME.get(id_to_code[i], str(id_to_code[i])) for i in range(num_classes)]
print("Classes:", num_classes, unique_codes)

# -----------------------------
# Train/test split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Standardise features (important for ANN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode labels
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# -----------------------------
# Model (baseline MLP)
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train_oh,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

print("\nClass ID ↔ WorldCover code mapping:")
for i in range(num_classes):
    print(f"  {i} -> {id_to_code[i]} ({class_names[i]})")

# ---- Plot 1: Confusion matrix ----
plt.figure(figsize=(8, 7))
plt.imshow(cm)
plt.title("Baseline ANN Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_baseline.png"), dpi=300)
plt.show()

# ---- Plot 1b: Normalised confusion matrix (row-normalised) ----
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
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_baseline_normalised.png"), dpi=300)
plt.show()


# ---- Plot 2: Per-class precision/recall/F1 ----
report = classification_report(
    y_test, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

prec = [report[name]["precision"] for name in class_names]
rec  = [report[name]["recall"] for name in class_names]
f1   = [report[name]["f1-score"] for name in class_names]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, prec, width, label="Precision")
plt.bar(x,         rec,  width, label="Recall")
plt.bar(x + width, f1,   width, label="F1-score")

plt.xticks(x, class_names, rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Baseline ANN Metrics by Class")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_metrics_baseline.png"), dpi=300)
plt.show()

# ---- Plot 3: Training curves ----
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline ANN Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_loss_baseline.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline ANN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_accuracy_baseline.png"), dpi=300)
plt.show()

model.save(os.path.join(RESULTS_DIR, "baseline_ann_worldcover.keras"))
pd.Series(unique_codes).to_csv(os.path.join(RESULTS_DIR, "worldcover_codes_used.csv"), index=False)
