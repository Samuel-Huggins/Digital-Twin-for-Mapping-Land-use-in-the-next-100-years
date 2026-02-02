import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "data/London2020_WorldCover_Samples.csv"  # you will download the export to here
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
y = np.array([code_to_id[c] for c in y_raw], dtype=int)

num_classes = len(unique_codes)
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

# -----------------------------
# Evaluation
# -----------------------------
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

print("\nConfusion matrix (class IDs):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report (class IDs):")
print(classification_report(y_test, y_pred, digits=3))

print("\nClass ID ↔ WorldCover code mapping:")
for i in range(num_classes):
    print(f"  {i} -> {id_to_code[i]}")

# Save model + scaler mapping info for reproducibility
model.save("baseline_ann_worldcover.keras")
pd.Series(unique_codes).to_csv("worldcover_codes_used.csv", index=False)
