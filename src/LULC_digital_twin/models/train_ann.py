from __future__ import annotations

from pathlib import Path
import random
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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


def infer_year_tag(train_csv: Path) -> str:
    """
    Extracts a year from filenames such as:
        EastAnglia_WorldCover_Samples_2020_Post.csv
        EastAnglia_WorldCover_Samples_with_slope_2021.csv

    Falls back to the stem if no year is found.
    """
    match = re.search(r"(19\d{2}|20\d{2})", train_csv.stem)
    return match.group(1) if match else train_csv.stem


def main() -> int:
    seed_everything(CFG.SEED)

    # -----------------------------
    # Experiment name
    # -----------------------------
    EXPERIMENT_TAG = "with_slope_roads_water_elevation_tpi"

    # -----------------------------
    # Load data
    # -----------------------------
    train_csv: Path = Path(CFG.TRAIN_CSV)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)

    if CFG.DEBUG:
        print("Loaded:", df.shape, "| path:", train_csv)

    # Drop non-feature columns such as .geo and system:index
    drop_cols = [
        c for c in df.columns
        if any(tok in c for tok in CFG.DROP_COLS_CONTAINING)
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna()

    if CFG.LABEL_COL not in df.columns:
        raise ValueError(f"Label column not found: {CFG.LABEL_COL}")

    # Use all exported features except label.
    # This preserves the exact CSV feature order for later raster prediction.
    feature_df = df.drop(columns=[CFG.LABEL_COL], errors="ignore")
    feature_cols = feature_df.columns.tolist()

    y_raw = df[CFG.LABEL_COL].astype(int).values
    X = feature_df.values

    if CFG.DEBUG:
        print("Feature columns:", feature_cols)
        print("Feature count:", X.shape[1])

    # -----------------------------
    # Map WC codes -> contiguous class IDs
    # -----------------------------
    unique_codes = sorted(np.unique(y_raw).tolist())
    code_to_id = {code: i for i, code in enumerate(unique_codes)}
    id_to_code = {i: code for code, i in code_to_id.items()}

    y = np.array([code_to_id[c] for c in y_raw], dtype=int)
    num_classes = len(unique_codes)

    class_names = [
        WC_NAME.get(id_to_code[i], str(id_to_code[i]))
        for i in range(num_classes)
    ]

    if CFG.DEBUG:
        print("Classes:", num_classes, unique_codes)

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
    # Model
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
    # Output directory
    # -----------------------------
    year_tag = infer_year_tag(train_csv)
    out_dir = CFG.RESULTS_DIR / EXPERIMENT_TAG / year_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Evaluation
    # -----------------------------
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification report:")
    print(report_text)

    # -----------------------------
    # Save reports and metadata
    # -----------------------------
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    metrics_summary = {
        "experiment_group": EXPERIMENT_TAG,
        "year": year_tag,
        "training_csv": str(train_csv),
        "num_features": int(X.shape[1]),
        "accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "num_test_samples": int(len(y_test)),
        "num_classes": int(num_classes),
        "worldcover_codes": [int(c) for c in unique_codes],
        "class_names": class_names,
        "feature_columns": feature_cols,
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save scaler and exact feature order for full-map inference
    joblib.dump(scaler, out_dir / "feature_scaler.joblib")

    pd.Series(feature_cols).to_csv(
        out_dir / "feature_columns.csv",
        index=False,
        header=False,
    )

    # Save WorldCover class code order
    pd.Series(unique_codes).to_csv(
        out_dir / CFG.CODES_FILENAME,
        index=False,
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    tick_marks = np.arange(len(class_names))

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("ANN Confusion Matrix: Slope + Roads + Water + Elevation + TPI")
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # -----------------------------
    # Training curves
    # -----------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Val accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_accuracy.png", dpi=300)
    plt.close()

    # -----------------------------
    # Save model
    # -----------------------------
    model.save(out_dir / CFG.MODEL_FILENAME)

    print(f"\nSaved outputs to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())