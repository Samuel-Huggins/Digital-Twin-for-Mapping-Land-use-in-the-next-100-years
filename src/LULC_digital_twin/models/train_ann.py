from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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


def main() -> int:
    seed_everything(CFG.SEED)

    # -----------------------------
    # Load data
    # -----------------------------
    train_csv: Path = Path(CFG.TRAIN_CSV)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)

    if CFG.DEBUG:
        print("Loaded:", df.shape, "| path:", train_csv)

    # Drop non-feature columns
    drop_cols = [c for c in df.columns if any(tok in c for tok in CFG.DROP_COLS_CONTAINING)]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna()

    # Use all exported features except label and raw road distance.
    # DIST_ROADS_LOG is kept because it is numerically better behaved for ANN training.
    feature_df = df.drop(columns=[CFG.LABEL_COL], errors="ignore")
    feature_df = feature_df.drop(columns=["DIST_ROADS_M"], errors="ignore")

    y_raw = df[CFG.LABEL_COL].astype(int).values
    X = feature_df.values

    if CFG.DEBUG:
        print("Feature columns:", list(feature_df.columns))
        print("Feature count:", X.shape[1])

    # Map WC codes -> contiguous class IDs
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
    year_tag = train_csv.stem.split("_")[-1]
    EXPERIMENT_TAG = "with_slope_roads_and_water"
    out_dir = CFG.RESULTS_DIR / EXPERIMENT_TAG / year_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Evaluation
    # -----------------------------
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))

    tick_marks = np.arange(len(class_names))

    # Confusion matrix (counts)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title("ANN Confusion Matrix: Slope + Roads + Water")
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_slope_roads_water.png", dpi=300)
    plt.close()

    # Training curves
    plt.figure()
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss_slope_roads_water.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Val accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_accuracy_slope_roads_water.png", dpi=300)
    plt.close()

    # Save model + class codes
    model.save(out_dir / CFG.MODEL_FILENAME)
    pd.Series(unique_codes).to_csv(out_dir / CFG.CODES_FILENAME, index=False)

    print(f"\nSaved outputs to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())