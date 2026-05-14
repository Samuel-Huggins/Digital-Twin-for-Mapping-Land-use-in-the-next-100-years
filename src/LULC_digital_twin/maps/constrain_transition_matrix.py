from __future__ import annotations

from pathlib import Path

import pandas as pd


TRANSITION_DIR = Path("results/historical_maps/transitions")

RAW_MATRIX = TRANSITION_DIR / "stable_transition_probability_matrix_2013_2019.csv"
OUT_MATRIX = TRANSITION_DIR / "constrained_transition_probability_matrix.csv"

CLASS_ORDER = [
    "Tree cover",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse",
    "Water",
    "Wetland",
]


def normalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    return df.div(row_sums.replace(0, 1), axis=0)


def main() -> int:
    if not RAW_MATRIX.exists():
        raise FileNotFoundError(RAW_MATRIX)

    raw = pd.read_csv(RAW_MATRIX, index_col=0)

    # Ensure all expected rows/columns exist.
    matrix = raw.reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0.0)

    # Start from the empirical matrix.
    constrained = matrix.copy()

    # -----------------------------
    # Rule 1: Shrubland is not predicted in the full-ROI maps.
    # Keep as zero to avoid inventing shrubland dynamics.
    # -----------------------------
    constrained.loc["Shrubland", :] = 0.0
    constrained.loc[:, "Shrubland"] = 0.0
    constrained.loc["Shrubland", "Shrubland"] = 1.0

    # -----------------------------
    # Rule 2: Built-up should be highly persistent.
    # Suppress unrealistic built-up reversion to vegetation/agriculture.
    # -----------------------------
    constrained.loc["Built-up", :] = 0.0
    constrained.loc["Built-up", "Built-up"] = 0.95
    constrained.loc["Built-up", "Bare / sparse"] = 0.02
    constrained.loc["Built-up", "Grassland"] = 0.01
    constrained.loc["Built-up", "Cropland"] = 0.01
    constrained.loc["Built-up", "Tree cover"] = 0.01

    # -----------------------------
    # Rule 3: Water is highly persistent.
    # Allow small exchange with wetland.
    # -----------------------------
    constrained.loc["Water", :] = 0.0
    constrained.loc["Water", "Water"] = 0.93
    constrained.loc["Water", "Wetland"] = 0.05
    constrained.loc["Water", "Tree cover"] = 0.01
    constrained.loc["Water", "Bare / sparse"] = 0.01

    # -----------------------------
    # Rule 4: Wetland should mostly remain wetland or move to water/grassland/tree.
    # Suppress direct wetland-to-built-up.
    # -----------------------------
    constrained.loc["Wetland", "Built-up"] = 0.01
    constrained.loc["Wetland", "Water"] = max(constrained.loc["Wetland", "Water"], 0.04)
    constrained.loc["Wetland", "Wetland"] = max(constrained.loc["Wetland", "Wetland"], 0.65)

    # -----------------------------
    # Rule 5: Development pressure can convert suitable open land to built-up,
    # but raw probabilities are too high, so cap them.
    # -----------------------------
    builtup_caps = {
        "Cropland": 0.06,
        "Grassland": 0.04,
        "Bare / sparse": 0.08,
        "Tree cover": 0.02,
        "Wetland": 0.01,
        "Water": 0.00,
    }

    for from_class, cap in builtup_caps.items():
        if from_class in constrained.index:
            constrained.loc[from_class, "Built-up"] = min(
                constrained.loc[from_class, "Built-up"],
                cap,
            )

    # -----------------------------
    # Rule 6: Preserve dominant self-transition for main land classes.
    # -----------------------------
    self_minima = {
        "Tree cover": 0.75,
        "Grassland": 0.70,
        "Cropland": 0.70,
        "Bare / sparse": 0.50,
        "Wetland": 0.65,
    }

    for cls, minimum in self_minima.items():
        constrained.loc[cls, cls] = max(constrained.loc[cls, cls], minimum)

    # Re-normalise so every non-empty row sums to 1.
    constrained = normalise_rows(constrained)

    constrained.to_csv(OUT_MATRIX)

    print(f"Wrote: {OUT_MATRIX}")
    print("\nConstrained transition probability matrix:")
    print(constrained.round(3))

    print("\nRow sums:")
    print(constrained.sum(axis=1).round(3))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())