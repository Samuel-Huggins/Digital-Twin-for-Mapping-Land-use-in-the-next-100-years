from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


BASE_YEAR = 2021
PROJECTION_YEARS = [2023, 2025, 2030]

INPUT_MAP = Path("results/historical_maps/predicted_lulc_2021_wc.tif")
TRANSITION_MATRIX = Path(
    "results/historical_maps/transitions/constrained_transition_probability_matrix.csv"
)

OUT_DIR = Path("results/projections")

NODATA = 255
SEED = 42

CLASS_NAME_TO_CODE = {
    "Tree cover": 10,
    "Shrubland": 20,
    "Grassland": 30,
    "Cropland": 40,
    "Built-up": 50,
    "Bare / sparse": 60,
    "Water": 80,
    "Wetland": 90,
}

CLASS_CODE_TO_NAME = {v: k for k, v in CLASS_NAME_TO_CODE.items()}

CLASS_CODES = np.array([10, 20, 30, 40, 50, 60, 80, 90], dtype=np.uint16)


def load_transition_matrix() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Loads transition probabilities and returns:

        {
            from_code: (to_codes, probabilities)
        }
    """
    if not TRANSITION_MATRIX.exists():
        raise FileNotFoundError(TRANSITION_MATRIX)

    matrix = pd.read_csv(TRANSITION_MATRIX, index_col=0)

    transition_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for from_class_name, row in matrix.iterrows():
        from_code = CLASS_NAME_TO_CODE[from_class_name]

        to_codes = []
        probs = []

        for to_class_name, prob in row.items():
            to_code = CLASS_NAME_TO_CODE[to_class_name]
            to_codes.append(to_code)
            probs.append(float(prob))

        to_codes_arr = np.array(to_codes, dtype=np.uint16)
        probs_arr = np.array(probs, dtype=np.float64)

        # Defensive normalisation against floating point rounding.
        total = probs_arr.sum()
        if total <= 0:
            probs_arr = np.zeros_like(probs_arr)
            probs_arr[to_codes_arr == from_code] = 1.0
        else:
            probs_arr = probs_arr / total

        transition_lookup[from_code] = (to_codes_arr, probs_arr)

    return transition_lookup


def project_one_step(
    current: np.ndarray,
    transition_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Projects one timestep forward using class transition probabilities.
    """
    projected = np.full(current.shape, NODATA, dtype=np.uint16)

    valid = current != NODATA
    projected[~valid] = NODATA

    for from_code in CLASS_CODES:
        mask = valid & (current == from_code)
        n = int(mask.sum())

        if n == 0:
            continue

        to_codes, probs = transition_lookup[from_code]

        sampled = rng.choice(
            to_codes,
            size=n,
            replace=True,
            p=probs,
        ).astype(np.uint16)

        projected[mask] = sampled

    return projected


def save_tif(array: np.ndarray, reference_profile: dict, out_path: Path) -> None:
    profile = reference_profile.copy()
    profile.update(
        count=1,
        dtype=rasterio.uint16,
        nodata=NODATA,
        compress="deflate",
        predictor=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.uint16), 1)


def save_png(array: np.ndarray, out_path: Path) -> None:
    masked = np.ma.array(array, mask=array == NODATA)

    plt.figure(figsize=(10, 10))
    plt.imshow(masked, interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04, label="WorldCover class code")
    plt.title(out_path.stem)
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def class_area_summary(array: np.ndarray, year: int) -> pd.DataFrame:
    valid = array != NODATA
    values, counts = np.unique(array[valid], return_counts=True)
    lookup = dict(zip(values.astype(int), counts.astype(int)))

    rows = []

    for code in CLASS_CODES:
        pixels = lookup.get(int(code), 0)
        rows.append(
            {
                "year": year,
                "class_code": int(code),
                "class_name": CLASS_CODE_TO_NAME[int(code)],
                "pixels": pixels,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transition_lookup = load_transition_matrix()
    rng = np.random.default_rng(SEED)

    if not INPUT_MAP.exists():
        raise FileNotFoundError(INPUT_MAP)

    with rasterio.open(INPUT_MAP) as src:
        current = src.read(1).astype(np.uint16)
        profile = src.profile.copy()

    summaries = []

    # Save baseline summary too
    summaries.append(class_area_summary(current, BASE_YEAR))

    previous_year = BASE_YEAR

    for target_year in PROJECTION_YEARS:
        steps = target_year - previous_year

        # Since your transition matrix was derived from two-year intervals
        # between 2013, 2015, 2017 and 2019, apply one transition step for
        # each two-year period.
        two_year_steps = max(1, round(steps / 2))

        projected = current.copy()

        for _ in range(two_year_steps):
            projected = project_one_step(projected, transition_lookup, rng)

        out_tif = OUT_DIR / f"projected_lulc_{target_year}_markov.tif"
        out_png = OUT_DIR / f"projected_lulc_{target_year}_markov.png"

        save_tif(projected, profile, out_tif)
        save_png(projected, out_png)

        summaries.append(class_area_summary(projected, target_year))

        print(f"Wrote: {out_tif}")
        print(f"Wrote: {out_png}")

        current = projected
        previous_year = target_year

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_csv = OUT_DIR / "markov_projection_class_pixel_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nWrote: {summary_csv}")
    print(summary_df.pivot(index="year", columns="class_name", values="pixels"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())