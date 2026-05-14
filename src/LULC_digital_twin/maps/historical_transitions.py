from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INPUT_DIR = Path("results/historical_maps")
OUT_DIR = INPUT_DIR / "transitions"

NODATA = 255

CLASS_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    80: "Water",
    90: "Wetland",
}

CLASS_CODES = list(CLASS_NAMES.keys())


def load_map(year: int) -> tuple[np.ndarray, rasterio.Affine, object]:
    path = INPUT_DIR / f"predicted_lulc_{year}_wc.tif"

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    return arr, transform, crs


def row_pixel_areas_km2(height: int, transform: rasterio.Affine, crs) -> np.ndarray:
    if crs is not None and not crs.is_geographic:
        pixel_area_m2 = abs(transform.a * transform.e)
        return np.full(height, pixel_area_m2 / 1_000_000, dtype=np.float64)

    rows = np.arange(height)
    lat = transform.f + (rows + 0.5) * transform.e

    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)

    metres_per_deg_lat = 110_574.0
    metres_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))

    pixel_width_m = pixel_width_deg * metres_per_deg_lon
    pixel_height_m = pixel_height_deg * metres_per_deg_lat

    return (pixel_width_m * pixel_height_m / 1_000_000).astype(np.float64)


def transition_summary(year_a: int, year_b: int) -> pd.DataFrame:
    arr_a, transform, crs = load_map(year_a)
    arr_b, _, _ = load_map(year_b)

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {year_a} {arr_a.shape}, {year_b} {arr_b.shape}")

    valid = (arr_a != NODATA) & (arr_b != NODATA)
    pixel_area_by_row = row_pixel_areas_km2(arr_a.shape[0], transform, crs)

    rows = []

    for from_code in CLASS_CODES:
        for to_code in CLASS_CODES:
            transition_mask = valid & (arr_a == from_code) & (arr_b == to_code)

            pixels = int(transition_mask.sum())
            area_km2 = 0.0

            if pixels > 0:
                for r in range(arr_a.shape[0]):
                    count_row = int(transition_mask[r, :].sum())
                    area_km2 += count_row * pixel_area_by_row[r]

            rows.append(
                {
                    "from_year": year_a,
                    "to_year": year_b,
                    "from_code": from_code,
                    "from_class": CLASS_NAMES[from_code],
                    "to_code": to_code,
                    "to_class": CLASS_NAMES[to_code],
                    "pixels": pixels,
                    "area_km2": area_km2,
                    "area_ha": area_km2 * 100,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    years = [2013, 2015, 2017, 2019, 2020, 2021]
    all_rows = []

    for year_a, year_b in zip(years[:-1], years[1:]):
        df = transition_summary(year_a, year_b)

        out_csv = OUT_DIR / f"transition_{year_a}_{year_b}.csv"
        df.to_csv(out_csv, index=False)

        matrix = df.pivot(
            index="from_class",
            columns="to_class",
            values="area_km2",
        ).fillna(0)

        matrix_out = OUT_DIR / f"transition_matrix_{year_a}_{year_b}_km2.csv"
        matrix.to_csv(matrix_out)

        print(f"\nTransition {year_a} → {year_b}, km²:")
        print(matrix.round(2))

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_transitions_long.csv", index=False)

    print(f"\nWrote transition outputs to: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())