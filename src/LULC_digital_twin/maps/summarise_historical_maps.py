from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INPUT_DIR = Path("results/historical_maps")
OUT_CSV = INPUT_DIR / "historical_class_area_summary.csv"

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


def row_pixel_areas_km2(src: rasterio.io.DatasetReader) -> np.ndarray:
    """
    Returns one pixel-area value per raster row, in km².

    If CRS units are metres, uses transform pixel size directly.
    If CRS is geographic degrees, approximates metres per degree using latitude.
    """
    transform = src.transform

    # Projected CRS, usually metres
    if src.crs is not None and not src.crs.is_geographic:
        pixel_area_m2 = abs(transform.a * transform.e)
        return np.full(src.height, pixel_area_m2 / 1_000_000, dtype=np.float64)

    # Geographic CRS, usually degrees
    # Approximate area per pixel row using latitude.
    rows = np.arange(src.height)

    # Pixel centre latitude for each row
    lat = transform.f + (rows + 0.5) * transform.e

    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)

    # Approx metres per degree.
    metres_per_deg_lat = 110_574.0
    metres_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))

    pixel_width_m = pixel_width_deg * metres_per_deg_lon
    pixel_height_m = pixel_height_deg * metres_per_deg_lat

    pixel_area_km2 = (pixel_width_m * pixel_height_m) / 1_000_000

    return pixel_area_km2.astype(np.float64)


def main() -> int:
    rows = []

    files = sorted(INPUT_DIR.glob("predicted_lulc_*_wc.tif"))

    if not files:
        raise FileNotFoundError(f"No predicted maps found in {INPUT_DIR}")

    for path in files:
        # Expected filename: predicted_lulc_2013_wc.tif
        year = int(path.stem.split("_")[2])

        with rasterio.open(path) as src:
            arr = src.read(1)
            pixel_area_by_row_km2 = row_pixel_areas_km2(src)

            print(
                f"{year}: CRS={src.crs}, "
                f"shape={src.height}x{src.width}, "
                f"pixel area approx range km²="
                f"{pixel_area_by_row_km2.min():.6f}–{pixel_area_by_row_km2.max():.6f}"
            )

        valid = arr != NODATA

        # Total valid area
        total_valid_km2 = 0.0
        for r in range(arr.shape[0]):
            valid_count_row = int(valid[r, :].sum())
            total_valid_km2 += valid_count_row * pixel_area_by_row_km2[r]

        for code, name in CLASS_NAMES.items():
            class_area_km2 = 0.0
            class_pixels = 0

            class_mask = arr == code

            for r in range(arr.shape[0]):
                count_row = int(class_mask[r, :].sum())
                class_pixels += count_row
                class_area_km2 += count_row * pixel_area_by_row_km2[r]

            area_ha = class_area_km2 * 100
            percent = (class_area_km2 / total_valid_km2 * 100) if total_valid_km2 > 0 else 0.0

            rows.append(
                {
                    "year": year,
                    "class_code": code,
                    "class_name": name,
                    "pixels": class_pixels,
                    "area_ha": area_ha,
                    "area_km2": class_area_km2,
                    "percent_of_valid_roi": percent,
                    "total_valid_km2": total_valid_km2,
                    "source_file": str(path),
                }
            )

    df = pd.DataFrame(rows).sort_values(["year", "class_code"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote: {OUT_CSV}")
    print("\nArea km²:")
    print(df.pivot(index="year", columns="class_name", values="area_km2").round(2))

    print("\nPercent of valid ROI:")
    print(df.pivot(index="year", columns="class_name", values="percent_of_valid_roi").round(2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())