from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, uniform_filter

BASE_YEAR = 2021
PROJECTION_YEARS = [2023, 2025, 2030, 2050, 2075, 2126]

INPUT_MAP = Path("results/historical_maps/predicted_lulc_2021_wc.tif")

FEATURE_RASTER = Path(
    "data/rasters/features/EastAnglia_Features_slope_roads_water_elevation_tpi_2021.tif"
)

TRANSITION_MATRIX = Path(
    "results/historical_maps/transitions/constrained_transition_probability_matrix.csv"
)

OUT_DIR = Path("results/projections_driver_weighted_neighbourhood")

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


def load_transition_matrix() -> dict[int, dict[int, float]]:
    """
    Loads constrained transition matrix.

    Returns:
        {
            from_code: {to_code: probability}
        }
    """
    if not TRANSITION_MATRIX.exists():
        raise FileNotFoundError(TRANSITION_MATRIX)

    matrix = pd.read_csv(TRANSITION_MATRIX, index_col=0)

    lookup: dict[int, dict[int, float]] = {}

    for from_class_name, row in matrix.iterrows():
        from_code = CLASS_NAME_TO_CODE[from_class_name]
        lookup[from_code] = {}

        for to_class_name, prob in row.items():
            to_code = CLASS_NAME_TO_CODE[to_class_name]
            lookup[from_code][to_code] = float(prob)

    return lookup


def get_band(src: rasterio.io.DatasetReader, name: str) -> np.ndarray:
    """
    Reads a band by its raster description/name.
    """
    descriptions = list(src.descriptions)

    if name not in descriptions:
        raise ValueError(
            f"Band {name!r} not found. Available descriptions: {descriptions}"
        )

    index = descriptions.index(name) + 1
    return src.read(index).astype(np.float32)


def normalise01(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Normalises array to 0-1 over valid pixels.
    """
    out = np.zeros(arr.shape, dtype=np.float32)

    vals = arr[valid_mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return out

    lo = np.nanpercentile(vals, 2)
    hi = np.nanpercentile(vals, 98)

    if hi <= lo:
        return out

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1).astype(np.float32)

    out[~valid_mask] = 0
    return out


def estimate_pixel_size_m(src: rasterio.io.DatasetReader) -> tuple[float, float]:
    """
    Estimates pixel height/width in metres.

    For EPSG:4326, approximates metres-per-degree at the raster centre latitude.
    """
    transform = src.transform

    if src.crs is not None and not src.crs.is_geographic:
        return abs(transform.e), abs(transform.a)

    centre_lat = transform.f + (src.height / 2) * transform.e

    metres_per_deg_lat = 110_574.0
    metres_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(centre_lat))

    pixel_h_m = abs(transform.e) * metres_per_deg_lat
    pixel_w_m = abs(transform.a) * metres_per_deg_lon

    return float(pixel_h_m), float(pixel_w_m)


def distance_to_builtup_m(
    class_map: np.ndarray,
    reference_src: rasterio.io.DatasetReader,
) -> np.ndarray:
    """
    Calculates distance to existing built-up pixels in metres.
    """
    built_mask = class_map == 50

    pixel_h_m, pixel_w_m = estimate_pixel_size_m(reference_src)

    # distance_transform_edt calculates distance to nearest zero.
    # We want distance to built-up, so built-up is zero and non-built-up is one.
    dist = distance_transform_edt(
        ~built_mask,
        sampling=(pixel_h_m, pixel_w_m),
    )

    return dist.astype(np.float32)

def builtup_neighbourhood_density(
    class_map: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Calculates local built-up density using a moving window.

    Output range:
        0 = no built-up nearby
        1 = neighbourhood is entirely built-up

    At 90 m resolution:
        3x3 window ≈ 270 m
        5x5 window ≈ 450 m
        7x7 window ≈ 630 m

    A 5x5 window is a good starting point for settlement-scale clustering.
    """
    built_mask = (class_map == 50).astype(np.float32)
    valid_mask = (class_map != NODATA).astype(np.float32)

    built_sum = uniform_filter(
        built_mask,
        size=window_size,
        mode="constant",
        cval=0.0,
    )

    valid_sum = uniform_filter(
        valid_mask,
        size=window_size,
        mode="constant",
        cval=0.0,
    )

    density = np.divide(
        built_sum,
        valid_sum,
        out=np.zeros_like(built_sum, dtype=np.float32),
        where=valid_sum > 0,
    )

    density[class_map == NODATA] = 0

    return np.clip(density, 0, 1).astype(np.float32)

def build_builtup_suitability(
    current_map: np.ndarray,
    features_path: Path,
) -> np.ndarray:
    """
    Builds a 0-1 suitability surface for built-up expansion.

    Higher = more suitable for built-up transition.
    """
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    with rasterio.open(features_path) as src:
        dist_roads_m = get_band(src, "DIST_ROADS_M")
        dist_water_m = get_band(src, "DIST_WATER_M")
        slope = get_band(src, "SLOPE")
        dist_builtup_m = distance_to_builtup_m(current_map, src)

    local_builtup_density = builtup_neighbourhood_density(
        current_map,
        window_size=5,
    )

    valid = current_map != NODATA

    # High when close to existing built-up.
    # 0 m -> close to 1, 3000+ m -> close to 0.
    builtup_closeness = 1.0 - np.clip(dist_builtup_m / 3000.0, 0, 1)

    # High when close to roads.
    # 0 m -> close to 1, 3000+ m -> close to 0.
    road_closeness = 1.0 - np.clip(dist_roads_m / 3000.0, 0, 1)

    # High when slope is low.
    # 0 degrees -> 1, 8+ degrees -> 0.
    slope_suitability = 1.0 - np.clip(slope / 8.0, 0, 1)

    # Low near persistent water.
    # 0 m from water -> 0, 1000+ m -> 1.
    water_avoidance = np.clip(dist_water_m / 1000.0, 0, 1)

    suitability = (
        0.30 * builtup_closeness
        + 0.25 * road_closeness
        + 0.30 * local_builtup_density
        + 0.10 * slope_suitability
        + 0.05 * water_avoidance
    )

    suitability = np.clip(suitability, 0, 1).astype(np.float32)

    # Built-up expansion should not directly occur into water.
    suitability[current_map == 80] = 0

    # Strongly suppress wetland conversion.
    suitability[current_map == 90] *= 0.15

    # Do not calculate outside valid ROI.
    suitability[~valid] = 0

    return suitability


def sample_non_builtup_targets(
    from_code: int,
    n: int,
    transition_lookup: dict[int, dict[int, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samples target classes excluding built-up.

    Used after built-up expansion pixels have been allocated by suitability.
    """
    row = transition_lookup[from_code]

    to_codes = []
    probs = []

    for to_code, prob in row.items():
        if to_code == 50:
            continue
        to_codes.append(to_code)
        probs.append(prob)

    to_codes_arr = np.array(to_codes, dtype=np.uint16)
    probs_arr = np.array(probs, dtype=np.float64)

    total = probs_arr.sum()

    if total <= 0:
        return np.full(n, from_code, dtype=np.uint16)

    probs_arr = probs_arr / total

    return rng.choice(
        to_codes_arr,
        size=n,
        replace=True,
        p=probs_arr,
    ).astype(np.uint16)


def project_one_step_driver_weighted(
    current: np.ndarray,
    transition_lookup: dict[int, dict[int, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Projects one step forward.

    Transitions into built-up are allocated to the most suitable pixels rather
    than sampled randomly across the landscape.
    """
    projected = np.full(current.shape, NODATA, dtype=np.uint16)
    valid = current != NODATA
    projected[~valid] = NODATA

    suitability = build_builtup_suitability(current, FEATURE_RASTER)

    for from_code in CLASS_CODES:
        from_mask = valid & (current == from_code)
        n_from = int(from_mask.sum())

        if n_from == 0:
            continue

        row = transition_lookup[from_code]

        # Existing built-up uses the transition matrix directly.
        # It is already highly persistent after constraint.
        if from_code == 50:
            to_codes = np.array(list(row.keys()), dtype=np.uint16)
            probs = np.array(list(row.values()), dtype=np.float64)
            probs = probs / probs.sum()

            sampled = rng.choice(
                to_codes,
                size=n_from,
                replace=True,
                p=probs,
            ).astype(np.uint16)

            projected[from_mask] = sampled
            continue

        builtup_prob = row.get(50, 0.0)

        candidate_indices = np.flatnonzero(from_mask.ravel())
        n_to_builtup = int(round(n_from * builtup_prob))

        # Keep within range.
        n_to_builtup = max(0, min(n_to_builtup, n_from))

        if n_to_builtup > 0:
            candidate_suitability = suitability.ravel()[candidate_indices]

            # Pick most suitable pixels for built-up transition.
            order = np.argsort(candidate_suitability)[::-1]
            builtup_indices = candidate_indices[order[:n_to_builtup]]
            remaining_indices = candidate_indices[order[n_to_builtup:]]

            projected.ravel()[builtup_indices] = 50
        else:
            remaining_indices = candidate_indices

        if remaining_indices.size > 0:
            sampled_remaining = sample_non_builtup_targets(
                from_code=from_code,
                n=int(remaining_indices.size),
                transition_lookup=transition_lookup,
                rng=rng,
            )

            projected.ravel()[remaining_indices] = sampled_remaining

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


def class_pixel_summary(array: np.ndarray, year: int) -> pd.DataFrame:
    valid = array != NODATA
    values, counts = np.unique(array[valid], return_counts=True)
    lookup = dict(zip(values.astype(int), counts.astype(int)))

    rows = []

    for code in CLASS_CODES:
        rows.append(
            {
                "year": year,
                "class_code": int(code),
                "class_name": CLASS_CODE_TO_NAME[int(code)],
                "pixels": lookup.get(int(code), 0),
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

    summaries = [class_pixel_summary(current, BASE_YEAR)]

    previous_year = BASE_YEAR

    for target_year in PROJECTION_YEARS:
        steps = target_year - previous_year

        # Matrix was derived from 2-year transitions: 2013→2015, 2015→2017, etc.
        two_year_steps = max(1, round(steps / 2))

        projected = current.copy()

        for _ in range(two_year_steps):
            projected = project_one_step_driver_weighted(
                current=projected,
                transition_lookup=transition_lookup,
                rng=rng,
            )

        out_tif = OUT_DIR / f"projected_lulc_{target_year}_driver_weighted_neighbourhood.tif"
        out_png = OUT_DIR / f"projected_lulc_{target_year}_driver_weighted_neighbourhood.png"

        save_tif(projected, profile, out_tif)
        save_png(projected, out_png)

        summaries.append(class_pixel_summary(projected, target_year))

        print(f"Wrote: {out_tif}")
        print(f"Wrote: {out_png}")

        current = projected
        previous_year = target_year

    summary = pd.concat(summaries, ignore_index=True)
    summary_csv = OUT_DIR / "driver_weighted_neighbourhood_projection_class_pixel_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"\nWrote: {summary_csv}")
    print(summary.pivot(index="year", columns="class_name", values="pixels"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())