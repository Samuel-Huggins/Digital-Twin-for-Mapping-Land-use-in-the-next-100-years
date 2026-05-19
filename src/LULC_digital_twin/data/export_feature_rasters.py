"""
Exports full-ROI multiband feature rasters for historical land-cover inference.

This script does NOT export WorldCover labels.
It exports the same feature stack used by the final ANN classifier so that
predict_map.py can classify the whole ROI for prior years.
"""

from __future__ import annotations

import time

import ee
from tqdm import tqdm

from LULC_digital_twin.config import CFG
from LULC_digital_twin.roi import build_roi, debug_point_checks
from LULC_digital_twin.landsat import get_annual_landsat_composite


EXPERIMENT_TAG = "slope_roads_water_elevation_tpi"

# Start with Landsat 8-era years first for sensor consistency.
INFERENCE_YEARS = [2015, 2017, 2019, 2020, 2021] #2013 was done individually


def wait_for_tasks(tasks: list[ee.batch.Task], poll_s: int = 30) -> None:
    """
    Blocks until all Earth Engine export tasks finish.
    """
    remaining = set(range(len(tasks)))

    with tqdm(total=len(tasks), desc="EE raster exports completed") as pbar:
        while remaining:
            finished_now = []

            for i in list(remaining):
                state = tasks[i].status().get("state")
                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    print(f"Task {i} finished with state: {state}")
                    finished_now.append(i)

            for i in finished_now:
                remaining.remove(i)
                pbar.update(1)

            if remaining:
                time.sleep(poll_s)


def get_road_distance_bands(roi: ee.Geometry, max_distance_m: int = 7_500) -> ee.Image:
    """
    Creates road proximity bands.

    Returns:
        DIST_ROADS_M
        DIST_ROADS_LOG
    """
    roads = (
        ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Europe")
        .filterBounds(roi)
    )

    road_mask = (
        ee.Image(0)
        .byte()
        .paint(featureCollection=roads, color=1)
        .selfMask()
        .clip(roi)
    )

    dist_roads_m = (
        road_mask
        .distance(ee.Kernel.euclidean(max_distance_m, "meters"))
        .rename("DIST_ROADS_M")
        .unmask(max_distance_m)
        .toFloat()
        .clip(roi)
    )

    dist_roads_log = (
        dist_roads_m
        .add(1)
        .log()
        .rename("DIST_ROADS_LOG")
        .toFloat()
        .clip(roi)
    )

    return dist_roads_m.addBands(dist_roads_log)


def get_terrain_bands(roi: ee.Geometry) -> ee.Image:
    """
    Creates terrain bands from SRTM.

    Returns:
        ELEVATION
        SLOPE
        TPI_300M
    """
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(roi)

    elevation = (
        dem
        .rename("ELEVATION")
        .toFloat()
        .clip(roi)
    )

    slope = (
        ee.Terrain.slope(dem)
        .rename("SLOPE")
        .toFloat()
        .clip(roi)
    )

    elev_mean_300m = elevation.focal_mean(
        radius=300,
        units="meters",
    )

    tpi_300m = (
        elevation
        .subtract(elev_mean_300m)
        .rename("TPI_300M")
        .toFloat()
        .clip(roi)
    )

    return elevation.addBands([slope, tpi_300m])


def get_water_context_bands(roi: ee.Geometry) -> ee.Image:
    """
    Creates hydrological context bands.

    Returns:
        DIST_WATER_M
        WATER_OCCURRENCE
    """
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)

    water_occurrence = (
        gsw
        .select("occurrence")
        .unmask(0)
        .rename("WATER_OCCURRENCE")
        .toFloat()
        .clip(roi)
    )

    persistent_water = water_occurrence.gte(80).selfMask()

    dist_water_m = (
        persistent_water
        .fastDistanceTransform(neighborhood=1024, units="pixels")
        .sqrt()
        .multiply(CFG.SCALE_M)
        .rename("DIST_WATER_M")
        .toFloat()
        .clip(roi)
    )

    return dist_water_m.addBands(water_occurrence)


def build_feature_stack(year: int, roi: ee.Geometry) -> ee.Image:
    """
    Builds the feature stack required for ANN prediction.

    CRITICAL:
    The band order must match feature_columns.csv exactly:

        BLUE
        BSI
        DIST_ROADS_LOG
        DIST_ROADS_M
        DIST_WATER_M
        ELEVATION
        GREEN
        NDBI
        NDMI
        NDVI
        NDWI
        NIR
        RED
        SLOPE
        SWIR1
        SWIR2
        TPI_300M
        WATER_OCCURRENCE
    """
    composite = get_annual_landsat_composite(year, roi, debug=CFG.DEBUG)

    blue = composite.select("BLUE")
    green = composite.select("GREEN")
    red = composite.select("RED")
    nir = composite.select("NIR")
    swir1 = composite.select("SWIR1")
    swir2 = composite.select("SWIR2")

    eps = ee.Image.constant(1e-6)

    ndvi = (
        nir.subtract(red)
        .divide(nir.add(red).add(eps))
        .rename("NDVI")
        .toFloat()
    )

    ndbi = (
        swir1.subtract(nir)
        .divide(swir1.add(nir).add(eps))
        .rename("NDBI")
        .toFloat()
    )

    ndwi = (
        green.subtract(nir)
        .divide(green.add(nir).add(eps))
        .rename("NDWI")
        .toFloat()
    )

    ndmi = (
        nir.subtract(swir1)
        .divide(nir.add(swir1).add(eps))
        .rename("NDMI")
        .toFloat()
    )

    bsi_num = swir1.add(red).subtract(nir.add(blue))
    bsi_den = swir1.add(red).add(nir.add(blue)).add(eps)

    bsi = (
        bsi_num
        .divide(bsi_den)
        .rename("BSI")
        .toFloat()
    )

    road_bands = get_road_distance_bands(roi)
    terrain_bands = get_terrain_bands(roi)
    water_bands = get_water_context_bands(roi)

    dist_roads_log = road_bands.select("DIST_ROADS_LOG")
    dist_roads_m = road_bands.select("DIST_ROADS_M")

    elevation = terrain_bands.select("ELEVATION")
    slope = terrain_bands.select("SLOPE")
    tpi_300m = terrain_bands.select("TPI_300M")

    dist_water_m = water_bands.select("DIST_WATER_M")
    water_occurrence = water_bands.select("WATER_OCCURRENCE")

    features = ee.Image.cat([
        blue,
        bsi,
        dist_roads_log,
        dist_roads_m,
        dist_water_m,
        elevation,
        green,
        ndbi,
        ndmi,
        ndvi,
        ndwi,
        nir,
        red,
        slope,
        swir1,
        swir2,
        tpi_300m,
        water_occurrence,
    ]).toFloat().clip(roi)

    return features


def main() -> int:
    # ee.Authenticate()
    ee.Initialize(project=CFG.EE_PROJECT)
    print("Earth Engine Initialised")

    roi, roi_meta = build_roi()

    print("Using ROI:", roi_meta.adm2_names, f"(ADM0={roi_meta.adm0_name})")
    print("ROI area (km^2):", roi_meta.area_km2)
    print("ROI bounds:", roi_meta.bounds)

    if CFG.DEBUG:
        sanity_cities = {
            "Norwich": (1.2974, 52.6309),
            "Ipswich": (1.1550, 52.0567),
            "Great Yarmouth": (1.7300, 52.6060),
            "Lowestoft": (1.7516, 52.4750),
            "King's Lynn": (0.3959, 52.7567),
            "Bury St Edmunds": (0.7187, 52.2420),
        }
        debug_point_checks(roi, sanity_cities)

    tasks: list[ee.batch.Task] = []

    for year in tqdm(INFERENCE_YEARS, desc="Submitting feature raster exports"):
        features = build_feature_stack(year, roi)

        print(f"{year}: feature bands:", features.bandNames().getInfo())

        desc = f"EastAnglia_Features_{EXPERIMENT_TAG}_{year}"
        prefix = f"EastAnglia_Features_{EXPERIMENT_TAG}_{year}"

        task = ee.batch.Export.image.toDrive(
            image=features,
            description=desc,
            folder=CFG.EXPORT_FOLDER,
            fileNamePrefix=prefix,
            region=roi,
            scale=30, #Previously Reduced to 90m scale
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )

        task.start()
        tasks.append(task)
        print(f"{year}: raster export started -> {task.status().get('state')}")

    wait_for_tasks(tasks, poll_s=30)
    print("All raster export tasks finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())