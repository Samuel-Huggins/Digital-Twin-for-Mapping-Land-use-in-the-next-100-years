#worldcover_export.py
"""
Exports stratified WorldCover-labelled samples for a Landsat annual composite over the configured ROI.
"""

from __future__ import annotations

import time
import ee
from tqdm import tqdm

from LULC_digital_twin.config import CFG
from LULC_digital_twin.roi import build_roi, debug_point_checks
from LULC_digital_twin.landsat import get_annual_landsat_composite


def wait_for_tasks(tasks: list[ee.batch.Task], poll_s: int = 30) -> None:
    """
    Blocks until all tasks finish. Updates a progress bar as tasks complete.
    """
    remaining = set(range(len(tasks)))

    with tqdm(total=len(tasks), desc="EE tasks completed") as pbar:
        while remaining:
            finished_now = []
            for i in list(remaining):
                state = tasks[i].status().get("state")
                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    finished_now.append(i)

            for i in finished_now:
                remaining.remove(i)
                pbar.update(1)

            if remaining:
                time.sleep(poll_s)

def get_road_distance_bands(roi: ee.Geometry, max_distance_m: int = 7_500) -> ee.Image:
    """
    Creates road proximity feature bands for the ROI.

    DIST_ROADS_M:
        Distance in metres to the nearest road, capped at max_distance_m.

    DIST_ROADS_LOG:
        log1p(DIST_ROADS_M), used as the ML-friendly version because distance
        values are usually highly skewed.

    Note:
        Earth Engine distance kernels have a max pixel size of 512.
        At 30 m scale, 10,000 m creates a ~667 pixel kernel, so 7,500 m
        is used to stay below the limit.
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

def main() -> int:
    # -
    # Init Earth Engine
    # -
    # ee.Authenticate()
    EXPERIMENT_TAG = "with_slope_roads_and_water"
    if CFG.FAIL_ON_MISSING_LABELS:
        print("FAIL_ON_MISSING_LABELS=True → Only years with ESA WorldCover labels (2020=v100, 2021=v200) are allowed.")

    ee.Initialize(project=CFG.EE_PROJECT)
    print("Earth Engine Initialised")

    # -
    # ROI (from roi.py)
    # -
    ROI, roi_meta = build_roi()

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
        debug_point_checks(ROI, sanity_cities)

    # -
    # Submit tasks (one per year)
    # -
    tasks: list[ee.batch.Task] = []

    for year in tqdm(CFG.YEARS, desc="Submitting EE exports"):
        composite = get_annual_landsat_composite(year, ROI, debug=CFG.DEBUG)
        # -
        # Debug-only: export quick ROI overlay TIFF
        # -
        if CFG.DEBUG:
            print("Composite bands:", composite.bandNames().getInfo())
            roi_fc = ee.FeatureCollection([ee.Feature(ROI)])

            rgb_vis = composite.select(["RED", "GREEN", "BLUE"]).visualize(min=0.0, max=0.3)
            roi_outline = ee.Image().byte().paint(roi_fc, 1, 3).visualize(palette=["FF0000"])
            debug_img = rgb_vis.blend(roi_outline)

            debug_desc = f"{CFG.ROI_EXPORT_DESC}_{EXPERIMENT_TAG}_{year}"
            debug_prefix = f"{CFG.ROI_EXPORT_PREFIX}_{EXPERIMENT_TAG}_{year}"

            debug_task = ee.batch.Export.image.toDrive(
                image=debug_img,
                description=debug_desc,
                folder=CFG.EXPORT_FOLDER,
                fileNamePrefix=debug_prefix,
                region=ROI,
                scale=CFG.SCALE_M,
                maxPixels=1e13,
            )
            debug_task.start()
            tasks.append(debug_task)
            print(f"{year}: DEBUG TIFF task started -> {debug_task.status().get('state')}")

        # Feature engineering (bands + indices)
        blue = composite.select("BLUE")
        green = composite.select("GREEN")
        red = composite.select("RED")
        nir = composite.select("NIR")
        swir1 = composite.select("SWIR1")

        # Small epsilon to avoid divide-by-zero edge cases
        eps = ee.Image.constant(1e-6)

        # Existing indices
        ndvi = nir.subtract(red).divide(nir.add(red).add(eps)).rename("NDVI")
        ndbi = swir1.subtract(nir).divide(swir1.add(nir).add(eps)).rename("NDBI")
        ndwi = green.subtract(nir).divide(green.add(nir).add(eps)).rename("NDWI")

        # New: Normalised Difference Moisture Index (NDMI)
        # Moisture in vegetation/land (helps wetlands + crop moisture separation)
        ndmi = nir.subtract(swir1).divide(nir.add(swir1).add(eps)).rename("NDMI")

        # New: Bare Soil Index (BSI)
        # Exposed soil/bare ground signal (helps bare vs cropland vs built-up edges)
        bsi_num = (swir1.add(red)).subtract(nir.add(blue))
        bsi_den = (swir1.add(red)).add(nir.add(blue)).add(eps)
        bsi = bsi_num.divide(bsi_den).rename("BSI")

        # New: Terrain slope from SRTM DEM
        # Static physical terrain feature, useful because land use is constrained
        # by gradient/suitability as well as spectral surface appearance.
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(ROI)
        slope = ee.Terrain.slope(dem).rename("SLOPE").toFloat()

        # New: Road proximity from GRIP4 Europe
        # Static infrastructure feature. This gives the classifier spatial context
        # linked to accessibility, urban pressure and human land-use patterns.
        road_distance = get_road_distance_bands(ROI)


        # New: Distance to persistent surface water
        # Uses JRC Global Surface Water occurrence.
        # occurrence is 0-100, where higher values indicate more frequent historical water presence.
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        water_occurrence = gsw.select("occurrence").unmask(0).clip(ROI)

        # Persistent water mask. 80 means water was observed in at least 80% of valid observations.
        persistent_water = water_occurrence.gte(80).selfMask()

        # Distance from each pixel to nearest persistent water pixel, in metres.
        # fastDistanceTransform gives squared distance in pixels, so sqrt() * scale gives metres.
        dist_water_m = (
            persistent_water
            .fastDistanceTransform(neighborhood=1024, units="pixels")
            .sqrt()
            .multiply(CFG.SCALE_M)
            .rename("DIST_WATER_M")
            .toFloat()
            .clip(ROI)
        )

        # Also keep occurrence itself as a contextual hydrology feature.
        water_occurrence = water_occurrence.rename("WATER_OCCURRENCE").toFloat()

        features = composite.addBands([
            ndvi,
            ndbi,
            ndwi,
            ndmi,
            bsi,
            slope,
            road_distance,
            dist_water_m,
            water_occurrence,
        ])

        if CFG.DEBUG:
            print("Feature bands:", features.bandNames().getInfo())
        # --- Labels (ESA WorldCover) ---
        if year == 2020:
            # v100 contains 2020 (ImageCollection -> take the first image)
            worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
        elif year == 2021:
            # v200 contains 2021 (ImageCollection -> take the first image)
            worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        else:
            if CFG.FAIL_ON_MISSING_LABELS:
                raise ValueError(
                    f"No ESA WorldCover labels available in v100/v200 for YEAR={year}. "
                    "WorldCover in EE covers 2020 (v100) and 2021 (v200)."
                )
            continue  # if you ever allow skipping

        labels = (
            worldcover.rename("label")   # <-- use rename, NOT name()
            .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
            .reproject(crs="EPSG:4326", scale=CFG.SCALE_M)
            .clip(ROI)
        )

        stack = features.addBands(labels)

        samples = stack.stratifiedSample(
            numPoints=CFG.POINTS_PER_CLASS,
            classBand="label",
            region=ROI,
            scale=CFG.SCALE_M,
            geometries=True,
            seed=CFG.SEED,
        )

        # -
        # Export CSV (always)
        # -
        csv_desc = f"{CFG.EXPORT_DESC}_{EXPERIMENT_TAG}_{year}"
        csv_prefix = f"{CFG.EXPORT_PREFIX}_{EXPERIMENT_TAG}_{year}"

        csv_task = ee.batch.Export.table.toDrive(
            collection=samples,
            description=csv_desc,
            folder=CFG.EXPORT_FOLDER,
            fileNamePrefix=csv_prefix,
            fileFormat="CSV",
        )
        csv_task.start()
        tasks.append(csv_task)
        print(f"{year}: CSV task started -> {csv_task.status().get('state')}")

    # -
    # Optional: wait for completion (pipeline mode)
    # -
    wait_for_tasks(tasks, poll_s=30)
    print("All EE tasks finished (COMPLETED/FAILED/CANCELLED).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())