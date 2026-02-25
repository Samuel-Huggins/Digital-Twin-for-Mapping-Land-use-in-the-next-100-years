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


def main() -> int:
    # -
    # Init Earth Engine
    # -
    # ee.Authenticate()
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
            roi_fc = ee.FeatureCollection([ee.Feature(ROI)])

            rgb_vis = composite.select(["RED", "GREEN", "BLUE"]).visualize(min=0.0, max=0.3)
            roi_outline = ee.Image().byte().paint(roi_fc, 1, 3).visualize(palette=["FF0000"])
            debug_img = rgb_vis.blend(roi_outline)

            debug_desc = f"{CFG.ROI_EXPORT_DESC}_{year}"
            debug_prefix = f"{CFG.ROI_EXPORT_PREFIX}_{year}"

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

        # -
        # Feature engineering (bands + indices)
        # -
        green = composite.select("GREEN")
        red = composite.select("RED")
        nir = composite.select("NIR")
        swir1 = composite.select("SWIR1")

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

        features = composite.addBands([ndvi, ndbi, ndwi])

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
        csv_desc = f"{CFG.EXPORT_DESC}_{year}"
        csv_prefix = f"{CFG.EXPORT_PREFIX}_{year}"

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