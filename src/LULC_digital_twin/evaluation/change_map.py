from __future__ import annotations

import time
import ee
from tqdm import tqdm

from LULC_digital_twin.config import CFG
from LULC_digital_twin.roi import build_roi
from LULC_digital_twin.landsat import get_annual_landsat_composite


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


def wait_for_tasks(tasks, poll_s: int = 15):
    """
    Poll Earth Engine export tasks and show tqdm progress.
    Advances 1 step per completed task (EE doesn't provide % progress).
    tasks: list[tuple[ee.batch.Task, str]] -> (task, description)
    """
    pbar = tqdm(total=len(tasks), desc="EE Exports", unit="task")
    finished = set()

    while len(finished) < len(tasks):
        for i, (task, desc) in enumerate(tasks):
            if i in finished:
                continue

            status = task.status()
            state = status.get("state", "UNKNOWN")

            if state in TERMINAL_STATES:
                finished.add(i)
                pbar.update(1)

                if state != "COMPLETED":
                    error_msg = status.get("error_message", "Unknown error")
                    tqdm.write(f"{desc}: {state} — {error_msg}")
                else:
                    tqdm.write(f"{desc}: COMPLETED")

        time.sleep(poll_s)

    pbar.close()


def build_feature_stack(year: int, roi: ee.Geometry) -> ee.Image:
    composite = get_annual_landsat_composite(year, roi, debug=CFG.DEBUG)

    green = composite.select("GREEN")
    red = composite.select("RED")
    nir = composite.select("NIR")
    swir1 = composite.select("SWIR1")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

    return composite.addBands([ndvi, ndbi, ndwi]).clip(roi)


def get_worldcover_labels(year: int, roi: ee.Geometry) -> ee.Image:
    if year == 2020:
        wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
    elif year == 2021:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    else:
        raise ValueError("WorldCover labels in this pipeline are only configured for 2020 and 2021.")

    return (
        wc.rename("label")
        .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=CFG.SCALE_M)
        .clip(roi)
    )


def majority_filter(img: ee.Image, radius_px: int = 1) -> ee.Image:
    kernel = ee.Kernel.square(radius_px, units="pixels", normalize=False)
    return img.reduceNeighborhood(
        reducer=ee.Reducer.mode(),
        kernel=kernel,
        optimization="window",
    )


def main() -> int:
    ee.Initialize(project=CFG.EE_PROJECT)

    roi, meta = build_roi()
    print("ROI:", meta.adm2_names, "| area_km2:", meta.area_km2)

    # --- TRAIN ON 2020 ---
    x2020 = build_feature_stack(2020, roi)
    y2020 = get_worldcover_labels(2020, roi)
    train_stack = x2020.addBands(y2020)

    samples_2020 = train_stack.stratifiedSample(
        numPoints=CFG.POINTS_PER_CLASS,
        classBand="label",
        region=roi,
        scale=CFG.SCALE_M,
        geometries=False,
        seed=CFG.SEED,
    )

    feature_bands = x2020.bandNames()
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=200,
        bagFraction=0.7,
        seed=CFG.SEED,
    ).train(
        features=samples_2020,
        classProperty="label",
        inputProperties=feature_bands,
    )

    # --- PREDICT BOTH YEARS (RAW) ---
    pred2020_raw = x2020.classify(classifier).rename("pred")
    x2021 = build_feature_stack(2021, roi)
    pred2021_raw = x2021.classify(classifier).rename("pred")

    # --- SPATIAL REGULARISATION (SMOOTHED) ---
    pred2020 = majority_filter(pred2020_raw, radius_px=1).rename("pred")
    pred2021 = majority_filter(pred2021_raw, radius_px=1).rename("pred")

    # --- CHANGE MAPS ---
    change_binary = pred2020.neq(pred2021).rename("changed")
    transition = pred2020.multiply(1000).add(pred2021).rename("transition")  # FIXED

    change_binary_raw = pred2020_raw.neq(pred2021_raw).rename("changed")
    transition_raw = pred2020_raw.multiply(1000).add(pred2021_raw).rename("transition")

    # --- EXPORTS ---
    out_folder = CFG.EXPORT_FOLDER
    scale = CFG.SCALE_M

    tasks: list[tuple[ee.batch.Task, str]] = []

    def export_image(img: ee.Image, desc: str, prefix: str):
        t = ee.batch.Export.image.toDrive(
            image=img.toInt16(),
            description=desc,
            folder=out_folder,
            fileNamePrefix=prefix,
            region=roi,
            scale=scale,
            maxPixels=1e13,
        )
        t.start()
        tasks.append((t, desc))
        print("Started:", desc)

    export_image(pred2020_raw, "Pred_LULC_2020_RF_raw", "Pred_LULC_2020_RF_raw")
    export_image(pred2021_raw, "Pred_LULC_2021_RF_raw", "Pred_LULC_2021_RF_raw")

    export_image(pred2020, "Pred_LULC_2020_RF_mode3x3", "Pred_LULC_2020_RF_mode3x3")
    export_image(pred2021, "Pred_LULC_2021_RF_mode3x3", "Pred_LULC_2021_RF_mode3x3")

    export_image(change_binary, "Pred_Change_2020_2021_RF_mode3x3_binary", "Pred_Change_2020_2021_RF_mode3x3_binary")
    export_image(transition, "Pred_Transition_2020_2021_RF_mode3x3", "Pred_Transition_2020_2021_RF_mode3x3")

    export_image(change_binary_raw, "Pred_Change_2020_2021_RF_raw_binary", "Pred_Change_2020_2021_RF_raw_binary")
    export_image(transition_raw, "Pred_Transition_2020_2021_RF_raw", "Pred_Transition_2020_2021_RF_raw")

    print("\nExports submitted. Waiting for completion...")
    wait_for_tasks(tasks, poll_s=20)
    print("All export tasks finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())