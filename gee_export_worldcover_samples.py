# gee_export_worldcover_samples.py
"""
Exports stratified WorldCover-labelled samples for a Landsat annual composite over the configured ROI.

- Uses config.CFG for all parameters (single source of truth).
- Uses roi.build_roi() to construct ROI.
- Uses landsat.get_annual_landsat_composite() for 1984–2012 (L5+L7) and 2013–present (L8+L9).
- In DEBUG mode:
    - prints extra diagnostics
    - runs city sanity checks (optional)
    - exports a quick RGB+ROI-outline debug TIFF
"""

import ee
from config import CFG
from roi import build_roi, debug_point_checks
from landsat import get_annual_landsat_composite


# -----------------------------
# Init Earth Engine
# -----------------------------
ee.Authenticate()
ee.Initialize(project=CFG.EE_PROJECT)
print("Earth Engine Initialised")


# -----------------------------
# ROI (from roi.py)
# -----------------------------
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


# -----------------------------
# Annual composite (era-aware)
# -----------------------------
for year in CFG.YEARS:
    composite = get_annual_landsat_composite(year, ROI, debug=CFG.DEBUG)


    # -----------------------------
    # Debug-only: export quick ROI overlay TIFF
    # -----------------------------
    if CFG.DEBUG:
        roi_fc = ee.FeatureCollection([ee.Feature(ROI)])

        # Works for all eras because landsat.py standardises these band names
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
        print("DEBUG TIFF export started:", debug_task.status())


    # -----------------------------
    # Feature engineering (bands + indices)
    # -----------------------------
    green = composite.select("GREEN")
    red   = composite.select("RED")
    nir   = composite.select("NIR")
    swir1 = composite.select("SWIR1")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

    features = composite.addBands([ndvi, ndbi, ndwi])

    if CFG.DEBUG:
        print("Feature bands:", features.bandNames().getInfo())


    # -----------------------------
    # Labels (ESA WorldCover) + stratified sampling
    # -----------------------------
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    labels = (
        worldcover.rename("label")
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

    if CFG.DEBUG:
        print("Sample count (server-side):", samples.size().getInfo())


    # -----------------------------
    # Export CSV (always)
    # -----------------------------
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
    print("CSV export started:", csv_task.status())
