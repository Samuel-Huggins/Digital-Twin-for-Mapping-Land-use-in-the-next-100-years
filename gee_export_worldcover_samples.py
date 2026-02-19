# gee_export_worldcover_samples.py
"""
Exports stratified WorldCover-labelled samples for a Landsat 8 composite over the configured ROI.

- Uses config.CFG for all parameters (single source of truth).
- Uses roi.build_roi() to construct ROI.
- In DEBUG mode:
    - prints extra diagnostics
    - runs city sanity checks (optional)
    - exports a quick RGB+ROI-outline debug TIFF
"""

import ee
from config import CFG
from roi import build_roi, debug_point_checks


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

# Always-on sanity info (small + useful)
print("Using ROI:", roi_meta.adm2_names, f"(ADM0={roi_meta.adm0_name})")
print("ROI area (km^2):", roi_meta.area_km2)
print("ROI bounds:", roi_meta.bounds)

if CFG.DEBUG:
    # Optional: quick “is this inside ROI?” checks
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
# Landsat 8 SR (mask + composite)
# -----------------------------
def mask_l8_sr(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    cloud_bit1 = 1 << 1  # dilated cloud
    cloud_bit2 = 1 << 2  # cirrus
    cloud_bit3 = 1 << 3  # cloud
    cloud_shadow = 1 << 4  # cloud shadow

    mask = (
        qa.bitwiseAnd(cloud_bit1).eq(0)
        .And(qa.bitwiseAnd(cloud_bit2).eq(0))
        .And(qa.bitwiseAnd(cloud_bit3).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow).eq(0))
    )

    # Surface reflectance scaling for Collection 2 Level-2 SR bands
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return optical.updateMask(mask).copyProperties(image, image.propertyNames())


l8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate(CFG.START_DATE, CFG.END_DATE)
    .filterBounds(ROI)
    .map(mask_l8_sr)
)

if CFG.DEBUG:
    print("Images after filtering:", l8.size().getInfo())

composite = l8.median().clip(ROI)


# -----------------------------
# Debug-only: export quick ROI overlay TIFF
# -----------------------------
if CFG.DEBUG:
    roi_fc = ee.FeatureCollection([ee.Feature(ROI)])

    rgb_vis = composite.select(["SR_B4", "SR_B3", "SR_B2"]).visualize(min=0.0, max=0.3)
    roi_outline = ee.Image().byte().paint(roi_fc, 1, 3).visualize(palette=["FF0000"])
    debug_img = rgb_vis.blend(roi_outline)

    debug_task = ee.batch.Export.image.toDrive(
        image=debug_img,
        description=CFG.ROI_EXPORT_DESC,
        folder=CFG.EXPORT_FOLDER,
        fileNamePrefix=CFG.ROI_EXPORT_PREFIX,
        region=ROI,
        scale=CFG.SCALE_M,
        maxPixels=1e13,
    )
    debug_task.start()
    print("DEBUG TIFF export started:", debug_task.status())


# -----------------------------
# Feature engineering (bands + indices)
# -----------------------------
green = composite.select("SR_B3")
red = composite.select("SR_B4")
nir = composite.select("SR_B5")
swir1 = composite.select("SR_B6")

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
csv_task = ee.batch.Export.table.toDrive(
    collection=samples,
    description=CFG.EXPORT_DESC,
    folder=CFG.EXPORT_FOLDER,
    fileNamePrefix=CFG.EXPORT_PREFIX,
    fileFormat="CSV",
)
csv_task.start()
print("CSV export started:", csv_task.status())
