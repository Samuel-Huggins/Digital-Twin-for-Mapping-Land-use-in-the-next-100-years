# gee_landsat_sample.py
"""
Minimal Landsat-8 (C2 L2) London ROI export using shared config.

Goal: quick EE sanity test (auth/init, mask, composite, export).
Config controls:
- EE project
- dates
- export folder + names
- debug verbosity
"""

import ee
from config import CFG

# -----------------------------
# Init Earth Engine
# -----------------------------
if CFG.EE_DO_AUTH:
    ee.Authenticate()

ee.Initialize(project=CFG.EE_PROJECT)
print("Google Earth Engine initialized successfully!")


# -----------------------------
# ROI: London test rectangle (from config)
# -----------------------------
london_roi = ee.Geometry.Rectangle(CFG.LONDON_ROI_BBOX)
if CFG.DEBUG:
    print("ROI bounds:", london_roi.getInfo())


# -----------------------------
# Cloud mask: Landsat 8 C2 L2 SR
# -----------------------------
def mask_l8_sr(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    cloud_bit1 = 1 << 1  # Dilated Cloud
    cloud_bit2 = 1 << 2  # Cirrus
    cloud_bit3 = 1 << 3  # Cloud
    cloud_shadow_bit = 1 << 4  # Cloud Shadow

    mask = (
        qa.bitwiseAnd(cloud_bit1).eq(0)
        .And(qa.bitwiseAnd(cloud_bit2).eq(0))
        .And(qa.bitwiseAnd(cloud_bit3).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    )

    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return optical.updateMask(mask).copyProperties(image, image.propertyNames())


# -----------------------------
# Build collection + composite
# -----------------------------
l8_sr = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate(CFG.START_DATE, CFG.END_DATE)
    .filterBounds(london_roi)
    .map(mask_l8_sr)
)

if CFG.DEBUG:
    print("Number of images after filtering:", l8_sr.size().getInfo())

l8_composite = l8_sr.median().clip(london_roi)

if CFG.DEBUG:
    print("Composite bands:", l8_composite.bandNames().getInfo())

    stats = l8_composite.select("SR_B4").reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=london_roi,
        scale=CFG.SCALE_M,
        maxPixels=1e7,
    )
    print("Band SR_B4 stats over ROI:", stats.getInfo())


# -----------------------------
# Export RGB to Drive
# -----------------------------
rgb = l8_composite.select(["SR_B4", "SR_B3", "SR_B2"])

task = ee.batch.Export.image.toDrive(
    image=rgb,
    description=CFG.LONDON_EXPORT_DESC,
    folder=CFG.EXPORT_FOLDER,
    fileNamePrefix=CFG.LONDON_EXPORT_PREFIX,
    region=london_roi,
    scale=CFG.SCALE_M,
    maxPixels=1e13,
)

task.start()
print("Task started with ID:", task.id)
print("Initial task status:", task.status())
