import ee

DEBUG_TOGGLE = True  # True = extra prints + optional debug TIFF. False = CSV only.

PROJECT = "digitaltwin-478518"
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"

POINTS_PER_CLASS = 500
EXPORT_FOLDER = "GEE_Exports"

CSV_DESC = "L8_WorldCover_EastAnglia_2020_Samples"
CSV_PREFIX = "EastAnglia2020_WorldCover_Samples"

DEBUG_TIFF_DESC = "EastAnglia_ROI_Debug"
DEBUG_TIFF_PREFIX = "EastAnglia_ROI_Debug"

# -----------------------------
# Init Earth Engine
# -----------------------------
ee.Authenticate()
ee.Initialize(project=PROJECT)
print("Earth Engine Initialised")

# -----------------------------
# ROI: Norfolkshire + Suffolk (GAUL admin2)
# -----------------------------
admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
uk_admin2 = admin2.filter(ee.Filter.eq("ADM0_NAME", "U.K. of Great Britain and Northern Ireland"))

norfolk = uk_admin2.filter(ee.Filter.eq("ADM2_NAME", "Norfolkshire"))
suffolk = uk_admin2.filter(ee.Filter.eq("ADM2_NAME", "Suffolk"))

ROI = norfolk.merge(suffolk).geometry()

# Always-on sanity info (small + useful)
print("ROI area (km^2):", ROI.area().divide(1e6).getInfo())
print("ROI bounds:", ROI.bounds().getInfo())

if DEBUG_TOGGLE:
    # Diagnostics to catch accidental empty geometries / wrong ADM names
    print("Norfolkshire area (km^2):", norfolk.geometry().area().divide(1e6).getInfo())
    print("Suffolk area (km^2):", suffolk.geometry().area().divide(1e6).getInfo())

# -----------------------------
# Landsat 8 SR (mask + composite)
# -----------------------------
def mask_l8_sr(image):
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

    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return optical.updateMask(mask).copyProperties(image, image.propertyNames())

l8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate(START_DATE, END_DATE)
    .filterBounds(ROI)
    .map(mask_l8_sr)
)

if DEBUG_TOGGLE:
    print("Images after filtering:", l8.size().getInfo())

composite = l8.median().clip(ROI)

# -----------------------------
# Debug-only: export quick ROI overlay TIFF
# (useful to visually confirm ROI coverage)
# -----------------------------
if DEBUG_TOGGLE:
    roi_fc = ee.FeatureCollection([ee.Feature(ROI)])

    rgb_vis = composite.select(["SR_B4", "SR_B3", "SR_B2"]).visualize(min=0.0, max=0.3)
    roi_outline = ee.Image().byte().paint(roi_fc, 1, 3).visualize(palette=["FF0000"])

    debug_img = rgb_vis.blend(roi_outline)

    debug_task = ee.batch.Export.image.toDrive(
        image=debug_img,
        description=DEBUG_TIFF_DESC,
        folder=EXPORT_FOLDER,
        fileNamePrefix=DEBUG_TIFF_PREFIX,
        region=ROI,
        scale=30,
        maxPixels=1e13
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

if DEBUG_TOGGLE:
    print("Feature bands:", features.bandNames().getInfo())

# -----------------------------
# Labels (ESA WorldCover) + Stratified sampling
# -----------------------------
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

labels = (
    worldcover.rename("label")
    .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
    .reproject(crs="EPSG:4326", scale=30)
    .clip(ROI)
)

stack = features.addBands(labels)

samples = stack.stratifiedSample(
    numPoints=POINTS_PER_CLASS,
    classBand="label",
    region=ROI,
    scale=30,
    geometries=True,
    seed=42
)

if DEBUG_TOGGLE:
    print("Sample count (server-side):", samples.size().getInfo())

# -----------------------------
# Export CSV (always)
# -----------------------------
csv_task = ee.batch.Export.table.toDrive(
    collection=samples,
    description=CSV_DESC,
    folder=EXPORT_FOLDER,
    fileNamePrefix=CSV_PREFIX,
    fileFormat="CSV"
)
csv_task.start()
print("CSV export started:", csv_task.status())
