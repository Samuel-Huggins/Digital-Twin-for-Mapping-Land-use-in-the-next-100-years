import ee

PROJECT = "digitaltwin-478518"
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"

ee.Authenticate()
ee.Initialize(project = PROJECT)
print("Earth Engine Initialised")

# Region of Interest - London
ROI = ee.Geometry.Rectangle([-0.35, 51.40, 0.10, 51.65])
#Samples per class (stratified)
POINTS_PER_CLASS = 500

EXPORT_FOLDER = "GEE_Exports"
EXPORT_DESC = "L8_WorldCover_London_2020_Samples"
EXPORT_PREFIX = "London2020_WorldCover_Samples"

def mask_l8_sr(image):
    qa = image.select("QA_PIXEL")
    cloud_bit1 = 1 << 1 # dilated clouds
    cloud_bit2 = 1 << 2 # cirrus
    cloud_bit3 = 1 << 3 # cloud
    cloud_shadow = 1 << 4 # cloud shadow

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

count = l8.size().getInfo()
print(f"Images after filtering: {count}")

composite = l8.median().clip(ROI)

# Landsat 8 bands:
# SR_B2 blue, SR_B3 green, SR_B4 red, SR_B5 nir, SR_B6 swir1, SR_B7 swir2
blue  = composite.select("SR_B2")
green = composite.select("SR_B3")
red   = composite.select("SR_B4")
nir   = composite.select("SR_B5")
swir1 = composite.select("SR_B6")

ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

features = composite.addBands([ndvi, ndbi, ndwi])

print("Feature bands:", features.bandNames().getInfo())

# ESA WorldCover v200 provides a discrete land cover map (10..100 codes).
# Using 2020 labels here for the 2020 Landsat composite.
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

# Resample to 30m-ish to match Landsat; mode is appropriate for categorical labels.
labels = (
    worldcover
    .rename("label")
    .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
    .reproject(crs="EPSG:4326", scale=30)
    .clip(ROI)
)

# Stack features + label into one image
stack = features.addBands(labels)

samples = stack.stratifiedSample(
    numPoints=POINTS_PER_CLASS,
    classBand="label",
    region=ROI,
    scale=30,
    geometries=True,
    seed=42
)

print("Sample count (server-side):", samples.size().getInfo())

task = ee.batch.Export.table.toDrive(
    collection=samples,
    description=EXPORT_DESC,
    folder=EXPORT_FOLDER,
    fileNamePrefix=EXPORT_PREFIX,
    fileFormat="CSV"
)
task.start()
print("Export started, Task ID:", task.id)
print("Initial task status:", task.status())