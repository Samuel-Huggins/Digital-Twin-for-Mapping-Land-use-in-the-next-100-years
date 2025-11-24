import ee

# -----------------------------
# 1. Authenticate & Initialize GEE
# -----------------------------
# First time: ee.Authenticate() will open a browser. After that, you can comment it out.
ee.Authenticate()
ee.Initialize(project='digitaltwin-478518')

print("Google Earth Engine initialized successfully!")

# -----------------------------
# 2. Define Region of Interest (ROI)
#    Here: a small box around central London
# -----------------------------
# [minLon, minLat, maxLon, maxLat]
london_roi = ee.Geometry.Rectangle([
    -0.35, 51.40,   # southwest corner
    0.10,  51.65    # northeast corner
])

print("ROI bounds:", london_roi.getInfo())

# -----------------------------
# 3. Cloud mask for Landsat 8 Collection 2 Level 2
# -----------------------------
def mask_l8_sr(image):
    """
    Mask clouds and cloud shadows using the QA_PIXEL band.
    This is a fairly standard approach for Landsat 8 C2 SR.
    """
    qa = image.select('QA_PIXEL')

    # Bitmasks for clouds and cloud shadows (see Landsat C2 docs)
    # Here we mask:
    #   bit 1 = Dilated Cloud
    #   bit 2 = Cirrus
    #   bit 3 = Cloud
    #   bit 4 = Cloud Shadow
    cloud_bit1 = 1 << 1
    cloud_bit2 = 1 << 2
    cloud_bit3 = 1 << 3
    cloud_shadow_bit = 1 << 4

    mask = (
        qa.bitwiseAnd(cloud_bit1).eq(0)
        .And(qa.bitwiseAnd(cloud_bit2).eq(0))
        .And(qa.bitwiseAnd(cloud_bit3).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    )

    # Scale factors for SR_* bands are 0.0000275 and -0.2 (Landsat C2)
    optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    return optical_bands.updateMask(mask).copyProperties(image, image.propertyNames())

# -----------------------------
# 4. Build Landsat 8 SR collection and composite
# -----------------------------
# You can adjust the time window as needed.
start_date = '2018-01-01'
end_date   = '2018-12-31'

l8_sr = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(start_date, end_date)
    .filterBounds(london_roi)
    .map(mask_l8_sr)
)

print("Number of images after filtering:", l8_sr.size().getInfo())

# Median composite for that year
l8_composite = l8_sr.median().clip(london_roi)

# -----------------------------
# 5. Quick sanity checks
# -----------------------------
# Print band names
print("Composite bands:", l8_composite.bandNames().getInfo())

# Get basic stats over the ROI for one band (e.g. SR_B4 = red)
stats = l8_composite.select('SR_B4').reduceRegion(
    reducer=ee.Reducer.mean().combine(
        reducer2=ee.Reducer.minMax(), sharedInputs=True
    ),
    geometry=london_roi,
    scale=30,
    maxPixels=1e7
)
print("Band SR_B4 stats over ROI:", stats.getInfo())

# -----------------------------
# 6. Export to Drive for visual inspection
# -----------------------------
task = ee.batch.Export.image.toDrive(
    image=l8_composite.select(['SR_B4', 'SR_B3', 'SR_B2']),  # RGB
    description='L8SR_London_2018_RGB',
    folder='GEE_Exports',  # Drive folder name
    fileNamePrefix='L8SR_London_2018_RGB',
    region=london_roi,
    scale=30,
    maxPixels=1e13
)

task.start()
print("Task started with ID:", task.id)

# --- NEW: check status once right after starting ---
print("Initial task status:", task.status())
