import ee

# -----------------------------
# Sensor-specific masking + renaming to common bands
# Output bands: BLUE, GREEN, RED, NIR, SWIR1, SWIR2
# -----------------------------

def _mask_c2_l2(image: ee.Image) -> ee.Image:
    """
    Common cloud/shadow mask for Landsat Collection 2 Level 2.
    Uses QA_PIXEL bits:
      bit 1: dilated cloud
      bit 2: cirrus
      bit 3: cloud
      bit 4: cloud shadow
    """
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return image.updateMask(mask)

def _scale_sr(image: ee.Image) -> ee.Image:
    """
    Scale surface reflectance bands for C2 L2.
    Scale: 0.0000275, Offset: -0.2
    """
    sr = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return image.addBands(sr, overwrite=True)

def _prep_l457(image: ee.Image) -> ee.Image:
    """
    Landsat 5 TM / Landsat 7 ETM+ band mapping:
      SR_B1 BLUE
      SR_B2 GREEN
      SR_B3 RED
      SR_B4 NIR
      SR_B5 SWIR1
      SR_B7 SWIR2
    """
    image = _mask_c2_l2(image)
    image = _scale_sr(image)
    return image.select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        ["BLUE",  "GREEN", "RED",  "NIR",  "SWIR1","SWIR2"]
    )

def _prep_l89(image: ee.Image) -> ee.Image:
    """
    Landsat 8/9 OLI/TIRS band mapping:
      SR_B2 BLUE
      SR_B3 GREEN
      SR_B4 RED
      SR_B5 NIR
      SR_B6 SWIR1
      SR_B7 SWIR2
    """
    image = _mask_c2_l2(image)
    image = _scale_sr(image)
    return image.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        ["BLUE",  "GREEN", "RED",  "NIR",  "SWIR1","SWIR2"]
    )

# -----------------------------
# Public API
# -----------------------------

def get_annual_landsat_composite(year: int, roi: ee.Geometry, debug: bool = False) -> ee.Image:
    """
    Returns an annual median composite clipped to ROI with consistent bands:
      BLUE, GREEN, RED, NIR, SWIR1, SWIR2

    Era rules:
      1984–2012: Landsat 5 + Landsat 7 merged (median handles L7 SLC-off gaps)
      2013–present: Landsat 8 + Landsat 9 merged
    """
    if year < 1984:
        raise ValueError("Landsat TM/ETM+/OLI era starts at 1984 for this pipeline.")
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    if 1984 <= year <= 2012:
        # Landsat 5 TM (ends 2013)
        l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
              .filterDate(start, end)
              .filterBounds(roi)
              .map(_prep_l457))

        # Landsat 7 ETM+ (SLC-off after 2003-05-31; median helps)
        l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
              .filterDate(start, end)
              .filterBounds(roi)
              .map(_prep_l457))

        col = l5.merge(l7)

        if debug:
            print(f"[DEBUG] {year} L5 count:", l5.size().getInfo())
            print(f"[DEBUG] {year} L7 count:", l7.size().getInfo())
            print(f"[DEBUG] {year} merged count:", col.size().getInfo())

    else:
        # 2013–present: Landsat 8 + 9
        l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterDate(start, end)
              .filterBounds(roi)
              .map(_prep_l89))

        l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
              .filterDate(start, end)
              .filterBounds(roi)
              .map(_prep_l89))

        col = l8.merge(l9)

        if debug:
            print(f"[DEBUG] {year} L8 count:", l8.size().getInfo())
            print(f"[DEBUG] {year} L9 count:", l9.size().getInfo())
            print(f"[DEBUG] {year} merged count:", col.size().getInfo())

    # Annual median composite
    composite = col.median().clip(roi)

    # Optional: sanity band list
    if debug:
        print(f"[DEBUG] {year} bands:", composite.bandNames().getInfo())

    return composite
