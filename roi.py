# roi.py
from __future__ import annotations
import ee
from dataclasses import dataclass
from typing import Dict, Tuple, List

from config import CFG

@dataclass(frozen=True)
class ROIMeta:
    adm0_name: str
    adm2_names: Tuple[str, ...]
    area_km2: float
    bounds: dict

def build_roi() -> tuple[ee.Geometry, ROIMeta]:
    """
    Returns (ROI_geometry, metadata).
    ROI is constructed from GAUL level2 by ADM0_NAME + ADM2_NAME(s).
    """
    admin2 = ee.FeatureCollection(CFG.GAUL_LEVEL2)
    uk_admin2 = admin2.filter(ee.Filter.eq("ADM0_NAME", CFG.GAUL_ADM0_NAME))

    parts: List[ee.FeatureCollection] = []
    for name in CFG.GAUL_ADM2_NAMES:
        fc = uk_admin2.filter(ee.Filter.eq("ADM2_NAME", name))
        parts.append(fc)

    merged_fc = parts[0]
    for fc in parts[1:]:
        merged_fc = merged_fc.merge(fc)

    roi = merged_fc.geometry()

    # Metadata (client-side)
    area_km2 = roi.area().divide(1e6).getInfo()
    bounds = roi.bounds().getInfo()

    meta = ROIMeta(
        adm0_name=CFG.GAUL_ADM0_NAME,
        adm2_names=CFG.GAUL_ADM2_NAMES,
        area_km2=area_km2,
        bounds=bounds,
    )
    return roi, meta

def debug_point_checks(roi: ee.Geometry, cities: Dict[str, Tuple[float, float]]) -> None:
    """
    Prints whether points fall inside ROI. Only call when CFG.DEBUG is True.
    """
    for name, (lon, lat) in cities.items():
        pt = ee.Geometry.Point([lon, lat])
        inside = roi.intersects(pt.buffer(50), ee.ErrorMargin(1)).getInfo()
        print(f"{name} inside ROI?: {inside}")
