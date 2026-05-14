from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_env(key: str, default_rel: str, project_root: Path) -> Path:
    raw = os.getenv(key, default_rel).strip()
    p = Path(raw)
    return p if p.is_absolute() else (project_root / p)


# For src/ layout: .../src/LULC_digital_twin/config.py -> parents[2] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Global toggle
    DEBUG: bool = _bool_env("DEBUG", True)
    FAIL_ON_MISSING_LABELS: bool = _bool_env("FAIL_ON_MISSING_LABELS", True)

    # Earth Engine
    EE_PROJECT: str = os.getenv("EE_PROJECT", "digitaltwin-478518")
    EXPORT_FOLDER: str = os.getenv("GEE_EXPORT_FOLDER", "GEE_Exports")

    YEARS: tuple[int, ...] = (2020, 2021)

    # Sampling
    POINTS_PER_CLASS: int = int(os.getenv("POINTS_PER_CLASS", "500"))
    SEED: int = int(os.getenv("SEED", "42"))
    SCALE_M: int = int(os.getenv("SCALE_M", "30"))

    # Drive export naming
    EXPORT_DESC: str = os.getenv("EXPORT_DESC", "L8_WorldCover_EastAnglia_Samples")
    EXPORT_PREFIX: str = os.getenv("EXPORT_PREFIX", "EastAnglia_WorldCover_Samples")

    # ROI export (debug only)
    ROI_EXPORT_DESC: str = os.getenv("ROI_EXPORT_DESC", "EastAnglia_ROI_debug")
    ROI_EXPORT_PREFIX: str = os.getenv("ROI_EXPORT_PREFIX", "EastAnglia_ROI_debug")

    # ROI dataset + filters (GAUL)
    GAUL_LEVEL2: str = "FAO/GAUL/2015/level2"
    GAUL_ADM0_NAME: str = "U.K. of Great Britain and Northern Ireland"
    GAUL_ADM2_NAMES: tuple[str, ...] = ("Norfolkshire", "Suffolk")

    # Paths (repo-root relative by default)
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = _path_env("DATA_DIR", "data", PROJECT_ROOT)
    RESULTS_DIR: Path = _path_env("RESULTS_DIR", "results", PROJECT_ROOT)

    # Training data (default points at latest year; you can override via env if needed)
    TRAIN_CSV: Path = Path(
        os.getenv("TRAIN_CSV", str(DATA_DIR / "EastAnglia_WorldCover_Samples_2021_Post.csv"))
    )
    LABEL_COL: str = "label"
    DROP_COLS_CONTAINING: tuple[str, ...] = (".geo", "system:index")

    # Training hyperparams
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    VAL_SPLIT: float = float(os.getenv("VAL_SPLIT", "0.2"))
    EPOCHS: int = int(os.getenv("EPOCHS", "20"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "128"))
    LR: float = float(os.getenv("LR", "0.001"))

    # Output filenames
    MODEL_FILENAME: str = os.getenv("MODEL_FILENAME", "ann_worldcover.keras")
    CODES_FILENAME: str = os.getenv("CODES_FILENAME", "worldcover_codes_used.csv")


CFG = Config()

# Ensure dirs exist
CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)
CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)