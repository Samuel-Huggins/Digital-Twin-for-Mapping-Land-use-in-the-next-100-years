# config.py
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Config:
    # Global toggle (one place to switch)
    DEBUG: bool = True

    # Earth Engine
    EE_PROJECT: str = "digitaltwin-478518"

    # Time window for compositing
    START_DATE: str = "2020-01-01"
    END_DATE: str = "2020-12-31"

    # Sampling
    POINTS_PER_CLASS: int = 500
    SEED: int = 42
    SCALE_M: int = 30

    # Drive export
    EXPORT_FOLDER: str = "GEE_Exports"
    EXPORT_DESC: str = "L8_WorldCover_EastAnglia_2020_Samples"
    EXPORT_PREFIX: str = "EastAnglia2020_WorldCover_Samples"

    # ROI export (only used when DEBUG=True)
    ROI_EXPORT_DESC: str = "EastAnglia_ROI_debug"
    ROI_EXPORT_PREFIX: str = "EastAnglia_ROI_debug"

    # ROI dataset + filters (GAUL)
    GAUL_LEVEL2: str = "FAO/GAUL/2015/level2"
    GAUL_ADM0_NAME: str = "U.K. of Great Britain and Northern Ireland"
    GAUL_ADM2_NAMES: tuple[str, ...] = ("Norfolkshire", "Suffolk")

    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")

    # Training data
    TRAIN_CSV: str = os.path.join(DATA_DIR, "EastAnglia2020_WorldCover_Samples.csv")
    LABEL_COL: str = "label"
    DROP_COLS_CONTAINING: tuple[str, ...] = (".geo", "system:index")

    # Training hyperparams
    SEED: int = 42
    TEST_SIZE: float = 0.2
    VAL_SPLIT: float = 0.2
    EPOCHS: int = 20
    BATCH_SIZE: int = 128
    LR: float = 1e-3

    # Output filenames
    MODEL_FILENAME: str = "baseline_ann_worldcover.keras"
    CODES_FILENAME: str = "worldcover_codes_used.csv"


CFG = Config()
