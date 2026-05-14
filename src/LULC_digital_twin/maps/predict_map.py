# src/LULC_digital_twin/maps/predict_map.py
"""
Generate a full-ROI predicted class map by running a trained Keras ANN over a
multiband feature-stack GeoTIFF exported from Google Earth Engine.

The feature raster band order MUST match the feature order used to train the model.

Typical usage from repo root:

  poetry run python -m LULC_digital_twin.maps.predict_map `
    --features data/rasters/features/EastAnglia_Features_slope_roads_water_elevation_tpi_2013.tif `
    --model results/with_slope_roads_water_elevation_tpi/2021/ann_worldcover_slope_roads_water_elevation_tpi.keras `
    --scaler results/with_slope_roads_water_elevation_tpi/2021/feature_scaler.joblib `
    --codes results/with_slope_roads_water_elevation_tpi/2021/worldcover_codes_used.csv `
    --out results/historical_maps/predicted_lulc_2013_wc.tif `
    --png results/historical_maps/predicted_lulc_2013_wc.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from rasterio.windows import Window

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ANN inference over a feature-stack GeoTIFF.")

    p.add_argument("--features", type=str, required=True, help="Path to multiband feature GeoTIFF.")
    p.add_argument("--model", type=str, required=True, help="Path to trained .keras model.")
    p.add_argument("--out", type=str, required=True, help="Output GeoTIFF path.")
    p.add_argument("--scaler", type=str, default=None, help="Optional joblib scaler used in training.")
    p.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Optional CSV containing WorldCover class codes in model output order.",
    )
    p.add_argument("--batch-size", type=int, default=8192, help="Batch size for model.predict.")
    p.add_argument("--blocksize", type=int, default=512, help="Fallback window size if no internal tiling.")
    p.add_argument("--nodata", type=int, default=255, help="Nodata value for output class map.")

    p.add_argument(
        "--png",
        type=str,
        default=None,
        help="Optional PNG output path for quick visualisation.",
    )
    p.add_argument(
        "--palette",
        type=str,
        default=None,
        help="Optional JSON file mapping class code -> [r,g,b].",
    )
    p.add_argument(
        "--save-probs",
        action="store_true",
        help="If set, also save per-class probability GeoTIFFs.",
    )
    p.add_argument(
        "--probs-outdir",
        type=str,
        default=None,
        help="Directory to save probability GeoTIFFs if --save-probs is set.",
    )

    return p.parse_args()


def _load_scaler(path: Optional[str]):
    if path is None:
        return None

    if joblib is None:
        raise RuntimeError("joblib is not available but --scaler was provided.")

    scaler_path = Path(path)

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    return joblib.load(scaler_path)


def _load_class_codes(path: Optional[str]) -> Optional[np.ndarray]:
    """
    Loads WorldCover class codes in the same order as the model output classes.

    Handles both:
    - headerless CSV: 10, 20, 30...
    - pandas Series CSV with accidental header: 0, 10, 20...
    """
    if path is None:
        return None

    codes_path = Path(path)

    if not codes_path.exists():
        raise FileNotFoundError(f"Class code file not found: {codes_path}")

    raw = pd.read_csv(codes_path, header=None).iloc[:, 0]

    codes = []
    for value in raw:
        try:
            code = int(value)
            codes.append(code)
        except ValueError:
            continue

    # WorldCover has no class code 0. If present, it is almost certainly
    # the pandas Series header accidentally read as a class code.
    if codes and codes[0] == 0:
        codes = codes[1:]

    return np.array(codes, dtype=np.uint16)


def _iter_windows(
    ds: rasterio.io.DatasetReader,
    fallback_blocksize: int,
) -> Tuple[Window, Tuple[int, int]]:
    """
    Yield windows covering the dataset.

    Prefer internal block windows if available; otherwise use regular fallback windows.
    """
    try:
        any_blocks = False

        for _, window in ds.block_windows(1):
            any_blocks = True
            yield window, (int(window.row_off), int(window.col_off))

        if any_blocks:
            return

    except Exception:
        pass

    h, w = ds.height, ds.width
    bs = int(fallback_blocksize)

    for row_off in range(0, h, bs):
        win_h = min(bs, h - row_off)

        for col_off in range(0, w, bs):
            win_w = min(bs, w - col_off)
            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=win_w,
                height=win_h,
            )
            yield window, (row_off, col_off)


def _make_valid_mask(block: np.ndarray, src_nodata: Optional[float]) -> np.ndarray:
    """
    block: (bands, h, w)

    Returns:
        (h, w) boolean mask where True means valid pixel.
    """
    invalid = np.isnan(block).any(axis=0)

    if src_nodata is not None:
        invalid = invalid | (block == src_nodata).any(axis=0)

    return ~invalid


def _predict_block(
    model: tf.keras.Model,
    block: np.ndarray,
    valid_mask: np.ndarray,
    scaler,
    batch_size: int,
    nodata_class: int,
    class_codes: Optional[np.ndarray] = None,
    save_probs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict classes for one raster block.

    If class_codes is provided, model class IDs are mapped back to WorldCover codes.
    """
    bands, h, w = block.shape
    class_map = np.full((h, w), nodata_class, dtype=np.uint16)

    idx = np.where(valid_mask.ravel())[0]

    if idx.size == 0:
        return class_map, None

    X = block.reshape(bands, -1).T
    Xv = X[idx, :].astype(np.float32, copy=False)

    if scaler is not None:
        Xv = scaler.transform(Xv)

    probs = model.predict(Xv, batch_size=batch_size, verbose=0)

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Unexpected model output shape {probs.shape}. Expected (n, num_classes).")

    pred_ids = np.argmax(probs, axis=1).astype(np.uint16)

    if class_codes is not None:
        if int(pred_ids.max()) >= len(class_codes):
            raise ValueError(
                f"Predicted class id {int(pred_ids.max())} exceeds class_codes length {len(class_codes)}."
            )
        pred = class_codes[pred_ids].astype(np.uint16)
    else:
        pred = pred_ids

    out_flat = class_map.ravel()
    out_flat[idx] = pred
    class_map = out_flat.reshape(h, w)

    if save_probs:
        num_classes = probs.shape[1]
        probs_map = np.zeros((num_classes, h, w), dtype=np.float32)

        for c in range(num_classes):
            tmp = np.full((h * w,), np.nan, dtype=np.float32)
            tmp[idx] = probs[:, c].astype(np.float32)
            probs_map[c] = tmp.reshape(h, w)

        return class_map, probs_map

    return class_map, None


def _save_png(
    tif_path: str,
    png_path: str,
    nodata: int,
    palette_path: Optional[str] = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot write PNG.")

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)

    mask = arr == nodata
    arr_disp = np.ma.array(arr, mask=mask)

    plt.figure(figsize=(10, 10))

    if palette_path:
        palette = json.loads(Path(palette_path).read_text(encoding="utf-8"))

        from matplotlib.colors import ListedColormap, BoundaryNorm  # type: ignore

        keys = sorted(int(k) for k in palette.keys())
        colors = []

        for k in keys:
            rgb = palette.get(str(k), [0, 0, 0])
            colors.append([c / 255.0 for c in rgb])

        cmap = ListedColormap(colors)
        boundaries = keys + [max(keys) + 1]
        norm = BoundaryNorm(boundaries, cmap.N)

        plt.imshow(arr_disp, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        plt.imshow(arr_disp, interpolation="nearest")
        plt.colorbar(fraction=0.046, pad=0.04, label="Class code")

    plt.title("Predicted land cover")
    plt.axis("off")

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def main() -> None:
    args = _parse_args()

    features_path = Path(args.features)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not features_path.exists():
        raise FileNotFoundError(f"Features GeoTIFF not found: {features_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = _load_scaler(args.scaler)
    class_codes = _load_class_codes(args.codes)

    model = tf.keras.models.load_model(model_path)

    probs_outdir = None

    if args.save_probs:
        probs_outdir = Path(args.probs_outdir) if args.probs_outdir else out_path.parent / "probs"
        probs_outdir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(features_path) as src:
        src_nodata = src.nodata
        bands = src.count

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint16,
            nodata=int(args.nodata),
            compress="deflate",
            predictor=2,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            prob_dsts = [] if args.save_probs else None

            for window, _ in _iter_windows(src, args.blocksize):
                block = src.read(
                    indexes=list(range(1, bands + 1)),
                    window=window,
                    out_dtype=np.float32,
                    masked=False,
                )

                valid_mask = _make_valid_mask(block, src_nodata)

                class_block, probs_block = _predict_block(
                    model=model,
                    block=block,
                    valid_mask=valid_mask,
                    scaler=scaler,
                    batch_size=args.batch_size,
                    nodata_class=int(args.nodata),
                    class_codes=class_codes,
                    save_probs=args.save_probs,
                )

                dst.write(class_block.astype(np.uint16), 1, window=window)

                if args.save_probs and probs_block is not None:
                    num_classes = probs_block.shape[0]

                    if prob_dsts is not None and len(prob_dsts) == 0:
                        prob_profile = profile.copy()
                        prob_profile.update(
                            count=1,
                            dtype=rasterio.float32,
                            nodata=np.nan,
                        )

                        for c in range(num_classes):
                            if class_codes is not None:
                                class_label = int(class_codes[c])
                                pth = probs_outdir / f"{out_path.stem}_prob_wc{class_label}.tif"
                            else:
                                pth = probs_outdir / f"{out_path.stem}_prob_class{c:02d}.tif"

                            prob_dsts.append(rasterio.open(pth, "w", **prob_profile))

                    for c in range(probs_block.shape[0]):
                        prob_dsts[c].write(probs_block[c].astype(np.float32), 1, window=window)

            if args.save_probs and prob_dsts:
                for ds in prob_dsts:
                    ds.close()

    if args.png is not None:
        _save_png(
            tif_path=str(out_path),
            png_path=args.png,
            nodata=int(args.nodata),
            palette_path=args.palette,
        )

    print(f"Wrote predicted GeoTIFF: {out_path}")

    if args.png:
        print(f"Wrote PNG: {args.png}")

    if args.save_probs:
        print(f"Wrote probability rasters to: {probs_outdir}")


if __name__ == "__main__":
    main()