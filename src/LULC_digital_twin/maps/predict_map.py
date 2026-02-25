# src/LULC_digital_twin/maps/predict_map.py
"""
Generate a full-ROI predicted class map (GeoTIFF + optional PNG) by running a trained
Keras ANN over a multiband feature-stack GeoTIFF exported from Google Earth Engine.

Typical usage (from repo root):
  poetry run python -m LULC_digital_twin.maps.predict_map \
    --features data/rasters/EastAnglia_features_2020.tif \
    --model results/2020/baseline_ann_worldcover.keras \
    --out results/2020/predicted_map_2020.tif \
    --png results/2020/predicted_map_2020.png

Optional (if you used scaling in training and saved a scaler):
  --scaler results/2020/feature_scaler.joblib

Notes:
- The band order in --features MUST match the feature order used to train the model.
- This script predicts in windows/blocks to avoid memory blow-ups.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

import tensorflow as tf

# Optional dependency; only needed if you pass --scaler
try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # noqa: N816

# Optional dependency for PNG rendering
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ANN inference over a feature-stack GeoTIFF.")
    p.add_argument("--features", type=str, required=True, help="Path to multiband feature GeoTIFF.")
    p.add_argument("--model", type=str, required=True, help="Path to trained .keras model.")
    p.add_argument("--out", type=str, required=True, help="Output GeoTIFF path (single-band class ids).")
    p.add_argument("--scaler", type=str, default=None, help="Optional joblib scaler used in training.")
    p.add_argument("--batch-size", type=int, default=8192, help="Batch size for model.predict.")
    p.add_argument("--blocksize", type=int, default=512, help="Fallback window size if no internal tiling.")
    p.add_argument("--nodata", type=int, default=255, help="Nodata value for output class map.")
    p.add_argument(
        "--png",
        type=str,
        default=None,
        help="Optional PNG output path for quick visualisation (requires matplotlib).",
    )
    p.add_argument(
        "--palette",
        type=str,
        default=None,
        help=(
            "Optional JSON file mapping class_id -> [r,g,b]. "
            "If omitted, PNG uses a default colormap."
        ),
    )
    p.add_argument(
        "--save-probs",
        action="store_true",
        help="If set, also save per-class probability GeoTIFF(s) (can be large).",
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
        raise RuntimeError("joblib is not available but --scaler was provided. Add joblib to dependencies.")
    scaler_path = Path(path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return joblib.load(scaler_path)


def _iter_windows(ds: rasterio.io.DatasetReader, fallback_blocksize: int) -> Tuple[Window, Tuple[int, int]]:
    """
    Yield windows (and their (row_off, col_off)) that cover the dataset.
    Prefer dataset internal block windows if tiled; otherwise use fallback grid windows.
    """
    # If the source is tiled, rasterio can iterate block windows which is efficient.
    try:
        # block_windows yields ((bidx, by, bx), window)
        any_blocks = False
        for (_, window) in ds.block_windows(1):
            any_blocks = True
            yield window, (int(window.row_off), int(window.col_off))
        if any_blocks:
            return
    except Exception:
        pass

    # Fallback: regular windows
    h, w = ds.height, ds.width
    bs = int(fallback_blocksize)
    for row_off in range(0, h, bs):
        win_h = min(bs, h - row_off)
        for col_off in range(0, w, bs):
            win_w = min(bs, w - col_off)
            window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
            yield window, (row_off, col_off)


def _make_valid_mask(block: np.ndarray, src_nodata: Optional[float]) -> np.ndarray:
    """
    block: (bands, h, w) float array
    Returns: (h, w) bool mask where True means valid pixel (no nodata/NaN in any band).
    """
    # NaNs are invalid
    invalid = np.isnan(block).any(axis=0)

    if src_nodata is not None:
        # nodata may be float; treat equal comparison carefully for integer nodata
        invalid = invalid | (block == src_nodata).any(axis=0)

    return ~invalid


def _predict_block(
    model: tf.keras.Model,
    block: np.ndarray,
    valid_mask: np.ndarray,
    scaler,
    batch_size: int,
    nodata_class: int,
    save_probs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict classes for one block.
    Returns:
      class_map: (h, w) uint8/uint16 array with nodata_class for invalid pixels
      probs_map: (num_classes, h, w) float32 array or None
    """
    bands, h, w = block.shape
    class_map = np.full((h, w), nodata_class, dtype=np.uint16)

    # Flatten valid pixels into (n, bands)
    idx = np.where(valid_mask.ravel())[0]
    if idx.size == 0:
        return class_map, None

    X = block.reshape(bands, -1).T  # (h*w, bands)
    Xv = X[idx, :].astype(np.float32, copy=False)

    if scaler is not None:
        # scaler expects 2D array
        Xv = scaler.transform(Xv)

    probs = model.predict(Xv, batch_size=batch_size, verbose=0)
    # probs shape: (n, num_classes) for softmax; or (n,1) for sigmoid binary (not expected here)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Unexpected model output shape {probs.shape}. Expected (n, num_classes).")

    pred = np.argmax(probs, axis=1).astype(np.uint16)

    # Fill class_map for valid pixels
    out_flat = class_map.ravel()
    out_flat[idx] = pred
    class_map = out_flat.reshape(h, w)

    if save_probs:
        num_classes = probs.shape[1]
        probs_map = np.zeros((num_classes, h, w), dtype=np.float32)
        # put probabilities back into raster positions
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
        raise RuntimeError("matplotlib is not available; cannot write PNG. Add matplotlib to dependencies.")
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
    # Mask nodata for display
    mask = arr == nodata
    arr_disp = np.ma.array(arr, mask=mask)

    plt.figure(figsize=(10, 10))
    if palette_path:
        palette = json.loads(Path(palette_path).read_text(encoding="utf-8"))
        # palette: { "0": [r,g,b], "1": [r,g,b], ... } or {0:[...],...}
        # Build a ListedColormap
        from matplotlib.colors import ListedColormap  # type: ignore

        # Determine max class
        keys = [int(k) for k in palette.keys()]
        max_k = max(keys) if keys else int(arr.max())
        colors = []
        for k in range(max_k + 1):
            rgb = palette.get(str(k), palette.get(k, [0, 0, 0]))
            colors.append([c / 255.0 for c in rgb])
        cmap = ListedColormap(colors)
        plt.imshow(arr_disp, cmap=cmap, interpolation="nearest")
    else:
        plt.imshow(arr_disp, interpolation="nearest")
        plt.colorbar(fraction=0.046, pad=0.04, label="Class ID")

    plt.title("Predicted land cover (class IDs)")
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

    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # Prepare probability output directory if needed
    probs_outdir = None
    if args.save_probs:
        if args.probs_outdir is None:
            probs_outdir = out_path.parent / "probs"
        else:
            probs_outdir = Path(args.probs_outdir)
        probs_outdir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(features_path) as src:
        src_nodata = src.nodata
        bands = src.count
        height, width = src.height, src.width

        # Create output profile (single band class IDs)
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint16,
            nodata=int(args.nodata),
            compress="deflate",
            predictor=2,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            # Optionally create probability outputs (one file per class)
            prob_dsts = None
            if args.save_probs:
                # We don't know num_classes until first prediction; lazily open when known.
                prob_dsts = []

            for window, _ in _iter_windows(src, args.blocksize):
                # Read as float32 for NaN handling
                block = src.read(
                    indexes=list(range(1, bands + 1)),
                    window=window,
                    out_dtype=np.float32,
                    masked=False,
                    # Leave resampling default; this is just reading
                )  # (bands, h, w)

                valid_mask = _make_valid_mask(block, src_nodata)

                class_block, probs_block = _predict_block(
                    model=model,
                    block=block,
                    valid_mask=valid_mask,
                    scaler=scaler,
                    batch_size=args.batch_size,
                    nodata_class=int(args.nodata),
                    save_probs=args.save_probs,
                )

                dst.write(class_block.astype(np.uint16), 1, window=window)

                if args.save_probs and probs_block is not None:
                    num_classes = probs_block.shape[0]
                    # Lazily create one GeoTIFF per class with same grid
                    if prob_dsts is not None and len(prob_dsts) == 0:
                        prob_profile = profile.copy()
                        prob_profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
                        for c in range(num_classes):
                            pth = probs_outdir / f"{out_path.stem}_prob_class{c:02d}.tif"
                            prob_dsts.append(rasterio.open(pth, "w", **prob_profile))
                    # Write each class prob band
                    for c in range(probs_block.shape[0]):
                        prob_dsts[c].write(probs_block[c].astype(np.float32), 1, window=window)

            # Close prob datasets
            if args.save_probs and prob_dsts:
                for ds in prob_dsts:
                    ds.close()

    # Optional quick PNG
    if args.png is not None:
        _save_png(str(out_path), args.png, nodata=int(args.nodata), palette_path=args.palette)

    print(f"Wrote predicted GeoTIFF: {out_path}")
    if args.png:
        print(f"Wrote PNG: {args.png}")
    if args.save_probs:
        print(f"Wrote probability rasters to: {probs_outdir}")


if __name__ == "__main__":
    main()