#!/usr/bin/env python3
"""
inference_feature.py
---------------------
Compute harmonic features for a single tile (i.e. a single .tif) 
based on the time series logic. We define a “window-based” approach 
or do it all at once if memory allows. We'll produce a multi-band TIF 
with [amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, ...] 
for each spectral index (NDVI, EVI, NBR, CRSWIR).
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import Affine
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

from src.utils import (
    compute_indices, 
    compute_quality_weights,
    robust_harmonic_fitting,
    scale_amplitude,
    scale_phase,
    scale_offset
)
from src.constants import (
    DEFAULT_NUM_HARMONICS,
    DEFAULT_MAX_ITER,
    AVAILABLE_INDICES
)

def setup_logger():
    logger = logging.getLogger("inference_feature")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)
    return logger

def read_bands(tif_path: Path):
    """
    Read the 6-band mosaic: B2, B4, B8, B11, B12, MSK_CLDPRB.
    Return them as a dict of arrays: 
       {
         "B2": <2D array>,
         "B4": <2D array>,
         ...
         "MSK_CLDPRB": <2D array>
       }
    We do not do extensive checks here. 
    """
    out = {}
    with rasterio.open(tif_path) as src:
        # We expect: B2=1, B4=2, B8=3, B11=4, B12=5, MSK_CLDPRB=6 (or some order)
        # Adjust if needed depending on how your mosaic is written
        # (Check the real order in your data!)
        band_order = {
            "B2": 1,
            "B4": 2,
            "B8": 3,
            "B11":4,
            "B12":5,
            "MSK_CLDPRB":6
        }
        for k,v in band_order.items():
            out[k] = src.read(v)
    return out

def compute_features_for_window(
    tif_paths: list[Path],
    dates: list[datetime],
    window: Window,
    profile: dict,
    num_harmonics: int,
    max_iter: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Read data in [time, H, W], compute spectral indices, do robust harmonic fitting,
    then return a stacked set of features for that window. We'll produce 
    #indices * (2*num_harmonics + 1) * 2 => amplitude + phase for each harmonic + offset, 
    plus we might store residual variance? You can define how many final bands you want.

    We'll store them scaled to uint16. Each index => 2*N_harm + 1 offsets, plus 2*N_harm phases 
    => for 2 harmonics => amplitude h1, amplitude h2, phase h1, phase h2, offset => 5 bands per index,
    plus 1 band of variance => 6 bands per index. 
    Or skip the variance if you prefer.

    Returns (band_count, H, W) array of uint16.
    """
    H = window.height
    W = window.width
    T = len(tif_paths)

    # Allocate array for bands
    # We'll store them as float for now (or int?), then do scaling at the end
    # B2, B4, B8, B11, B12 => shape (T,H,W)
    arr_b2  = np.zeros((T,H,W), dtype=np.float32)
    arr_b4  = np.zeros((T,H,W), dtype=np.float32)
    arr_b8  = np.zeros((T,H,W), dtype=np.float32)
    arr_b11 = np.zeros((T,H,W), dtype=np.float32)
    arr_b12 = np.zeros((T,H,W), dtype=np.float32)
    arr_cld = np.zeros((T,H,W), dtype=np.float32)

    # Read the window from each TIF
    for i, tifp in enumerate(tif_paths):
        with rasterio.open(tifp) as src:
            # read the 6 bands
            b2  = src.read(1, window=window).astype(np.float32)
            b4  = src.read(2, window=window).astype(np.float32)
            b8  = src.read(3, window=window).astype(np.float32)
            b11 = src.read(4, window=window).astype(np.float32)
            b12 = src.read(5, window=window).astype(np.float32)
            cld = src.read(6, window=window).astype(np.float32)
            arr_b2[i]  = b2
            arr_b4[i]  = b4
            arr_b8[i]  = b8
            arr_b11[i] = b11
            arr_b12[i] = b12
            arr_cld[i] = cld

    # Convert cloud band to weights in [0..1]
    qa_weights = compute_quality_weights(arr_cld, logger=logger)

    # Now compute the indices (ndvi, evi, etc.)
    ndvi, evi, nbr, crswir = compute_indices(
        arr_b2, arr_b4, arr_b8, arr_b11, arr_b12, logger=logger
    )

    # We can store them in a dict for convenience
    indices_dict = {
        "ndvi": ndvi,
        "evi": evi,
        "nbr": nbr,
        "crswir": crswir
    }

    # We'll compute harmonic fitting per index 
    # each call => amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_resid
    results = {}
    for idx_name in AVAILABLE_INDICES:
        cube = indices_dict[idx_name]
        (
            amp_h1, amp_h2, phs_h1, phs_h2, offset_map, var_map
        ) = robust_harmonic_fitting(
            cube, qa_weights, dates, 
            num_harmonics=num_harmonics, 
            max_iter=max_iter,
            logger=logger
        )
        results[idx_name] = {
            "amp_h1": amp_h1, "amp_h2": amp_h2,
            "phs_h1": phs_h1, "phs_h2": phs_h2,
            "offset": offset_map, "var": var_map
        }

    # We'll stack them in a specific band order
    # For each index in AVAILABLE_INDICES:
    #  band1=amp_h1, band2=amp_h2, band3=phs_h1, band4=phs_h2, band5=offset, band6=var
    # So total bands = 6 * number_of_indices
    out_bands = []
    for idx_name in AVAILABLE_INDICES:
        dic = results[idx_name]
        # Scale amplitude
        a1 = scale_amplitude(dic["amp_h1"])
        a2 = scale_amplitude(dic["amp_h2"])
        p1 = scale_phase(dic["phs_h1"])
        p2 = scale_phase(dic["phs_h2"])
        o  = scale_offset(dic["offset"])
        # variance we might just store as amplitude?
        # or do a separate scale, say clamp var to [0..some max]
        var_clamped = np.clip(dic["var"], 0, 2.0) # arbitrary
        v  = scale_array_to_uint16(var_clamped, 0.0, 2.0)

        out_bands.extend([a1, a2, p1, p2, o, v])

    # shape => (band_count, H, W)
    output_cube = np.stack(out_bands, axis=0)
    return output_cube

def generate_windows(width, height, block_size=1024):
    """Yield rasterio windows for chunked processing."""
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            w = min(block_size, width - x)
            h = min(block_size, height - y)
            yield Window(x, y, w, h)

def main():
    parser = argparse.ArgumentParser(description="Compute harmonic features on a single or multiple TIFs.")
    parser.add_argument("--config-dir", type=str, required=True,
                        help="Directory with the tile_config_XXX.json and metadata.json.")
    parser.add_argument("--tile-idx", type=int, required=True,
                        help="Which tile config index to process.")
    parser.add_argument("--num-harmonics", type=int, default=DEFAULT_NUM_HARMONICS,
                        help="Number of harmonics to fit.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help="Max IRLS iteration.")
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Block/window size for chunked processing.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers for window processing.")
    args = parser.parse_args()

    logger = setup_logger()

    # Load metadata
    config_dir = Path(args.config_dir)
    meta_path = config_dir / "metadata.json"
    if not meta_path.exists():
        logger.error(f"Cannot find metadata.json in {config_dir}")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    # load tile config
    tile_config_path = config_dir / f"tile_config_{args.tile_idx:03d}.json"
    if not tile_config_path.exists():
        logger.error(f"Cannot find tile_config_{args.tile_idx:03d}.json")
        sys.exit(1)
    with open(tile_config_path) as f:
        tile_cfg = json.load(f)

    # We only have 1 TIF path and 1 date in this "simple" approach, 
    # but maybe we want to gather multiple TIFs and multiple dates. 
    # For now, let's assume there's only 1 TIF in each config. 
    tif_path = Path(tile_cfg["tif_path"])
    date_str = tile_cfg["date"]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # If you have a list of TIFs/dates, you'd parse them differently.

    if not tif_path.exists():
        logger.error(f"Input TIF not found: {tif_path}")
        sys.exit(1)

    logger.info(f"Computing features for tile {args.tile_idx}, TIF={tif_path}, date={date_obj}")

    # For an actual multi-date scenario, you'd have a list of TIFs and date_obj. 
    # We'll just do a single date, so the "harmonic fitting" is somewhat meaningless. 
    # Realistically, you want multiple TIFs across time. 
    # But let's pretend we have a timeseries of length 1 => can't fit a harmonic. 
    # Possibly we want each tile config to list multiple TIFs. 
    # We'll do a single for demonstration.

    tif_paths = [tif_path]   # single element
    dates = [date_obj]       # single element => not enough for a real harmonic, but let's proceed

    # Read the base profile for output
    with rasterio.open(tif_path) as src:
        base_profile = src.profile
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

    # We'll output a multi-band TIF with shape (#features, height, width)
    # We'll chunk it window by window. 
    out_name = f"features_{tile_cfg['tile_idx']:03d}.tif"
    out_path = Path(meta["output_dir"]) / out_name

    # Prepare the final profile
    # We have 6 bands per index, 4 indices => 24 bands total
    # But if we do 2 harmonics => ( amp_h1, amp_h2, phs_h1, phs_h2, offset, var ) => 6 bands 
    # times 4 indices = 24 bands
    # If you had more or fewer indices, adjust accordingly.
    nb_indices = len(AVAILABLE_INDICES)
    nb_bands = nb_indices * 6

    out_profile = base_profile.copy()
    out_profile.update({
        "count": nb_bands,
        "dtype": "uint16",
        "compress": "lzw",
        "predictor": 2,  # sometimes good for floating data
        "BIGTIFF": "YES" # if large
    })

    logger.info(f"Writing features to {out_path} with {nb_bands} bands.")
    logger.info(f"Window size = {args.block_size}, parallel workers = {args.max_workers}.")

    # Create output in 'w' mode
    with rasterio.open(out_path, "w", **out_profile) as dst:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            all_windows = list(generate_windows(width, height, block_size=args.block_size))
            for w in all_windows:
                f = executor.submit(
                    compute_features_for_window,
                    tif_paths, dates, w, base_profile,
                    args.num_harmonics, args.max_iter,
                    logger
                )
                futures[f] = w

            for fut in tqdm(as_completed(futures), total=len(all_windows), desc="Processing windows"):
                w = futures[fut]
                try:
                    res_cube = fut.result()  # shape (nb_bands, w.height, w.width)
                    # write to dst
                    for band_i in range(nb_bands):
                        dst.write(res_cube[band_i], band_i+1, window=w)
                except Exception as e:
                    logger.error(f"Window {w} failed: {e}")

    logger.info("Feature extraction done.")

if __name__ == "__main__":
    main()
