#!/usr/bin/env python3
"""
process_soft_labels.py
------------------------
Processes 2-band soft label probability maps (P(Deciduous), P(Evergreen))
by incorporating forest structure information from a Canopy Height Model (CHM).

1. Reads an input 2-band soft label tile.
2. Reads the corresponding area from a CHM, reprojecting it to match the tile.
3. Creates a binary forest mask based on a height threshold.
4. Applies Gaussian smoothing to the mask to get a P(Forest) probability map.
5. Calculates final 3-class probabilities:
   - P(NonForest) = 1 - P(Forest)
   - P(Deciduous) = P_orig(Decid) * P(Forest)
   - P(Evergreen) = P_orig(Everg) * P(Forest)
6. Saves the resulting 3-band [P(NonForest), P(Deciduous), P(Evergreen)] map.

Uses parallelism to process multiple tiles efficiently.
"""

import sys
import argparse
import logging
import numpy as np
import rasterio
import rasterio.warp
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("process_soft_labels")

DEFAULT_THRESHOLD = 5.0 # Default meters threshold for CHM -> Forest
DEFAULT_SIGMA = 1.5     # Default sigma for Gaussian smoothing
DEFAULT_WORKERS = 4

# --- Top-level Worker Function (for pickling) ---

def process_single_label_tile_worker(
    input_label_tile_path: Path,
    chm_path: Path,
    output_dir: Path,
    height_threshold: float,
    sigma: float
) -> str:
    """Function executed by each worker process to process one soft label tile."""
    try:
        output_tile_path = output_dir / input_label_tile_path.name
        # Optional: Skip if output already exists
        # if output_tile_path.exists():
        #     return f"Skipped (already exists): {input_label_tile_path.name}"

        with rasterio.open(input_label_tile_path) as src_label:
            profile = src_label.profile
            label_bounds = src_label.bounds
            label_crs = src_label.crs
            label_transform = src_label.transform
            height, width = src_label.height, src_label.width

            if profile['count'] != 2:
                raise ValueError(f"Input tile {input_label_tile_path.name} has {profile['count']} bands, expected 2.")

            # Read original probabilities P(Decid), P(Evergreen)
            p_decid_orig = src_label.read(1)
            p_everg_orig = src_label.read(2)

            # Prepare array for reprojected CHM data
            chm_aligned = np.zeros((height, width), dtype=np.float32) # Use float for CHM

            # Open CHM and reproject the relevant window
            with rasterio.open(chm_path) as src_chm:
                chm_crs = src_chm.crs
                chm_transform = src_chm.transform

                # Check if CRS match, otherwise reproject
                if label_crs != chm_crs:
                    # Calculate transform and dimensions for reprojection
                    # This ensures the reprojected CHM aligns pixel-to-pixel with the label tile
                    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                        chm_crs, label_crs, src_chm.width, src_chm.height, *src_chm.bounds,
                        dst_width=width, dst_height=height, dst_bounds=label_bounds
                    )

                    rasterio.warp.reproject(
                        source=rasterio.band(src_chm, 1),
                        destination=chm_aligned,
                        src_transform=chm_transform,
                        src_crs=chm_crs,
                        dst_transform=label_transform, # Use label tile's transform
                        dst_crs=label_crs,
                        resampling=rasterio.warp.Resampling.bilinear # Or nearest if preferred
                    )
                else:
                    # CRS match, just read the window directly
                    window = src_chm.window(*label_bounds)
                    chm_aligned = src_chm.read(1, window=window, out_shape=(height, width), resampling=rasterio.warp.Resampling.bilinear)

            # Handle potential NoData in CHM (replace with 0, assuming non-forest)
            chm_nodata = src_chm.nodata
            if chm_nodata is not None:
                 chm_aligned[chm_aligned == chm_nodata] = 0.0
            chm_aligned = np.nan_to_num(chm_aligned, nan=0.0)

            # 1. Create binary forest mask
            forest_mask_binary = (chm_aligned >= height_threshold).astype(np.float32)

            # 2. Apply Gaussian smoothing to get P(Forest)
            p_forest = gaussian_filter(forest_mask_binary, sigma=sigma)
            p_forest = np.clip(p_forest, 0.0, 1.0) # Ensure strictly within [0, 1]

            # 3. Calculate final 3 probabilities
            p_non_forest = np.clip(1.0 - p_forest, 0.0, 1.0)
            p_decid_final = np.clip(p_decid_orig * p_forest, 0.0, 1.0)
            p_everg_final = np.clip(p_everg_orig * p_forest, 0.0, 1.0)

            # 4. Prepare output profile and write
            out_profile = profile.copy()
            out_profile.update({
                'count': 3,
                'dtype': 'float32',
                'nodata': None # Typically no nodata for probability maps
            })

            with rasterio.open(output_tile_path, 'w', **out_profile) as dst:
                dst.write(p_non_forest, 1)
                dst.write(p_decid_final, 2)
                dst.write(p_everg_final, 3)
                # Optional: Set band descriptions
                dst.set_band_description(1, 'P(NonForest)')
                dst.set_band_description(2, 'P(Deciduous)')
                dst.set_band_description(3, 'P(Evergreen)')

        return f"Successfully processed {input_label_tile_path.name}"

    except Exception as e:
        # Log error using standard logging if possible, or print
        # logging.error(f"Worker failed on tile {input_label_tile_path.name}: {e}", exc_info=True)
        print(f"ERROR: Worker failed on tile {input_label_tile_path.name}: {e}")
        return f"Failed to process {input_label_tile_path.name}"

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Process 2-band soft labels using a CHM to create 3-band probability maps.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input 2-band soft label tiles (GeoTIFF).")
    parser.add_argument("--chm", type=str, required=True,
                        help="Path to the Canopy Height Model (CHM) GeoTIFF.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output 3-band probability map tiles (GeoTIFF).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Height threshold (meters) for CHM to define forest (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"Sigma value for Gaussian filter smoothing of the forest mask (default: {DEFAULT_SIGMA}).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    input_dir = Path(args.input_dir)
    chm_path = Path(args.chm)
    output_dir = Path(args.output_dir)

    # Basic input validation
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    if not chm_path.is_file():
        logger.error(f"CHM file not found: {chm_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input label tiles
    input_files = list(input_dir.glob('*.tif*'))
    if not input_files:
        logger.error(f"No .tif or .tiff files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} input tiles in {input_dir}. Processing with CHM: {chm_path.name}")

    # --- Parallel Execution ---
    logger.info(f"Starting parallel processing with {args.workers} workers.")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_single_label_tile_worker,
                tile_path,
                chm_path,
                output_dir,
                args.threshold,
                args.sigma
            )
            for tile_path in input_files
        ]

        for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing Soft Labels"):
            results.append(future.result())

    logger.info("All tiles processed.")
    success_count = sum(1 for res in results if "Successfully" in res)
    failure_count = len(results) - success_count
    logger.info(f"Processing Summary: Success={success_count}, Failures={failure_count}")
    if failure_count > 0:
         logger.warning(f"Check log/output for details on the {failure_count} failures.")

if __name__ == "__main__":
    main() 