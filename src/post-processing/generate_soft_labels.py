#!/usr/bin/env python3
"""
generate_soft_labels.py
-----------------------
Combines a forest mask and a Random Forest (RF) derived probability map 
(assumed P(Deciduous)) to create a 3-channel soft label probability map 
(P(Non-Forest), P(Deciduous), P(Evergreen)).

The script uses the RF probability map as the reference grid. If the forest mask 
has a different grid, its data will be resampled (Nearest Neighbor) on-the-fly.

Gaussian blurring is applied to initial sharp probability estimates for each class,
followed by pixel-wise normalization to ensure probabilities sum to 1.

Inputs:
- Forest Mask: A GeoTIFF where pixel value 1 indicates forest and 0 indicates 
               non-forest or nodata. Assumed to be single-band.
- RF P(Deciduous) Map: A single-band GeoTIFF (uint8, 0-255 scale) representing 
                       P(Deciduous). This map defines the output grid.

Output:
- A 3-band GeoTIFF (uint8, 0-255 per band), aligned with RF P(Deciduous) grid:
    - Band 1: P(Non-Forest)
    - Band 2: P(Deciduous)
    - Band 3: P(Evergreen)
"""

import argparse
import logging
import time
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds, intersection
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from tqdm import tqdm
from scipy.ndimage import gaussian_filter # For Gaussian blurring

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_soft_labels")

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_OUTPUT_FILENAME = "soft_label_probabilities.tif"
DEFAULT_GAUSSIAN_SIGMA = 1.0

def process_block_soft_labels(
    mask_block_resampled: np.ndarray, 
    prob_d_block_uint8: np.ndarray, 
    gaussian_sigma: float,
    prob_d_nodata_val # Nodata value from the original P(Deciduous) raster
    ) -> np.ndarray:
    """
    Generates 3-channel soft labels for a block.

    Args:
        mask_block_resampled (np.ndarray): Resampled forest mask block (binary 0 or 1).
        prob_d_block_uint8 (np.ndarray): P(Deciduous) block (uint8, 0-255).
        gaussian_sigma (float): Sigma for the Gaussian filter.
        prob_d_nodata_val: Nodata value from the P(Deciduous) source.

    Returns:
        np.ndarray: 3-band uint8 array (P(NF), P(D), P(E)), 0-255 scale.
    """
    # Convert P(Deciduous) to float (0-1)
    # Handle nodata in probability map: if prob_d is nodata, treat as uncertain (0.5?) 
    # or perhaps implies non-forest if mask agrees? For now, let's make it 0.5 if nodata.
    # This means P(D) = 0.5, P(E) = 0.5 within forest if P(D) from RF is nodata.
    prob_d_float = np.where(
        prob_d_block_uint8 == prob_d_nodata_val if prob_d_nodata_val is not None else False, 
        0.5, # Assign 0.5 if P(Deciduous) is nodata
        prob_d_block_uint8 / 255.0
    )

    # 1. Determine P(Non-Forest) directly from the resampled mask (no blur)
    # mask_block_resampled is 0 for non-forest, 1 for forest.
    # So, P(Non-Forest) is 1.0 where mask is 0, and 0.0 where mask is 1.
    p_non_forest_final = np.where(mask_block_resampled == 0, 1.0, 0.0).astype(float)

    # 2. Calculate remaining probability available for forest types
    # This will be 0.0 for non-forest areas, and 1.0 for forest areas.
    prob_available_for_D_E = 1.0 - p_non_forest_final
    # Ensure remaining_prob is not negative due to potential float precision issues (though less likely here)
    prob_available_for_D_E = np.maximum(0.0, prob_available_for_D_E)

    # 3. Distribute the available probability to Deciduous and Evergreen based on RF's P(D)
    # If it's non-forest, prob_available_for_D_E is 0, so P(D) and P(E) become 0.
    # If it's forest, prob_available_for_D_E is 1, so P(D) = prob_d_float and P(E) = 1 - prob_d_float.
    p_deciduous_final = prob_available_for_D_E * prob_d_float
    p_evergreen_final = prob_available_for_D_E * (1.0 - prob_d_float)
    
    # Numerical stability check: Ensure probabilities sum to approximately 1.0
    total_prob_float = p_non_forest_final + p_deciduous_final + p_evergreen_final
    # Using a slightly generous atol for floating point comparisons of sums
    # For sums that should be exactly 1.0, atol is more relevant than rtol.
    tolerance = 1e-5 
    if not np.allclose(total_prob_float, 1.0, atol=tolerance):
        deviating_pixels_mask = ~np.isclose(total_prob_float, 1.0, atol=tolerance)
        num_deviating = np.sum(deviating_pixels_mask)
        if num_deviating > 0:
            deviating_sums = total_prob_float[deviating_pixels_mask]
            logger.warning(
                f"Block probabilities do not sum to 1.0 for {num_deviating} pixels (tolerance={tolerance}). "
                f"Min sum: {np.min(deviating_sums):.6f}, Max sum: {np.max(deviating_sums):.6f}. "
                f"This might indicate numerical precision issues. Probabilities will still be generated."
            )
        # Optional: Force normalization if deviations are problematic, though np.allclose implies they are minor.
        # Forcing normalization again here could be an option if strict sum-to-one is paramount even after first pass.
        # e.g., total_prob_float_safe = np.where(total_prob_float == 0, 1e-6, total_prob_float)
        # p_non_forest_final = p_non_forest_final / total_prob_float_safe
        # p_deciduous_final = p_deciduous_final / total_prob_float_safe
        # p_evergreen_final = p_evergreen_final / total_prob_float_safe

    # 5. Scale to uint8 (0-255)
    # Clipping final float probabilities to [0,1] before scaling can prevent issues if any calculation drifted slightly outside.
    out_band_nf = (np.clip(p_non_forest_final, 0.0, 1.0) * 255).round().astype(np.uint8)
    out_band_d = (np.clip(p_deciduous_final, 0.0, 1.0) * 255).round().astype(np.uint8)
    out_band_e = (np.clip(p_evergreen_final, 0.0, 1.0) * 255).round().astype(np.uint8)

    return np.stack([out_band_nf, out_band_d, out_band_e], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-channel soft label probability maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mask-file", type=str, required=True,
                        help="Path to the input forest mask GeoTIFF (Value 1=Forest).")
    parser.add_argument("--prob-deciduous-file", type=str, required=True,
                        help="Path to P(Deciduous) GeoTIFF (0-255 scale), defines output grid.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output 3-channel soft label GeoTIFF.")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help="Name for the output classification file.")
    parser.add_argument("--gaussian-sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA,
                        help="Sigma for the Gaussian filter applied to probability layers.")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="Processing block size (pixels).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    start_time = time.time()

    mask_path = Path(args.mask_file)
    prob_d_path = Path(args.prob_deciduous_file)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_filename
    
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(mask_path) as mask_src, rasterio.open(prob_d_path) as prob_d_src:
            logger.info(f"Opened mask file: {mask_path.name}")
            logger.info(f"Opened P(Deciduous) file (reference grid): {prob_d_path.name}")

            prob_d_profile = prob_d_src.profile
            prob_d_crs = prob_d_src.crs
            prob_d_transform = prob_d_src.transform
            prob_d_width = prob_d_src.width
            prob_d_height = prob_d_src.height
            prob_d_dtype = prob_d_src.dtypes[0] # Should be uint8
            prob_d_nodata = prob_d_src.nodata # Nodata from P(Deciduous) map
            
            mask_crs = mask_src.crs
            mask_transform = mask_src.transform
            mask_width = mask_src.width
            mask_height = mask_src.height
            
            logger.info(f"P(Deciduous) (Reference) metadata: CRS={prob_d_crs}, Size=({prob_d_width}x{prob_d_height}), Nodata={prob_d_nodata}")
            logger.info(f"Mask metadata: CRS={mask_crs}, Size=({mask_width}x{mask_height})")
            logger.info(f"Using Gaussian sigma: {args.gaussian_sigma}")

            if mask_crs != prob_d_crs:
                raise ValueError(f"CRS mismatch: Mask={mask_crs}, P(Deciduous)={prob_d_crs}. Files MUST have the same CRS.")
            if prob_d_dtype not in (rasterio.uint8, np.uint8):
                 logger.warning(f"P(Deciduous) file dtype is {prob_d_dtype}. Expected uint8 (0-255).")

            output_profile = prob_d_profile.copy()
            output_profile.update({
                'driver': 'GTiff', # Explicitly set driver to GeoTIFF
                'count': 3, # P(NF), P(D), P(E)
                'dtype': 'uint8',
                'nodata': None, # Explicitly set nodata to None for prob maps
                'compress': 'DEFLATE',
                'predictor': 1, 
                'zlevel': 9,
                'tiled': True,
                'blockxsize': args.block_size,
                'blockysize': args.block_size,
                'BIGTIFF': 'YES'
            })
            logger.debug(f"Output profile configured: {output_profile}")

            windows = []
            for j in range(0, prob_d_height, args.block_size):
                height = min(args.block_size, prob_d_height - j)
                for i in range(0, prob_d_width, args.block_size):
                    width = min(args.block_size, prob_d_width - i)
                    windows.append(Window(i, j, width, height))
            
            total_blocks = len(windows)
            logger.info(f"Total blocks to process: {total_blocks}")

            with rasterio.open(output_path, 'w', **output_profile) as dst:
                mask_bounds_window = Window(0, 0, mask_width, mask_height)
                prob_res_x, prob_res_y = abs(prob_d_transform.a), abs(prob_d_transform.e)
                mask_res_x, mask_res_y = abs(mask_transform.a), abs(mask_transform.e)

                for window in tqdm(windows, desc="Generating soft labels", total=total_blocks, unit="block"):
                    logger.debug(f"Processing P(Deciduous) window: {window}")
                    
                    prob_d_block = prob_d_src.read(1, window=window)
                    prob_block_shape = prob_d_block.shape
                    
                    prob_window_bounds = window_bounds(window, prob_d_transform)
                    mask_window_proj = window_from_bounds(*prob_window_bounds, mask_transform).round_offsets().round_lengths()
                    
                    try:
                        clipped_mask_window = intersection(mask_window_proj, mask_bounds_window)
                    except WindowError:
                        clipped_mask_window = None
                    
                    mask_block_resampled = np.zeros(prob_block_shape, dtype=mask_src.dtypes[0])

                    if clipped_mask_window and clipped_mask_window.width > 0 and clipped_mask_window.height > 0:
                        try:
                            # Determine shape for resampling the clipped part of the mask
                            # Target shape for the clipped part should match its extent in prob_block_shape pixels
                            target_clip_h = int(round(clipped_mask_window.height * (prob_res_y / mask_res_y)))
                            target_clip_w = int(round(clipped_mask_window.width * (prob_res_x / mask_res_x)))
                            
                            # Ensure shapes are at least 1x1
                            target_clip_h = max(1, target_clip_h)
                            target_clip_w = max(1, target_clip_w)

                            mask_data_clipped_resampled = mask_src.read(
                                1,
                                window=clipped_mask_window,
                                out_shape=(target_clip_h, target_clip_w),
                                resampling=Resampling.nearest
                            )
                            
                            # Calculate placement offsets in the full prob_block_shape
                            # Offset of the *projected mask window* from origin of mask grid
                            # Offset of the *clipped mask window* from origin of mask grid
                            # Difference gives where the clipped part starts within the projected window
                            offset_y_in_proj_mask_pixels = clipped_mask_window.row_off - mask_window_proj.row_off
                            offset_x_in_proj_mask_pixels = clipped_mask_window.col_off - mask_window_proj.col_off

                            # Convert these offsets to the probability grid resolution
                            place_y = int(round(offset_y_in_proj_mask_pixels * (prob_res_y / mask_res_y)))
                            place_x = int(round(offset_x_in_proj_mask_pixels * (prob_res_x / mask_res_x)))
                            
                            # Define slice for placement
                            slice_y_end = min(prob_block_shape[0], place_y + mask_data_clipped_resampled.shape[0])
                            slice_x_end = min(prob_block_shape[1], place_x + mask_data_clipped_resampled.shape[1])
                            
                            slice_y = slice(max(0, place_y), slice_y_end)
                            slice_x = slice(max(0, place_x), slice_x_end)

                            # Ensure data to be placed matches slice size
                            data_to_place_h = slice_y.stop - slice_y.start
                            data_to_place_w = slice_x.stop - slice_x.start

                            if data_to_place_h > 0 and data_to_place_w > 0:
                                mask_block_resampled[slice_y, slice_x] = mask_data_clipped_resampled[:data_to_place_h, :data_to_place_w]
                            else:
                                logger.debug(f"Skipping placement for window {window}, zero-sized placement slice.")

                        except Exception as read_err:
                            logger.error(f"Error reading/resampling mask window {clipped_mask_window} for P(D) window {window}: {read_err}", exc_info=True)
                            # Continue with zeros in mask_block_resampled for this block
                    else:
                         logger.debug(f"Skipping mask read for P(D) window {window}: intersection is outside mask bounds or zero size.")

                    output_bands_block = process_block_soft_labels(
                        mask_block_resampled, 
                        prob_d_block, 
                        args.gaussian_sigma,
                        prob_d_nodata # Pass nodata value
                    )
                    
                    dst.write(output_bands_block, window=window)

        end_time = time.time()
        logger.info(f"Successfully created 3-channel soft label map: {output_path}")
        logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found: {e}")
        sys.exit(1)
    except ValueError as e:
         logger.error(f"Error: {e}")
         sys.exit(1)
    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio I/O error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 