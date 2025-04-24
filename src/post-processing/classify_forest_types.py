#!/usr/bin/env python3
"""
classify_forest_types.py
--------------------------
Combines a forest mask and a probability map to create a final classified 
forest type map. The script uses the **probability map as the reference grid**. 
If the forest mask has a different grid (dimensions, transform, resolution), 
its data will be resampled (Nearest Neighbor) on-the-fly to match the 
probability map's grid block by block.

Inputs:
- Forest Mask: A GeoTIFF where pixel value 1 indicates forest and 0 indicates 
               non-forest or nodata. Assumed to be single-band (uint8 or uint16)
               in the same CRS as the probability map.
- Probability Map: A single-band GeoTIFF (uint8, 0-255 scale) representing the 
                   probability of a pixel belonging to the *first* class 
                   defined in the classification model (e.g., output of 
                   compress_rf_probabilities.py). 
                   **Crucially, this script assumes the probability map represents P(Deciduous).**
                   This map defines the output grid.

Output Classes (2-bit GeoTIFF, matching probability map grid):
- 0: No Forest / Nodata / Unknown (Resampled mask == 0 OR Resampled mask == 1 but probability is nodata)
- 1: Deciduous Forest (Resampled mask == 1 AND Input probability >= 0.5 [128/255])
- 2: Evergreen Forest (Resampled mask == 1 AND Input probability < 0.5 [128/255])
"""

import argparse
import logging
import time
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, from_bounds as window_from_bounds, intersection
from rasterio.enums import Resampling # Import Resampling enum
from rasterio.errors import WindowError # Import WindowError
from tqdm import tqdm

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("classify_forest_types")

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_OUTPUT_FILENAME = "forest_classification.tif"
PROBABILITY_THRESHOLD = 128 # Corresponds to P=0.5 on a 0-255 scale

def classify_block(mask_block, prob_block, prob_nodata):
    """
    Classifies a block based on forest mask and probability (assumed P(Deciduous)).
    Both input blocks are assumed to be aligned and have the same shape.
    
    Args:
        mask_block (np.ndarray): Input forest mask block (e.g., uint8/uint16: 0 or 1), 
                                 potentially resampled. Should have same shape as prob_block.
        prob_block (np.ndarray): Input probability block (uint8: 0-255), assumed P(Deciduous).
                                 Should have same shape as mask_block.
        prob_nodata (float/int or None): Nodata value for the probability raster.

    Returns:
        np.ndarray: Classified block (uint8: 0, 1, or 2), same shape as inputs.
    """
    # Check if shapes match (important if resampling occurred)
    if mask_block.shape != prob_block.shape:
        raise ValueError(f"Internal error: mask_block shape {mask_block.shape} != prob_block shape {prob_block.shape}")
        
    # Initialize output block with 0 (No Forest / Nodata / Unknown)
    output_block = np.zeros_like(prob_block, dtype=np.uint8) # Use prob_block shape as reference
    
    # Find pixels that are forest in the (potentially resampled) mask
    forest_pixels = (mask_block == 1)

    # Identify forest pixels where the probability value is valid (not nodata)
    if prob_nodata is not None:
        # Ensure prob_nodata has the same dtype as prob_block for comparison
        if np.issubdtype(prob_block.dtype, np.floating) and not isinstance(prob_nodata, float):
            prob_nodata_cmp = float(prob_nodata)
        elif np.issubdtype(prob_block.dtype, np.integer) and not isinstance(prob_nodata, int):
             prob_nodata_cmp = int(prob_nodata)
        else:
             prob_nodata_cmp = prob_nodata
        valid_prob_pixels = forest_pixels & (prob_block != prob_nodata_cmp)
    else:
        valid_prob_pixels = forest_pixels # Assume all probability values are valid if nodata is None
        
    # Class 1: Deciduous (mask=1, P(Deciduous) >= threshold)
    deciduous_pixels = valid_prob_pixels & (prob_block >= PROBABILITY_THRESHOLD)
    output_block[deciduous_pixels] = 1

    # Class 2: Evergreen (mask=1, P(Deciduous) < threshold)
    evergreen_pixels = valid_prob_pixels & (prob_block < PROBABILITY_THRESHOLD)
    output_block[evergreen_pixels] = 2
    
    # Pixels where resampled mask was 0 remain 0.
    # Pixels where resampled mask was 1 but prob was nodata also remain 0.

    return output_block

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify forest types using a mask (1=Forest) and a probability map (assumed P(Deciduous)). "
            "Uses the probability map as the reference grid, resampling the mask if necessary."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
    )
    parser.add_argument("--mask-file", type=str, required=True, 
                        help="Path to the input forest mask GeoTIFF (Value 1=Forest, 0=Non-Forest/Nodata). Must be in the same CRS as the probability map.")
    parser.add_argument("--prob-file", type=str, required=True,
                        help="Path to the input probability GeoTIFF (0-255 scale), assumed to be P(Deciduous). This defines the output grid.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output classified GeoTIFF.")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help="Name for the output classification file.")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="Processing block size (pixels), applied to the probability map grid.")
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
    prob_path = Path(args.prob_file)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_filename
    
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Open input files and validate ---
        with rasterio.open(mask_path) as mask_src, rasterio.open(prob_path) as prob_src:
            logger.info(f"Opened mask file: {mask_path.name}")
            logger.info(f"Opened probability file (reference grid): {prob_path.name}")

            # Get metadata
            prob_profile = prob_src.profile
            prob_crs = prob_src.crs
            prob_transform = prob_src.transform
            prob_width = prob_src.width
            prob_height = prob_src.height
            prob_dtype = prob_src.dtypes[0]
            prob_nodata = prob_src.nodata
            
            mask_crs = mask_src.crs
            mask_transform = mask_src.transform
            mask_width = mask_src.width
            mask_height = mask_src.height
            mask_dtype = mask_src.dtypes[0]
            mask_nodata = mask_src.nodata # Store mask nodata if needed later, though classification uses 1
            
            logger.info(f"Prob (Reference) metadata: CRS={prob_crs}, Size=({prob_width}x{prob_height}), Transform={prob_transform}, Dtype={prob_dtype}, Nodata={prob_nodata}")
            logger.info(f"Mask metadata: CRS={mask_crs}, Size=({mask_width}x{mask_height}), Transform={mask_transform}, Dtype={mask_dtype}, Nodata={mask_nodata}")
            logger.info(f"Using Probability Threshold: {PROBABILITY_THRESHOLD}/255 (>= {PROBABILITY_THRESHOLD/255.0:.3f}) for Deciduous (Class 1)")

            # --- Input Validation ---
            if mask_src.count != 1 or prob_src.count != 1:
                raise ValueError("Input files must be single-band rasters.")
            if mask_dtype not in (rasterio.uint8, rasterio.uint16, np.uint8, np.uint16): # Check rasterio and numpy types
                 logger.warning(f"Mask file dtype is {mask_dtype}. Expected uint8 or uint16. Proceeding, but ensure values are 0 and 1.")
            if prob_dtype not in (rasterio.uint8, np.uint8):
                 logger.warning(f"Probability file dtype is {prob_dtype}. Expected uint8 (0-255). Ensure threshold is appropriate.")
                 
            # CRITICAL: Check CRS match
            if mask_crs != prob_crs:
                raise ValueError(f"CRS mismatch: Mask={mask_crs}, Prob={prob_crs}. Files MUST have the exact same CRS.")
            
            # Check pixel size match (optional, but good practice)
            prob_res_x, prob_res_y = abs(prob_transform.a), abs(prob_transform.e)
            mask_res_x, mask_res_y = abs(mask_transform.a), abs(mask_transform.e)
            if not (np.allclose(prob_res_x, mask_res_x) and np.allclose(prob_res_y, mask_res_y)):
                 logger.warning(f"Pixel resolutions differ. Prob: ({prob_res_x:.2f}, {prob_res_y:.2f}), Mask: ({mask_res_x:.2f}, {mask_res_y:.2f}). Mask data will be resampled.")
            else:
                 logger.info("Pixel resolutions match.")

            # Check for Lambert 93 (EPSG:2154) - Informational
            try:
                if prob_crs and prob_crs.is_valid and prob_crs.to_epsg() == 2154:
                    logger.info("Reference CRS is EPSG:2154 (Lambert 93).")
                else:
                    logger.warning(f"Reference CRS is not EPSG:2154 (Lambert 93). Found: {prob_crs}")
            except AttributeError:
                 logger.warning(f"Could not verify reference CRS EPSG code. CRS: {prob_crs}")


            # --- Prepare output profile ---
            # Base the output profile on the PROBABILITY map
            output_profile = prob_profile.copy() 
            output_profile.update({
                'driver': 'GTiff',
                'dtype': 'uint8',      # Classified output (0, 1, 2)
                'count': 1,
                'nodata': None,        # Explicitly set nodata to None as 0 is a valid class
                'compress': 'DEFLATE', # Good lossless compression for thematic maps
                'predictor': 1,        # No predictor or PREDICTOR=1 is often fine for thematic
                'zlevel': 9,           # Max compression
                'tiled': True,         # Essential for large files
                'blockxsize': args.block_size, # Match processing block size
                'blockysize': args.block_size, # Match processing block size
                'BIGTIFF': 'YES'       # Safer for potentially large outputs
            })
            # NBITS=2 creation option optimizes storage for 3 values (0, 1, 2)
            creation_options = {'nbits': 2} 
            logger.debug(f"Output profile configured: {output_profile}")
            logger.debug(f"Output creation options: {creation_options}")

            # --- Process block by block using PROBABILITY map windows ---
            logger.info(f"Starting classification using probability map grid with block size {args.block_size}x{args.block_size}.")
            
            # Generate windows manually based on the desired block size
            windows = []
            for j in range(0, prob_height, args.block_size):
                height = min(args.block_size, prob_height - j)
                for i in range(0, prob_width, args.block_size):
                    width = min(args.block_size, prob_width - i)
                    windows.append(Window(i, j, width, height))

            total_blocks = len(windows)
            logger.info(f"Total blocks to process: {total_blocks}")
            
            # Write output using nbits=2
            with rasterio.open(output_path, 'w', **output_profile, **creation_options) as dst:
                # Define the full bounds window for the mask
                mask_bounds_window = Window(0, 0, mask_width, mask_height)
                
                for window in tqdm(windows, desc="Classifying blocks", total=total_blocks, unit="block"):
                    logger.debug(f"Processing prob window: {window}")
                    
                    # 1. Read probability block
                    prob_block = prob_src.read(1, window=window)
                    prob_block_shape = prob_block.shape # Store shape for resampling target
                    
                    # 2. Calculate geographic bounds of the probability window
                    prob_window_bounds = window_bounds(window, prob_transform)
                    
                    # 3. Calculate the corresponding window in the mask CRS/transform
                    mask_window = window_from_bounds(*prob_window_bounds, mask_transform) 
                    mask_window = mask_window.round_offsets().round_lengths()

                    logger.debug(f" Target mask window (approx): {mask_window}")

                    # 4. Intersect the target mask window with the actual mask bounds
                    try:
                        clipped_mask_window = intersection(mask_window, mask_bounds_window)
                        logger.debug(f" Clipped mask window: {clipped_mask_window}")
                    except WindowError:
                        # This occurs if the calculated mask_window is completely outside mask_bounds_window
                        logger.debug(f" Mask window {mask_window} is outside mask bounds. Setting clipped window to None.")
                        clipped_mask_window = None # Indicate no valid intersection

                    # Initialize the resampled mask block with zeros (nodata/non-forest)
                    # Important: Use the shape of the *probability* block as the target shape
                    mask_block_resampled = np.zeros(prob_block_shape, dtype=mask_dtype) 

                    # 5. If the clipped window is valid (has width and height > 0), read and resample
                    if clipped_mask_window and clipped_mask_window.width > 0 and clipped_mask_window.height > 0:
                        try:
                            # Read the data from the *valid intersection* part of the mask
                            mask_data_clipped = mask_src.read(
                                1, 
                                window=clipped_mask_window,
                                # Resample the *clipped* data to the size the *clipped window*
                                # would occupy within the full target probability block shape.
                                # This requires calculating the shape based on intersection.
                                out_shape=(
                                    int(round(clipped_mask_window.height * (prob_res_y / mask_res_y))),
                                    int(round(clipped_mask_window.width * (prob_res_x / mask_res_x)))
                                ),
                                resampling=Resampling.nearest 
                            )
                            
                            # Calculate where this clipped data should be placed in the full output block
                            # Offsets relative to the start of the original (unclipped) mask_window
                            offset_y = clipped_mask_window.row_off - mask_window.row_off
                            offset_x = clipped_mask_window.col_off - mask_window.col_off
                            
                            # Convert these pixel offsets in the mask grid to pixel offsets in the prob grid
                            # (taking resampling into account)
                            place_y = int(round(offset_y * (prob_res_y / mask_res_y)))
                            place_x = int(round(offset_x * (prob_res_x / mask_res_x)))
                            
                            # Ensure indices are within bounds of the target block
                            slice_y = slice(max(0, place_y), min(prob_block_shape[0], place_y + mask_data_clipped.shape[0]))
                            slice_x = slice(max(0, place_x), min(prob_block_shape[1], place_x + mask_data_clipped.shape[1]))
                            
                            # Ensure the shapes match for assignment
                            if (slice_y.stop - slice_y.start) == mask_data_clipped.shape[0] and \
                               (slice_x.stop - slice_x.start) == mask_data_clipped.shape[1]:
                                mask_block_resampled[slice_y, slice_x] = mask_data_clipped
                            else:
                                # This case might happen due to rounding errors or edge cases
                                # Try to place the valid part of the resampled data
                                read_h, read_w = mask_data_clipped.shape
                                place_h = slice_y.stop - slice_y.start
                                place_w = slice_x.stop - slice_x.start
                                logger.warning(f"Shape mismatch during placement for window {window}. Placing intersection. "
                                             f"Read shape: {(read_h, read_w)}, Place shape: {(place_h, place_w)}")
                                mask_block_resampled[slice_y, slice_x] = mask_data_clipped[:place_h, :place_w]

                        except Exception as read_err:
                            logger.error(f"Error reading/resampling mask window {clipped_mask_window} for prob window {window}: {read_err}")
                            # Decide how to handle: skip block? fill with nodata? For now, re-raise.
                            raise read_err
                    else:
                         logger.debug(f"Skipping mask read for prob window {window}: intersection is outside mask bounds.")

                    # 6. Classify the block using probability and resampled mask
                    output_block = classify_block(mask_block_resampled, prob_block, prob_nodata)

                    # 7. Write the resulting classification block to the destination window
                    dst.write(output_block.astype(rasterio.uint8), 1, window=window) # Write to the original prob window

        end_time = time.time()
        logger.info(f"Successfully created classification map: {output_path}")
        logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found: {e}")
        sys.exit(1)
    except ValueError as e: # Catch validation errors and others
         logger.error(f"Error: {e}")
         sys.exit(1)
    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio I/O error: {e}")
        sys.exit(1)
    except Exception as e:
        # Log the full traceback for unexpected errors
        logger.exception(f"An unexpected error occurred: {e}") 
        sys.exit(1)

if __name__ == "__main__":
    main() 