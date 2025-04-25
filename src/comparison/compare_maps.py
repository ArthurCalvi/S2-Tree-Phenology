#!/usr/bin/env python3
"""
compare_maps.py
-----------------
Compares a custom forest classification map with a reference map (DLT or BDForet)
stratified by eco-regions, focusing only on forest classes.

Inputs:
- Custom Map: GeoTIFF classified into (0: Non-Forest, 1: Deciduous, 2: Evergreen).
- Reference Map (DLT or BDForet): GeoTIFF with forest classes (see class definitions).
- Eco-regions Map: GeoTIFF where pixel values represent distinct regions (e.g., 1-11).
                  THIS MAP DEFINES THE REFERENCE GRID.

Process:
1. Opens all three input maps.
2. Determines reference map type (DLT or BDForet) based on argument.
3. Checks if CRS and transform/resolution match the Eco-regions map.
4. If not, uses rasterio.vrt.WarpedVRT to align Custom and Reference maps
   to the Eco-regions grid on-the-fly using Nearest Neighbor resampling,
   passing the correct nodata value for the reference map.
5. Processes the maps block by block based on the Eco-regions grid.
6. For each block:
   a. Identifies pixels classified as FOREST (class 1 or 2) in BOTH maps
      and belonging to a valid eco-region.
   b. Calculates a 2x2 confusion matrix (Forest Type 1 vs Forest Type 2)
      for the valid pixels within the block (count_both_forest).
   c. Calculates Overall Accuracy, Cohen's Kappa, Precision, Recall, and F1
      for the block's 2x2 matrix.
   d. Determines the geometry (bounding box) of the block.
   e. Calculates counts and percentages for agreement/disagreement on FOREST presence:
      - perc_custom_only_forest: Pct. Forest in Custom map only.
      - perc_ref_only_forest: Pct. Forest in Reference map only.
      - perc_both_forest: Pct. Forest in both maps.
      - perc_both_nonforest: Pct. Non-forest in both maps (within comparable area).
   f. Stores the metrics, percentages, counts (optional), and geometry object.
7. Saves the collected per-block metrics, percentages, counts, and geometries into a Parquet file
   using GeoPandas (preserving geometry type and CRS).

Output:
- Log messages detailing the comparison process.
- A Parquet file saved to the specified output directory with the filename
  formatted as '<output_filename>_vs_<ref_type>.parquet',
  containing metrics (OA, Kappa, P, R, F1), agreement percentages,
  agreement counts, and the native geometry object for each processed block.
"""

import argparse
import logging
import time
import sys
# import csv # Removed
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds # Added bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm
import warnings # Added to suppress GeoPandas warning

# Import GeoPandas AFTER checking for its existence (or handle ImportError)
try:
    import geopandas as gpd
    from shapely.geometry import box
    import pyarrow # Check if pyarrow is available for parquet
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("GeoPandas and/or PyArrow not found. Per-block Parquet output is disabled. Install GeoPandas and PyArrow to enable.")


# --- Configuration & Logging ---
# Set default logging level to WARNING
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("compare_maps")

DEFAULT_BLOCK_SIZE = 1024
# --- Class Definitions ---
# Custom Map: Assumed fixed
CUSTOM_CLASSES = {1: 'Deciduous', 2: 'Evergreen'} # Forest classes only for CM
CUSTOM_NODATA_PLACEHOLDER = -9999 # Placeholder if original nodata is None

# Reference Maps
DLT_CLASSES_ALL = {0: 'Non-Forest/Other', 1: 'Broadleaved', 2: 'Coniferous'}
DLT_CLASSES_FOREST = {1: 'Broadleaved', 2: 'Coniferous'}
DLT_NODATA = 255

BDFORET_CLASSES_ALL = {1: 'Deciduous', 2: 'Evergreen', 0: 'Nodata'}
BDFORET_CLASSES_FOREST = {1: 'Deciduous', 2: 'Evergreen'}
BDFORET_NODATA = 0

N_COMPARE_CLASSES = 2 # Comparing 2 forest types

ECO_REGION_CLASSES = {
    1: 'Alps',
    2: 'Central Massif',
    3: 'Corsica',
    4: 'Greater Crystalline and Oceanic West',
    5: 'Greater Semi-Continental East',
    6: 'Jura',
    7: 'Mediterranean',
    8: 'Oceanic Southwest',
    9: 'Pyrenees',
    10: 'Semi-Oceanic North Center',
    11: 'Vosges'
}

def calculate_metrics_from_cm(cm):
    """Calculates OA, Kappa, Precision, Recall, F1 from a confusion matrix.
    
    Returns NaNs if total pixels is 0.
    """
    n_classes = cm.shape[0]
    metrics = {}
    
    total = np.sum(cm)
    if total == 0:
        # Return NaN values if no valid pixels
        metrics['oa'] = np.nan
        metrics['kappa'] = np.nan
        for i in range(n_classes):
            metrics[f'precision_{i}'] = np.nan
            metrics[f'recall_{i}'] = np.nan
            metrics[f'f1_{i}'] = np.nan
        return metrics

    # Overall Accuracy (OA)
    oa = np.trace(cm) / total
    metrics['oa'] = oa

    # Cohen's Kappa
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    expected_prop = np.sum((sum_rows * sum_cols)) / (total * total)

    # Handle division by zero or 1-expected_prop == 0
    if (1 - expected_prop) == 0:
        # Kappa is 1 if perfect agreement (oa=1), 0 if agreement equals chance
        kappa = 1.0 if np.isclose(oa, 1.0) else 0.0
    else:
        kappa = (oa - expected_prop) / (1 - expected_prop)
    metrics['kappa'] = kappa

    # Precision, Recall, F1 per class (Indices 0 and 1 for the 2x2 matrix)
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        # Precision (Positive Predictive Value)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'precision_{i}'] = precision
        
        # Recall (Sensitivity, True Positive Rate)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'recall_{i}'] = recall
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f'f1_{i}'] = f1
        
    return metrics

def needs_warping(src_profile, ref_profile):
    """Check if CRS or transform differ significantly."""
    if src_profile['crs'] != ref_profile['crs']:
        return True
    # Check transform (affine components a, b, d, e)
    src_t = src_profile['transform']
    ref_t = ref_profile['transform']
    if not np.allclose(
        [src_t.a, src_t.b, src_t.d, src_t.e],
        [ref_t.a, ref_t.b, ref_t.d, ref_t.e]
    ):
        return True
    return False

def main():
    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas/PyArrow is required for this script but not installed. Exiting.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(
        description=(
            "Compare a custom classification map with a reference map (DLT or BDForet), "
            "calculating forest comparison metrics per block and saving to Parquet (with geometry). "
            "Eco-regions map defines the reference grid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--custom-map", type=str, required=True,
                        help="Path to the custom classification GeoTIFF (0:NF, 1:Decid, 2:Ever).")
    parser.add_argument("--ref-map", type=str, required=True,
                        help="Path to the reference GeoTIFF (DLT or BDForet).")
    parser.add_argument("--ref-type", type=str, required=True, choices=['DLT', 'BDForet'],
                        help="Type of the reference map ('DLT' or 'BDForet').")
    parser.add_argument("--eco-map", type=str, required=True,
                        help="Path to the eco-regions GeoTIFF (int values >= 1). Defines the reference grid.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output Parquet file.")
    parser.add_argument("--output-filename", type=str, required=True,
                        help="Base filename for the output Parquet file (e.g., 'comparison_metrics', _vs_<ref_type>.parquet will be added).")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="Processing block size (pixels) based on the eco-regions map grid.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    start_time = time.time()

    custom_map_path = Path(args.custom_map)
    ref_map_path = Path(args.ref_map)
    eco_map_path = Path(args.eco_map)
    output_dir = Path(args.output_dir)
    output_filename = f"{args.output_filename}_vs_{args.ref_type}.parquet"
    output_parquet_path = output_dir / output_filename

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output Parquet will be saved to: {output_parquet_path}")

    # --- Determine Reference Map Configuration ---
    if args.ref_type == 'DLT':
        ref_classes_forest = DLT_CLASSES_FOREST
        ref_classes_all = DLT_CLASSES_ALL
        ref_nodata_val_orig = DLT_NODATA
        ref_forest_values = (1, 2)
        logger.info(f"Reference map type: DLT. Forest classes: {ref_classes_forest}. Nodata: {ref_nodata_val_orig}")
    elif args.ref_type == 'BDForet':
        ref_classes_forest = BDFORET_CLASSES_FOREST
        ref_classes_all = BDFORET_CLASSES_ALL
        ref_nodata_val_orig = BDFORET_NODATA
        ref_forest_values = (1, 2)
        logger.info(f"Reference map type: BDForet. Forest classes: {ref_classes_forest}. Nodata: {ref_nodata_val_orig}")
    else:
        # Should be caught by argparse choices, but good practice
        logger.error(f"Invalid reference map type: {args.ref_type}")
        sys.exit(1)

    # Define labels for the 2x2 comparison matrix (still useful for metric keys)
    CM_CLASSES_ROW = {0: CUSTOM_CLASSES[1], 1: CUSTOM_CLASSES[2]} # Rows: Custom map forest classes
    CM_CLASSES_COL = {0: ref_classes_forest[1], 1: ref_classes_forest[2]} # Cols: Ref map forest classes

    try:
        # --- Open input files ---
        with rasterio.open(eco_map_path) as eco_src, \
             rasterio.open(custom_map_path) as custom_src_orig, \
             rasterio.open(ref_map_path) as ref_src_orig:

            logger.info(f"Opened Eco-regions map (Reference Grid): {eco_map_path.name}")
            logger.info(f"Opened Custom map: {custom_map_path.name}")
            logger.info(f"Opened Reference ({args.ref_type}) map: {ref_map_path.name}")

            # --- Get Metadata and Validate ---
            eco_profile = eco_src.profile
            eco_crs = eco_profile['crs'] # Store CRS for GeoDataFrame
            eco_transform = eco_profile['transform'] # Store transform for bounds calculation
            eco_nodata = eco_src.nodata
            logger.info(f"Eco-regions: CRS={eco_crs}, Size=({eco_profile['width']}x{eco_profile['height']}), Transform={eco_transform}, Nodata={eco_nodata}, Dtype={eco_src.dtypes[0]}")
            # logger.warning(f"Using Eco-regions map CRS ({eco_crs}) for block geometries. CRS info is NOT stored in the output Parquet file.") # Removed warning

            custom_profile_orig = custom_src_orig.profile
            custom_nodata_orig = custom_src_orig.nodata if custom_src_orig.nodata is not None else CUSTOM_NODATA_PLACEHOLDER
            logger.info(f"Custom Map (Original): CRS={custom_profile_orig['crs']}, Size=({custom_profile_orig['width']}x{custom_profile_orig['height']}), Transform={custom_profile_orig['transform']}, Nodata={custom_nodata_orig if custom_nodata_orig != CUSTOM_NODATA_PLACEHOLDER else 'None'}")

            ref_profile_orig = ref_src_orig.profile
            ref_nodata_orig = ref_nodata_val_orig
            logger.info(f"Reference Map ({args.ref_type} Original): CRS={ref_profile_orig['crs']}, Size=({ref_profile_orig['width']}x{ref_profile_orig['height']}), Transform={ref_profile_orig['transform']}, Nodata={ref_nodata_orig}")
            logger.info(f"Reference Map classes (All): {ref_classes_all}")

            if eco_src.count != 1 or custom_src_orig.count != 1 or ref_src_orig.count != 1:
                raise ValueError("All input files must be single-band rasters.")

            # --- Check for Alignment and Setup WarpedVRT if needed ---
            custom_needs_warp = needs_warping(custom_profile_orig, eco_profile)
            ref_needs_warp = needs_warping(ref_profile_orig, eco_profile)

            custom_src = custom_src_orig
            if custom_needs_warp:
                logger.info("Custom map grid differs from Eco-regions map. Using WarpedVRT for alignment.")
                custom_src = WarpedVRT(custom_src_orig,
                                       crs=eco_crs, # Use eco_crs
                                       transform=eco_transform, # Use eco_transform
                                       width=eco_profile['width'],
                                       height=eco_profile['height'],
                                       resampling=Resampling.nearest,
                                       nodata=custom_nodata_orig if custom_nodata_orig != CUSTOM_NODATA_PLACEHOLDER else None)
                logger.debug(f"Custom map WarpedVRT nodata: {custom_src.nodata}")

            ref_src = ref_src_orig
            if ref_needs_warp:
                logger.info(f"Reference ({args.ref_type}) map grid differs from Eco-regions map. Using WarpedVRT for alignment.")
                ref_src = WarpedVRT(ref_src_orig,
                                    crs=eco_crs, # Use eco_crs
                                    transform=eco_transform, # Use eco_transform
                                    width=eco_profile['width'],
                                    height=eco_profile['height'],
                                    resampling=Resampling.nearest,
                                    nodata=ref_nodata_orig)
                logger.debug(f"Reference ({args.ref_type}) map WarpedVRT nodata: {ref_src.nodata}")

            # --- Initialize results storage ---
            block_results = []

            # --- Log nodata handling strategy ---
            logger.info("--- Comparison Strategy & Nodata Handling ---")
            logger.info(f"Comparison focuses on FOREST pixels only (Custom map classes {list(CUSTOM_CLASSES.keys())} vs Ref map classes {ref_forest_values}).")
            logger.info(f"Eco-regions nodata: {eco_nodata}. Pixels with this value in Eco-regions map will be EXCLUDED.")
            custom_nodata_val = custom_src.nodata if hasattr(custom_src, 'nodata') else custom_nodata_orig
            if custom_nodata_val == CUSTOM_NODATA_PLACEHOLDER:
                custom_nodata_val = None
            ref_nodata_val = ref_src.nodata if hasattr(ref_src, 'nodata') else ref_nodata_orig
            logger.info(f"Custom map nodata: {custom_nodata_val}. Pixels with value 0 or nodata will be EXCLUDED from forest comparison.")
            logger.info(f"Reference ({args.ref_type}) map nodata: {ref_nodata_val}. Pixels with nodata or non-forest values will be EXCLUDED from forest comparison.")
            logger.info("----------------------------------------------")

            # --- Process block by block using ECO-REGIONS map windows ---
            block_size = args.block_size
            eco_height = eco_profile['height']
            eco_width = eco_profile['width']
            logger.info(f"Starting per-block comparison using eco-regions grid with block size {block_size}x{block_size}.")
            
            windows = []
            for j in range(0, eco_height, block_size):
                height = min(block_size, eco_height - j)
                for i in range(0, eco_width, block_size):
                    width = min(block_size, eco_width - i)
                    windows.append(Window(i, j, width, height))

            total_blocks = len(windows)
            logger.info(f"Total blocks to process: {total_blocks}")

            for window in tqdm(windows, desc="Comparing blocks", total=total_blocks, unit="block"):
                logger.debug(f"Processing window: {window}")

                eco_block = eco_src.read(1, window=window)
                custom_block = custom_src.read(1, window=window)
                ref_block = ref_src.read(1, window=window)

                shapes = [arr.shape for arr in (eco_block, custom_block, ref_block)]
                min_height = min(s[0] for s in shapes)
                min_width = min(s[1] for s in shapes)

                if min_height <= 0 or min_width <= 0:
                    logger.debug(f"Skipping window {window} due to zero dimension after intersection.")
                    continue

                eco_block_crop = eco_block[:min_height, :min_width]
                custom_block_crop = custom_block[:min_height, :min_width]
                ref_block_crop = ref_block[:min_height, :min_width]

                # --- Create masks for valid comparison area --- 
                eco_valid_mask = (eco_block_crop >= 1)
                if eco_nodata is not None:
                    eco_valid_mask &= (eco_block_crop != eco_nodata)

                custom_valid_data_mask = (custom_block_crop != custom_nodata_val) if custom_nodata_val is not None else np.ones_like(custom_block_crop, dtype=bool)
                ref_valid_data_mask = (ref_block_crop != ref_nodata_val) if ref_nodata_val is not None else np.ones_like(ref_block_crop, dtype=bool)
                
                # Mask for pixels valid in all three inputs (eco-region and data)
                valid_comparison_mask = eco_valid_mask & custom_valid_data_mask & ref_valid_data_mask
                total_comparable_pixels = np.sum(valid_comparison_mask)
                
                # --- Identify forest/non-forest within the valid comparison mask --- 
                custom_is_forest = ((custom_block_crop == 1) | (custom_block_crop == 2)) & valid_comparison_mask
                ref_is_forest = ((ref_block_crop == 1) | (ref_block_crop == 2)) & valid_comparison_mask
                
                # --- Calculate Agreement Counts ---                
                count_custom_only_forest = np.sum(custom_is_forest & ~ref_is_forest)
                count_ref_only_forest = np.sum(ref_is_forest & ~custom_is_forest)
                count_both_forest = np.sum(custom_is_forest & ref_is_forest)
                count_both_nonforest = np.sum(~custom_is_forest & ~ref_is_forest)
                
                # --- Calculate Agreement Percentages --- 
                if total_comparable_pixels > 0:
                    perc_custom_only_forest = (count_custom_only_forest / total_comparable_pixels) * 100.0
                    perc_ref_only_forest = (count_ref_only_forest / total_comparable_pixels) * 100.0
                    perc_both_forest = (count_both_forest / total_comparable_pixels) * 100.0
                    perc_both_nonforest = (count_both_nonforest / total_comparable_pixels) * 100.0
                else:
                    # Assign NaN or 0 if no comparable pixels
                    perc_custom_only_forest = np.nan
                    perc_ref_only_forest = np.nan
                    perc_both_forest = np.nan
                    perc_both_nonforest = np.nan
                
                # --- Calculate 2x2 CM and Metrics for FOREST vs FOREST pixels --- 
                # Use pixels where *both* are forest (count_both_forest)
                final_mask_2x2 = custom_is_forest & ref_is_forest # Same as count_both_forest mask
                valid_custom_2x2 = custom_block_crop[final_mask_2x2] # Forest classes (1, 2)
                valid_ref_2x2 = ref_block_crop[final_mask_2x2]       # Forest classes (1, 2)

                block_cm_2x2 = np.zeros((N_COMPARE_CLASSES, N_COMPARE_CLASSES), dtype=np.uint64)
                if count_both_forest > 0:
                    cm_custom_idx = valid_custom_2x2 - 1
                    cm_ref_idx = valid_ref_2x2 - 1
                    np.add.at(block_cm_2x2, (cm_custom_idx, cm_ref_idx), 1)

                # Calculate metrics based on the 2x2 forest-only comparison
                block_metrics = calculate_metrics_from_cm(block_cm_2x2)

                # --- Get Block Geometry --- 
                block_bounds = window_bounds(window, eco_transform)
                block_geometry = box(*block_bounds)

                # --- Store results for this block --- 
                block_data = {
                    'row_off': window.row_off,
                    'col_off': window.col_off,
                    'height': window.height,
                    'width': window.width,
                    'total_comparable_pixels': total_comparable_pixels,
                    # Agreement Percentages
                    'perc_custom_only_forest': perc_custom_only_forest,
                    'perc_ref_only_forest': perc_ref_only_forest,
                    'perc_both_forest': perc_both_forest,
                    'perc_both_nonforest': perc_both_nonforest,
                    # Agreement Counts (Optional - kept for reference)
                    'count_custom_only_forest': count_custom_only_forest,
                    'count_ref_only_forest': count_ref_only_forest,
                    'count_both_forest': count_both_forest,
                    'count_both_nonforest': count_both_nonforest,
                    # 2x2 Forest Comparison Metrics
                    'oa_2x2': block_metrics['oa'],
                    'kappa_2x2': block_metrics['kappa'],
                    f'precision_{CM_CLASSES_ROW[0]}': block_metrics['precision_0'],
                    f'recall_{CM_CLASSES_ROW[0]}': block_metrics['recall_0'],
                    f'f1_{CM_CLASSES_ROW[0]}': block_metrics['f1_0'],
                    f'precision_{CM_CLASSES_ROW[1]}': block_metrics['precision_1'],
                    f'recall_{CM_CLASSES_ROW[1]}': block_metrics['recall_1'],
                    f'f1_{CM_CLASSES_ROW[1]}': block_metrics['f1_1'],
                    # 2x2 Forest Confusion Matrix Counts
                    f'cm_{CM_CLASSES_ROW[0]}_vs_{CM_CLASSES_COL[0]}': block_cm_2x2[0, 0],
                    f'cm_{CM_CLASSES_ROW[0]}_vs_{CM_CLASSES_COL[1]}': block_cm_2x2[0, 1],
                    f'cm_{CM_CLASSES_ROW[1]}_vs_{CM_CLASSES_COL[0]}': block_cm_2x2[1, 0],
                    f'cm_{CM_CLASSES_ROW[1]}_vs_{CM_CLASSES_COL[1]}': block_cm_2x2[1, 1],
                    # Geometry
                    'geometry': block_geometry
                }
                block_results.append(block_data)

            # Close VRTs if they were created
            if custom_needs_warp:
                custom_src.close()
            if ref_needs_warp:
                ref_src.close()

            # --- Save results to Parquet --- 
            if block_results:
                 logger.info(f"Converting {len(block_results)} block results to GeoDataFrame for Parquet export...")
                 try:
                     gdf = gpd.GeoDataFrame(block_results, geometry='geometry', crs=eco_crs)
                     logger.info(f"Attempting to save results to: {output_parquet_path}")
                     gdf.to_parquet(output_parquet_path, index=False)
                     logger.info(f"Per-block comparison metrics, percentages, and counts saved successfully to: {output_parquet_path}")
                 except Exception as e:
                     logger.error(f"Failed to create or write Parquet file to {output_parquet_path}: {e}")
            else:
                 logger.warning("No valid blocks processed, Parquet file not saved.")

        end_time = time.time()
        logger.info(f"Comparison finished in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or Value Error: {e}")
        sys.exit(1)
    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio I/O error: {e}")
        sys.exit(1)
    except ImportError:
        logger.error("GeoPandas/PyArrow import failed unexpectedly. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 