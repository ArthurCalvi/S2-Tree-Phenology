#!/usr/bin/env python3
"""
compare_maps.py
-----------------
Compares a custom forest classification map with a reference map (e.g., Copernicus DLT)
stratified by eco-regions.

Inputs:
- Custom Map: GeoTIFF classified into (0: Non-Forest, 1: Deciduous, 2: Evergreen).
- Reference Map (DLT): GeoTIFF classified into (e.g., 0: Non-Forest/Other, 1: Broadleaved, 2: Coniferous).
- Eco-regions Map: GeoTIFF where pixel values represent distinct regions (e.g., 1-11).
                  THIS MAP DEFINES THE REFERENCE GRID.

Process:
1. Opens all three input maps.
2. Checks if CRS and transform/resolution match the Eco-regions map.
3. If not, uses rasterio.vrt.WarpedVRT to align Custom and Reference maps
   to the Eco-regions grid on-the-fly using Nearest Neighbor resampling.
4. Processes the maps block by block based on the Eco-regions grid.
5. Accumulates pixel counts for a 3x3 confusion matrix for each valid eco-region ID.
6. Calculates Overall Accuracy and Cohen's Kappa for each region's matrix.
7. Logs results and saves metrics to a CSV file.

Output:
- Log messages detailing the comparison process and results per region.
- A CSV file containing the metrics (OA, Kappa) for each eco-region.
"""

import argparse
import logging
import time
import sys
import csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm

# --- Configuration & Logging ---
# Set default logging level to WARNING
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("compare_maps")

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_OUTPUT_FILENAME = "comparison_metrics.csv"
# Define expected class values (adjust if necessary based on actual map values)
CUSTOM_CLASSES = {0: 'Non-Forest', 1: 'Deciduous', 2: 'Evergreen'}
# IMPORTANT: Verify the meaning of DLT values, especially 0
DLT_CLASSES = {0: 'Non-Forest/Other', 1: 'Broadleaved', 2: 'Coniferous'}
N_CLASSES = 3 # Should match the number of classes in CUSTOM_CLASSES and DLT_CLASSES

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
    """Calculates OA, Kappa, Precision, Recall, F1 from a confusion matrix."""
    n_classes = cm.shape[0]
    metrics = {}
    
    total = np.sum(cm)
    if total == 0:
        # Return default values if no valid pixels
        metrics['oa'] = 0.0
        metrics['kappa'] = 0.0
        for i in range(n_classes):
            metrics[f'precision_{i}'] = 0.0
            metrics[f'recall_{i}'] = 0.0
            metrics[f'f1_{i}'] = 0.0
        return metrics

    # Overall Accuracy (OA)
    oa = np.trace(cm) / total
    metrics['oa'] = oa

    # Cohen's Kappa
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    expected_prop = np.sum((sum_rows * sum_cols)) / (total * total)

    if (1 - expected_prop) == 0:
        kappa = 1.0 if oa == 1.0 else 0.0
    else:
        kappa = (oa - expected_prop) / (1 - expected_prop)
    metrics['kappa'] = kappa

    # Precision, Recall, F1 per class
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        #tn = total - tp - fp - fn # Not typically needed for these metrics
        
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
    parser = argparse.ArgumentParser(
        description=(
            "Compare a custom classification map with a reference map (e.g., DLT), "
            "stratified by eco-regions. Eco-regions map defines the reference grid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--custom-map", type=str, required=True,
                        help="Path to the custom classification GeoTIFF (0:NF, 1:Decid, 2:Ever).")
    parser.add_argument("--ref-map", type=str, required=True,
                        help="Path to the reference (DLT) GeoTIFF (e.g., 0:NF/Other, 1:Broad, 2:Conif).")
    parser.add_argument("--eco-map", type=str, required=True,
                        help="Path to the eco-regions GeoTIFF (int values >= 1). Defines the reference grid.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output metrics CSV file.")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help="Name for the output metrics CSV file.")
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
    output_path = output_dir / args.output_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Open input files ---
        with rasterio.open(eco_map_path) as eco_src, \
             rasterio.open(custom_map_path) as custom_src_orig, \
             rasterio.open(ref_map_path) as ref_src_orig:

            logger.info(f"Opened Eco-regions map (Reference Grid): {eco_map_path.name}")
            logger.info(f"Opened Custom map: {custom_map_path.name}")
            logger.info(f"Opened Reference (DLT) map: {ref_map_path.name}")

            # --- Get Metadata and Validate ---
            eco_profile = eco_src.profile
            eco_nodata = eco_src.nodata
            # Use dtypes tuple for single-band rasters
            eco_dtype = eco_src.dtypes[0] 
            logger.info(f"Eco-regions: CRS={eco_profile['crs']}, Size=({eco_profile['width']}x{eco_profile['height']}), Transform={eco_profile['transform']}, Nodata={eco_nodata}, Dtype={eco_dtype}")

            custom_profile_orig = custom_src_orig.profile
            custom_nodata_orig = custom_src_orig.nodata
            logger.info(f"Custom Map (Original): CRS={custom_profile_orig['crs']}, Size=({custom_profile_orig['width']}x{custom_profile_orig['height']}), Transform={custom_profile_orig['transform']}, Nodata={custom_nodata_orig}")

            ref_profile_orig = ref_src_orig.profile
            ref_nodata_orig = ref_src_orig.nodata
            logger.info(f"Reference Map (Original): CRS={ref_profile_orig['crs']}, Size=({ref_profile_orig['width']}x{ref_profile_orig['height']}), Transform={ref_profile_orig['transform']}, Nodata={ref_nodata_orig}")
            logger.warning(f"Assuming Reference Map classes: {DLT_CLASSES}. Verify value '0'.")


            if eco_src.count != 1 or custom_src_orig.count != 1 or ref_src_orig.count != 1:
                raise ValueError("All input files must be single-band rasters.")

            # --- Check for Alignment and Setup WarpedVRT if needed ---
            custom_needs_warp = needs_warping(custom_profile_orig, eco_profile)
            ref_needs_warp = needs_warping(ref_profile_orig, eco_profile)

            custom_src = custom_src_orig # Default to original source
            if custom_needs_warp:
                logger.info("Custom map grid differs from Eco-regions map. Using WarpedVRT for alignment.")
                custom_src = WarpedVRT(custom_src_orig,
                                       crs=eco_profile['crs'],
                                       transform=eco_profile['transform'],
                                       width=eco_profile['width'],
                                       height=eco_profile['height'],
                                       resampling=Resampling.nearest,
                                       nodata=custom_nodata_orig) # Pass original nodata
                logger.debug(f"Custom map WarpedVRT nodata: {custom_src.nodata}")


            ref_src = ref_src_orig # Default to original source
            if ref_needs_warp:
                logger.info("Reference map grid differs from Eco-regions map. Using WarpedVRT for alignment.")
                ref_src = WarpedVRT(ref_src_orig,
                                    crs=eco_profile['crs'],
                                    transform=eco_profile['transform'],
                                    width=eco_profile['width'],
                                    height=eco_profile['height'],
                                    resampling=Resampling.nearest,
                                    nodata=ref_nodata_orig) # Pass original nodata
                logger.debug(f"Reference map WarpedVRT nodata: {ref_src.nodata}")

            # --- Initialize results storage ---
            # Dictionary to hold confusion matrix for each eco-region ID
            region_cms = {}

            # --- Log nodata handling strategy ---
            logger.info("--- Nodata Handling ---")
            logger.info(f"Eco-regions nodata: {eco_nodata}. Pixels with this value will be EXCLUDED.")
            custom_nodata_val = custom_src.nodata if hasattr(custom_src, 'nodata') else custom_nodata_orig
            ref_nodata_val = ref_src.nodata if hasattr(ref_src, 'nodata') else ref_nodata_orig
            logger.info(f"Custom map nodata: {custom_nodata_val}. Pixels with this value will be treated as Class 0 (Non-Forest).")
            logger.info(f"Reference map nodata: {ref_nodata_val}. Pixels with this value will be treated as Class 0 (Non-Forest).")
            logger.info("-----------------------")


            # --- Process block by block using ECO-REGIONS map windows ---
            # Manually generate windows based on block_size argument
            block_size = args.block_size
            eco_height = eco_profile['height']
            eco_width = eco_profile['width']
            logger.info(f"Starting comparison using eco-regions grid with block size {block_size}x{block_size}.")
            
            windows = []
            for j in range(0, eco_height, block_size):
                height = min(block_size, eco_height - j)
                for i in range(0, eco_width, block_size):
                    width = min(block_size, eco_width - i)
                    windows.append(Window(i, j, width, height))

            #windows = [window for ij, window in eco_src.block_windows()] # Replaced by manual generation
            total_blocks = len(windows)
            logger.info(f"Total blocks to process: {total_blocks}")

            for window in tqdm(windows, desc="Comparing blocks", total=total_blocks, unit="block"):
                logger.debug(f"Processing window: {window}")

                # Read blocks - WarpedVRT handles reprojection if needed
                eco_block = eco_src.read(1, window=window)
                custom_block = custom_src.read(1, window=window)
                ref_block = ref_src.read(1, window=window)

                # --- Pixel-wise comparison within the block ---
                # 1. Identify pixels belonging to a valid eco-region (ID >= 1 and not eco_nodata)
                eco_valid_mask = (eco_block >= 1)
                if eco_nodata is not None:
                    eco_valid_mask &= (eco_block != eco_nodata)

                # 2. Handle nodata in custom and reference maps: treat as class 0 (Non-Forest)
                custom_block_proc = custom_block.copy()
                if custom_nodata_val is not None:
                    custom_block_proc[custom_block == custom_nodata_val] = 0

                ref_block_proc = ref_block.copy()
                if ref_nodata_val is not None:
                    ref_block_proc[ref_block == ref_nodata_val] = 0

                # 3. Ensure class values are within expected range AFTER nodata conversion
                class_valid_mask = (custom_block_proc >= 0) & (custom_block_proc < N_CLASSES) & \
                                   (ref_block_proc >= 0) & (ref_block_proc < N_CLASSES)

                # 4. Combine masks: must be valid eco-region AND have valid class values
                final_mask = eco_valid_mask & class_valid_mask

                # Extract valid data using the final mask
                valid_eco = eco_block[final_mask]
                valid_custom = custom_block_proc[final_mask] # Use processed blocks
                valid_ref = ref_block_proc[final_mask]     # Use processed blocks

                # --- Accumulate counts for confusion matrices ---
                unique_regions = np.unique(valid_eco)
                for region_id in unique_regions:
                    if region_id not in region_cms:
                         # Initialize 3x3 matrix (row: custom, col: ref)
                        region_cms[region_id] = np.zeros((N_CLASSES, N_CLASSES), dtype=np.uint64)

                    # Find pixels belonging to the current region_id
                    region_mask = (valid_eco == region_id)
                    custom_vals_region = valid_custom[region_mask]
                    ref_vals_region = valid_ref[region_mask]

                    # Update the confusion matrix for this region
                    # np.add.at is efficient for this
                    np.add.at(region_cms[region_id], (custom_vals_region, ref_vals_region), 1)

            # Close VRTs if they were created
            if custom_needs_warp:
                custom_src.close()
            if ref_needs_warp:
                ref_src.close()

            # --- Calculate and Log Metrics ---
            logger.info("Calculating and reporting metrics per eco-region...")
            results_data = []
            sorted_region_ids = sorted(region_cms.keys())

            if not sorted_region_ids:
                 logger.warning("No valid overlapping pixels found across all inputs.")
                 # Close VRTs before exiting if they were created
                 if custom_needs_warp: custom_src.close()
                 if ref_needs_warp: ref_src.close()
                 sys.exit(0) # Exit gracefully if no data

            for region_id in sorted_region_ids:
                cm = region_cms[region_id]
                calculated_metrics = calculate_metrics_from_cm(cm)
                total_pixels = np.sum(cm)

                logger.info(f"--- Eco-Region ID: {region_id} ---")
                logger.info(f" Total valid pixels: {total_pixels:,}") # Added comma formatting
                logger.info(f" Overall Accuracy (OA): {calculated_metrics['oa']:.4f}")
                logger.info(f" Cohen's Kappa: {calculated_metrics['kappa']:.4f}")
                # Log the confusion matrix
                cm_log_header = "  CM (Rows: Custom, Cols: Ref) | " + " | ".join([DLT_CLASSES.get(i, f'Col_{i}') for i in range(N_CLASSES)])
                logger.info(cm_log_header)
                logger.info("  " + "-" * (len(cm_log_header)-2))
                for i in range(N_CLASSES):
                   row_str = f"  {CUSTOM_CLASSES.get(i, f'Row_{i}'):<25} | " + " | ".join([f"{int(cm[i, j]):,}" for j in range(N_CLASSES)])
                   logger.info(row_str)
                
                # Log class metrics
                for i in range(N_CLASSES):
                     class_name = CUSTOM_CLASSES.get(i, f'Class_{i}')
                     logger.info(f"    {class_name:>18}: Precision={calculated_metrics[f'precision_{i}']:.4f}, Recall={calculated_metrics[f'recall_{i}']:.4f}, F1={calculated_metrics[f'f1_{i}']:.4f}")
                logger.info("-" * 40)

                # Prepare data for CSV and final summary
                region_id_int = int(region_id) # Ensure integer for lookup
                region_name = ECO_REGION_CLASSES.get(region_id_int, f"Unknown_{region_id_int}")
                region_result = {
                    'eco_region_id': region_id_int,
                    'eco_region_name': region_name,
                    'total_pixels': total_pixels,
                    'overall_accuracy': calculated_metrics['oa'],
                    'kappa': calculated_metrics['kappa'],
                }
                # Add class metrics
                for i in range(N_CLASSES):
                    region_result[f'precision_{i}'] = calculated_metrics[f'precision_{i}']
                    region_result[f'recall_{i}'] = calculated_metrics[f'recall_{i}']
                    region_result[f'f1_{i}'] = calculated_metrics[f'f1_{i}']
                # Add CM counts
                for r in range(N_CLASSES):
                    for c in range(N_CLASSES):
                        region_result[f'cm_{r}{c}'] = cm[r, c]
                
                results_data.append(region_result)

            # --- Save results to CSV ---
            if results_data:
                 # Ensure fieldnames capture all added metrics and CM counts
                 fieldnames = list(results_data[0].keys()) # Dynamically get keys from first result
                 with open(output_path, 'w', newline='') as csvfile:
                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                     writer.writeheader()
                     writer.writerows(results_data)
                 logger.info(f"Comparison metrics saved to: {output_path}")

                 # --- Print Final Summary Table to Console ---
                 print("\n--- Final Comparison Summary ---")
                 # Determine column widths dynamically or set reasonable defaults
                 headers = ["Region ID", "Region Name", "Pixels", "OA", "Kappa"] + \
                           [f"P_{i}" for i in range(N_CLASSES)] + \
                           [f"R_{i}" for i in range(N_CLASSES)] + \
                           [f"F1_{i}" for i in range(N_CLASSES)]
                 # Example fixed widths - adjust as needed
                 col_widths = [9, 30, 12, 6, 6] + [6]*N_CLASSES*3
                 header_str = " | ".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
                 print(header_str)
                 print("-" * len(header_str))

                 for result in results_data:
                      row_vals = [result['eco_region_id'], result['eco_region_name'], 
                                  f"{result['total_pixels']:,}", 
                                  f"{result['overall_accuracy']:.3f}", f"{result['kappa']:.3f}"] + \
                                 [f"{result[f'precision_{i}']:.3f}" for i in range(N_CLASSES)] + \
                                 [f"{result[f'recall_{i}']:.3f}" for i in range(N_CLASSES)] + \
                                 [f"{result[f'f1_{i}']:.3f}" for i in range(N_CLASSES)]
                      row_str = " | ".join([f"{str(v):<{w}}" for v, w in zip(row_vals, col_widths)])
                      print(row_str)
                 print("-" * len(header_str))
                 print(f"Class Legend: {CUSTOM_CLASSES}")
                 print("P=Precision, R=Recall, F1=F1-Score")
                 print("---------------------------------")

            else:
                 logger.warning("No results generated, CSV file not saved and no summary printed.")


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
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main() 