#!/usr/bin/env python3
"""
inspect_raster.py
-----------------
Opens a raster file (GeoTIFF) and prints its basic metadata 
(CRS, dtype, nodata value, dimensions) and the unique values 
found in a sample block of the data.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import rasterio

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inspect_raster")

def inspect_raster(raster_path: Path):
    """
    Opens a raster file, prints metadata, and unique values from the first block.

    Args:
        raster_path (Path): Path to the input raster file.
    """
    if not raster_path.is_file():
        logger.error(f"Error: Input file not found: {raster_path}")
        sys.exit(1)

    try:
        with rasterio.open(raster_path) as src:
            logger.info(f"Inspecting file: {raster_path.name}")

            # --- Print Metadata ---
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  Dimensions (Width x Height): {src.width} x {src.height}")
            logger.info(f"  Number of Bands: {src.count}")
            logger.info(f"  Data Type (dtype): {src.dtypes[0]}")
            logger.info(f"  Nodata Value: {src.nodata}")
            logger.info(f"  Bounding Box: {src.bounds}")
            logger.info(f"  Transform: {src.transform}")

            if src.count > 1:
                 logger.warning("Raster has multiple bands. Inspecting only the first band.")
            
            # --- Inspect Sample Data (First Block) ---
            # Get the window corresponding to the first block
            window = next(src.block_windows())[1] # Get the first window
            logger.info(f"Reading data from the first block: {window}")
            
            data_sample = src.read(1, window=window) # Read the first band within the window

            unique_values, counts = np.unique(data_sample, return_counts=True)
            
            logger.info("--- Unique Values in Sample Block ---")
            if unique_values.size > 50:
                 logger.warning(f"Found {unique_values.size} unique values. Displaying the first 50.")
                 unique_values = unique_values[:50]
                 counts = counts[:50] # Keep counts consistent

            for value, count in zip(unique_values, counts):
                logger.info(f"  Value: {value} - Count: {count}")
            logger.info("------------------------------------")

    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio I/O error opening or reading {raster_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Inspect a raster file's metadata and sample data.")
    parser.add_argument("raster_file", type=str,
                        help="Path to the input raster (GeoTIFF) file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    raster_path = Path(args.raster_file)
    inspect_raster(raster_path)

if __name__ == "__main__":
    main() 