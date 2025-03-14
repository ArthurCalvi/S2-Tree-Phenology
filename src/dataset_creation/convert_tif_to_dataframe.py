#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Training Tiles to Dataframe

This script reads training tiles in GeoTIFF format and converts them to a pandas DataFrame
for machine learning model training. It extracts data only for pixels where the phenology
band value is not zero.

The resulting DataFrame contains:
- Features from harmonic analysis (amplitudes, phases, offsets) for 4 indices (ndvi, evi, nbr, crswir)
- Reference data (phenology, genus, species, source, year)
- Tile ID extracted from the filename

The output is saved as a parquet file for efficient storage and fast access.

Usage:
    python convert_tif_to_dataframe.py --input_dir path/to/tiles --output_path path/to/output.parquet
    
    # With defaults (recommended):
    python convert_tif_to_dataframe.py
"""

import os
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


# Configure logging
def setup_logging(loglevel="INFO"):
    """
    Set up logging configuration.
    Console output: INFO level with progress bar (no DEBUG)
    File output: DEBUG level (all messages)
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/tif_to_dataframe_{timestamp}.log"
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler with INFO level (no DEBUG messages)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Create file handler with DEBUG level (all messages including DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set specific levels for external libraries to reduce noise
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('fiona').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    
    logging.info(f"Log file: {log_file}")
    return log_file


def extract_tile_id(filename):
    """Extract tile ID from filename."""
    match = re.search(r'tile_(\d+)_training', filename)
    if match:
        return int(match.group(1))
    return None


def process_tile(tile_path):
    """
    Process a single training tile and extract data where phenology != 0.
    
    Args:
        tile_path (str): Path to the training tile GeoTIFF file
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted data
    """
    tile_id = extract_tile_id(os.path.basename(tile_path))
    if tile_id is None:
        logging.warning(f"Could not extract tile ID from {tile_path}, skipping")
        return None
    
    logging.debug(f"Processing tile {tile_id} at {tile_path}")
    
    try:
        with rasterio.open(tile_path) as src:
            # Get band descriptions
            band_names = [src.descriptions[i-1] or f"Band_{i}" for i in range(1, src.count + 1)]
            logging.debug(f"Band names: {band_names}")
            
            # Find the index of the phenology band
            try:
                phenology_idx = band_names.index("phenology") + 1  # 1-indexed in rasterio
                logging.debug(f"Phenology band index: {phenology_idx}")
            except ValueError:
                logging.error(f"Phenology band not found in {tile_path}, skipping")
                return None
            
            # Read all bands
            data = src.read()
            
            # Get the phenology band
            phenology = data[phenology_idx-1]
            
            # Find indices where phenology != 0
            valid_indices = np.where(phenology != 0)
            
            if len(valid_indices[0]) == 0:
                logging.warning(f"No valid pixels found in tile {tile_id}")
                return None
            
            # Extract values for valid pixels
            tile_data = {
                'tile_id': tile_id,
                'row': valid_indices[0],
                'col': valid_indices[1],
            }
            
            # Add values for each band
            for i, band_name in enumerate(band_names):
                tile_data[band_name] = data[i][valid_indices]
            
            # Create DataFrame
            result_df = pd.DataFrame(tile_data)
            
            # Log message to both console and file
            message = f"Extracted {len(result_df)} valid pixels from tile {tile_id}"
            logging.info(message)
            
            return result_df
            
    except Exception as e:
        logging.error(f"Error processing tile {tile_path}: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert training tiles (GeoTIFF) to dataframe for machine learning"
    )
    parser.add_argument(
        "--input_dir",
        default="data/training/training_tiles2023",
        help="Directory containing training tile GeoTIFF files (default: data/training/training_tiles2023)"
    )
    parser.add_argument(
        "--output_path",
        default="results/datasets/training_dataset.parquet",
        help="Path to save the output parquet file (default: results/datasets/training_dataset.parquet)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to this number of tiles (for testing)"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Print summary of arguments
    print(f"Converting training tiles to DataFrame")
    print(f"Input directory: {args.input_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Log level: {args.loglevel}\n")
    
    if args.limit:
        print(f"Limited to {args.limit} tiles (for testing)")
    
    # Set up logging
    log_file = setup_logging(args.loglevel)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Find all training tile files
    input_dir = Path(args.input_dir)
    tile_files = list(input_dir.glob("*_training.tif"))
    logging.info(f"Found {len(tile_files)} training tile files")
    
    # Limit the number of tiles if requested
    if args.limit and args.limit < len(tile_files):
        tile_files = tile_files[:args.limit]
        logging.info(f"Limited to first {args.limit} tiles")
    
    # Process each tile with progress bar
    logging.info("Processing tiles:")
    all_dfs = []
    
    # Use tqdm for progress bar, but hide the iteration details for cleaner console output
    for tile_file in tqdm(tile_files, desc="Progress", ncols=100):
        df = process_tile(tile_file)
        if df is not None and not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        logging.error("No valid data extracted from any tiles")
        return
    
    # Combine all DataFrames
    logging.info("Combining results from all tiles...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Final DataFrame size: {len(final_df)} rows, {len(final_df.columns)} columns")
    
    # Save the DataFrame to parquet
    logging.info(f"Saving DataFrame to {args.output_path}...")
    final_df.to_parquet(args.output_path, index=False)
    
    # Print summary statistics
    logging.info("Summary statistics:")
    logging.info(f"  Total pixels: {len(final_df)}")
    logging.info(f"  Unique tiles: {final_df['tile_id'].nunique()}")
    
    if 'phenology' in final_df.columns:
        phenology_counts = final_df['phenology'].value_counts()
        logging.info(f"  Phenology distribution: {phenology_counts.to_dict()}")
    
    logging.info(f"Complete log saved to {log_file}")
    print(f"\nConversion complete. DataFrame saved to {args.output_path}")


if __name__ == "__main__":
    main() 