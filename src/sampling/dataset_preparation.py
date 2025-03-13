#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Preparation Script

Takes input from:
- Tiles (2.5km polygons)
- Final dataset with phenology, genus, species etc.
- Features TIFF with harmonic features

Creates a training dataset by:
1. Cropping features to tile extent
2. Adding categorical bands from dataset (phenology, genus, species, source, year)
3. Saving each tile as a separate GeoTIFF with proper band names

The features bands are, in order:
- ndvi (amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_residual)
- evi (amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_residual)
- nbr (amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_residual)
- crswir (amplitude_h1, amplitude_h2, phase_h1, phase_h2, offset, var_residual)

Additional bands added from the dataset are:
- phenology (1=deciduous, 2=evergreen)
- genus (mapped to integers)
- species (mapped to integers)
- source (mapped to integers)
- year (integer, no mapping needed)
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import tqdm
from shapely.geometry import box
from rasterio.windows import from_bounds

# Set up logging
def setup_logging(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {loglevel}", file=sys.stderr)
        sys.exit(1)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/dataset_preparation_{timestamp}.log"
    
    # Configure logging to both console and file, but with different levels
    # File handler gets all logs at the specified level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    
    # Console handler only gets INFO and above (no WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Custom filter to exclude WARNING level logs from console
    class NoWarningFilter(logging.Filter):
        def filter(self, record):
            return record.levelno != logging.WARNING
    console_handler.addFilter(NoWarningFilter())
    
    # Set formatter for both handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplication
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific levels for external libraries to reduce noise
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('fiona').setLevel(logging.WARNING)
    logging.getLogger('geopandas').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Set GDAL's specific log level to WARNING only
    os.environ['CPL_DEBUG'] = 'OFF'  # Turn off GDAL debug messages
    os.environ['GDAL_LOG_LEVEL'] = 'WARNING'  # Set GDAL log level
    
    logging.info(f"Logging to {log_file}")
    return log_file

def create_category_mappings(gdf):
    """Create mappings from categorical values to integers."""
    mappings = {}
    
    # Phenology mapping is fixed (1=deciduous, 2=evergreen)
    mappings["phenology"] = {"deciduous": 1, "evergreen": 2}
    
    # For genus, species, and source, create a mapping from unique values
    for column in ["genus", "species", "source"]:
        unique_values = sorted(gdf[column].dropna().unique())
        # Start mapping from 1 since 0 is nodata
        mappings[column] = {val: i + 1 for i, val in enumerate(unique_values)}
        logging.info(f"Created mapping for {column}: {len(unique_values)} unique values")
    
    return mappings

def get_tile_extent(geometry, src_crs, dst_crs):
    """Get the extent of a tile geometry in the target CRS."""
    # If CRS already match, no need to reproject
    if src_crs == dst_crs:
        minx, miny, maxx, maxy = geometry.bounds
        return box(minx, miny, maxx, maxy)
    
    # Create a temporary GeoDataFrame with the geometry
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=src_crs)
    # Reproject to target CRS
    gdf = gdf.to_crs(dst_crs)
    # Get bounds
    minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
    # Return box geometry in target CRS
    return box(minx, miny, maxx, maxy)

def clip_feature_raster(feature_path, tile_geometry, tile_id, output_dir, tile_crs):
    """Clip the feature raster to a tile geometry."""
    logging.info(f"Clipping features for tile {tile_id}")
    
    with rasterio.open(feature_path) as src:
        # Log the bounds of the raster
        raster_bounds = src.bounds
        logging.debug(f"Raster bounds: {raster_bounds} (CRS: {src.crs})")
        
        # Log the bounds of the tile geometry
        tile_bounds = tile_geometry.bounds
        logging.debug(f"Tile bounds: {tile_bounds} (CRS: {tile_crs})")
        
        # Reproject tile geometry to raster CRS if needed
        tile_geom_in_raster_crs = get_tile_extent(tile_geometry, tile_crs, src.crs)
        
        # Log the reprojected bounds
        logging.debug(f"Reprojected tile bounds: {tile_geom_in_raster_crs.bounds} (CRS: {src.crs})")
        
        # Get the bounding box of the geometry in the raster's CRS
        bbox = [tile_geom_in_raster_crs]
        
        # Clip the raster using the tile geometry
        try:
            # Calculate the window from the tile geometry
            logging.debug(f"Calculating window for tile {tile_id}")
            minx, miny, maxx, maxy = tile_geom_in_raster_crs.bounds
            
            # Snap bounds to the source grid to ensure perfect alignment
            src_transform = src.transform
            col_start, row_start = ~src_transform * (minx, miny)
            col_stop, row_stop = ~src_transform * (maxx, maxy)
            
            # Convert to integer pixel coordinates
            col_start = int(col_start)
            row_start = int(row_start)
            col_stop = int(col_stop + 0.5)  # Round up
            row_stop = int(row_stop + 0.5)  # Round up
            
            # Convert back to world coordinates that align exactly with pixels
            aligned_minx, aligned_miny = src_transform * (col_start, row_start)
            aligned_maxx, aligned_maxy = src_transform * (col_stop, row_stop)
            logging.debug(f"Original bounds: {(minx, miny, maxx, maxy)}")
            logging.debug(f"Aligned bounds: {(aligned_minx, aligned_miny, aligned_maxx, aligned_maxy)}")
            
            # Create window using the aligned coordinates
            window = from_bounds(
                aligned_minx, aligned_miny, aligned_maxx, aligned_maxy, src.transform
            )
            
            # Read data from the window first
            windowed_data = src.read(window=window)
            windowed_transform = src.window_transform(window)
            
            # Use windowed data directly without masking to preserve exact alignment
            out_image = windowed_data
            out_transform = windowed_transform
            
            out_meta = src.meta.copy()
            logging.debug("Updating metadata")
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "predictor": 2
            })
            
            # Define the output path
            output_path = os.path.join(output_dir, f"tile_{tile_id}_features.tif")
            
            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            logging.info(f"Saved clipped features to {output_path}")
            return output_path, out_meta, out_image, out_transform
        
        except Exception as e:
            logging.error(f"Error clipping features for tile {tile_id}: {e}")
            return None, None, None, None

def rasterize_attributes(gdf, attribute, mapping, transform, shape, crs):
    """Rasterize a specific attribute from the GeoDataFrame."""
    logging.debug(f"Rasterizing attribute: {attribute}, shape: {shape}")
    
    # Filter out rows with missing values
    valid_gdf = gdf[~gdf[attribute].isna()].copy()
    logging.debug(f"Valid rows with non-null {attribute}: {len(valid_gdf)}/{len(gdf)}")
    
    if len(valid_gdf) == 0:
        logging.debug(f"No valid values for {attribute}, returning zeros")
        return np.zeros(shape, dtype=np.uint32)
    
    # Apply mapping if provided
    if mapping:
        logging.debug(f"Applying mapping for {attribute}")
        valid_gdf['value'] = valid_gdf[attribute].map(mapping)
    else:
        logging.debug(f"No mapping for {attribute}, using values directly")
        valid_gdf['value'] = valid_gdf[attribute].astype(np.uint32)
    
    # Create list of (geometry, value) pairs for rasterize
    logging.debug(f"Creating geometry-value pairs for {attribute}")
    shapes = [(geom, value) for geom, value in zip(valid_gdf.geometry, valid_gdf['value'])]
    logging.debug(f"Created {len(shapes)} geometry-value pairs")
    
    # Log some info about the shapes
    if len(shapes) > 0:
        sample_geom = shapes[0][0]
        sample_val = shapes[0][1]
        logging.debug(f"Sample geometry type: {sample_geom.geom_type}, valid: {sample_geom.is_valid}, value: {sample_val}")
    
    # Rasterize the geometries with their values
    logging.debug(f"Beginning rasterization for {attribute}")
    try:
        rasterized = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,  # nodata value
            dtype=np.uint32,
            all_touched=False
        )
        logging.debug(f"Rasterization complete for {attribute}, shape: {rasterized.shape}")
        return rasterized
    except Exception as e:
        logging.error(f"Error during rasterization of {attribute}: {e}")
        logging.debug("Error details:", exc_info=True)
        # Return zeros as fallback
        return np.zeros(shape, dtype=np.uint32)

def create_training_tile(tile_id, tile_geometry, tile_crs, dataset_gdf, feature_path, mappings, output_dir):
    """Create a training dataset tile with features and rasterized attributes."""
    # Clip the feature raster to the tile geometry
    clipped_path, meta, image, transform = clip_feature_raster(feature_path, tile_geometry, tile_id, output_dir, tile_crs)
    
    if clipped_path is None:
        return None
    
    # Filter dataset to geometries that intersect the tile
    tile_box = tile_geometry
    intersecting_gdf = dataset_gdf[dataset_gdf.intersects(tile_box)].copy()
    
    logging.info(f"Tile {tile_id} has {len(intersecting_gdf)} intersecting polygons")
    
    if len(intersecting_gdf) == 0:
        logging.warning(f"No polygons intersect tile {tile_id}, skipping additional bands")
        return clipped_path
    
    # Make sure GDF has the same CRS as the raster
    with rasterio.open(clipped_path) as src:
        if intersecting_gdf.crs != src.crs:
            intersecting_gdf = intersecting_gdf.to_crs(src.crs)
    
    # Prepare output file
    output_path = os.path.join(output_dir, f"tile_{tile_id}_training.tif")
    
    # Update metadata for the output file
    out_meta = meta.copy()
    # Number of original bands in feature raster
    n_feature_bands = out_meta['count']
    # Add 5 new bands (phenology, genus, species, source, year)
    out_meta.update({
        'count': n_feature_bands + 5,
        'dtype': 'uint32'  # Use uint32 for all bands
    })
    
    # Define band names
    indices = ["ndvi", "evi", "nbr", "crswir"]
    measures = ["amplitude_h1", "amplitude_h2", "phase_h1", "phase_h2", "offset", "var_residual"]
    
    band_names = []
    for idx in indices:
        for measure in measures:
            band_names.append(f"{idx}_{measure}")
    
    # Add extra band names
    band_names.extend(["phenology", "genus", "species", "source", "year"])
    
    # Create output file with feature bands copied
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        # Copy original bands
        logging.debug("Copying original bands")
        for i in range(1, n_feature_bands + 1):
            logging.debug(f"Copying band {i}")
            # The shape issue: out_image has shape (bands, height, width)
            # So we need to select the correct band (i-1) without slicing
            dst.write(image[i-1], i)
            # Add band name as metadata
            dst.set_band_description(i, band_names[i-1])
        
        # Add categorical bands
        shape = (out_meta['height'], out_meta['width'])
        
        # Band 25: Phenology (1=deciduous, 2=evergreen)
        phenology_raster = rasterize_attributes(
            intersecting_gdf, 'phenology', mappings['phenology'], transform, shape, src.crs
        )
        dst.write(phenology_raster, n_feature_bands + 1)
        dst.set_band_description(n_feature_bands + 1, 'phenology')
        
        # Band 26: Genus
        genus_raster = rasterize_attributes(
            intersecting_gdf, 'genus', mappings['genus'], transform, shape, src.crs
        )
        dst.write(genus_raster, n_feature_bands + 2)
        dst.set_band_description(n_feature_bands + 2, 'genus')
        
        # Band 27: Species
        species_raster = rasterize_attributes(
            intersecting_gdf, 'species', mappings['species'], transform, shape, src.crs
        )
        dst.write(species_raster, n_feature_bands + 3)
        dst.set_band_description(n_feature_bands + 3, 'species')
        
        # Band 28: Source
        source_raster = rasterize_attributes(
            intersecting_gdf, 'source', mappings['source'], transform, shape, src.crs
        )
        dst.write(source_raster, n_feature_bands + 4)
        dst.set_band_description(n_feature_bands + 4, 'source')
        
        # Band 29: Year (no mapping needed)
        # Convert year to integer first (might be string in the dataset)
        intersecting_gdf['year'] = intersecting_gdf['year'].astype(str).str.extract('(\d+)').astype(float).fillna(0).astype(int)
        year_raster = rasterize_attributes(
            intersecting_gdf, 'year', None, transform, shape, src.crs
        )
        dst.write(year_raster, n_feature_bands + 5)
        dst.set_band_description(n_feature_bands + 5, 'year')
    
    # Clean up the intermediate file
    os.remove(clipped_path)
    
    logging.info(f"Created training tile at {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset by combining features and vector data."
    )
    parser.add_argument(
        "--tiles_path",
        required=True,
        default="results/datasets/tiles_2_5_km_final.parquet",
        help="Path to the parquet file with tiles"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        default="results/datasets/final_dataset.parquet",
        help="Path to the final dataset parquet file"
    )
    parser.add_argument(
        "--features_path",
        required=True,
        default="data/features",
        help="Path to the directory containing feature raster data"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        default="results/training_tiles",
        help="Directory to save the output training tiles"
    )
    parser.add_argument(
        "--mappings_path",
        default=None,
        help="Optional path to save the categorical mappings as JSON"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to this number of tiles (for testing)"
    )
    parser.add_argument(
        "--tile_ids",
        type=str,
        default=None,
        help="Comma-separated list of tile IDs to process (for testing specific tiles)"
    )
    parser.add_argument(
        "--force_crs",
        type=str,
        default=None,
        help="Force a specific CRS (e.g., 'EPSG:2154') for the tiles and dataset if CRS issues are detected"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.loglevel)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find the features file
    if os.path.isdir(args.features_path):
        # Look for a VRT file first
        vrt_files = list(Path(args.features_path).glob("*.vrt"))
        if vrt_files:
            features_file = str(vrt_files[0])
            logging.info(f"Using VRT file: {features_file}")
        else:
            # Look for a TIF file
            tif_files = list(Path(args.features_path).glob("*.tif"))
            if tif_files:
                features_file = str(tif_files[0])
                logging.info(f"Using TIF file: {features_file}")
            else:
                logging.error(f"No VRT or TIF files found in {args.features_path}")
                sys.exit(1)
    else:
        # Direct path to a file
        if os.path.exists(args.features_path):
            features_file = args.features_path
            logging.info(f"Using features file: {features_file}")
        else:
            logging.error(f"Features file not found: {args.features_path}")
            sys.exit(1)
    
    # Get the CRS of the feature raster
    with rasterio.open(features_file) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        logging.info(f"Features raster CRS: {raster_crs}")
        logging.debug(f"Features raster bounds: {raster_bounds}")
    
    # Load the tiles dataset
    logging.info(f"Loading tiles from {args.tiles_path}")
    tiles_gdf = gpd.read_parquet(args.tiles_path)
    logging.info(f"Loaded {len(tiles_gdf)} tiles")
    logging.debug(f"Tiles CRS: {tiles_gdf.crs}")
    
    # Load the final dataset
    logging.info(f"Loading dataset from {args.dataset_path}")
    dataset_gdf = gpd.read_parquet(args.dataset_path)
    logging.info(f"Loaded {len(dataset_gdf)} polygons")
    logging.debug(f"Dataset CRS: {dataset_gdf.crs}")
    
    # Create index on tile_id to speed up filtering
    if "tile_id" in dataset_gdf.columns:
        logging.info("Creating index on tile_id column to speed up filtering")
        dataset_gdf = dataset_gdf.set_index("tile_id", drop=False)
    
    # Handle CRS alignment
    if args.force_crs:
        logging.warning(f"Forcing CRS to {args.force_crs} as requested")
        target_crs = args.force_crs
        tiles_gdf.set_crs(target_crs, allow_override=True, inplace=True)
        dataset_gdf.set_crs(target_crs, allow_override=True, inplace=True)
    else:
        # Use the raster CRS as the target
        target_crs = raster_crs
        logging.info(f"Using raster CRS as target: {target_crs}")
    
    # Reproject tiles and dataset to match the raster CRS if needed
    if tiles_gdf.crs != target_crs:
        logging.info(f"Reprojecting tiles from {tiles_gdf.crs} to {target_crs}")
        tiles_gdf = tiles_gdf.to_crs(target_crs)
    
    if dataset_gdf.crs != target_crs:
        logging.info(f"Reprojecting dataset from {dataset_gdf.crs} to {target_crs}")
        dataset_gdf = dataset_gdf.to_crs(target_crs)
    
    # Make sure tile_id column exists in tiles GDF
    if "tile_id" not in tiles_gdf.columns:
        logging.debug("Adding tile_id column from index")
        tiles_gdf["tile_id"] = tiles_gdf.index
    
    # Create mappings for categorical values
    logging.info("Creating categorical mappings")
    mappings = create_category_mappings(dataset_gdf)
    
    # Save mappings if requested
    if args.mappings_path:
        with open(args.mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        logging.info(f"Saved mappings to {args.mappings_path}")
    else:
        # Default path in output directory
        mappings_path = os.path.join(args.output_dir, "categorical_mappings.json")
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        logging.info(f"Saved mappings to {mappings_path}")
    
    # Filter tiles if specific IDs were requested
    if args.tile_ids:
        tile_ids = [int(tile_id.strip()) for tile_id in args.tile_ids.split(',')]
        tiles_gdf = tiles_gdf[tiles_gdf.index.isin(tile_ids)]
        logging.info(f"Filtered to {len(tiles_gdf)} requested tiles")
    
    # Limit the number of tiles if requested
    if args.limit and args.limit < len(tiles_gdf):
        tiles_gdf = tiles_gdf.head(args.limit)
        logging.info(f"Limited to first {args.limit} tiles")
    
    # Process each tile
    processed_tiles = 0
    skipped_tiles = 0
    logging.info(f"Starting to process {len(tiles_gdf)} tiles...")
    for idx, tile_row in tqdm.tqdm(tiles_gdf.iterrows(), total=len(tiles_gdf), desc="Processing tiles"):
        tile_id = tile_row.tile_id if "tile_id" in tiles_gdf.columns else idx
        tile_geometry = tile_row.geometry
        
        # Log the tile bounds for debugging
        logging.debug(f"Processing tile {tile_id} with bounds: {tile_geometry.bounds}")
        
        try:
            # Filter dataset to only include polygons with this tile_id
            if "tile_id" in dataset_gdf.index.names:
                # Use .loc for faster filtering with index
                tile_dataset = dataset_gdf.loc[[tile_id]].copy() if tile_id in dataset_gdf.index else dataset_gdf[dataset_gdf["tile_id"] == tile_id].copy()
            else:
                tile_dataset = dataset_gdf[dataset_gdf["tile_id"] == tile_id].copy()
            logging.debug(f"Tile {tile_id} has {len(tile_dataset)} associated polygons")
            
            if len(tile_dataset) == 0:
                logging.warning(f"No polygons found for tile {tile_id}, skipping")
                skipped_tiles += 1
                continue
            
            # Clip the feature raster to the tile geometry
            logging.debug(f"Clipping features for tile {tile_id}")
            with rasterio.open(features_file) as src:
                # Reproject tile geometry to raster CRS if needed
                tile_geom_in_raster_crs = get_tile_extent(tile_geometry, tiles_gdf.crs, src.crs)
                
                # Log the reprojected bounds
                logging.debug(f"Original tile bounds: {tile_geometry.bounds} (CRS: {tiles_gdf.crs})")
                logging.debug(f"Reprojected tile bounds: {tile_geom_in_raster_crs.bounds} (CRS: {src.crs})")
                
                # Check if the tile geometry is valid and not empty
                if not tile_geom_in_raster_crs.is_valid or tile_geom_in_raster_crs.is_empty:
                    logging.error(f"Invalid or empty tile geometry for tile {tile_id} after reprojection")
                    skipped_tiles += 1
                    continue
                
                # Check if the tile intersects with any of the actual TIF files
                # Find all TIF files in the features directory
                features_dir = Path(args.features_path).parent if not os.path.isdir(args.features_path) else Path(args.features_path)
                tif_files = list(features_dir.glob("*.tif"))
                
                intersects_any_tif = False
                for tif_file in tif_files:
                    with rasterio.open(tif_file) as tif_src:
                        tif_bbox = box(tif_src.bounds.left, tif_src.bounds.bottom, tif_src.bounds.right, tif_src.bounds.top)
                        if tif_bbox.intersects(tile_geom_in_raster_crs):
                            intersects_any_tif = True
                            logging.debug(f"Tile {tile_id} intersects with {tif_file.name}")
                            break
                
                if not intersects_any_tif:
                    logging.warning(f"Tile {tile_id} does not intersect with any TIF files. Skipping.")
                    skipped_tiles += 1
                    continue
                
                # Get the bounding box of the geometry in the raster's CRS
                bbox = [tile_geom_in_raster_crs]
                logging.debug(f"Using tile geometry for masking: {tile_geom_in_raster_crs.wkt[:100]}...")
                
                try:
                    # Calculate the window from the tile geometry
                    logging.debug(f"Calculating window for tile {tile_id}")
                    minx, miny, maxx, maxy = tile_geom_in_raster_crs.bounds
                    
                    # Snap bounds to the source grid to ensure perfect alignment
                    src_transform = src.transform
                    col_start, row_start = ~src_transform * (minx, miny)
                    col_stop, row_stop = ~src_transform * (maxx, maxy)
                    
                    # Convert to integer pixel coordinates
                    col_start = int(col_start)
                    row_start = int(row_start)
                    col_stop = int(col_stop + 0.5)  # Round up
                    row_stop = int(row_stop + 0.5)  # Round up
                    
                    # Convert back to world coordinates that align exactly with pixels
                    aligned_minx, aligned_miny = src_transform * (col_start, row_start)
                    aligned_maxx, aligned_maxy = src_transform * (col_stop, row_stop)
                    logging.debug(f"Original bounds: {(minx, miny, maxx, maxy)}")
                    logging.debug(f"Aligned bounds: {(aligned_minx, aligned_miny, aligned_maxx, aligned_maxy)}")
                    
                    # Create window using the aligned coordinates
                    window = from_bounds(
                        aligned_minx, aligned_miny, aligned_maxx, aligned_maxy, src.transform
                    )
                    
                    # Read data from the window first
                    windowed_data = src.read(window=window)
                    windowed_transform = src.window_transform(window)
                    
                    # Use windowed data directly without masking to preserve exact alignment
                    out_image = windowed_data
                    out_transform = windowed_transform
                    
                    out_meta = src.meta.copy()
                    logging.debug("Updating metadata")
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "lzw",
                        "predictor": 2
                    })
                    
                    # Prepare output file
                    output_path = os.path.join(args.output_dir, f"tile_{tile_id}_training.tif")
                    logging.debug(f"Output path: {output_path}")
                    
                    # Update metadata for the output file
                    n_feature_bands = out_meta['count']
                    logging.debug(f"Number of feature bands: {n_feature_bands}")
                    # Add 5 new bands (phenology, genus, species, source, year)
                    out_meta.update({
                        'count': n_feature_bands + 5,
                        'dtype': 'uint32'  # Use uint32 for all bands
                    })
                    logging.debug(f"Updated band count to {out_meta['count']}")
                    
                    # Define band names
                    indices = ["ndvi", "evi", "nbr", "crswir"]
                    measures = ["amplitude_h1", "amplitude_h2", "phase_h1", "phase_h2", "offset", "var_residual"]
                    
                    band_names = []
                    for idx_name in indices:
                        for measure in measures:
                            band_names.append(f"{idx_name}_{measure}")
                    
                    # Add extra band names
                    band_names.extend(["phenology", "genus", "species", "source", "year"])
                    logging.debug(f"Defined {len(band_names)} band names")
                    
                    # Create output file with feature bands copied
                    logging.debug(f"Opening output file for writing: {output_path}")
                    with rasterio.open(output_path, 'w', **out_meta) as dst:
                        # Copy original bands
                        logging.debug("Copying original bands")
                        for i in range(1, n_feature_bands + 1):
                            logging.debug(f"Copying band {i}")
                            # The shape issue: out_image has shape (bands, height, width)
                            # So we need to select the correct band (i-1) without slicing
                            dst.write(out_image[i-1], i)
                            # Add band name as metadata
                            dst.set_band_description(i, band_names[i-1])
                        
                        # Add categorical bands
                        shape = (out_meta['height'], out_meta['width'])
                        logging.debug(f"Preparing to rasterize categorical bands, shape: {shape}")
                        
                        # Band 25: Phenology (1=deciduous, 2=evergreen)
                        logging.debug("Rasterizing phenology band")
                        phenology_raster = rasterize_attributes(
                            tile_dataset, 'phenology', mappings['phenology'], out_transform, shape, target_crs
                        )
                        logging.debug("Writing phenology band")
                        dst.write(phenology_raster, n_feature_bands + 1)
                        dst.set_band_description(n_feature_bands + 1, 'phenology')
                        
                        # Band 26: Genus
                        logging.debug("Rasterizing genus band")
                        genus_raster = rasterize_attributes(
                            tile_dataset, 'genus', mappings['genus'], out_transform, shape, target_crs
                        )
                        logging.debug("Writing genus band")
                        dst.write(genus_raster, n_feature_bands + 2)
                        dst.set_band_description(n_feature_bands + 2, 'genus')
                        
                        # Band 27: Species
                        logging.debug("Rasterizing species band")
                        species_raster = rasterize_attributes(
                            tile_dataset, 'species', mappings['species'], out_transform, shape, target_crs
                        )
                        logging.debug("Writing species band")
                        dst.write(species_raster, n_feature_bands + 3)
                        dst.set_band_description(n_feature_bands + 3, 'species')
                        
                        # Band 28: Source
                        logging.debug("Rasterizing source band")
                        source_raster = rasterize_attributes(
                            tile_dataset, 'source', mappings['source'], out_transform, shape, target_crs
                        )
                        logging.debug("Writing source band")
                        dst.write(source_raster, n_feature_bands + 4)
                        dst.set_band_description(n_feature_bands + 4, 'source')
                        
                        # Band 29: Year (no mapping needed)
                        logging.debug("Preparing year data")
                        # Convert year to integer first (might be string in the dataset)
                        tile_dataset['year'] = tile_dataset['year'].astype(str).str.extract('(\d+)').astype(float).fillna(0).astype(int)
                        logging.debug("Rasterizing year band")
                        year_raster = rasterize_attributes(
                            tile_dataset, 'year', None, out_transform, shape, target_crs
                        )
                        logging.debug("Writing year band")
                        dst.write(year_raster, n_feature_bands + 5)
                        dst.set_band_description(n_feature_bands + 5, 'year')
                    
                    logging.debug(f"Created training tile at {output_path}")
                    processed_tiles += 1
                    if processed_tiles % 10 == 0:  # Log every 10 tiles
                        logging.info(f"Processed {processed_tiles} tiles so far, skipped {skipped_tiles}")
                
                except Exception as e:
                    logging.error(f"Error processing tile {tile_id}: {e}")
                    logging.debug("Error details:", exc_info=True)
            
        except Exception as e:
            logging.error(f"Error processing tile {tile_id}: {e}", exc_info=True)
    
    logging.info(f"Completed! Processed {processed_tiles} out of {len(tiles_gdf)} tiles, skipped {skipped_tiles}")
    logging.info(f"Complete log saved to {log_file}")

if __name__ == "__main__":
    main() 