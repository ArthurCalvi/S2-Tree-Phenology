#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds a reference GeoDataFrame from:
 - tiles (2.5km polygons, with 'effective_pixels')
 - in-situ polygons (priority data)
 - BD Forêt polygons (supplementary data)
Output includes for each polygon:
 tile_id, geometry, source, effective_pixels, phenology, genus, species, year

Note: 'effective_pixels' represents the count of 10m × 10m pixels within each polygon.

Rules:
 - Always include in-situ polygons that intersect each tile
 - BD Forêt polygons are used only outside a 100m buffer of in-situ polygons
 - If BD Forêt polygon overlaps an in-situ polygon of a different phenology => skip
"""

import argparse
import logging
import sys
import os
import datetime
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
import contextily as ctx  # Add contextily import for basemaps

# For intersection/difference we might want to allow multi-geometry:
import warnings
warnings.filterwarnings("ignore", message="`keep_geom_type=True`")

def main():
    parser = argparse.ArgumentParser(
        description="Build reference dataset from tiles, in-situ, and BD Foret."
    )
    parser.add_argument(
        "--tiles",
        required=True,
        default="results/datasets/tiles_2_5_km_final.parquet",
        help="Path to the parquet file with tiles (e.g. results/datasets/tiles_2_5_km_final.parquet)."
    )
    parser.add_argument(
        "--species",
        required=True,
        default="data/species/processed/france_species_with_ecoregions.parquet",
        help="Path to the in-situ dataset (e.g. data/species/processed/france_species_with_ecoregions.parquet)."
    )
    parser.add_argument(
        "--bdforet",
        required=True,
        default="data/species/processed/bdforet_with_ecoregions.parquet",
        help="Path to BD Foret file (e.g. data/species/processed/bdforet_with_ecoregions.parquet)."
    )
    parser.add_argument(
        "--output",
        required=True,
        default="results/datasets/final_dataset.parquet",
        help="Path to output file (e.g. results/datasets/final_dataset.parquet or .geojson)."
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=50.0,
        help="Buffer in meters for excluding BD Forêt polygons near in-situ polygons."
    )
    parser.add_argument(
        "--negative_buffer",
        type=float,
        default=50.0,
        help="Negative buffer in meters for BD Forêt polygons to prevent bad annotations on forest edges."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)."
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated list of NomSER regions to process (default: all regions)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from last saved state."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all regions even if they exist in output."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="results/datasets/intermediate",
        help="Directory for intermediate results (used for incremental processing)."
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to output PDF report with dataset metrics (e.g. results/reports/dataset_report.pdf)."
    )
    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {args.loglevel}", file=sys.stderr)
        sys.exit(1)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/dataset_creation_{timestamp}.log"
    
    # Configure logging to both console and file
    # File handler gets all logs at the specified level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    
    # Console handler also gets all logs at the specified level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
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
    logging.getLogger('shapely').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Set GDAL's specific log level to WARNING only
    os.environ['CPL_DEBUG'] = 'OFF'  # Turn off GDAL debug messages
    os.environ['GDAL_LOG_LEVEL'] = 'WARNING'  # Set GDAL log level
    
    logging.info(f"Logging to {log_file}")

    # Create work directory if needed
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Track processing state
    state_file = work_dir / "processing_state.json"
    processed_regions = set()
    
    # Load previous state if resuming
    if args.resume and state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                processed_regions = set(state.get('processed_regions', []))
            logging.info(f"Resuming with {len(processed_regions)} previously processed regions")
        except Exception as e:
            logging.warning(f"Failed to load state file: {e}. Starting fresh.")
    
    # 1. Load data
    logging.info("Loading tiles, species, and BD Forêt datasets...")
    tiles_gdf = gpd.read_parquet(args.tiles)
    species_gdf = gpd.read_parquet(args.species)
    bdforet_gdf = gpd.read_parquet(args.bdforet)

    logging.debug(f"Tiles columns: {tiles_gdf.columns}")
    logging.debug(f"In-situ species columns: {species_gdf.columns}")
    logging.debug(f"BD Forêt columns: {bdforet_gdf.columns}")
    
    logging.debug(f"Tiles CRS: {tiles_gdf.crs}")
    logging.debug(f"In-situ species CRS: {species_gdf.crs}")
    logging.debug(f"BD Forêt CRS: {bdforet_gdf.crs}")
    
    # Fix the CRS of the tiles dataset if coordinates look like Lambert-93
    if tiles_gdf.crs == "EPSG:4326" or str(tiles_gdf.crs).lower().find("wgs") >= 0:
        # Check if the coordinates look like they're already in Lambert-93
        if len(tiles_gdf) > 0:
            sample_x = tiles_gdf.iloc[0].geometry.centroid.x
            sample_y = tiles_gdf.iloc[0].geometry.centroid.y
            
            # Lambert-93 coordinates for France typically have these ranges
            if 600000 <= sample_x <= 1200000 and 6000000 <= sample_y <= 7200000:
                logging.warning("Detected Lambert-93 coordinates despite WGS 84 CRS declaration")
                logging.info("Setting CRS directly to EPSG:2154 without reprojection")
                tiles_gdf.set_crs("EPSG:2154", allow_override=True, inplace=True)

    # Ensure same CRS
    target_crs = "EPSG:2154"  # Lambert-93 projection for France (metric)
    
    # Only reproject if we need to
    if tiles_gdf.crs != target_crs:
        logging.info(f"Reprojecting tiles to {target_crs}...")
        tiles_gdf = tiles_gdf.to_crs(target_crs)
    else:
        logging.info(f"Tiles already in target CRS: {target_crs}")
    
    if species_gdf.crs != target_crs:
        logging.info(f"Reprojecting in-situ species to {target_crs}...")
        species_gdf = species_gdf.to_crs(target_crs)

    if bdforet_gdf.crs != target_crs:
        logging.info(f"Reprojecting BD Forêt to {target_crs}...")
        bdforet_gdf = bdforet_gdf.to_crs(target_crs)

    # If the tiles GDF lacks a "tile_id" column, create one from the index
    if "tile_id" not in tiles_gdf.columns:
        logging.debug("No 'tile_id' column found in tiles. Using index as tile_id.")
        tiles_gdf["tile_id"] = tiles_gdf.index

    # Determine regions to process
    available_regions = sorted(tiles_gdf["NomSER"].unique())
    logging.info(f"Available regions: {available_regions}")
    
    if args.regions:
        regions_to_process = [r.strip() for r in args.regions.split(',')]
        # Validate requested regions
        invalid_regions = [r for r in regions_to_process if r not in available_regions]
        if invalid_regions:
            logging.error(f"Invalid regions requested: {invalid_regions}. Available regions: {available_regions}")
            sys.exit(1)
    else:
        regions_to_process = available_regions

    # Skip already processed regions if resuming and not forcing
    if args.resume and not args.force:
        regions_to_process = [r for r in regions_to_process if r not in processed_regions]
        logging.info(f"After skipping processed regions, {len(regions_to_process)} regions remain to process")

    # Prepare final result GDF
    final_columns = [
        "tile_id", "geometry", "source", "effective_pixels", 
        "phenology", "genus", "species", "year", "NomSER"
    ]
    final_crs = target_crs
    final_gdf = gpd.GeoDataFrame(columns=final_columns, crs=final_crs)
    
    # If output exists and we're resuming, load it
    if args.resume and Path(args.output).exists():
        try:
            output_ext = os.path.splitext(args.output)[1].lower()
            if output_ext in [".parquet"]:
                existing_final = gpd.read_parquet(args.output)
            elif output_ext in [".geojson", ".json"]:
                existing_final = gpd.read_file(args.output)
            else:
                existing_final = gpd.read_file(args.output)
            
            # Keep data for regions we're not reprocessing
            if "NomSER" in existing_final.columns:
                keep_regions = [r for r in existing_final["NomSER"].unique() if r not in regions_to_process]
                if keep_regions:
                    keep_data = existing_final[existing_final["NomSER"].isin(keep_regions)]
                    final_gdf = pd.concat([final_gdf, keep_data], ignore_index=True)
                    logging.info(f"Loaded {len(keep_data)} existing polygons from {len(keep_regions)} regions")
        except Exception as e:
            logging.warning(f"Failed to load existing output: {e}. Starting with empty result.")

    # Process each region
    for region in regions_to_process:
        logging.info(f"Processing region: {region}")
        
        # Filter tiles for this region
        region_tiles = tiles_gdf[tiles_gdf["NomSER"] == region]
        logging.info(f"Region {region} has {len(region_tiles)} tiles")
        
        # Filter species and BD Forêt data for this region if possible
        if "NomSER" in species_gdf.columns and "NomSER" in bdforet_gdf.columns:
            region_species = species_gdf[species_gdf["NomSER"] == region]
            region_bdforet = bdforet_gdf[bdforet_gdf["NomSER"] == region]
            logging.info(f"Filtered to {len(region_species)} in-situ polygons and {len(region_bdforet)} BD Forêt polygons for region {region}")
            
            # Log more detailed info about BD Forêt polygons for this region
            if len(region_bdforet) > 0 and "phen_en" in region_bdforet.columns:
                bdforet_phens = region_bdforet["phen_en"].value_counts()
                logging.debug(f"BD Forêt phenology distribution for region {region}:")
                for phen, count in bdforet_phens.items():
                    logging.debug(f"  - {phen}: {count} polygons")
                
                if "genus_en" in region_bdforet.columns:
                    bdforet_genera = region_bdforet["genus_en"].value_counts().nlargest(5)
                    logging.debug(f"Top 5 BD Forêt genera for region {region}:")
                    for genus, count in bdforet_genera.items():
                        logging.debug(f"  - {genus}: {count} polygons")
        else:
            # Use all data
            region_species = species_gdf
            region_bdforet = bdforet_gdf
        
        # Process tiles in this region
        region_results = gpd.GeoDataFrame(columns=final_columns, crs=final_crs)
        
        for tile_row in tqdm(region_tiles.itertuples(), total=len(region_tiles), desc=f"Processing tiles in {region}"):
            tile_id = tile_row.tile_id
            tile_geom = tile_row.geometry
            eff_pixels = getattr(tile_row, "effective_pixels", 0.0)

            # Clip in-situ polygons to tile
            clipped_in_situ = gpd.clip(region_species, tile_geom, keep_geom_type=False)
            logging.debug(f"Tile {tile_id}: {len(clipped_in_situ)} in-situ polygons intersect.")

            # Build rows from in-situ polygons
            if len(clipped_in_situ) > 0:
                clipped_in_situ["tile_id"] = tile_id
                clipped_in_situ["source"] = "in-situ"
                # Calculate effective_pixels based on area (1 pixel = 10m × 10m = 100 sq meters)
                clipped_in_situ["effective_pixels"] = clipped_in_situ.geometry.area / 100.0
                clipped_in_situ["NomSER"] = region

                # rename columns to final
                clipped_in_situ["phenology"] = clipped_in_situ["phen_en"]
                clipped_in_situ["genus"]     = clipped_in_situ["genus_en"]
                clipped_in_situ["species"]   = clipped_in_situ["specie_en"]
                # year stays year, or rename if needed

                # select columns
                final_in_situ = clipped_in_situ[
                    ["tile_id", "geometry", "source", "effective_pixels",
                     "phenology", "genus", "species", "year", "NomSER"]
                ].copy()

                # append to region results
                region_results = pd.concat([region_results, final_in_situ], ignore_index=True)

            # Now handle BD Forêt:
            # 1) buffer the in-situ polygons by 100m to exclude them
            if len(clipped_in_situ) > 0:
                # unary_union for the geometry
                in_situ_union = clipped_in_situ.geometry.unary_union
                # buffer outward by user-chosen distance
                in_situ_buffered = in_situ_union.buffer(args.buffer)
            else:
                # if no in-situ polygons, no buffer to exclude
                in_situ_buffered = None

            # First clip BD Forêt polygons to tile
            clipped_bdforet = gpd.clip(region_bdforet, tile_geom, keep_geom_type=False)
            logging.debug(f"Tile {tile_id}: {len(clipped_bdforet)} BD Forêt polygons intersect.")

            # Collect the polygons that pass all checks
            bdforet_accepted = []
            # Counters for debugging
            phenology_mismatch_count = 0
            buffer_exclusion_count = 0
            empty_geometry_count = 0
            negative_buffer_empty_count = 0

            # If no in-situ polygons in this tile, accept all BD Forêt polygons
            if len(clipped_in_situ) == 0:
                bdforet_accepted = clipped_bdforet.index.tolist()
                logging.debug(f"Tile {tile_id}: No in-situ polygons, accepting all {len(clipped_bdforet)} BD Forêt polygons")
            else:
                # Create spatial index for in-situ data for faster intersection checks
                if len(clipped_in_situ) > 0:  # Need to check again as it might be empty
                    in_situ_sindex = clipped_in_situ.sindex
                    logging.debug(f"Tile {tile_id}: Processing {len(clipped_bdforet)} BD Forêt polygons with {len(clipped_in_situ)} in-situ polygons")
                    
                    # Process each BD Forêt polygon
                    for bd_row in clipped_bdforet.itertuples():
                        bd_geom = bd_row.geometry
                        if bd_geom is None or bd_geom.is_empty:
                            empty_geometry_count += 1
                            continue
                        bd_phen = getattr(bd_row, "phen_en", None)
                        bd_genus = getattr(bd_row, "genus_en", "unknown")
                        bd_species = getattr(bd_row, "specie_en", "unknown")
                        
                        # Quickly find potential intersections using spatial index
                        possible_matches_idx = list(in_situ_sindex.intersection(bd_geom.bounds))
                        if not possible_matches_idx:
                            # No intersection with any in-situ polygon - accept & continue
                            bdforet_accepted.append(bd_row.Index)
                            continue
                        
                        # Get the actual in-situ polygons that intersect
                        possible_matches = clipped_in_situ.iloc[possible_matches_idx]
                        precise_matches = possible_matches[possible_matches.intersects(bd_geom)]
                        
                        if len(precise_matches) == 0:
                            # No actual intersection with any in-situ polygon - accept
                            bdforet_accepted.append(bd_row.Index)
                            continue
                        
                        # Check if any phenology mismatch exists
                        mismatch_found = False
                        for ins_row in precise_matches.itertuples():
                            ins_phen = getattr(ins_row, "phen_en", None)
                            ins_genus = getattr(ins_row, "genus_en", "unknown")
                            ins_species = getattr(ins_row, "specie_en", "unknown")
                            
                            if ins_phen != bd_phen:
                                mismatch_found = True
                                logging.debug(f"Tile {tile_id}: Phenology mismatch - BD Forêt: {bd_genus} {bd_species} ({bd_phen}) vs In-situ: {ins_genus} {ins_species} ({ins_phen})")
                                break
                        
                        if mismatch_found:
                            # Skip this polygon entirely due to phenology mismatch
                            phenology_mismatch_count += 1
                            continue
                        
                        # Apply negative buffer to BD Forêt polygon
                        negatively_buffered_bd_geom = bd_geom.buffer(-args.negative_buffer)
                        if negatively_buffered_bd_geom.is_empty:
                            # Skip if negative buffer makes polygon disappear
                            negative_buffer_empty_count += 1
                            logging.debug(f"Tile {tile_id}: BD Forêt polygon ({bd_genus} {bd_species}) became empty after negative buffer - excluded")
                            continue
                        
                        # Check buffer constraint - is this polygon entirely outside the buffer?
                        if in_situ_buffered and not in_situ_buffered.is_empty:
                            if negatively_buffered_bd_geom.intersects(in_situ_buffered):
                                # Need to perform the difference operation
                                diff_geom = negatively_buffered_bd_geom.difference(in_situ_buffered)
                                if diff_geom.is_empty:
                                    # Polygon completely within buffer - skip
                                    buffer_exclusion_count += 1
                                    logging.debug(f"Tile {tile_id}: BD Forêt polygon ({bd_genus} {bd_species}) completely within buffer - excluded")
                                    continue
                                else:
                                    # Update the geometry in the original DataFrame
                                    clipped_bdforet.loc[bd_row.Index, 'geometry'] = diff_geom
                                    logging.debug(f"Tile {tile_id}: BD Forêt polygon ({bd_genus} {bd_species}) partially within buffer - trimmed and accepted")
                            else:
                                # Not intersecting with buffer, but still need to update with negatively buffered geometry
                                clipped_bdforet.loc[bd_row.Index, 'geometry'] = negatively_buffered_bd_geom
                        else:
                            # No in_situ_buffered geometry, just update with negatively buffered geometry
                            clipped_bdforet.loc[bd_row.Index, 'geometry'] = negatively_buffered_bd_geom
                        
                        # All checks passed, accept this polygon
                        bdforet_accepted.append(bd_row.Index)
                else:
                    # If clipped_in_situ became empty, accept all BD Forêt polygons
                    bdforet_accepted = clipped_bdforet.index.tolist()
                    logging.debug(f"Tile {tile_id}: Clipped in-situ became empty, accepting all {len(clipped_bdforet)} BD Forêt polygons")
            
            # Log summary of filtering for this tile
            logging.debug(f"Tile {tile_id}: BD Forêt filtering summary - " 
                         f"Total: {len(clipped_bdforet)}, "
                         f"Accepted: {len(bdforet_accepted)}, "
                         f"Rejected: {len(clipped_bdforet) - len(bdforet_accepted)} "
                         f"(Phenology mismatch: {phenology_mismatch_count}, "
                         f"Buffer exclusion: {buffer_exclusion_count}, "
                         f"Empty geometry: {empty_geometry_count}, "
                         f"Negative buffer empty: {negative_buffer_empty_count})")

            # Build rows from accepted BD Forêt polygons
            accepted_bdforet = clipped_bdforet.loc[bdforet_accepted].copy() if bdforet_accepted else clipped_bdforet.head(0)
            if len(accepted_bdforet) > 0:
                accepted_bdforet["tile_id"] = tile_id
                accepted_bdforet["source"] = "bdforet"
                # Calculate effective_pixels based on area (1 pixel = 10m × 10m = 100 sq meters)
                accepted_bdforet["effective_pixels"] = accepted_bdforet.geometry.area / 100.0
                accepted_bdforet["NomSER"] = region

                accepted_bdforet["phenology"] = accepted_bdforet["phen_en"]
                accepted_bdforet["genus"]     = accepted_bdforet["genus_en"]
                accepted_bdforet["species"]   = accepted_bdforet["specie_en"]
                # year stays year

                final_bdforet = accepted_bdforet[
                    ["tile_id", "geometry", "source", "effective_pixels",
                     "phenology", "genus", "species", "year", "NomSER"]
                ].copy()

                region_results = pd.concat([region_results, final_bdforet], ignore_index=True)

        # Save region results to intermediate file
        region_file = work_dir / f"{region}_results.parquet"
        if len(region_results) > 0:
            # Fix the year column type issue
            if 'year' in region_results.columns:
                # Convert year to string to avoid type conversion issues
                region_results['year'] = region_results['year'].astype(str)
            
            region_results.to_parquet(region_file)
            logging.info(f"Saved {len(region_results)} polygons for region {region} to {region_file}")
        
        # Add to final results
        final_gdf = pd.concat([final_gdf, region_results], ignore_index=True)
        
        # Update processed regions list
        processed_regions.add(region)
        with open(state_file, 'w') as f:
            json.dump({'processed_regions': list(processed_regions)}, f)
        
        logging.info(f"Completed region {region}, {len(processed_regions)}/{len(available_regions)} regions processed")

    # 4. Save final
    logging.info(f"Final dataset has {len(final_gdf)} polygons.")
    
    # Log detailed summary of the final dataset
    insitu_count = len(final_gdf[final_gdf["source"] == "in-situ"])
    bdforet_count = len(final_gdf[final_gdf["source"] == "bdforet"])
    logging.info(f"Final dataset composition: {insitu_count} in-situ polygons, {bdforet_count} BD Forêt polygons")
    
    # Log phenology distribution
    if "phenology" in final_gdf.columns:
        phen_counts = final_gdf["phenology"].value_counts()
        logging.info(f"Phenology distribution in final dataset:")
        for phen, count in phen_counts.items():
            insitu_phen = len(final_gdf[(final_gdf["source"] == "in-situ") & (final_gdf["phenology"] == phen)])
            bdforet_phen = len(final_gdf[(final_gdf["source"] == "bdforet") & (final_gdf["phenology"] == phen)])
            logging.info(f"  - {phen}: {count} total ({insitu_phen} in-situ, {bdforet_phen} BD Forêt)")
    
    # Log genus distribution (top 10)
    if "genus" in final_gdf.columns:
        genus_counts = final_gdf["genus"].value_counts().nlargest(10)
        logging.info(f"Top 10 genus distribution in final dataset:")
        for genus, count in genus_counts.items():
            insitu_genus = len(final_gdf[(final_gdf["source"] == "in-situ") & (final_gdf["genus"] == genus)])
            bdforet_genus = len(final_gdf[(final_gdf["source"] == "bdforet") & (final_gdf["genus"] == genus)])
            logging.info(f"  - {genus}: {count} total ({insitu_genus} in-situ, {bdforet_genus} BD Forêt)")
    
    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext in [".parquet"]:
        final_gdf.to_parquet(args.output, index=False)
    elif out_ext in [".geojson", ".json"]:
        final_gdf.to_file(args.output, driver="GeoJSON")
    else:
        # default to GPKG if extension is .gpkg or something else
        final_gdf.to_file(args.output, driver="GPKG")

    # 5. Generate report if requested
    if args.report:
        generate_report(final_gdf, args.report)
        logging.info(f"Report generated at {args.report}")

    logging.info("Done.")

def generate_report(gdf, report_path):
    """
    Generate a PDF report with dataset metrics.
    
    Args:
        gdf: GeoDataFrame with the final dataset
        report_path: Path where the PDF report will be saved
    """
    logging.info("Generating report...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Get list of regions
    regions = sorted(gdf["NomSER"].unique())
    
    # Dictionary to store metrics
    metrics = {
        "regions": {},
        "overall": {}
    }
    
    # Calculate overall metrics
    metrics["overall"] = calculate_region_metrics(gdf)
    
    # Calculate metrics per region
    for region in regions:
        region_gdf = gdf[gdf["NomSER"] == region]
        metrics["regions"][region] = calculate_region_metrics(region_gdf)
    
    # Generate PDF
    with PdfPages(report_path) as pdf:
        # Title page
        plt.figure(figsize=(11.7, 8.3))  # A4 landscape
        plt.axis('off')
        plt.text(0.5, 0.5, "Dataset Metrics Report", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=24, fontweight='bold')
        plt.text(0.5, 0.4, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=16)
        plt.text(0.5, 0.3, f"Total polygons: {len(gdf)}",
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=16)
        pdf.savefig()
        plt.close()
        
        # Overall metrics page followed by overall map
        create_metrics_page(metrics["overall"], "Overall Metrics", pdf)
        create_region_map(gdf, "All Regions", pdf)
        
        # Region metrics pages - each region's metrics page followed by its map
        for region in regions:
            region_gdf = gdf[gdf["NomSER"] == region]
            create_metrics_page(metrics["regions"][region], f"Region: {region}", pdf)
            create_region_map(region_gdf, f"Region: {region}", pdf)
            
        # Comparative pages at the end
        create_comparative_page(metrics, pdf)

def calculate_region_metrics(gdf):
    """Calculate metrics for a specific region or the overall dataset."""
    metrics = {}
    
    # Effective pixels by source
    metrics["effective_pixels"] = {}
    for source in ["in-situ", "bdforet"]:
        source_gdf = gdf[gdf["source"] == source]
        metrics["effective_pixels"][source] = source_gdf["effective_pixels"].sum()
    metrics["effective_pixels"]["total"] = gdf["effective_pixels"].sum()
    
    # Phenology distribution (by effective pixels)
    if "phenology" in gdf.columns:
        metrics["phenology"] = {}
        for phen in gdf["phenology"].unique():
            phen_gdf = gdf[gdf["phenology"] == phen]
            phen_pixels = phen_gdf["effective_pixels"].sum()
            metrics["phenology"][phen] = phen_pixels
            
            # Log phenology distribution details
            total_species_for_phen = phen_gdf["species"].nunique()
            top_species_for_phen = phen_gdf.groupby("species")["effective_pixels"].sum().nlargest(3)
            logging.debug(f"Phenology '{phen}': {phen_pixels/1000000:.2f}M pixels, {total_species_for_phen} species")
            for sp, px in top_species_for_phen.items():
                logging.debug(f"  - {sp}: {px/1000000:.2f}M pixels ({px/phen_pixels*100:.1f}% of {phen})")
    
    # Verify phenology totals match overall total
    if "phenology" in metrics:
        pheno_total = sum(metrics["phenology"].values())
        total_pixels = metrics["effective_pixels"]["total"]
        if abs(pheno_total - total_pixels) > 0.1:  # small tolerance for floating point
            logging.warning(f"Phenology total ({pheno_total}) doesn't match overall total ({total_pixels})")
            logging.warning(f"Difference: {pheno_total - total_pixels} pixels")
    
    # Genus distribution (top 10 by effective pixels)
    if "genus" in gdf.columns:
        genus_pixels = gdf.groupby("genus")["effective_pixels"].sum()
        metrics["genus"] = genus_pixels.nlargest(10).to_dict()
    
    # Species distribution (top 10 by effective pixels)
    if "species" in gdf.columns and "phenology" in gdf.columns:
        # Group by both species and phenology
        species_pheno_pixels = gdf.groupby(["species", "phenology"])["effective_pixels"].sum().reset_index()
        # Get top 10 species by total pixels
        top_species = gdf.groupby("species")["effective_pixels"].sum().nlargest(10).index
        # Filter to only include top species
        species_pheno_filtered = species_pheno_pixels[species_pheno_pixels["species"].isin(top_species)]
        # Create dictionary with species and phenology info
        metrics["species_with_pheno"] = species_pheno_filtered.values.tolist()
        # Also keep the original species metrics
        species_pixels = gdf.groupby("species")["effective_pixels"].sum()
        metrics["species"] = species_pixels.nlargest(10).to_dict()
        
        # Add extra debugging for top species phenology distribution
        top_species_total = species_pixels.nlargest(10).sum()
        logging.debug(f"Top 10 species represent {top_species_total/total_pixels*100:.1f}% of total effective pixels")
    
    return metrics

def create_metrics_page(metrics, title, pdf):
    """Create a PDF page with metrics for a region or overall."""
    plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    
    # Create a grid for organizing plots
    gs = gridspec.GridSpec(2, 2)
    
    # Title
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Effective pixels by source (in millions)
    ax1 = plt.subplot(gs[0, 0])
    sources = list(metrics["effective_pixels"].keys())
    values = [v / 1000000 for v in metrics["effective_pixels"].values()]  # Convert to millions
    ax1.bar(sources, values)
    ax1.set_title("Effective Pixels by Source")
    ax1.set_ylabel("Number of Effective Pixels (Millions)")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    # Remove x-axis ticks
    ax1.tick_params(axis='x', which='both', bottom=False)
    
    # Phenology distribution (by effective pixels)
    if "phenology" in metrics:
        ax2 = plt.subplot(gs[0, 1])
        labels = list(metrics["phenology"].keys())
        sizes = list(metrics["phenology"].values())
        total = sum(sizes)
        # Only create pie chart if there's data
        if total > 0:
            ax2.pie([s/total for s in sizes], labels=None, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Phenology Distribution (by Effective Pixels)")
            if len(labels) > 0:
                ax2.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        # Remove all ticks and labels for pie chart
        ax2.axis('off')
    
    # Genus distribution (horizontal bar) by effective pixels
    if "genus" in metrics:
        ax3 = plt.subplot(gs[1, 0])
        genus_data = sorted(metrics["genus"].items(), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in genus_data]
        values = [x[1] / 1000000 for x in genus_data]  # Convert to millions
        y_pos = np.arange(len(labels))
        ax3.barh(y_pos, values)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.invert_yaxis()  # labels read top-to-bottom
        ax3.set_title("Top Genera (by Effective Pixels)")
        ax3.set_xlabel("Effective Pixels (Millions)")
    
    # Species distribution (text only due to potentially long names)
    if "species" in metrics:
        ax4 = plt.subplot(gs[1, 1])
        ax4.axis('off')
        
        # Use the new species with phenology data if available
        if "species_with_pheno" in metrics:
            # First summarize by species to get totals for sorting
            species_summary = {}
            for species, phenology, pixels in metrics["species_with_pheno"]:
                if species not in species_summary:
                    species_summary[species] = 0
                species_summary[species] += pixels
            
            # Sort species by total pixel count
            sorted_species = sorted(species_summary.items(), key=lambda x: x[1], reverse=True)
            
            # Create mapping of species to phenologies
            species_to_phenos = {}
            for species, phenology, pixels in metrics["species_with_pheno"]:
                if species not in species_to_phenos:
                    species_to_phenos[species] = set()
                species_to_phenos[species].add(phenology)
            
            # Format the text with species and their phenologies
            species_text = "Top Species (by Effective Pixels):\n\n"
            for species, total_count in sorted_species:
                # List of phenologies for this species (without pixel counts)
                pheno_text = ", ".join(sorted(species_to_phenos[species]))
                species_text += f"{species}: {total_count/1000000:.2f}M pixels ({pheno_text})\n"
        else:
            # Fallback to original display if new data format not available
            species_data = sorted(metrics["species"].items(), key=lambda x: x[1], reverse=True)
            species_text = "Top Species (by Effective Pixels):\n\n"
            for species, count in species_data:
                species_text += f"{species}: {count/1000000:.2f}M pixels\n"
                
        ax4.text(0, 0.95, species_text, verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.close()

def create_comparative_page(metrics, pdf):
    """Create a page comparing metrics across regions."""
    plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    
    # Title
    plt.suptitle("Regional Comparison", fontsize=16, fontweight='bold', y=0.98)
    
    # Prepare data for regions
    regions = list(metrics["regions"].keys())
    
    # Effective pixels comparison (in millions)
    ax1 = plt.subplot(111)
    
    # Data for grouped bar chart (converted to millions)
    in_situ_values = [metrics["regions"][r]["effective_pixels"]["in-situ"] / 1000000 for r in regions]
    bdforet_values = [metrics["regions"][r]["effective_pixels"]["bdforet"] / 1000000 for r in regions]
    
    # Set width of bars
    barWidth = 0.35
    r1 = np.arange(len(regions))
    r2 = [x + barWidth for x in r1]
    
    # Create grouped bars
    ax1.bar(r1, in_situ_values, width=barWidth, label='In-situ')
    ax1.bar(r2, bdforet_values, width=barWidth, label='BD Forêt')
    
    # Add labels and legend
    ax1.set_xlabel('')
    ax1.set_ylabel('Effective Pixels (Millions)')
    ax1.set_title('Effective Pixels by Region and Source')
    ax1.set_xticks([r + barWidth/2 for r in range(len(regions))])
    ax1.set_xticklabels(regions, rotation=45, ha='right')
    ax1.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.close()
    
    # Add heatmap page for phenology distribution by region
    if all("phenology" in metrics["regions"][r] for r in regions):
        plt.figure(figsize=(11.7, 8.3))  # A4 landscape
        plt.suptitle("Phenology Distribution by Region (Effective Pixels)", fontsize=16, fontweight='bold', y=0.98)
        
        # Collect all phenology types
        all_phenology = set()
        for r in regions:
            all_phenology.update(metrics["regions"][r]["phenology"].keys())
        all_phenology = sorted(all_phenology)
        
        # Create data for heatmap
        heatmap_data = []
        for r in regions:
            row = []
            for phen in all_phenology:
                # Get pixels in millions
                pixel_count = metrics["regions"][r]["phenology"].get(phen, 0) / 1000000
                row.append(pixel_count)
            heatmap_data.append(row)
        
        ax = plt.subplot(111)
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", 
                   xticklabels=all_phenology, yticklabels=regions, ax=ax)
        ax.set_title("Effective Pixels (Millions) by Phenology and Region")
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()

def create_region_map(gdf, title, pdf):
    """Create a map visualization of tiles for a region with contextily background."""
    if len(gdf) == 0:
        logging.warning(f"No data to plot for {title}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape
    
    # Set title
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Get tile boundaries for the region - we'll focus on these
    tiles = gdf[["tile_id", "geometry"]].drop_duplicates("tile_id")
    
    # Try to read the tiles from the provided file path for more accurate boundaries
    tiles_file = "results/datasets/tiles_2_5_km_final.parquet"
    try:
        if os.path.exists(tiles_file):
            # Load the full tiles dataset
            full_tiles_gdf = gpd.read_parquet(tiles_file)
            # Filter to only the tile_ids in our current region
            region_tile_ids = tiles["tile_id"].unique()
            full_tiles_for_region = full_tiles_gdf[full_tiles_gdf.index.isin(region_tile_ids)]
            if len(full_tiles_for_region) > 0:
                # Use these tiles instead if we found matches
                tiles = full_tiles_for_region.reset_index(names="tile_id")
                logging.info(f"Using {len(tiles)} tiles from source file for {title}")
            else:
                logging.warning(f"No matching tiles found in source file for {title}, using derived tiles")
        else:
            logging.warning(f"Tiles file {tiles_file} not found, using derived tiles")
    except Exception as e:
        logging.warning(f"Error loading tiles from {tiles_file}: {e}. Using derived tiles.")
    
    # Convert to Web Mercator (EPSG:3857) for compatibility with contextily
    tiles_webmerc = tiles.to_crs(epsg=3857)
    
    # Plot filled tiles (focus on this as requested)
    tiles_webmerc.plot(ax=ax, color='black', alpha=0.5, edgecolor='black', linewidth=1)
    
    # Add contextily basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        logging.warning(f"Failed to add contextily basemap: {e}")
        
    # Remove axes ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Save figure
    plt.tight_layout()
    pdf.savefig()
    plt.close()

if __name__ == "__main__":
    main()
