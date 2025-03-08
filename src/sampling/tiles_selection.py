#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sampling script for 2.5x2.5 km tiles over France with the full monthly mosaics.
We:
1) Filter out tiles that have fewer than 10 in-situ pixels.
2) Attempt to balance coverage among eco-regions according to their relative area.
3) Optionally add more tiles for improved spatial distribution.

Author: Arthur (adaptation by o1-pro)
"""

import argparse
import geopandas as gpd
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
import warnings

# Try to import tabulate for nice table formatting, with fallback
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    logging.warning("tabulate package not found. Tables will use basic formatting.")
    logging.warning("Install with: pip install tabulate")

warnings.filterwarnings("ignore", category=UserWarning, module="shapely")

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.constants import MAPPING_SER_ECO_REGIONS, MAPPING_ECO_REGIONS_FR_EN, FOREST_COVER_RATIO_BY_REGION


# Set up logging
def setup_logging(log_file):
    """Set up logging to file and console."""
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler with detailed output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler with minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

def load_data(
    tiles_path: str,
    ecoregion_path: str,
    force_crs: str = "EPSG:2154"  # Add parameter to force CRS
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads the tiles GeoDataFrame and the eco-region GeoDataFrame.
    Both will be renamed to English in the 'NomSER' column by the time
    they exit this function.
    
    Args:
        tiles_path: Path to tiles parquet file
        ecoregion_path: Path to eco-region shapefile
        force_crs: CRS to force on input data (default: EPSG:2154 Lambert-93)
    """

    tiles_gdf = gpd.read_parquet(tiles_path)
    eco_gdf = gpd.read_file(ecoregion_path)

    # Force setting CRS to Lambert-93 without reprojection for tiles
    # This assumes the coordinates are actually Lambert-93 but were incorrectly tagged
    logging.info(f"Forcing CRS to {force_crs} for tiles dataset without reprojection")
    tiles_gdf.set_crs(force_crs, allow_override=True, inplace=True)
    
    # Ensure eco_gdf has the same CRS
    if eco_gdf.crs != force_crs:
        logging.info(f"Converting eco-regions from {eco_gdf.crs} to {force_crs}")
        eco_gdf = eco_gdf.to_crs(force_crs)
    
    # 1) Convert ecoregions to "greco" if 'codeser' exists, then dissolve
    if 'codeser' in eco_gdf.columns:
        eco_gdf['greco'] = eco_gdf['codeser'].apply(
            lambda x: x[0] if isinstance(x, str) and len(x) > 0 else x
        )
        eco_gdf = eco_gdf.dissolve(by='greco', aggfunc='first').reset_index().iloc[1:]
        eco_gdf['NomSER'] = eco_gdf['NomSER'].apply(
            lambda x: MAPPING_SER_ECO_REGIONS.get(x.replace(' ', '_'), x.replace(' ', '_')) if isinstance(x, str) else x
        )
        eco_gdf['NomSER'] = eco_gdf['NomSER'].apply(
            lambda x: MAPPING_ECO_REGIONS_FR_EN.get(x, x) if isinstance(x, str) else x
        )
        
    # 3) If NomSER is present in tiles, convert short French => English
    if 'NomSER' in tiles_gdf.columns:
        # The tile dataset is already in short French like 'Alpes', 'Corse', 'Méditerranée'.
        # So we directly map it to English with MAPPING_ECO_REGIONS_FR_EN
        tiles_gdf['NomSER'] = tiles_gdf['NomSER'].apply(
            lambda x: MAPPING_ECO_REGIONS_FR_EN.get(x, x) if isinstance(x, str) else x
        )
    else:
        logging.debug("No 'NomSER' column found in tiles_gdf. Cannot rename to English.")

    # Verify CRS for debugging
    logging.debug(f"Final tiles CRS: {tiles_gdf.crs}")
    logging.debug(f"Final eco-regions CRS: {eco_gdf.crs}")

    return tiles_gdf, eco_gdf



def filter_tiles_by_in_situ(
    tiles_gdf: gpd.GeoDataFrame,
    min_pixels: int = 10
) -> gpd.GeoDataFrame:
    """
    Filters tiles that have at least 'min_pixels' from the in-situ dataset.

    Args:
        tiles_gdf (gpd.GeoDataFrame): The grid cell GeoDataFrame.
        min_pixels (int): The minimum number of in-situ pixels required.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with tiles meeting the requirement.
    """
    # Each tile's 'perc' is the percentage coverage of the in-situ dataset.
    # The tile area is (2.5 * 2.5) = 6.25 km^2, in meter square it is 2,500 * 2,500 = 6,250,000 m^2 if using EPSG:2154
    # But for counting in-situ *pixels*, we can assume each tile is 625,000 pixels if each pixel is 10x10 m, etc.
    # For simplicity, we can interpret 'perc' as percent coverage. So effective_pixels = (perc / 100) * total_possible_pixels_in_tile.
    # The user is working with 250 x 250 = 62,500 possible 10-m pixels in a 2.5 km tile.
    # So effective_pixels = (perc / 100) * 62500
    # Then we keep those with effective_pixels > min_pixels

    tiles_gdf = tiles_gdf.copy()
    tiles_gdf["effective_pixels"] = (tiles_gdf["perc"] / 100.0) * 62500
    return tiles_gdf[tiles_gdf["effective_pixels"] >= min_pixels]


def assign_tiles_to_ecoregions(
    tiles_gdf: gpd.GeoDataFrame,
    ecoregions_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns each tile to an eco-region based on tile centroid.

    Args:
        tiles_gdf (gpd.GeoDataFrame): Tiles to be assigned.
        ecoregions_gdf (gpd.GeoDataFrame): Polygons for eco-regions.

    Returns:
        gpd.GeoDataFrame: tiles_gdf with an added 'NomSER' column for the eco-region name.
    """
    tiles_gdf = tiles_gdf.copy()
    # Compute centroids
    centroids = tiles_gdf.geometry.centroid
    # Spatial join with eco-regions
    centroids_gdf = gpd.GeoDataFrame(tiles_gdf.drop(columns="geometry"), 
                                     geometry=centroids, 
                                     crs=tiles_gdf.crs)
    joined = gpd.sjoin(centroids_gdf, ecoregions_gdf, how="left", op="within")
    # joined might have a field like 'NomSER' or whatever your column is. 
    # Rename accordingly if needed.
    tiles_gdf["NomSER"] = joined["NomSER"]
    return tiles_gdf


def visualize_reference_region_tiles(
    reference_region: str,
    original_tiles: gpd.GeoDataFrame,
    added_tiles: gpd.GeoDataFrame,
    ecoregions_gdf: gpd.GeoDataFrame,
    output_path: str
) -> str:
    """
    Creates a visualization of the reference region showing original filtered tiles
    and newly added tiles.
    
    Args:
        reference_region: Name of the reference region
        original_tiles: Original filtered tiles in the reference region
        added_tiles: Newly added tiles to the reference region
        ecoregions_gdf: Eco-regions GeoDataFrame
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get just the reference region polygon
    ref_region_poly = ecoregions_gdf[ecoregions_gdf['NomSER'] == reference_region]
    
    # Plot the reference region
    ref_region_poly.plot(
        ax=ax,
        color='lightblue',
        alpha=0.5,
        edgecolor='blue',
        linewidth=1
    )
    
    # Plot original filtered tiles
    original_tiles_in_region = original_tiles[original_tiles['NomSER'] == reference_region]
    original_tiles_in_region.plot(
        ax=ax,
        color='gray',
        markersize=25,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7,
        label='Original filtered tiles'
    )
    
    # Plot newly added tiles
    if len(added_tiles) > 0:
        added_tiles.plot(
            ax=ax,
            color='red',
            markersize=25,
            edgecolor='darkred',
            linewidth=1,
            label='Newly added tiles'
        )
    
    # Set title and legend
    plt.title(f'Tile Selection in Reference Region: {reference_region}')
    plt.legend(loc='upper right')
    
    # Add count information
    plt.figtext(
        0.02, 0.02, 
        f"Original tiles: {len(original_tiles_in_region)} | Newly added tiles: {len(added_tiles)}",
        fontsize=10, 
        bbox={"facecolor":"white", "alpha":0.8, "pad":5}
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def balance_coverage_with_spatial_distribution(
    tiles_gdf: gpd.GeoDataFrame,
    ecoregions_gdf: gpd.GeoDataFrame,
    all_tiles_gdf: gpd.GeoDataFrame = None,
    min_distance: float = 25000,
    min_pixels: int = 0
) -> gpd.GeoDataFrame:
    """
    Balances tile coverage across eco-regions while ensuring good spatial distribution.
    First identifies the baseline density (tiles per km²) from the eco-region with 
    the highest quality coverage, then tries to match that density in other regions
    while maintaining spatial separation between selected tiles.
    
    Uses effective forest area (total area * forest cover ratio) to determine 
    the number of tiles needed for each region.

    Args:
        tiles_gdf (gpd.GeoDataFrame): Tiles that passed the initial filtering.
        ecoregions_gdf (gpd.GeoDataFrame): Eco-regions polygons.
        all_tiles_gdf (gpd.GeoDataFrame, optional): Full set of tiles to consider for additional selection.
            If None, uses tiles_gdf. Default is None.
        min_distance (float): Minimum distance between selected tiles.
        min_pixels (int): Minimum number of pixels for additional tiles.

    Returns:
        gpd.GeoDataFrame: Selected tiles with balanced coverage and spatial distribution.
    """
    # Use all_tiles_gdf for additional candidates if provided, otherwise use tiles_gdf
    if all_tiles_gdf is None:
        all_tiles_gdf = tiles_gdf.copy()
    
    # Calculate eco-region areas
    ecoregions_gdf = ecoregions_gdf.copy()
    ecoregions_gdf["area_km2"] = ecoregions_gdf.geometry.area / 1_000_000  # Convert to km²
    eco_areas = ecoregions_gdf.set_index("NomSER")["area_km2"].to_dict()
    logging.debug(f"Eco-areas: {eco_areas}")

    
    # Remove None keys from eco_areas if they exist
    if None in eco_areas:
        logging.warning("Found None value in eco-region names. Removing it.")
        eco_areas.pop(None, None)
    
    # Group tiles by eco-region, skipping any with None or NaN values
    region_groups = tiles_gdf.dropna(subset=["NomSER"]).groupby("NomSER")
    
    # Calculate initial density of high-quality tiles by region, using effective forest area
    region_counts = region_groups.size()
    region_densities = {}
    
    logging.debug("Initial tile densities by eco-region (based on effective forest area):")
    for region, count in region_counts.items():
        if region not in eco_areas:
            logging.warning(f"Region '{region}' not found in eco-regions shapefile. Skipping.")
            # Debug: Check if there are any similar region names that might match
            close_matches = [r for r in eco_areas.keys() if r is not None and (
                r.lower() in region.lower() or region.lower() in r.lower()
            )]
            if close_matches:
                logging.debug(f"  Possible matches for '{region}': {close_matches}")
            continue
        
        # Convert region name to English for accessing forest cover ratio
        region_en = MAPPING_ECO_REGIONS_FR_EN.get(region, region)
        
        # Get forest cover ratio for this region (default to 0.5 if not found)
        forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region_en, 0.5)
        
        # Calculate effective forest area
        effective_forest_area = eco_areas[region] * forest_ratio
        
        # Calculate density based on effective forest area
        density = count / effective_forest_area if effective_forest_area > 0 else 0
        region_densities[region] = density
        logging.debug(f"  {region}: {count} tiles / {effective_forest_area:.2f} km² effective forest area = {density:.6f} tiles/km²")
    
    # Find the eco-region with highest quality coverage as reference
    if not region_densities:
        logging.warning("No valid eco-regions with tiles found.")
        return tiles_gdf
    
    reference_region = max(region_densities.items(), key=lambda x: x[1])[0]
    reference_density = region_densities[reference_region]
    
    logging.info(f"Reference eco-region: {reference_region} with density {reference_density:.6f} tiles/km²")
    
    # NEW STEP: Add 10 more tiles to the reference region for better spatial coverage
    logging.info(f"Adding 10 more tiles to reference region {reference_region} for improved spatial coverage")
    
    # Initialize selected tiles with the current tiles
    selected_tiles = tiles_gdf.copy()
    
    # Store original tiles in reference region before adding new ones (for visualization)
    original_ref_tiles = selected_tiles[selected_tiles["NomSER"] == reference_region].copy()
    
    # Get candidates for the reference region that are not already selected
    reference_candidates = all_tiles_gdf[
        (all_tiles_gdf["NomSER"] == reference_region) & 
        (~all_tiles_gdf.index.isin(selected_tiles.index))
    ].copy()
    
    # Filter to have at least min_pixels if specified
    if min_pixels > 0:
        reference_candidates["effective_pixels"] = reference_candidates["effective_pixels"].fillna(0)
        reference_candidates = reference_candidates[reference_candidates["effective_pixels"] >= min_pixels]
    
    # Add 10 tiles to reference region, maintaining spatial distribution
    added_to_reference = 0
    # Create empty GeoDataFrame with geometry column
    added_ref_tiles = gpd.GeoDataFrame(
        data=[],
        columns=['geometry'],
        geometry='geometry',
        crs=selected_tiles.crs
    )
    
    with tqdm(total=10, desc=f"Adding tiles to {reference_region}", unit="tile") as pbar:
        while added_to_reference < 10 and not reference_candidates.empty:
            # Calculate minimum distance to any already selected tile
            if len(selected_tiles) > 0:
                # Use a vectorized approach to calculate distances
                distances = []
                for candidate in reference_candidates.itertuples():
                    # Calculate minimum distance from this candidate to any selected tile
                    min_dist = float('inf')
                    for selected in selected_tiles.itertuples():
                        dist = candidate.geometry.centroid.distance(selected.geometry.centroid)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
                # Use .loc to avoid SettingWithCopyWarning
                reference_candidates.loc[:, "min_dist"] = distances
            else:
                reference_candidates.loc[:, "min_dist"] = float('inf')
            
            # Sort by distance (descending) and then by effective pixels (descending)
            reference_candidates = reference_candidates.sort_values(
                by=["min_dist", "effective_pixels"], 
                ascending=[False, False]
            )
            
            # Select the best candidate
            if len(reference_candidates) > 0:
                best_candidate = reference_candidates.iloc[0]
                
                # Only add if it meets minimum distance requirement or we have relaxed criteria
                if best_candidate["min_dist"] >= min_distance or len(reference_candidates) <= 1:
                    use_anyway = (best_candidate["min_dist"] < min_distance) and \
                                 (added_to_reference < 2) and \
                                 (len(reference_candidates) <= 1)
                    
                    if best_candidate["min_dist"] >= min_distance or use_anyway:
                        # Add to selected tiles
                        best_candidate_df = best_candidate.to_frame().T
                        selected_tiles = pd.concat(
                            [selected_tiles, best_candidate_df],
                            ignore_index=True
                        )
                        # Also add to our tracking of newly added reference tiles 
                        added_ref_tiles = pd.concat(
                            [added_ref_tiles, best_candidate_df],
                            ignore_index=True
                        )
                        added_to_reference += 1
                        pbar.update(1)
                        
                        dist_msg = f"{best_candidate['min_dist']:.1f}m from nearest" if \
                                    best_candidate["min_dist"] < float('inf') else "no nearby tiles"
                        logging.debug(f"Added tile to reference region with {best_candidate['effective_pixels']:.1f} pixels, {dist_msg}")
                
                # Remove this candidate
                reference_candidates = reference_candidates.iloc[1:]
                
                # If we can't find tiles that meet distance requirement, try with reduced distance
                if added_to_reference < 10 and reference_candidates.empty:
                    relaxed_distance = min_distance / 2
                    logging.debug(f"Could not add all 10 tiles with {min_distance}m separation. Trying with {relaxed_distance}m")
                    
                    # Get candidates again
                    reference_candidates = all_tiles_gdf[
                        (all_tiles_gdf["NomSER"] == reference_region) & 
                        (~all_tiles_gdf.index.isin(selected_tiles.index))
                    ].copy()
                    
                    if min_pixels > 0:
                        reference_candidates["effective_pixels"] = reference_candidates["effective_pixels"].fillna(0)
                        reference_candidates = reference_candidates[reference_candidates["effective_pixels"] >= min_pixels]
                    
                    # Calculate distances again with relaxed criteria
                    if len(selected_tiles) > 0:
                        distances = []
                        for candidate in reference_candidates.itertuples():
                            # Use vectorized distance calculation  
                            all_distances = [candidate.geometry.centroid.distance(selected.geometry.centroid) 
                                            for selected in selected_tiles.itertuples()]
                            min_dist = min(all_distances) if all_distances else float('inf')
                            distances.append(min_dist)
                        reference_candidates.loc[:, "min_dist"] = distances
                        reference_candidates = reference_candidates[reference_candidates["min_dist"] >= relaxed_distance]
            else:
                break
    
    logging.info(f"Added {added_to_reference} tiles to reference region {reference_region}")
    
    # Create and save visualization of reference region tiles
    if added_to_reference > 0:
        # Create directory for visualizations if it doesn't exist
        vis_dir = "results/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = f"{vis_dir}/reference_region_{reference_region}_{timestamp}.png"
        
        try:
            # Create the visualization
            logging.info(f"Creating visualization of reference region {reference_region}...")
            vis_file = visualize_reference_region_tiles(
                reference_region,
                original_ref_tiles,
                added_ref_tiles,
                ecoregions_gdf,
                vis_path
            )
            logging.info(f"Visualization saved to: {vis_file}")
        except Exception as e:
            logging.error(f"Failed to create reference region visualization: {str(e)}")
    
    # Recalculate reference density with additional tiles
    updated_reference_count = selected_tiles[selected_tiles["NomSER"] == reference_region].shape[0]
    
    # Get effective forest area for reference region
    region_en = MAPPING_ECO_REGIONS_FR_EN.get(reference_region, reference_region)
    forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region_en, 0.5)
    effective_forest_area = eco_areas[reference_region] * forest_ratio
    
    # Update reference density
    updated_reference_density = updated_reference_count / effective_forest_area if effective_forest_area > 0 else 0
    logging.info(f"Updated reference density: {updated_reference_density:.6f} tiles/km² ({updated_reference_count} tiles)")
    
    # Calculate target number of tiles for each region based on updated reference density and effective forest area
    target_counts = {}
    additional_needed = {}
    effective_forest_areas = {}
    forest_ratios = {}
    
    logging.debug("Target tile counts by eco-region (based on effective forest area):")
    # Use list() to make a copy of the keys for safe iteration
    for region in list(eco_areas.keys()):
        if region is None:
            continue
        
        # Get total area
        area = eco_areas[region]
        
        # Convert French region name to English for accessing the forest cover ratio dictionary
        region_en = MAPPING_ECO_REGIONS_FR_EN.get(region, region)
        
        # Get forest cover ratio for this region (default to 0.5 if not found)
        forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region_en, 0.5)
        
        # Calculate effective forest area
        effective_forest_area = area * forest_ratio
        
        # Store values for summary table
        effective_forest_areas[region] = effective_forest_area
        forest_ratios[region] = forest_ratio
        
        # Calculate target based on updated reference density applied to effective forest area
        target = int(updated_reference_density * effective_forest_area)
        
        # Get current count from the updated selected_tiles dataframe
        current = selected_tiles[selected_tiles["NomSER"] == region].shape[0]
        needed = max(0, target - current)
        
        target_counts[region] = target
        additional_needed[region] = needed
        
        status = "✓" if needed == 0 else f"needs {needed} more"
        logging.debug(f"  {region} ({region_en}): {current}/{target} tiles ({status}), " 
                     f"area: {area:.2f} km², forest cover: {forest_ratio:.2f}, "
                     f"effective forest area: {effective_forest_area:.2f} km²")
    
    # Log a summary table of eco-region info
    logging.info("\nEco-region Summary (sorted by effective forest area):")
    
    # Prepare data for tabulate
    table_data = []
    headers = ["Eco-region", "Total Area (km²)", "Forest Cover", "Effective Area (km²)", 
               "Current Tiles", "Target Tiles", "Needed"]
    
    # Sort regions by effective forest area for the summary table
    sorted_regions = sorted(
        [r for r in eco_areas.keys() if r is not None],
        key=lambda r: effective_forest_areas.get(r, 0),
        reverse=True
    )
    
    for region in sorted_regions:
        area = eco_areas.get(region, 0)
        forest_ratio = forest_ratios.get(region, 0)
        effective_area = effective_forest_areas.get(region, 0)
        current = selected_tiles[selected_tiles["NomSER"] == region].shape[0]
        target = target_counts.get(region, 0)
        needed = additional_needed.get(region, 0)
        
        table_data.append([
            region, 
            f"{area:.2f}", 
            f"{forest_ratio:.2f}", 
            f"{effective_area:.2f}", 
            current, 
            target, 
            needed
        ])
    
    # Add a total row
    all_current = sum(selected_tiles[selected_tiles["NomSER"] == region].shape[0] for region in sorted_regions)
    table_data.append([
        "TOTAL", 
        f"{sum(eco_areas.values()):.2f}", 
        "-", 
        f"{sum(effective_forest_areas.values()):.2f}", 
        all_current, 
        sum(target_counts.values()), 
        sum(additional_needed.values())
    ])
    
    # Format and display the table
    if HAS_TABULATE:
        # Use tabulate for nice formatting
        table_str = tabulate(
            table_data, 
            headers=headers, 
            tablefmt="grid", 
            numalign="right", 
            stralign="left"
        )
        logging.info("\n" + table_str)
    else:
        # Fallback to basic formatting
        logging.info("=" * 100)
        logging.info(f"{'Eco-region':<20} {'Total Area (km²)':<15} {'Forest Cover':<15} {'Effective Area (km²)':<20} {'Current Tiles':<15} {'Target Tiles':<15} {'Needed':<10}")
        logging.info("-" * 100)
        
        for row in table_data:
            region, area, forest_ratio, effective_area, current, target, needed = row
            logging.info(f"{region:<20} {area:<15} {forest_ratio:<15} {effective_area:<20} {str(current):<15} {str(target):<15} {str(needed):<10}")
        
        logging.info("=" * 100)
    
    logging.info(f"\nTotal additional tiles needed: {sum(additional_needed.values())}")

    # Prepare for spatial selection of remaining regions
    # Sort remaining tiles by in-situ pixel count (highest first)
    # Use the full set of tiles for additional candidates
    additional_candidates = all_tiles_gdf.copy()
    additional_candidates["effective_pixels"] = additional_candidates["effective_pixels"].fillna(0)
    
    # Filter additional_candidates to have at least min_pixels
    if min_pixels > 0:
        additional_candidates = additional_candidates[additional_candidates["effective_pixels"] >= min_pixels]
        logging.debug(f"Filtered potential additional tiles to {len(additional_candidates)} with >= {min_pixels} pixels")
    
    # We'll add tiles to regions that need them, starting from regions with lowest current density
    regions_to_fill = [r for r, n in additional_needed.items() if n > 0]
    regions_to_fill.sort(key=lambda r: selected_tiles[selected_tiles["NomSER"] == r].shape[0] / effective_forest_areas.get(r, 1))
    
    # Count total tiles needed
    total_needed = sum(additional_needed.values())
    logging.info(f"Adding tiles to balance coverage: {total_needed} tiles needed across {len(regions_to_fill)} eco-regions")
    logging.info(f"Maintaining {min_distance/1000:.1f}km separation between tiles")
    
    # Track how many candidates we have per region (for debug info)
    region_candidate_counts = {}
    for region in regions_to_fill:
        region_candidates = additional_candidates[additional_candidates["NomSER"] == region]
        region_candidate_counts[region] = len(region_candidates)
        logging.debug(f"Region {region}: {len(region_candidates)} candidate tiles available")
    
    # Use tqdm for progress display
    with tqdm(total=total_needed, desc="Adding tiles", unit="tile") as pbar:
        for region in regions_to_fill:
            needed = additional_needed[region]
            if needed <= 0:
                continue
                
            logging.debug(f"Working on {region}: need {needed} more tiles")
            
            # Get candidates for this region that are not already selected
            # Create an explicit copy to avoid SettingWithCopyWarning
            candidates = additional_candidates[
                (additional_candidates["NomSER"] == region) & 
                (~additional_candidates.index.isin(selected_tiles.index))
            ].copy()
            
            if candidates.empty:
                logging.debug(f"No additional candidate tiles available for {region}")
                continue
            
            # Add tiles one by one, ensuring spatial separation
            added = 0
            
            while added < needed and not candidates.empty:
                # Calculate minimum distance to any already selected tile
                if len(selected_tiles) > 0:
                    # Use a vectorized approach to calculate distances between each candidate and all selected tiles
                    distances = []
                    for candidate in candidates.itertuples():
                        # Calculate minimum distance from this candidate to any selected tile
                        min_dist = float('inf')
                        for selected in selected_tiles.itertuples():
                            dist = candidate.geometry.centroid.distance(selected.geometry.centroid)
                            min_dist = min(min_dist, dist)
                        distances.append(min_dist)
                    # Use .loc to avoid SettingWithCopyWarning
                    candidates.loc[:, "min_dist"] = distances
                else:
                    candidates.loc[:, "min_dist"] = float('inf')
                
                # Sort by distance (descending) and then by effective pixels (descending)
                candidates = candidates.sort_values(
                    by=["min_dist", "effective_pixels"], 
                    ascending=[False, False]
                )
                
                # Select the best candidate
                best_candidate = candidates.iloc[0]
                
                # Only add if it meets minimum distance requirement
                if best_candidate["min_dist"] >= min_distance or len(candidates) <= 1:
                    # If we only have one candidate left or can't meet distance requirement
                    # and we've added < 10% of what we need, use it anyway
                    use_anyway = (best_candidate["min_dist"] < min_distance) and \
                                 (added < 0.1 * needed) and \
                                 (len(candidates) <= 1)
                    
                    if best_candidate["min_dist"] >= min_distance or use_anyway:
                        selected_tiles = pd.concat(
                            [selected_tiles, best_candidate.to_frame().T],
                            ignore_index=True
                        )
                        added += 1
                        pbar.update(1)
                        
                        dist_msg = f"{best_candidate['min_dist']:.1f}m from nearest" if \
                                    best_candidate["min_dist"] < float('inf') else "no nearby tiles"
                        logging.debug(f"Added tile with {best_candidate['effective_pixels']:.1f} pixels, {dist_msg}")
                
                # Remove this candidate
                candidates = candidates.iloc[1:]
                
                # If we can't find tiles that meet distance requirement, we might need to relax it
                if candidates.empty and added < needed:
                    if added == 0:
                        # If we couldn't add any tiles, try with a reduced distance requirement
                        relaxed_distance = min_distance / 2
                        logging.debug(f"Could not add any tiles with {min_distance}m separation. Trying with {relaxed_distance}m")
                        
                        # Get candidates again
                        candidates = additional_candidates[
                            (additional_candidates["NomSER"] == region) & 
                            (~additional_candidates.index.isin(selected_tiles.index))
                        ].copy()
                        
                        # Calculate distances
                        if len(selected_tiles) > 0:
                            distances = []
                            for candidate in candidates.itertuples():
                                # Use vectorized distance calculation  
                                all_distances = [candidate.geometry.centroid.distance(selected.geometry.centroid) 
                                                for selected in selected_tiles.itertuples()]
                                min_dist = min(all_distances) if all_distances else float('inf')
                                distances.append(min_dist)
                            candidates.loc[:, "min_dist"] = distances
                        else:
                            candidates.loc[:, "min_dist"] = float('inf')
                            
                        # Filter by relaxed distance
                        candidates = candidates[candidates["min_dist"] >= relaxed_distance]
                        
                        if not candidates.empty:
                            # Sort candidates
                            candidates = candidates.sort_values(
                                by=["min_dist", "effective_pixels"], 
                                ascending=[False, False]
                            )
                            
                            # Add best candidate
                            best_candidate = candidates.iloc[0]
                            selected_tiles = pd.concat(
                                [selected_tiles, best_candidate.to_frame().T],
                                ignore_index=True
                            )
                            added += 1
                            pbar.update(1)
                            logging.debug(f"Added tile with relaxed distance: {best_candidate['min_dist']:.1f}m")
                    else:
                        logging.debug(f"Could only add {added}/{needed} tiles while maintaining distance requirements")
            
            logging.info(f"Added {added}/{needed} tiles to {region}")
    
    # Calculate final statistics
    final_counts = selected_tiles.groupby("NomSER").size()
    logging.debug("Final tile distribution after balancing with spatial distribution:")
    for region in sorted([r for r in eco_areas.keys() if r is not None], key=str):
        area = eco_areas.get(region, 0)
        count = final_counts.get(region, 0)
        density = count / area if area > 0 else 0
        target = target_counts.get(region, 0)
        percent = (count / target * 100) if target > 0 else 0
        logging.debug(f"  {region}: {count} tiles ({density:.6f} tiles/km², {percent:.1f}% of target)")
    
    # Remove temporary columns before returning
    if "min_dist" in selected_tiles.columns:
        selected_tiles = selected_tiles.drop(columns=["min_dist"])
    if "centroid" in selected_tiles.columns:
        selected_tiles = selected_tiles.drop(columns=["centroid"])
        
    return selected_tiles

def create_visualization(
    selected_tiles: gpd.GeoDataFrame,
    filtered_tiles: gpd.GeoDataFrame,
    all_tiles: gpd.GeoDataFrame,
    eco_regions: gpd.GeoDataFrame,
    output_path: str
) -> str:
    """
    Creates a visualization of the selected tiles overlaid on eco-regions.
    
    Args:
        selected_tiles: Final selected tiles
        filtered_tiles: Tiles that passed the initial filtering
        all_tiles: All tiles in the original dataset
        eco_regions: Eco-regions polygons
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Choose color palette for eco-regions based on matplotlib version
    if hasattr(mpl, 'colormaps'):  # New matplotlib version (>=3.6)
        base_cmap = mpl.colormaps['Pastel1']
        # Create a new colormap with the right number of colors
        colors = base_cmap(np.linspace(0, 1, len(eco_regions)))
        cmap = ListedColormap(colors)
    else:  # Older matplotlib version
        cmap = plt.cm.get_cmap('Pastel1', len(eco_regions))
    
    # Plot eco-regions with a colormap
    eco_regions_plot = eco_regions.plot(
        column='NomSER',
        ax=ax,
        alpha=0.6,
        cmap=cmap,
        edgecolor='gray',
        linewidth=0.5
    )
    
    # Create a dictionary to map eco-region names to colors
    unique_regions = eco_regions['NomSER'].unique()
    if hasattr(mpl, 'colormaps'):
        region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
    else:
        region_colors = {region: cmap(i) for i, region in enumerate(unique_regions)}
    
    # Plot all tiles as small dots with edges for better visibility
    if len(all_tiles) > 0:
        all_tiles.plot(
            ax=ax,
            color='lightgray',
            markersize=2,
            alpha=0.3,
            edgecolor='gray',
            linewidth=0.2
        )
    
    # We're skipping the filtered tiles as requested
    
    # Plot selected tiles with edges for better visibility
    selected_tiles.plot(
        ax=ax,
        color='red',
        markersize=10,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Create manual legend for tile types
    legend_elements = [
        mpatches.Patch(color='lightgray', alpha=0.3, edgecolor='gray', label=f'All tiles ({len(all_tiles)})'),
        mpatches.Patch(color='red', edgecolor='black', label=f'Selected tiles ({len(selected_tiles)})')
    ]
    
    # Add eco-region legend patches
    eco_region_patches = []
    for region in sorted(unique_regions):
        if region is not None:
            color = region_colors.get(region, 'gray')
            eco_region_patches.append(
                mpatches.Patch(color=color, alpha=0.6, edgecolor='gray', 
                              label=f'{region}')
            )
    
    # Add tile type legend
    ax.legend(handles=legend_elements, loc='lower right', title="Tile Selection")
    
    # Add eco-region legend to the top right
    if eco_region_patches:
        # Create second legend for eco-regions
        eco_legend = plt.legend(
            handles=eco_region_patches, 
            loc='upper left', 
            bbox_to_anchor=(1, 1), 
            title="Eco-Regions", 
            fontsize=8
        )
        # Add the second legend manually
        ax.add_artist(eco_legend)
    
    # Set title and remove axis ticks for cleaner appearance
    plt.title('Selected Tiles by Eco-Region')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis borders for an even cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_pdf_report(
    selected_tiles: gpd.GeoDataFrame,
    filtered_tiles: gpd.GeoDataFrame,
    eco_regions: gpd.GeoDataFrame,
    visualization_path: str,
    output_path: str,
    min_pixels: int
) -> str:
    """
    Generates a PDF report with key metrics and visualization.
    
    Args:
        selected_tiles: Final selected tiles
        filtered_tiles: Tiles that passed the initial filtering
        eco_regions: Eco-regions polygons
        visualization_path: Path to the visualization image
        output_path: Path to save the PDF report
        min_pixels: Minimum pixel threshold used for filtering
        
    Returns:
        Path to the saved PDF report
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from datetime import datetime
    
    # Create document with portrait orientation
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create title style
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,  # Center alignment
        spaceAfter=20
    )
    
    # Create elements list for the document
    elements = []
    
    # Add title
    elements.append(Paragraph("Tile Selection Report", title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add date
    date_style = ParagraphStyle(
        name='Date',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,  # Center alignment
    )
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {date_str}", date_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Add summary metrics
    elements.append(Paragraph("Summary Metrics", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Calculate metrics
    total_tiles = len(filtered_tiles)
    selected_count = len(selected_tiles)
    
    # Create metrics table
    metrics_data = [
        ["Metric", "Value"],
        ["Total tiles available (after filtering)", str(total_tiles)],
        ["Total tiles selected", str(selected_count)],
        ["Selection ratio", f"{selected_count/total_tiles:.2%}"],
        ["Minimum in-situ pixels threshold", str(min_pixels)]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Calculate distribution by eco-region
    selected_by_region = selected_tiles.groupby('NomSER').size()
    
    # Calculate eco-region areas and forest coverage
    eco_regions['area_km2'] = eco_regions.geometry.area / 1_000_000
    eco_areas = eco_regions.set_index('NomSER')['area_km2'].to_dict()
    
    # Prepare the simplified eco-region distribution table
    elements.append(Paragraph("Eco-Region Distribution", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Prepare data for simplified distribution table
    distribution_data = [["Eco-Region", "Selected Tiles", "Effective Forest Area (km²)", "Tiles per 1000 km² (forest)"]]
    
    for region in sorted(eco_areas.keys(), key=lambda x: str(x) if x is not None else ""):
        if region is None:
            continue
            
        area = eco_areas.get(region, 0)
        selected_count = selected_by_region.get(region, 0)
        
        # Convert region name to English for accessing forest cover ratio
        region_en = MAPPING_ECO_REGIONS_FR_EN.get(region, region)
        
        # Get forest cover ratio for this region (default to 0.5 if not found)
        forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region_en, 0.5)
        
        # Calculate effective forest area
        effective_forest_area = area * forest_ratio
        
        # Calculate density per 1000 km² of effective forest area
        forest_density = (selected_count / effective_forest_area) * 1000 if effective_forest_area > 0 else 0
        
        distribution_data.append([
            str(region),
            str(selected_count),
            f"{effective_forest_area:.2f}",
            f"{forest_density:.2f}"
        ])
    
    # Add total row
    total_area = sum(area for region, area in eco_areas.items() if region is not None)
    total_selected = selected_by_region.sum()
    
    # Calculate total effective forest area
    total_effective_forest = 0
    for region in eco_areas.keys():
        if region is None:
            continue
        area = eco_areas.get(region, 0)
        region_en = MAPPING_ECO_REGIONS_FR_EN.get(region, region)
        forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region_en, 0.5)
        total_effective_forest += area * forest_ratio
    
    total_forest_density = (total_selected / total_effective_forest) * 1000 if total_effective_forest > 0 else 0
    
    distribution_data.append([
        "TOTAL",
        str(total_selected),
        f"{total_effective_forest:.2f}",
        f"{total_forest_density:.2f}"
    ])
    
    # Create table
    col_widths = [1.7*inch, 1.2*inch, 1.8*inch, 1.8*inch]
    distribution_table = Table(distribution_data, colWidths=col_widths)
    distribution_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    
    elements.append(distribution_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add visualization
    elements.append(Paragraph("Visualization", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    img = Image(visualization_path)
    available_width = 7.5*inch  # Width for the image in portrait mode
    img.drawWidth = available_width
    img.drawHeight = (available_width / img.imageWidth) * img.imageHeight  # Maintain aspect ratio
    elements.append(img)
    
    # Add caption
    caption_style = ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=9,
        alignment=1,  # Center alignment
    )
    caption = "Figure 1: Selected tiles (red) overlaid on eco-regions of France. Blue dots represent filtered tiles with sufficient in-situ data."
    elements.append(Paragraph(caption, caption_style))
    
    # Build PDF
    doc.build(elements)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Select 2.5-km tiles over France with complete monthly Sentinel-2 coverage."
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="data/species/val_train_tiles_2_5_km.parquet",
        help="Path to the parquet file containing 2.5km tiles with columns: 'perc', 'NomSER', geometry..."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        default="data/species/ser_l93_new",
        help="Path to the shapefile/GeoJSON of French eco-regions (including 'NomSER')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/datasets/tiles_2_5_km_final.parquet",
        help="Output filename for the final selection of tiles."
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=1000,
        help="Minimum in-situ pixel count required to keep a tile."
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=5000,
        help="Minimum distance (in meters) between selected tiles for spatial distribution."
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/tiles_selection.log",
        help="Path for detailed log file."
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate PDF report with visualization and metrics."
    )

    args = parser.parse_args()
    
    # Set up logging
    log_file = args.log
    # Add timestamp to log file to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_timestamp = log_file.replace('.log', f'_{timestamp}.log')
    logger = setup_logging(log_file_with_timestamp)
    
    logging.info("=== Tile Selection Script ===")
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load data
    logging.info("Loading data...")
    tiles_gdf, eco_gdf = load_data(
        args.tiles, 
        args.ecoregions,
        force_crs="EPSG:2154"  # Force Lambert-93 CRS
    )
    logging.info(f"Loaded {len(tiles_gdf)} total tiles and {len(eco_gdf)} eco-regions")

    # 2. Filter tiles by in-situ coverage
    logging.info(f"Filtering tiles with >= {args.min_pixels} pixels from the in-situ dataset...")
    filtered_tiles = filter_tiles_by_in_situ(tiles_gdf, args.min_pixels)
    logging.info(f"Number of tiles after filtering: {len(filtered_tiles)} (removed {len(tiles_gdf) - len(filtered_tiles)})")

    # 3. Assign these tiles to eco-regions if needed (based on centroid)
    if "NomSER" not in filtered_tiles.columns or filtered_tiles["NomSER"].isna().all():
        logging.info("Assigning tiles to eco-regions based on centroids...")
        filtered_tiles = assign_tiles_to_ecoregions(filtered_tiles, eco_gdf)
        
    # Make sure to assign eco-regions to all tiles for later use
    if "NomSER" not in tiles_gdf.columns or tiles_gdf["NomSER"].isna().all():
        logging.info("Assigning all tiles to eco-regions based on centroids...")
        tiles_gdf = assign_tiles_to_ecoregions(tiles_gdf, eco_gdf)
    
    # Log eco-region distribution
    eco_counts = filtered_tiles.groupby("NomSER").size().sort_values(ascending=False)
    logging.debug("Tiles per eco-region after filtering:")
    for region, count in eco_counts.items():
        if pd.isna(region):
            logging.debug(f"  Unassigned: {count} tiles")
        else:
            logging.debug(f"  {region}: {count} tiles")
    
    # Get eco-region areas
    eco_gdf["area_km2"] = eco_gdf.geometry.area / 1_000_000  # Convert to km²
    eco_areas = eco_gdf.set_index("NomSER")["area_km2"].sort_values(ascending=False)
    logging.debug("Eco-region areas (km²):")
    for region, area in eco_areas.items():
        if pd.isna(region):
            logging.debug(f"  Unassigned: {area:.2f} km²")
        else:
            logging.debug(f"  {region}: {area:.2f} km²")

    # 4. Balance coverage with spatial distribution 
    # - pass both filtered_tiles (high quality) and all tiles (for additional selection)
    # - use a lower min_pixels threshold for additional tiles
    logging.info("Balancing eco-region coverage with spatial distribution...")
    
    # Calculate effective_pixels for all tiles
    tiles_gdf["effective_pixels"] = (tiles_gdf["perc"] / 100.0) * 62500
    
    # Use a lower min_pixels threshold for additional tiles (5% of the main threshold)
    additional_tiles_threshold = max(10, int(args.min_pixels * 0.001))
    logging.info(f"Using threshold of {additional_tiles_threshold} pixels for additional tiles")
    
    final_tiles = balance_coverage_with_spatial_distribution(
        filtered_tiles, 
        eco_gdf,
        all_tiles_gdf=tiles_gdf,
        min_distance=args.min_distance,
        min_pixels=additional_tiles_threshold
    )
    
    # Log final statistics
    logging.info(f"Final selection: {len(final_tiles)} tiles")
    final_eco_counts = final_tiles.groupby("NomSER").size()
    final_eco_distribution = []
    
    # Safe iteration over keys, handling None values
    for region in sorted([r for r in eco_areas.keys() if r is not None], key=str):
        if region in final_eco_counts:
            final_eco_distribution.append(f"{region}: {final_eco_counts[region]}")
    
    logging.info(f"Distribution: {', '.join(final_eco_distribution)}")
    
    # Save the final selection
    logging.info(f"Saving final selection to {args.output}")
    
    # Remove non-serializable columns
    columns_to_drop = ["centroid", "min_dist"]
    for col in columns_to_drop:
        if col in final_tiles.columns:
            final_tiles = final_tiles.drop(columns=[col])
    
    # Make sure the CRS is explicitly set to EPSG:2154 before saving
    if final_tiles.crs is None or final_tiles.crs != "EPSG:2154":
        logging.info("Setting CRS to EPSG:2154 (Lambert-93) before saving")
        final_tiles.set_crs("EPSG:2154", inplace=True)
    
    # Display the head of the final dataset
    logging.info("\nSample of selected tiles:")
    print(final_tiles.head().to_string())
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get output extension
    out_ext = os.path.splitext(args.output)[1].lower()
    
    if out_ext == '.parquet':
        # Save directly to parquet (modern GeoPandas handles geometry columns correctly)
        logging.info("Saving to parquet format with proper geometry serialization")
        try:
            # Try saving with to_parquet
            final_tiles.to_parquet(args.output, index=False)
            
            # Verify the saved file by reading it back
            try:
                logging.info("Verifying the saved parquet file...")
                test_read = gpd.read_parquet(args.output)
                logging.info(f"Successfully verified parquet file. Contains {len(test_read)} tiles.")
                
                # Additional check - count by region
                if 'NomSER' in test_read.columns:
                    region_counts = test_read.groupby('NomSER').size()
                    logging.info(f"Region counts in verified parquet: {dict(region_counts)}")
            except Exception as verify_err:
                logging.error(f"Error verifying parquet file: {verify_err}")
                logging.warning("Saving backup GeoJSON file due to verification failure")
                # Save backup GeoJSON
                backup_geojson = args.output.replace('.parquet', '_backup.geojson')
                final_tiles.to_file(backup_geojson, driver="GeoJSON")
                logging.info(f"Backup GeoJSON saved to {backup_geojson}")
        except Exception as e:
            logging.error(f"Error saving to parquet: {e}")
            # Save as GeoJSON instead
            backup_file = args.output.replace('.parquet', '.geojson')
            logging.warning(f"Saving as GeoJSON instead to {backup_file}")
            final_tiles.to_file(backup_file, driver="GeoJSON")
            logging.info(f"GeoJSON backup saved with {len(final_tiles)} tiles")
            
    elif out_ext in ['.geojson', '.json']:
        # Save to GeoJSON with explicit CRS
        logging.info("Saving to GeoJSON format")
        final_tiles.to_file(args.output, driver="GeoJSON")
    else:
        # Default to GPKG if extension is something else
        logging.info(f"Unrecognized extension '{out_ext}', defaulting to GeoPackage format")
        final_tiles.to_file(args.output.replace(out_ext, '.gpkg'), driver="GPKG")
    
    logging.info(f"Final dataset has {len(final_tiles)} tiles.")
    
    # 7. Create visualization and report
    if args.report:
        # Generate paths for visualization and PDF report
        vis_output = args.output.replace('.parquet', '_visualization.png')
        report_output = args.output.replace('.parquet', '_report.pdf')
        
        # Check if matplotlib and reportlab are available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            logging.info("Creating visualization...")
            vis_path = create_visualization(
                final_tiles,
                filtered_tiles,
                tiles_gdf,
                eco_gdf,
                vis_output
            )
            logging.info(f"Visualization saved to: {vis_path}")
            
            try:
                # Try to import reportlab
                from reportlab.lib.pagesizes import letter
                
                logging.info("Generating PDF report...")
                report_path = generate_pdf_report(
                    final_tiles,
                    filtered_tiles,
                    eco_gdf,
                    vis_path,
                    report_output,
                    args.min_pixels
                )
                
                logging.info(f"PDF report saved to: {report_path}")
            except ImportError:
                logging.warning("reportlab not installed. Cannot generate PDF report.")
                logging.warning("Install with: pip install reportlab")
                
        except ImportError:
            logging.warning("matplotlib not installed. Cannot create visualization.")
            logging.warning("Install with: pip install matplotlib")
    
    logging.info("Done.")
    logging.info(f"Log file saved to: {log_file_with_timestamp}")


if __name__ == "__main__":
    main()
