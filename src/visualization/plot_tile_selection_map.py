#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np
import os
import sys
import logging

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
from src.constants import MAPPING_SER_ECO_REGIONS, MAPPING_ECO_REGIONS_FR_EN
from src.utils import science_style
except ImportError:
    logging.error("Could not import constants. Make sure src/constants.py exists and the script is run from the project root or PYTHONPATH is set.")
    # Provide fallback empty dicts if import fails, though processing will likely fail
    MAPPING_SER_ECO_REGIONS = {}
    MAPPING_ECO_REGIONS_FR_EN = {}

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_ecoregions(eco_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocesses eco-regions GeoDataFrame to match the 11 main regions with English names."""
    logging.info("Preprocessing eco-regions...")
    eco_gdf = eco_gdf.copy()
    processed = False

    # 1) Dissolve based on 'greco' if 'codeser' exists
    if 'codeser' in eco_gdf.columns:
        logging.debug("Found 'codeser' column, creating 'greco' and dissolving...")
        eco_gdf['greco'] = eco_gdf['codeser'].apply(
            lambda x: x[0] if isinstance(x, str) and len(x) > 0 else x
        )
        # Preserve necessary columns during dissolve, EXCLUDING geometry (dissolve handles it)
        agg_funcs = {col: 'first' for col in eco_gdf.columns if col not in ['greco', 'geometry']}
        # agg_funcs['geometry'] = lambda x: x.unary_union # Let dissolve handle geometry
        
        # Perform dissolve - geometry is automatically aggregated with unary_union
        eco_gdf = eco_gdf.dissolve(by='greco', aggfunc=agg_funcs).reset_index()
        if 'greco' in eco_gdf.columns and len(eco_gdf) > 1:
            # Sometimes an initial row with index 0 might be invalid after dissolve
            if eco_gdf.iloc[0].geometry is None or eco_gdf.iloc[0]['NomSER'] is None:
                 eco_gdf = eco_gdf.iloc[1:]
        processed = True
        logging.debug(f"Dissolved into {len(eco_gdf)} regions based on 'greco'.")

    # 2) Standardize 'NomSER' using mappings from constants
    if 'NomSER' in eco_gdf.columns:
        logging.debug("Standardizing 'NomSER' using constants...")
        # First map detailed french names to broader french categories if needed
        if MAPPING_SER_ECO_REGIONS:
             eco_gdf['NomSER'] = eco_gdf['NomSER'].apply(
                 lambda x: MAPPING_SER_ECO_REGIONS.get(str(x).replace(' ', '_'), str(x)) if pd.notna(x) else x
             )
        # Then map broader french categories to English names
        if MAPPING_ECO_REGIONS_FR_EN:
             eco_gdf['NomSER'] = eco_gdf['NomSER'].apply(
                 lambda x: MAPPING_ECO_REGIONS_FR_EN.get(str(x), str(x)) if pd.notna(x) else x
             )
        logging.debug("Applied French to English mapping to 'NomSER'.")
        processed = True
    else:
        logging.warning("Column 'NomSER' not found in eco-regions. Cannot standardize names.")

    if not processed:
        logging.warning("Could not preprocess eco-regions (no 'codeser' or 'NomSER' found?). Using as is.")

    # Drop intermediate 'greco' column if it exists
    if 'greco' in eco_gdf.columns:
        eco_gdf = eco_gdf.drop(columns=['greco'])

    # Final check for valid regions
    eco_gdf = eco_gdf.dropna(subset=['NomSER', 'geometry'])
    logging.info(f"Preprocessing complete. Resulting eco-regions: {eco_gdf['NomSER'].unique().tolist()}")
    return eco_gdf

def create_visualization(
    selected_tiles: gpd.GeoDataFrame,
    eco_regions: gpd.GeoDataFrame,
    output_path: str,
    selected_tiles_label: str = "Selected Tiles"
) -> str:
    """
    Creates a publication-ready visualization of selected tiles overlaid on eco-regions.

    Args:
        selected_tiles: Final selected tiles GeoDataFrame.
        eco_regions: Preprocessed eco-regions polygons GeoDataFrame (expects 'NomSER' column with standardized English names).
        output_path: Path to save the visualization.
        selected_tiles_label: Label for selected tiles (not shown in legend by default).

    Returns:
        Path to the saved visualization.
    """
    logging.info("Starting visualization creation...")

    with science_style():
        fig = plt.figure(figsize=(10, 10.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[8.5, 1.5], hspace=0.05)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.axis('off')

        num_regions = len(eco_regions['NomSER'].unique())
        pastel_colors = [
            "#c6def1", "#fadadd", "#e6f5c9", "#e0cef1", "#f1e1c6", "#def1c6",
            "#f1d4c6", "#c6f1e7", "#f5e2c9", "#f1c6d8", "#d9f1c6",
        ]
        colors_list = (pastel_colors * (1 + num_regions // len(pastel_colors)))[:num_regions]
        cmap = ListedColormap(colors_list)

        logging.info(f"Plotting {len(eco_regions)} processed eco-regions...")
        eco_regions.plot(
            column='NomSER',
            ax=ax_map,
            alpha=0.8,
            cmap=cmap,
            edgecolor='white',
            linewidth=1.0
        )

        unique_regions = sorted([str(r) for r in eco_regions['NomSER'].unique() if r is not None])
        region_colors = {region: colors_list[i % len(colors_list)] for i, region in enumerate(unique_regions)}

        bounds = eco_regions.total_bounds
        x_min, y_min, x_max, y_max = bounds
        margin = 0.05
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        ax_map.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_map.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_map.set_aspect('equal', adjustable='box')

        logging.info(f"Plotting {len(selected_tiles)} selected tiles with black edges...")
        selected_tiles.plot(
            ax=ax_map,
            color='black',
            markersize=7,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )

        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(False)
        ax_map.set_title('Tile Distribution by Eco-Region', fontsize=14, pad=10)

        logging.info("Creating eco-region legend...")
        num_eco_regions = len(unique_regions)
        ncols = max(4, (num_eco_regions + 2) // 3)
        eco_region_patches = [
            mpatches.Patch(color=region_colors.get(region, 'gray'), alpha=0.8, edgecolor='white', linewidth=0.5, label=str(region))
            for region in unique_regions
        ]
        eco_legend = ax_legend.legend(
            handles=eco_region_patches,
            loc='center',
            title="Eco-Regions",
            ncol=ncols,
            fontsize=9,
            title_fontsize=11,
            frameon=True,
            framealpha=0.95,
            edgecolor='black',
            labelspacing=0.8,
            columnspacing=1.0
        )
        ax_legend.add_artist(eco_legend)

        fig.tight_layout(rect=[0, 0.03, 1, 0.97], pad=1.0)

        logging.info(f"Saving visualization to {output_path}...")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    logging.info("Visualization saved successfully.")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create a publication-ready map of selected tiles.")
    parser.add_argument(
        "--selected_tiles",
        type=str,
        required=True,
        help="Path to the GeoDataFrame file (e.g., Parquet) of selected tiles."
    )
    parser.add_argument(
        "--ecoregions",
        type=str,
        required=True,
        help="Path to the GeoDataFrame file (e.g., Shapefile, GeoJSON) of eco-regions. Must contain a 'NomSER' column."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output visualization (e.g., 'map.png')."
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:2154",
        help="Target Coordinate Reference System (e.g., 'EPSG:2154' for Lambert-93). Data will be projected if needed."
    )

    args = parser.parse_args()

    # Load data
    logging.info(f"Loading selected tiles from {args.selected_tiles}")
    selected_tiles_gdf = gpd.read_parquet(args.selected_tiles)

    logging.info(f"Loading eco-regions from {args.ecoregions}")
    eco_regions_gdf = gpd.read_file(args.ecoregions)

    # Ensure CRS consistency before preprocessing
    target_crs = args.crs
    logging.info(f"Ensuring all layers are in CRS: {target_crs}")
    if selected_tiles_gdf.crs != target_crs:
        logging.info(f"Projecting selected tiles to {target_crs}...")
        selected_tiles_gdf = selected_tiles_gdf.to_crs(target_crs)
    if eco_regions_gdf.crs != target_crs:
        logging.info(f"Projecting eco-regions to {target_crs}...")
        eco_regions_gdf = eco_regions_gdf.to_crs(target_crs)

    # Preprocess eco-regions
    processed_eco_regions_gdf = preprocess_ecoregions(eco_regions_gdf)

    # Check for 'NomSER' column after preprocessing
    if 'NomSER' not in processed_eco_regions_gdf.columns:
        logging.error("Eco-regions file must contain a 'NomSER' column, even after preprocessing.")
        return
    if processed_eco_regions_gdf.empty:
         logging.error("Preprocessing resulted in empty eco-regions GeoDataFrame.")
         return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
         os.makedirs(output_dir, exist_ok=True)

    # Create visualization
    create_visualization(
        selected_tiles=selected_tiles_gdf,
        eco_regions=processed_eco_regions_gdf,
        output_path=args.output
    )

if __name__ == "__main__":
    main() 
