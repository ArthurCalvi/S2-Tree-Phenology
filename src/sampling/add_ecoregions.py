#!/usr/bin/env python3
"""
Script to add ecoregion information (NomSER column in English) to the species and BD Forêt datasets.
This script performs spatial joins between the datasets and the ecoregion polygons,
then adds the NomSER column with English names using the mapping dictionaries.
"""

import os
import sys
import argparse
import logging
import geopandas as gpd
from tabulate import tabulate

# Add the project root directory to the Python path to make imports work
# regardless of where the script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.constants import MAPPING_SER_ECO_REGIONS, MAPPING_ECO_REGIONS_FR_EN
except ImportError:
    # If that fails, try a relative import
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from src.constants import MAPPING_SER_ECO_REGIONS, MAPPING_ECO_REGIONS_FR_EN

# Define the mappings directly in case imports still fail
if 'MAPPING_SER_ECO_REGIONS' not in globals():
    MAPPING_SER_ECO_REGIONS = {
        'Côtes_et_plateaux_de_la_Manche': 'Centre Nord semi-océanique',
        'Ardenne_primaire': 'Grand Est semi-continental',
        'Préalpes_du_Nord': 'Alpes',
        'Garrigues': 'Méditerranée',
        'Massif_vosgien_central': 'Vosges',
        'Premier_plateau_du_Jura': 'Jura',
        'Piémont_pyrénéen': 'Pyrénées',
        'Terres_rouges': 'Sud-Ouest océanique',
        'Corse_occidentale': 'Corse',
        "Châtaigneraie_du_Centre_et_de_l'Ouest": 'Massif central',
        'Ouest-Bretagne_et_Nord-Cotentin': 'Grand Ouest cristallin et océanique',
        'Total': 'Total'
    }

if 'MAPPING_ECO_REGIONS_FR_EN' not in globals():
    MAPPING_ECO_REGIONS_FR_EN = {
        "Grand Ouest cristallin et océanique": "Greater Crystalline and Oceanic West",
        "Centre Nord semi-océanique": "Semi-Oceanic North Center",
        "Grand Est semi-continental": "Greater Semi-Continental East",
        "Vosges": "Vosges",
        "Jura": "Jura",
        "Sud-Ouest océanique": "Oceanic Southwest",
        "Massif central": "Central Massif",
        "Alpes": "Alps",
        "Pyrénées": "Pyrenees",
        "Méditerranée": "Mediterranean",
        "Corse": "Corsica"
    }

def setup_logging(log_file=None):
    """Set up logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler with minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

def load_ecoregions(ecoregion_path: str) -> gpd.GeoDataFrame:
    """
    Load and prepare the ecoregion GeoDataFrame.
    
    Args:
        ecoregion_path: Path to the ecoregion shapefile/GeoJSON
        
    Returns:
        Prepared ecoregion GeoDataFrame with standardized NomSER column
    """
    eco_gdf = gpd.read_file(ecoregion_path)
    
    # Convert ecoregions to "greco" if 'codeser' exists, then dissolve
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
    
    logging.info(f"Loaded ecoregions with names: {eco_gdf['NomSER'].unique()}")
    return eco_gdf

def assign_to_ecoregions(gdf: gpd.GeoDataFrame, ecoregions_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assigns each geometry in the input GeoDataFrame to an eco-region based on centroid.

    Args:
        gdf: Input GeoDataFrame to be assigned ecoregions
        ecoregions_gdf: Polygons for eco-regions

    Returns:
        GeoDataFrame with an added 'NomSER' column for the eco-region name in English
    """
    # Ensure same CRS
    if gdf.crs != ecoregions_gdf.crs:
        gdf = gdf.to_crs(ecoregions_gdf.crs)
        logging.info(f"Reprojected input data to match ecoregion CRS: {ecoregions_gdf.crs}")
    
    # Compute centroids
    centroids = gdf.geometry.centroid
    
    # Spatial join with eco-regions
    centroids_gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), 
                                     geometry=centroids, 
                                     crs=gdf.crs)
    joined = gpd.sjoin(centroids_gdf, ecoregions_gdf, how="left", predicate="within")
    
    # Add NomSER column to original GeoDataFrame
    result = gdf.copy()
    result["NomSER"] = joined["NomSER"]
    
    # Count points per region
    region_counts = result["NomSER"].value_counts()
    logging.info("Distribution of points across ecoregions:")
    for region, count in region_counts.items():
        logging.info(f"  {region}: {count} points")
    
    # Check for unassigned points
    unassigned = result[result["NomSER"].isna()]
    if len(unassigned) > 0:
        logging.warning(f"Warning: {len(unassigned)} points could not be assigned to any ecoregion")
        
    return result

def process_datasets(species_path: str, bdforet_path: str, ecoregion_path: str, 
                     output_species: str, output_bdforet: str) -> None:
    """
    Process both datasets to add ecoregion information.
    
    Args:
        species_path: Path to the species dataset
        bdforet_path: Path to the BD Forêt dataset
        ecoregion_path: Path to the ecoregion dataset
        output_species: Path to save the processed species dataset
        output_bdforet: Path to save the processed BD Forêt dataset
    """
    # Load ecoregions
    logging.info(f"Loading ecoregions from {ecoregion_path}")
    ecoregions_gdf = load_ecoregions(ecoregion_path)
    
    # Process species dataset
    logging.info(f"Processing species dataset from {species_path}")
    species_gdf = gpd.read_file(species_path)
    logging.info(f"Species dataset has {len(species_gdf)} rows and columns: {list(species_gdf.columns)}")
    
    species_with_regions = assign_to_ecoregions(species_gdf, ecoregions_gdf)
    
    # Process BD Forêt dataset
    logging.info(f"Processing BD Forêt dataset from {bdforet_path}")
    if bdforet_path.endswith('.parquet'):
        bdforet_gdf = gpd.read_parquet(bdforet_path)
    else:
        bdforet_gdf = gpd.read_file(bdforet_path)
    logging.info(f"BD Forêt dataset has {len(bdforet_gdf)} rows and columns: {list(bdforet_gdf.columns)}")
    
    bdforet_with_regions = assign_to_ecoregions(bdforet_gdf, ecoregions_gdf)
    
    # Save results
    os.makedirs(os.path.dirname(output_species), exist_ok=True)
    os.makedirs(os.path.dirname(output_bdforet), exist_ok=True)
    
    # Save species dataset
    if output_species.endswith('.parquet'):
        species_with_regions.to_parquet(output_species)
    else:
        species_with_regions.to_file(output_species)
    logging.info(f"Saved species dataset with ecoregions to {output_species}")
    
    # Save BD Forêt dataset
    if output_bdforet.endswith('.parquet'):
        bdforet_with_regions.to_parquet(output_bdforet)
    else:
        bdforet_with_regions.to_file(output_bdforet)
    logging.info(f"Saved BD Forêt dataset with ecoregions to {output_bdforet}")
    
    # Display sample of results
    print("\nSample of species dataset with ecoregions:")
    species_sample = species_with_regions.head(5).copy()
    if 'geometry' in species_sample.columns:
        # Create a new column for displaying geometry instead of modifying the geometry column
        species_sample['geometry_str'] = species_sample['geometry'].apply(lambda x: str(x)[:50] + '...' if x else None)
        display_cols = ['NomSER'] + [col for col in species_sample.columns if col != 'NomSER' and col != 'geometry' and col != 'geometry_str'][:4] + ['geometry_str']
    else:
        display_cols = ['NomSER'] + [col for col in species_sample.columns if col != 'NomSER'][:4]
    
    print(tabulate(species_sample[display_cols], headers='keys', tablefmt='grid', showindex=True))
    
    print("\nSample of BD Forêt dataset with ecoregions:")
    bdforet_sample = bdforet_with_regions.head(5).copy()
    if 'geometry' in bdforet_sample.columns:
        # Create a new column for displaying geometry instead of modifying the geometry column
        bdforet_sample['geometry_str'] = bdforet_sample['geometry'].apply(lambda x: str(x)[:50] + '...' if x else None)
        display_cols = ['NomSER'] + [col for col in bdforet_sample.columns if col != 'NomSER' and col != 'geometry' and col != 'geometry_str'][:4] + ['geometry_str']
    else:
        display_cols = ['NomSER'] + [col for col in bdforet_sample.columns if col != 'NomSER'][:4]
    
    print(tabulate(bdforet_sample[display_cols], headers='keys', tablefmt='grid', showindex=True))

def main():
    parser = argparse.ArgumentParser(
        description="Add ecoregion information (NomSER column in English) to species and BD Forêt datasets"
    )
    parser.add_argument(
        "--species",
        default="data/species/in-situ dataset/france_species.shp",
        help="Path to the in-situ species dataset"
    )
    parser.add_argument(
        "--bdforet",
        default="data/species/bdforet_10_FF1_FF2_EN_year.parquet",
        help="Path to the BD Forêt dataset"
    )
    parser.add_argument(
        "--ecoregions",
        default="data/species/ser_l93_new",
        help="Path to the ecoregion shapefile/GeoJSON"
    )
    parser.add_argument(
        "--output-species",
        default="data/species/processed/france_species_with_ecoregions.parquet",
        help="Output path for the processed species dataset"
    )
    parser.add_argument(
        "--output-bdforet",
        default="data/species/processed/bdforet_with_ecoregions.parquet",
        help="Output path for the processed BD Forêt dataset"
    )
    parser.add_argument(
        "--log",
        default="logs/add_ecoregions.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger = setup_logging(args.log)
    
    # Check if files exist
    for path, name in [(args.species, "Species dataset"), 
                       (args.bdforet, "BD Forêt dataset"), 
                       (args.ecoregions, "Ecoregions dataset")]:
        if not os.path.exists(path):
            logging.error(f"Error: {name} file not found at {path}")
            sys.exit(1)
    
    # Process datasets
    process_datasets(
        args.species, 
        args.bdforet, 
        args.ecoregions,
        args.output_species,
        args.output_bdforet
    )
    
    logging.info("Processing complete!")

if __name__ == "__main__":
    main() 