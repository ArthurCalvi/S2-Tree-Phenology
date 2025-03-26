#!/usr/bin/env python3
"""
This script adds an 'eco_region' column to the training_dataset_w_corsica.parquet file
by joining it with the final_dataset.parquet file on the tile_id column.

The eco-region information comes from the NomSER column in the final_dataset.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_eco_region():
    """
    Add eco-region column to the training dataset by joining with the final dataset.
    """
    logger.info("Loading datasets...")
    
    # Define file paths
    training_dataset_path = 'results/datasets/training_dataset_w_corsica.parquet'
    final_dataset_path = 'results/datasets/final_dataset.parquet'
    output_path = 'results/datasets/training_dataset_w_ecoregion.parquet'
    
    # Load datasets
    logger.info(f"Loading training dataset from {training_dataset_path}")
    training_df = pd.read_parquet(training_dataset_path)
    
    logger.info(f"Loading final dataset from {final_dataset_path}")
    final_df = pd.read_parquet(final_dataset_path)
    
    # Create a mapping of tile_id to eco-region (NomSER)
    logger.info("Creating tile_id to eco-region mapping...")
    # Extract unique tile_id to NomSER mappings
    tile_to_ecoregion = final_df[['tile_id', 'NomSER']].drop_duplicates()
    
    # Check if there are multiple eco-regions per tile_id
    tile_counts = tile_to_ecoregion['tile_id'].value_counts()
    multi_region_tiles = tile_counts[tile_counts > 1].index
    
    if len(multi_region_tiles) > 0:
        logger.warning(f"Found {len(multi_region_tiles)} tiles with multiple eco-regions. Using the most common eco-region for each.")
        # For tiles with multiple eco-regions, take the most common one
        def get_most_common_region(tile_id):
            regions = tile_to_ecoregion[tile_to_ecoregion['tile_id'] == tile_id]['NomSER']
            return regions.value_counts().index[0]
            
        # Create a clean mapping with one eco-region per tile
        clean_mapping = {}
        for tile_id in tile_to_ecoregion['tile_id'].unique():
            clean_mapping[tile_id] = get_most_common_region(tile_id)
        
        # Convert to dataframe
        tile_to_ecoregion = pd.DataFrame({
            'tile_id': list(clean_mapping.keys()),
            'eco_region': list(clean_mapping.values())
        })
    else:
        # Rename NomSER to eco_region for clarity
        tile_to_ecoregion = tile_to_ecoregion.rename(columns={'NomSER': 'eco_region'})
    
    # Add eco-region to the training dataset
    logger.info("Adding eco-region column to training dataset...")
    
    # Check how many unique tile_ids in training dataset
    training_tile_ids = training_df['tile_id'].unique()
    logger.info(f"Training dataset has {len(training_tile_ids)} unique tile_ids")
    
    # Check how many of these are in our mapping
    mapped_tiles = set(training_tile_ids).intersection(set(tile_to_ecoregion['tile_id']))
    logger.info(f"Found eco-region mapping for {len(mapped_tiles)} out of {len(training_tile_ids)} tile_ids")
    
    # Add the eco-region column through a merge operation
    result_df = training_df.merge(tile_to_ecoregion, on='tile_id', how='left')
    
    # Check for unmapped tile_ids
    unmapped_mask = result_df['eco_region'].isna()
    unmapped_count = unmapped_mask.sum()
    
    if unmapped_count > 0:
        logger.warning(f"{unmapped_count} rows ({unmapped_count/len(result_df)*100:.2f}%) have no eco-region mapping")
        # Optionally, fill with a default value
        result_df.loc[unmapped_mask, 'eco_region'] = 'Unknown'
    
    # Save the resulting dataset
    logger.info(f"Saving result to {output_path}")
    result_df.to_parquet(output_path, index=False)
    
    logger.info(f"Done! Added eco-region column to training dataset")
    logger.info(f"Output saved to: {output_path}")
    
    # Print some stats about the resulting dataset
    logger.info(f"Resulting dataset has {len(result_df)} rows and {len(result_df.columns)} columns")
    logger.info(f"Eco-region distribution:")
    for region, count in result_df['eco_region'].value_counts().items():
        logger.info(f"  {region}: {count} rows ({count/len(result_df)*100:.2f}%)")

if __name__ == "__main__":
    add_eco_region() 