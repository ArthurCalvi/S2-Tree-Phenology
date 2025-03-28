#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to add sample weights to a training dataset based on eco-region distribution.

This script computes weights for each sample in the dataset to balance the 
training across eco-regions. Weights are calculated based on the effective 
forest area of each eco-region and the number of pixels available in the dataset.

Author: Arthur
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.constants import EFFECTIVE_FOREST_AREA_BY_REGION

# Set up logging
def setup_logging(log_file=None):
    """Set up logging to file and console."""
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler with minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def compute_weights(df, eco_region_col="eco_region"):
    """
    Compute weights for each pixel based on eco-region distribution.
    
    Args:
        df: DataFrame containing the training data
        eco_region_col: Name of the column containing eco-region names
        
    Returns:
        DataFrame with an additional 'weight' column
    """
    # Count samples per eco-region
    eco_region_counts = df[eco_region_col].value_counts()
    total_samples = len(df)
    
    # Create a dictionary to map eco-regions to weights
    eco_region_weights = {}
    
    # Calculate the total effective forest area across all eco-regions
    total_effective_forest_area = sum(EFFECTIVE_FOREST_AREA_BY_REGION.values())
    
    logging.info("\nCalculating weights for each eco-region:")
    logging.info(f"{'Eco-region':<35} {'Samples':<10} {'% of Dataset':<15} {'% of Forest Area':<20} {'Weight':<10}")
    logging.info("-" * 90)
    
    for eco_region in eco_region_counts.index:
        # Get count and percentage of dataset for this eco-region
        count = eco_region_counts[eco_region]
        dataset_fraction = count / total_samples
        
        # Get effective forest area and its percentage of total
        effective_area = EFFECTIVE_FOREST_AREA_BY_REGION.get(eco_region, 0)
        area_fraction = effective_area / total_effective_forest_area if total_effective_forest_area > 0 else 0
        
        # Calculate weight: ideal fraction (by area) / actual fraction (in dataset)
        # This gives higher weights to underrepresented regions and lower weights to overrepresented ones
        weight = area_fraction / dataset_fraction if dataset_fraction > 0 else 1.0
        
        # Store in dictionary
        eco_region_weights[eco_region] = weight
        
        # Log information
        logging.info(f"{eco_region:<35} {count:<10} {dataset_fraction*100:>6.2f}%        "
                    f"{area_fraction*100:>6.2f}%             {weight:>6.4f}")
    
    # Apply weights to the dataset
    df['weight'] = df[eco_region_col].map(eco_region_weights)
    
    # Normalize weights so they sum to the number of samples
    # This keeps the effective sample size the same
    df['weight'] = df['weight'] * (total_samples / df['weight'].sum())
    
    logging.info("\nWeight statistics:")
    logging.info(f"Min weight: {df['weight'].min():.4f}")
    logging.info(f"Max weight: {df['weight'].max():.4f}")
    logging.info(f"Mean weight: {df['weight'].mean():.4f}")
    logging.info(f"Sum of weights: {df['weight'].sum():.1f} (should be close to {total_samples})")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Add sample weights to training dataset based on eco-region distribution."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/datasets/training_dataset_w_ecoregion.parquet",
        help="Path to the input parquet file containing the training dataset with eco-region information."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/datasets/training_datasets_pixels.parquet",
        help="Output path for the dataset with weights added."
    )
    parser.add_argument(
        "--eco-region-col",
        type=str,
        default="eco_region",
        help="Name of the column containing eco-region information."
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/add_weights.log",
        help="Path for log file."
    )

    args = parser.parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log.replace('.log', f'_{timestamp}.log')
    logger = setup_logging(log_file)
    
    # Log start
    logging.info("=== Adding Sample Weights to Training Dataset ===")
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load dataset
    logging.info(f"Loading dataset from {args.input}")
    try:
        df = pd.read_parquet(args.input)
        logging.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Check if eco-region column exists
    if args.eco_region_col not in df.columns:
        logging.error(f"Error: Column '{args.eco_region_col}' not found in dataset")
        logging.info(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Compute and add weights
    logging.info("Computing weights based on eco-region distribution...")
    df = compute_weights(df, args.eco_region_col)
    
    # Save output
    logging.info(f"Saving dataset with weights to {args.output}")
    try:
        df.to_parquet(args.output, index=False)
        logging.info(f"Successfully saved {len(df)} samples with weights")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")
        sys.exit(1)
    
    logging.info("Done!")
    logging.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 