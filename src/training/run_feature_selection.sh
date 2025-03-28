#!/bin/bash

# Feature Selection for Sentinel-2 Tree Phenology Classification
# This script runs various feature selection methods to find the best subset of features
# Author: Generated with Claude AI
# Date: April 2024

# Set the base output directory
BASE_OUTPUT_DIR="results/feature_selection"

# Make sure directories exist
mkdir -p $BASE_OUTPUT_DIR
mkdir -p logs

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

echo "Starting feature selection analysis..."

# Option 1: Run all methods with full dataset
# This will take a long time but provides comprehensive results
echo "Running comprehensive feature selection (this may take several hours)..."
python src/training/feature_selection.py --method all --output "${BASE_OUTPUT_DIR}/comprehensive"

# Option 2: Run with a sample to get quick results
echo "Running quick analysis with a sample of 100,000 pixels..."
python src/training/feature_selection.py --method incremental --sample_size 100000 --output "${BASE_OUTPUT_DIR}/quick_sample"

# Option 3: Run only RFE with different feature counts
echo "Running RFE with different feature counts..."
for n_features in 5 10 15 20; do
    echo "Testing with $n_features features..."
    python src/training/feature_selection.py --method rfe --n_features $n_features --output "${BASE_OUTPUT_DIR}/rfe_${n_features}"
done

# Option 4: Run by eco-region (create a separate subset for each eco-region)
echo "Running analysis by eco-region..."
python - << EOF
import pandas as pd
import os

# Load the dataset
df = pd.read_parquet('results/datasets/training_datasets_pixels.parquet')

# Create eco-region based outputs
eco_regions = df['eco_region'].unique()
for eco_region in eco_regions:
    # Create a filtered dataset
    eco_df = df[df['eco_region'] == eco_region]
    
    # Skip eco-regions with too few samples
    if len(eco_df) < 1000:
        print(f"Skipping eco-region {eco_region} with only {len(eco_df)} samples.")
        continue
    
    # Save as parquet for faster processing
    eco_file = f'results/datasets/eco_{eco_region.replace(" ", "_")}.parquet'
    eco_df.to_parquet(eco_file, index=False)
    
    print(f"Created dataset for eco-region {eco_region} with {len(eco_df)} samples.")
EOF

# Run feature selection for each eco-region
for eco_file in results/datasets/eco_*.parquet; do
    eco_name=$(basename $eco_file .parquet | sed 's/eco_//')
    echo "Running feature selection for eco-region: $eco_name"
    
    # Use a custom dataset path and output directory
    python src/training/feature_selection.py \
        --method incremental \
        --output "${BASE_OUTPUT_DIR}/eco_${eco_name}" \
        --dataset_path "$eco_file"
done

echo "Feature selection analysis completed!" 