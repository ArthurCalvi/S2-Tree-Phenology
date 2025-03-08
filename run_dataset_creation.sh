#!/bin/bash

# Run the dataset creation script with the correct file paths
# Using the GeoJSON version of the tiles file since the parquet version has issues

python -m src.sampling.dataset_creation \
  --tiles results/datasets/tiles_2_5_km_final.geojson \
  --species data/species/processed/france_species_with_ecoregions.parquet \
  --bdforet data/species/processed/bdforet_with_ecoregions.parquet \
  --output results/datasets/final_dataset.parquet \
  --loglevel INFO \
  --work_dir results/datasets/intermediate

# If you want to process only specific regions, uncomment and modify the following line:
# --regions "Alps,Vosges,Jura"

# If you want to resume processing from where it left off, add the --resume flag:
# --resume

# If you want to force reprocessing of all regions, add the --force flag:
# --force

# If you want to change the buffer distance (default is 100m), add the --buffer flag:
# --buffer 200.0 