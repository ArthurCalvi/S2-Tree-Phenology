#!/bin/bash

# Script to run the dataset preparation pipeline
# This extracts features for each tile and adds categorical data from the dataset

# Default values
TILES_PATH="results/datasets/tiles_2_5_km_final.parquet"
DATASET_PATH="results/datasets/final_dataset.parquet"
FEATURES_PATH="data/features"
OUTPUT_DIR="results/training_tiles"
MAPPINGS_PATH="results/training_tiles/categorical_mappings.json"
LOGLEVEL="INFO"
LIMIT=""
TILE_IDS=""
FORCE_CRS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiles)
      TILES_PATH="$2"
      shift 2
      ;;
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --features)
      FEATURES_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mappings)
      MAPPINGS_PATH="$2"
      shift 2
      ;;
    --loglevel)
      LOGLEVEL="$2"
      shift 2
      ;;
    --limit)
      LIMIT="--limit $2"
      shift 2
      ;;
    --tile_ids)
      TILE_IDS="--tile_ids $2"
      shift 2
      ;;
    --force_crs)
      FORCE_CRS="--force_crs $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting dataset preparation..."
echo "Tiles: $TILES_PATH"
echo "Dataset: $DATASET_PATH"
echo "Features: $FEATURES_PATH"
echo "Output: $OUTPUT_DIR"

python src/sampling/dataset_preparation.py \
  --tiles_path "$TILES_PATH" \
  --dataset_path "$DATASET_PATH" \
  --features_path "$FEATURES_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mappings_path "$MAPPINGS_PATH" \
  --loglevel "$LOGLEVEL" \
  $FORCE_CRS \
  $LIMIT $TILE_IDS

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Dataset preparation completed successfully!"
  echo "Output saved to: $OUTPUT_DIR"
  echo "Mappings saved to: $MAPPINGS_PATH"
else
  echo "Dataset preparation failed!"
fi 