#!/bin/bash

# Master script to run the dataset preparation workflow
# Runs the dataset preparation script to create training tiles

set -e  # Exit on error

# Default values
TILES_PATH="results/datasets/tiles_2_5_km_final.parquet"
DATASET_PATH="results/datasets/final_dataset.parquet"
FEATURES_DIR="data/features"
OUTPUT_DIR="results/training_tiles"
MAPPINGS_PATH="results/training_tiles/categorical_mappings.json"
LOGLEVEL="INFO"
LIMIT=""
TILE_IDS=""
VERBOSE=""
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
      FEATURES_DIR="$2"
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
    --verbose)
      VERBOSE="--verbose"
      shift
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

# Run the dataset preparation script
echo "=== Running dataset preparation ==="
echo "Tiles: $TILES_PATH"
echo "Dataset: $DATASET_PATH"
echo "Features: $FEATURES_DIR"
echo "Output: $OUTPUT_DIR"

src/sampling/prepare_training_dataset.sh \
  --tiles "$TILES_PATH" \
  --dataset "$DATASET_PATH" \
  --features "$FEATURES_DIR" \
  --output "$OUTPUT_DIR" \
  --mappings "$MAPPINGS_PATH" \
  --loglevel "$LOGLEVEL" \
  $FORCE_CRS \
  $LIMIT $TILE_IDS

# Check if the dataset preparation was successful
if [ $? -ne 0 ]; then
  echo "Dataset preparation failed!"
  exit 1
fi

echo ""
echo "=== Workflow completed successfully! ==="
echo "Output saved to: $OUTPUT_DIR"
echo "Mappings saved to: $MAPPINGS_PATH" 