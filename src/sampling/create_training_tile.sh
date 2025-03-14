#!/bin/bash
#SBATCH --job-name=cr-tt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1         # For parallel processing within each tile
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00             # 10 hours max
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.err

echo "Running Training tile creation" 
set -x 

# Load necessary modules
source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module unload gcc/10.1.0
module load gcc/11.3.1
module load gdal

# Define paths
REPO_DIR="$WORK/S2-Tree-Phenology"
TILES_PATH="$REPO_DIR/results/datasets/tiles_2_5_km_final.parquet"
DATASET_PATH="$REPO_DIR/results/datasets/final_dataset.parquet"
FEATURES_DIR="$SCRATCH/features2023"
OUTPUT_DIR="$SCRATCH/training_tiles2023"
MAPPINGS_PATH="$SCRATCH/training_tiles2023/categorical_mappings.json"
FEATURES_VRT="$SCRATCH/features2023/features.vrt"
LOGLEVEL="DEBUG"

# Optional arguments that can be passed to the script
LIMIT=""
TILE_IDS=""
FORCE_CRS=""
VERBOSE=""

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
    --force_crs)
      FORCE_CRS="--force_crs $2"
      shift 2
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname $MAPPINGS_PATH)"
echo "Features directory: $FEATURES_DIR"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Create VRT file for features
echo "=== Creating VRT file for features ==="
cd "$REPO_DIR"

python src/sampling/create_features_vrt.py \
  --features_dir "$FEATURES_DIR" \
  --output_vrt "$FEATURES_VRT" \
  --loglevel "$LOGLEVEL" \
  $VERBOSE

# Check if VRT creation was successful
if [ $? -ne 0 ]; then
  echo "VRT file creation failed!"
  exit 1
fi
echo "VRT file created successfully at $FEATURES_VRT"

# Step 2: Run dataset preparation using the VRT file
echo "=== Running dataset preparation ==="
echo "Tiles: $TILES_PATH"
echo "Dataset: $DATASET_PATH"
echo "Features VRT: $FEATURES_VRT"
echo "Output: $OUTPUT_DIR"

python src/sampling/dataset_preparation.py \
  --tiles_path "$TILES_PATH" \
  --dataset_path "$DATASET_PATH" \
  --features_path "$FEATURES_VRT" \
  --output_dir "$OUTPUT_DIR" \
  --mappings_path "$MAPPINGS_PATH" \
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