#!/bin/bash
#SBATCH --job-name=rf-inf       # Job name for RF inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2      # RF inference is likely CPU-bound per tile, no internal parallelism needed here
#SBATCH --partition=prepost   # Adjust partition if needed
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00         # Adjust time estimate (e.g., 2 hours)
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.out # Adjust log path
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.err  # Adjust log path
#SBATCH --array=0-80%2        # ADJUST: Set XXX to (number of feature tiles - 1), YY to concurrency limit

echo "### Running RF Phenology Inference - Task ${SLURM_ARRAY_TASK_ID} ###"
set -x

source $HOME/.bashrc
module load gdal # Ensure necessary modules are loaded (GDAL likely needed)
# Add other modules if your environment requires them (e.g., Python environment activation)

# --- User Configuration ---
INPUT_DIR="$SCRATCH/features2023"  # ADJUST: Path to feature tiles
OUTPUT_DIR="$SCRATCH/rf_inference_phenology" # ADJUST: Path for output probability maps
MODEL_PATH="$WORK/S2-Tree-Phenology/results/final_model/phenology_model_selected_features_20250411.joblib"             # ADJUST: Path to your .joblib model file
BLOCK_SIZE=2048                           # Processing block size within a tile
# --- End User Configuration ---

# Validate paths (optional but recommended)
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found."
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file '$MODEL_PATH' not found."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run RF inference for the assigned tile index
python $WORK/S2-Tree-Phenology/src/inference/inference_rf_on_tiles.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL_PATH}" \
    --tile-idx ${SLURM_ARRAY_TASK_ID} \
    --block-size ${BLOCK_SIZE} \
    # --workers argument is ignored when --tile-idx is set, so no need to specify it
    # Add --verbose if needed:
    # --verbose

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Error during RF inference for tile index ${SLURM_ARRAY_TASK_ID}. Exit code: $exit_code"
    exit $exit_code
fi

echo "### RF Inference Task ${SLURM_ARRAY_TASK_ID} Finished ###" 