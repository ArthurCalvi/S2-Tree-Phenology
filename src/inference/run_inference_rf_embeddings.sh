#!/bin/bash
#SBATCH --job-name=rf-emb-inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-80%4  # ADJUST upper bound to n_tiles-1 and throttle as needed

set -euo pipefail
set -x

echo "### Running RF Embedding Inference - Task ${SLURM_ARRAY_TASK_ID} ###"

source "$HOME/.bashrc"
module load gdal
# module load pytorch-gpu/py3/2.5.0  # Uncomment if your env requires it

# --- User Configuration ---
INPUT_DIR="$SCRATCH/alphaearth_embeddings_tiles"   # ADJUST path to AlphaEarth embedding tiles
OUTPUT_DIR="$SCRATCH/rf_inference_embeddings"      # ADJUST destination for probability tiles
MODEL_PATH="$WORK/S2-Tree-Phenology/results/final_model/rf_embeddings_embeddings_all_YYYYMMDDTHHMMSSZ.joblib"  # ADJUST model filename
FEATURES_FILE="$WORK/S2-Tree-Phenology/results/final_model/features_embeddings_all.txt"                        # ADJUST features file from training
BLOCK_SIZE=2048
RF_N_JOBS=8            # Set <= cpus-per-task or leave blank to use model default
SAVE_CLASSES=1         # Set to 1 to save class raster, 0 to skip
# --- End User Configuration ---

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found." >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file '$MODEL_PATH' not found." >&2
  exit 1
fi
if [ ! -f "$FEATURES_FILE" ]; then
  echo "Error: Features file '$FEATURES_FILE' not found." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD=(
  python "$WORK/S2-Tree-Phenology/src/inference/inference_rf_embeddings.py"
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL_PATH"
  --features-file "$FEATURES_FILE"
  --tile-idx "${SLURM_ARRAY_TASK_ID}"
  --block-size "$BLOCK_SIZE"
)

if [ -n "${RF_N_JOBS:-}" ]; then
  CMD+=(--rf-n-jobs "$RF_N_JOBS")
fi

if [ "$SAVE_CLASSES" -eq 1 ]; then
  CMD+=(--save-classes)
fi

"${CMD[@]}"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Error during RF embedding inference for tile index ${SLURM_ARRAY_TASK_ID}. Exit code: $exit_code" >&2
    exit $exit_code
fi

echo "### RF Embedding Inference Task ${SLURM_ARRAY_TASK_ID} Finished ###"
