#!/bin/bash
#SBATCH --job-name=train-rf-emb-topk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

set -x
source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK_DIR/results"
DATASET_PATH="$RESULTS_DIR/datasets/training_datasets_pixels_embedding.parquet"
OUT_DIR="$RESULTS_DIR/final_model"
mkdir -p "$OUT_DIR"
cd "$WORK_DIR"

# Use selected features from prior selection job; fall back to RF importance if missing
FEATURES_DIR="$RESULTS_DIR/feature_selection_embeddings"
FEATURES_FILE="$FEATURES_DIR/features_embeddings_selected.txt"
K=14
if [ -f "$FEATURES_FILE" ]; then
  TMP_LIST="$OUT_DIR/features_embeddings_topk_k${K}.list"
  head -n $K "$FEATURES_FILE" > "$TMP_LIST"
  echo "Using selected features from $FEATURES_FILE (top-$K) -> $TMP_LIST"
  python src/training/train_rf_embeddings.py \
    --dataset_path "$DATASET_PATH" \
    --config topk \
    --k $K \
    --features_file "$TMP_LIST" \
    --output_dir "$OUT_DIR"
else
  echo "Selected features file not found at $FEATURES_FILE; falling back to RF importance top-$K"
  python src/training/train_rf_embeddings.py \
    --dataset_path "$DATASET_PATH" \
    --config topk \
    --k $K \
    --output_dir "$OUT_DIR"
fi

echo "Done: features_*, metrics_*, eco_metrics_* saved under $OUT_DIR"
