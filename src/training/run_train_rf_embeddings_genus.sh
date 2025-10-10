#!/bin/bash
#SBATCH --job-name=train-rf-emb-genus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
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
OUT_DIR="$RESULTS_DIR/final_model_genus"
mkdir -p "$OUT_DIR"
cd "$WORK_DIR"

python src/training/train_rf_embeddings_genus.py \
  --dataset_path "$DATASET_PATH" \
  --config all \
  --output_dir "$OUT_DIR"

echo "Done: genus metrics & artefacts saved under $OUT_DIR"
