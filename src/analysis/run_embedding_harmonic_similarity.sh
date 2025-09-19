#!/bin/bash
#SBATCH --job-name=sim-emb-harm-k14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Embedding–Harmonic Similarity (Top‑14) ###"
set -xeuo pipefail

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK_DIR/results"

HARM_PARQUET="$RESULTS_DIR/datasets/training_datasets_pixels.parquet"
EMB_PARQUET="$RESULTS_DIR/datasets/training_datasets_pixels_embedding.parquet"
TILES_PARQUET="$RESULTS_DIR/datasets/tiles_2_5_km_final.parquet"
TOPK_FILE="$RESULTS_DIR/feature_selection_embeddings/features_embeddings_topk_k14.txt"
OUT_DIR="$RESULTS_DIR/analysis_similarity"

mkdir -p "$OUT_DIR"
cd "$WORK_DIR"

python src/analysis/compute_embedding_harmonic_similarity.py \
  --harmonics "$HARM_PARQUET" \
  --embeddings "$EMB_PARQUET" \
  --tiles "$TILES_PARQUET" \
  --topk "$TOPK_FILE" \
  --out "$OUT_DIR" \
  --splits 5 \
  --min-tile-samples 20

echo "### Done: similarity metrics saved under $OUT_DIR ###"

