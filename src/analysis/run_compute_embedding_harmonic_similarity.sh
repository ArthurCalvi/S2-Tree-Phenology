#!/usr/bin/env bash
set -euo pipefail

# Compute linear similarity (R^2) between Top-14 embeddings and full harmonic base
# Aggregates per tile (with centroids) and per eco-region.

python src/analysis/compute_embedding_harmonic_similarity.py \
  --harmonics results/datasets/training_datasets_pixels.parquet \
  --embeddings results/datasets/training_datasets_pixels_embedding.parquet \
  --tiles results/datasets/tiles_2_5_km_final.parquet \
  --topk results/final_model/features_embeddings_topk_k14.txt \
  --out results/analysis_similarity \
  --splits 5
