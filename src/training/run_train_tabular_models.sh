#!/bin/bash
#SBATCH --job-name=train-tabular-models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

set -euo pipefail
echo "### Training tabular baselines for HARM-14 and EMB-14 ###"

source "$HOME/.bashrc"
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

WORK_DIR="$WORK/S2-Tree-Phenology"
TUNING_DIR="$WORK_DIR/results/tuning_tabular"
OUTPUT_BASE="$WORK_DIR/results/trained_tabular"

DATASET_HARM="$WORK_DIR/results/datasets/training_datasets_pixels.parquet"
DATASET_EMB="$WORK_DIR/results/datasets/training_datasets_pixels_embedding.parquet"

HARM_FEATURES="$WORK_DIR/results/feature_selection_harmonic/features_harmonic_topk_k14.txt"
EMB_FEATURES="$WORK_DIR/results/feature_selection_embeddings/features_embeddings_topk_k14.txt"

mkdir -p "$WORK_DIR/logs"
mkdir -p "$OUTPUT_BASE/harmonics"
mkdir -p "$OUTPUT_BASE/embeddings"

cd "$WORK_DIR"

ESTIMATORS=("logreg" "linear_svm")

for ESTIMATOR in "${ESTIMATORS[@]}"; do
  echo "--- Training $ESTIMATOR (HARM-14) ---"
  PARAM_JSON="$TUNING_DIR/$ESTIMATOR/phenology/harmonic_${ESTIMATOR}_tuning.json"
  if [[ ! -f "$PARAM_JSON" ]]; then
    echo "Missing tuning JSON for $ESTIMATOR at $PARAM_JSON" >&2
    exit 1
  fi

  python src/training/tune_tabular_models.py \
    --estimator "$ESTIMATOR" \
    --features-file "$HARM_FEATURES" \
    --dataset-path "$DATASET_HARM" \
    --output-dir "$OUTPUT_BASE/harmonics" \
    --target-column phenology \
    --param-json "$PARAM_JSON" \
    --results-name "harmonic_${ESTIMATOR}_metrics_full.json" \
    --best-model-name "harmonic_${ESTIMATOR}_model.joblib"

  echo "--- Training $ESTIMATOR (EMB-14) ---"
  python src/training/tune_tabular_models.py \
    --estimator "$ESTIMATOR" \
    --features-file "$EMB_FEATURES" \
    --dataset-path "$DATASET_EMB" \
    --output-dir "$OUTPUT_BASE/embeddings" \
    --target-column phenology \
    --param-json "$PARAM_JSON" \
    --results-name "embeddings_${ESTIMATOR}_metrics_full.json" \
    --best-model-name "embeddings_${ESTIMATOR}_model.joblib"
done

echo "### Training complete. Models under $OUTPUT_BASE/(harmonics|embeddings) ###"
