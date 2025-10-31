#!/bin/bash
#SBATCH --job-name=tune-tabular-models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

set -euo pipefail
echo "### Tuning tabular baselines (logreg, linear SVM, ExtraTrees, HistGB) ###"

source "$HOME/.bashrc"
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

WORK_DIR="$WORK/S2-Tree-Phenology"
DATASET_PATH="$WORK_DIR/results/datasets/training_datasets_pixels.parquet"
HARM_FEATURES="$WORK_DIR/results/feature_selection_harmonic/features_harmonic_topk_k14.txt"
OUTPUT_DIR="$WORK_DIR/results/tuning_tabular"

mkdir -p "$WORK_DIR/logs"
mkdir -p "$OUTPUT_DIR"

cd "$WORK_DIR"

ESTIMATORS=("logreg" "linear_svm" "extra_trees" "histgb")

for ESTIMATOR in "${ESTIMATORS[@]}"; do
  echo "--- Running $ESTIMATOR ---"
  RESULTS_NAME="harmonic_${ESTIMATOR}_tuning.json"
  MODEL_NAME="harmonic_${ESTIMATOR}_best.joblib"
  python src/training/tune_tabular_models.py \
    --estimator "$ESTIMATOR" \
    --features-file "$HARM_FEATURES" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --target-column phenology \
    --n-splits 5 \
    --factor 3 \
    --min-resources 700000 \
    --results-name "$RESULTS_NAME" \
    --best-model-name "$MODEL_NAME"
done

echo "### Tuning complete. Artifacts under $OUTPUT_DIR/<estimator>/phenology ###"
