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
K=14
mkdir -p "$OUT_DIR"
cd "$WORK_DIR"

# Use selected features from prior selection job; fall back to RF importance if missing
FEATURES_DIR="$RESULTS_DIR/feature_selection_embeddings"
FEATURES_FILE_TOPK="$FEATURES_DIR/features_embeddings_topk_k${K}.txt"
FEATURES_FILE="$FEATURES_DIR/features_embeddings_selected.txt"

FEATURES_ARGS=()
if [ -f "$FEATURES_FILE_TOPK" ]; then
  TMP_LIST="$OUT_DIR/features_embeddings_topk_k${K}.list"
  cp "$FEATURES_FILE_TOPK" "$TMP_LIST"
  echo "Using precomputed top-$K features from $FEATURES_FILE_TOPK -> $TMP_LIST"
  FEATURES_ARGS+=(--features_file "$TMP_LIST")
elif [ -f "$FEATURES_FILE" ]; then
  TMP_LIST="$OUT_DIR/features_embeddings_topk_k${K}.list"
  head -n $K "$FEATURES_FILE" > "$TMP_LIST"
  echo "Using selected features from $FEATURES_FILE (top-$K) -> $TMP_LIST"
  FEATURES_ARGS+=(--features_file "$TMP_LIST")
else
  echo "Selected features file not found at $FEATURES_FILE; falling back to RF importance top-$K"
  FEATURES_ARGS=()
fi

CMD=(
  python src/training/train_rf_embeddings.py
  --dataset_path "$DATASET_PATH"
  --config topk
  --k "$K"
  --output_dir "$OUT_DIR"
)

if [ ${#FEATURES_ARGS[@]} -gt 0 ]; then
  CMD+=("${FEATURES_ARGS[@]}")
fi

"${CMD[@]}"
status=$?
if [ $status -ne 0 ]; then
  echo "Training script failed with exit code $status"
  exit $status
fi

METRICS_FILE="$OUT_DIR/metrics_embeddings_topk_k${K}.json"
if [ ! -f "$METRICS_FILE" ]; then
  echo "Metrics file not found at $METRICS_FILE"
  exit 1
fi

RUN_TS=$(python - <<'PY'
import json, sys
from pathlib import Path
metrics_path = Path(sys.argv[1])
data = json.loads(metrics_path.read_text())
print(data.get("timestamp","unknown").replace(":","").replace(".","-"))
PY
"$METRICS_FILE")

if [ "$RUN_TS" = "unknown" ]; then
  RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
fi

ARCHIVE_BASE="$OUT_DIR/archive/embeddings_topk_k${K}"
RUN_DIR="$ARCHIVE_BASE/$RUN_TS"
mkdir -p "$RUN_DIR"

copy_if_exists () {
  local src="$1"
  if [ -f "$src" ]; then
    cp "$src" "$RUN_DIR/"
  fi
}

copy_if_exists "$METRICS_FILE"
copy_if_exists "$OUT_DIR/model_metadata_embeddings_topk_k${K}.json"
copy_if_exists "$OUT_DIR/features_embeddings_topk_k${K}.txt"
copy_if_exists "$OUT_DIR/fold_metrics_embeddings_topk_k${K}.csv"
copy_if_exists "$OUT_DIR/cv_predictions_embeddings_topk_k${K}.parquet"
copy_if_exists "$OUT_DIR/eco_metrics_embeddings_topk_k${K}.csv"

MODEL_FILE=$(ls -1t "$OUT_DIR"/rf_embeddings_embeddings_topk_k${K}_*.joblib 2>/dev/null | head -n1)
if [ -n "$MODEL_FILE" ]; then
  cp "$MODEL_FILE" "$RUN_DIR/"
fi

ln -sfn "$RUN_DIR" "$OUT_DIR/latest_embeddings_topk_k${K}"

echo "Done: features_*, metrics_*, eco_metrics_* saved under $OUT_DIR"
echo "Archived run under $RUN_DIR"
