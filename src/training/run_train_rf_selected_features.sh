#!/bin/bash
#SBATCH --job-name=train-selected-features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Training phenology model with selected features ###"
set -x

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

# Create logs directory if it doesn't exist
mkdir -p $WORK/S2-Tree-Phenology/logs

# Define paths
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK/S2-Tree-Phenology/results"
FINAL_MODEL_DIR="$RESULTS_DIR/final_model"

# Create results directories if they don't exist
mkdir -p $FINAL_MODEL_DIR

# Change to working directory
cd $WORK_DIR

# Run the training script with selected features
python $WORK_DIR/src/training/train_rf_selected_features.py \
    --features "ndvi_amplitude_h1,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,nbr_amplitude_h1,nbr_phase_h1_cos,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual" \
    --output_dir $FINAL_MODEL_DIR \
    --model_name "phenology_model_selected_features" \
    --metrics_name "phenology_metrics_selected_features.json" \
    --eco_metrics_name "phenology_eco_metrics_selected_features.csv" \
    --dataset_path $RESULTS_DIR/datasets/training_datasets_pixels.parquet
status=$?
if [ $status -ne 0 ]; then
    echo "Training script failed with exit code $status"
    exit $status
fi

METRICS_FILE="$FINAL_MODEL_DIR/phenology_metrics_selected_features.json"
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

ARCHIVE_BASE="$FINAL_MODEL_DIR/archive/selected_features"
RUN_DIR="$ARCHIVE_BASE/$RUN_TS"
mkdir -p "$RUN_DIR"

copy_if_exists () {
    local src="$1"
    if [ -f "$src" ]; then
        cp "$src" "$RUN_DIR/"
    fi
}

MODEL_BASE="phenology_model_selected_features"
MODEL_FILE=$(ls -1t "$FINAL_MODEL_DIR"/${MODEL_BASE}_*.joblib 2>/dev/null | head -n1)
CONFIG_FILE=$(ls -1t "$FINAL_MODEL_DIR"/${MODEL_BASE}_*_config.json 2>/dev/null | head -n1)
FOLD_FILE=$(ls -1t "$FINAL_MODEL_DIR"/${MODEL_BASE}_*_fold_metrics.csv 2>/dev/null | head -n1)
CV_FILE=$(ls -1t "$FINAL_MODEL_DIR"/${MODEL_BASE}_*_cv_predictions.parquet 2>/dev/null | head -n1)

copy_if_exists "$METRICS_FILE"
copy_if_exists "$FINAL_MODEL_DIR/phenology_eco_metrics_selected_features.csv"

if [ -n "$MODEL_FILE" ]; then
    cp "$MODEL_FILE" "$RUN_DIR/"
fi
if [ -n "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$RUN_DIR/"
fi
if [ -n "$FOLD_FILE" ]; then
    cp "$FOLD_FILE" "$RUN_DIR/"
fi
if [ -n "$CV_FILE" ]; then
    cp "$CV_FILE" "$RUN_DIR/"
fi

ln -sfn "$RUN_DIR" "$FINAL_MODEL_DIR/latest_selected_features"

echo "Archived run under $RUN_DIR"

# To run in test mode with a smaller dataset, uncomment below:
# python $WORK_DIR/src/training/train_rf_selected_features.py \
#     --features "ndvi_amplitude_h1,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_amplitude_h2,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_offset,crswir_var_residual" \
#     --output_dir $FINAL_MODEL_DIR \
#     --test \
#     --test_size 10000 
