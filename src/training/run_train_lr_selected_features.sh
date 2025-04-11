#!/bin/bash
#SBATCH --job-name=logistic-features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Training logistic regression model with selected features ###"
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
FINAL_MODEL_DIR="$RESULTS_DIR/final_logistic_model"

# Create results directories if they don't exist
mkdir -p $FINAL_MODEL_DIR

# Change to working directory
cd $WORK_DIR

# Run the training script with selected features
python $WORK_DIR/src/training/train_lr_selected_features.py \
    --features "ndvi_amplitude_h1,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,nbr_amplitude_h1,nbr_phase_h1_cos,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual" \
    --output_dir $FINAL_MODEL_DIR \
    --model_name "logistic_phenology_model_selected_features" \
    --metrics_name "logistic_phenology_metrics_selected_features.json" \
    --eco_metrics_name "logistic_phenology_eco_metrics_selected_features.csv" \
    --C 0.1 \
    --dataset_path $RESULTS_DIR/datasets/training_datasets_pixels.parquet

# To run in test mode with a smaller dataset, uncomment below:
# python $WORK_DIR/src/training/train_lr_selected_features.py \
#     --features "ndvi_amplitude_h1,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_amplitude_h2,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_offset,crswir_var_residual" \
#     --output_dir $FINAL_MODEL_DIR \
#     --C 0.1 \
#     --test \
#     --test_size 10000 