#!/bin/bash
#SBATCH --job-name=train-selected-features-genus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost # Or your specific Jenzay partition
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00 # Adjust time as needed for genus model
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Training GENUS model with selected features ###"
set -x

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0 # Ensure this environment has necessary scikit-learn, pandas etc.
module load gdal

# Create logs directory if it doesn't exist
mkdir -p $WORK/S2-Tree-Phenology/logs

# Define paths
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK/S2-Tree-Phenology/results"
FINAL_MODEL_GENUS_DIR="$RESULTS_DIR/final_model_genus" # Specific directory for genus models

# Create results directories if they don't exist
mkdir -p $FINAL_MODEL_GENUS_DIR

# Change to working directory
cd $WORK_DIR

# Define the selected features for genus (same as default in the python script)
FEATURES="ndvi_amplitude_h1,ndvi_amplitude_h2,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_phase_h1_sin,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual"

# Run the training script for genus
python $WORK_DIR/src/training/train_rf_selected_features_genus.py \
    --features "$FEATURES" \
    --output_dir $FINAL_MODEL_GENUS_DIR \
    --model_name "genus_model_selected_features" \
    --metrics_name "genus_metrics_selected_features.json" \
    --eco_metrics_name "genus_eco_metrics_selected_features.csv" \
    --dataset_path $RESULTS_DIR/datasets/training_datasets_pixels.parquet

echo "### Genus selected features training script finished ###" 