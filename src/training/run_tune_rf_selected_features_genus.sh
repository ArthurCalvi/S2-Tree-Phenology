#!/bin/bash
#SBATCH --job-name=tune-selected-features-genus # Changed job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 # HalvingGridSearchCV can use multiple cores (n_jobs=-1 in script)
#SBATCH --partition=prepost # Or adjust partition as needed
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00 # Increased time for tuning, adjust as needed
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out # Updated output path
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err # Updated error path

echo "### Tuning genus model hyperparameters with selected features ###"
set -x

source $HOME/.bashrc
module purge
# Load necessary modules (adjust versions if needed)
module load pytorch-gpu/py3/2.2.0 # Or appropriate python/scikit-learn env
module load gdal

# Create logs directory if it doesn't exist (using $WORK defined in env)
mkdir -p $WORK/S2-Tree-Phenology/logs

# Define paths
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK/S2-Tree-Phenology/results"
# Updated tuning directory to be specific to genus
TUNING_DIR="$RESULTS_DIR/tuning_selected/genus"
DATASET_PATH="$RESULTS_DIR/datasets/training_datasets_pixels.parquet"

# Create results directories if they don't exist
mkdir -p $TUNING_DIR

# Change to working directory
cd $WORK_DIR

# Define the selected features for genus
FEATURES="ndvi_amplitude_h1,ndvi_amplitude_h2,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_phase_h1_sin,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual"

# Run the tuning script for genus
python $WORK_DIR/src/training/tune_rf_selected_features.py \
    --features "$FEATURES" \
    --target_column "genus" \
    --output_dir $TUNING_DIR \
    --results_name "tuning_results_genus_selected.json" \
    --best_model_name "best_genus_model_selected.joblib" \
    --dataset_path $DATASET_PATH \
    --n_splits 5 \
    --factor 3 # Uses the default min_resources (adjusted based on target in python script)
    # Add --min_resources <number> here if you want to override the default

echo "### Genus tuning script finished ###" 