#!/bin/bash
#SBATCH --job-name=recursive-feature-selection-genus-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00 # Reduced time for test run
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Running Recursive Feature Selection for Tree Genus ###"
set -x

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

# Define paths
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK/S2-Tree-Phenology/results"
FEATURE_SEL_DIR_BASE="$RESULTS_DIR/feature_selection"
OUTPUT_DIR="$FEATURE_SEL_DIR_BASE/feature_selection_genus"
DATASET_PATH="$RESULTS_DIR/datasets/training_datasets_pixels.parquet"

# Create logs directory if it doesn't exist (relative to WORK_DIR)
mkdir -p $WORK_DIR/logs

# Create results directory for this specific run
mkdir -p $OUTPUT_DIR

# Change to working directory
cd $WORK_DIR

# Run the feature selection script in test mode for genus
python $WORK_DIR/src/training/recursive_feature_selection.py \
    --target_column genus \
    --output $OUTPUT_DIR \
    --min_features 8 \
    --step_sizes 7,6,5,4,3,2,1 \
    --dataset_path $DATASET_PATH

echo "### Genus Feature Selection Test Run Complete ###" 