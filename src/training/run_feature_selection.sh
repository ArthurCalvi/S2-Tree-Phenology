#!/bin/bash
#SBATCH --job-name=recursive-feature-selection
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Running Recursive Feature Selection for Tree Phenology ###"
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
FEATURE_SEL_DIR="$RESULTS_DIR/feature_selection"

# Create results directories if they don't exist
mkdir -p $FEATURE_SEL_DIR

# Change to working directory
cd $WORK_DIR

# Run the feature selection script
# You can add/modify these arguments as needed
python $WORK_DIR/src/training/recursive_feature_selection.py \
    --output $FEATURE_SEL_DIR \
    --min_features 8 \
    --step_sizes 6,5,4,3,2,1 \
    --dataset_path $RESULTS_DIR/datasets/training_datasets_pixels.parquet

# To run in test mode with a smaller dataset, uncomment below:
# python $WORK_DIR/src/training/recursive_feature_selection.py \
#     --output $FEATURE_SEL_DIR \
#     --test \
#     --test_size 10000 