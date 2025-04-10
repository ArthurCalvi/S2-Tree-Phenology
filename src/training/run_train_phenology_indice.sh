#!/bin/bash
#SBATCH --job-name=train-phenology
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Running Tree Phenology Training ###"
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
MODELS_DIR="$RESULTS_DIR/models"

# Create results directories if they don't exist
mkdir -p $MODELS_DIR

# Run training script
cd $WORK_DIR

# Run the main training script
# Add any command line arguments as needed
python $WORK_DIR/src/training/train_rf_phenology_indice.py 