#!/bin/bash
#SBATCH --job-name=train-unet-selected
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=8         # Request 8 CPUs for GPU task (Decreased from 10)
#SBATCH --partition=gpu_p2       # Specify the GPU partition
#SBATCH --time=14:00:00           # Adjust time if needed
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Training UNet phenology model with selected features (GPU) ###"
set -x

source $HOME/.bashrc
module purge
# Load modules suitable for GPU PyTorch on Jean Zay
module load pytorch-gpu/py3/2.2.0 # Keep this or use the latest recommended PyTorch module
module load gdal                  # Keep GDAL dependency

# Set TMPDIR to job-specific scratch directory (Recommended on Jean Zay)
export TMPDIR=$JOBSCRATCH

# Define paths using Jean Zay work directory
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK_DIR/results/final_model" # Output directory for this run
TILE_DIR="/lustre/fsn1/projects/rech/ego/uyr48jk/training_tiles2023" # Tile directory

# Create results directories if they don't exist
mkdir -p $RESULTS_DIR
mkdir -p $WORK/S2-Tree-Phenology/logs # Ensure logs directory exists

# Change to working directory
cd $WORK_DIR

# Run the UNet training script with selected features in test mode
python $WORK_DIR/src/training/train_unet_selected_features.py \
    --features "ndvi_amplitude_h1,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,nbr_amplitude_h1,nbr_phase_h1_cos,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual" \
    --output_dir $RESULTS_DIR \
    --model_name "phenology_unet_selected_features_14" \
    --tile_dir $TILE_DIR \
    --epochs 100 # Maybe reduce epochs for a quick test run
    --augment 

echo "### Job Finished ###" 