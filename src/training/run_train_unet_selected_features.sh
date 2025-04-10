#!/bin/bash
#SBATCH --job-name=train-unet-selected
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=10        # Request more CPUs for GPU task
#SBATCH --partition=gpu_p13       # Specify the GPU partition
#SBATCH --time=10:00:00           # Adjust time if needed
#SBATCH --output=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ego/uyr48jk/slurm_logs/%x_%j.err

echo "### Training UNet phenology model with selected features (GPU) ###"
set -x

source $HOME/.bashrc
module purge
# Load modules suitable for GPU PyTorch on Jean Zay
module load pytorch-gpu/py3/2.2.0 # Keep this or use the latest recommended PyTorch module
module load gdal                  # Keep GDAL dependency

# Define paths using Jean Zay work directory
WORK_DIR="$WORK/S2-Tree-Phenology"
RESULTS_DIR="$WORK_DIR/results/final_model" # Output directory for this run
TILE_DIR="$WORK_DIR/data/training/training_tiles2023_w_corsica/training_tiles2023" # Tile directory

# Create results directories if they don't exist
mkdir -p $RESULTS_DIR
mkdir -p $WORK/S2-Tree-Phenology/logs # Ensure logs directory exists

# Change to working directory
cd $WORK_DIR

# Run the UNet training script with selected features in test mode
python $WORK_DIR/src/training/train_unet_selected_features.py \
    --features "ndvi_amplitude_h1,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_amplitude_h2,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_offset,crswir_var_residual" \
    --output_dir $RESULTS_DIR \
    --model_name "phenology_unet_selected_features" \
    --tile_dir $TILE_DIR \
    --epochs 50 # Maybe reduce epochs for a quick test run

echo "### Job Finished ###" 