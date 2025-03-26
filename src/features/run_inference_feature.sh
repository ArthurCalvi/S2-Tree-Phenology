#!/bin/bash
#SBATCH --job-name=ru-f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # For parallel processing within each tile
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00             # 20 hours max
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-280%20            # Will be adjusted based on number of tiles

echo "### Running Phenology Inference - Task ${SLURM_ARRAY_TASK_ID} ###"
set -x

source $HOME/.bashrc
module load gdal
module load pytorch-gpu/py3/2.2.0

# Config directory where prepare_inference_feature.py output is stored
CONFIG_DIR="$WORK/S2-Tree-Phenology/configs"

# Run inference for the assigned tile
python $WORK/S2-Tree-Phenology/src/features/inference_feature.py \
    --config-dir ${CONFIG_DIR} \
    --tile-idx ${SLURM_ARRAY_TASK_ID} \
    --block-size 1024 \
    --max-workers 4 \
    --num-harmonics 2 \
    --max-iter 1 