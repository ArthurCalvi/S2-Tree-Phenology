#!/bin/bash
#SBATCH --job-name=pr-f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=00:01:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.err

echo "### Running $SLURM_JOB_NAME ###"
set -x

source $HOME/.bashrc
module load gdal
module load pytorch-gpu/py3/2.2.0

python $WORK/S2-Tree-Phenology/src/features/prepare_inference_feature.py \
    --input-dir /lustre/fsn1/projects/rech/ego/uyr48jk/mosaic2023 \
    --output-dir $WORK/S2-Tree-Phenology/configs \
    --year 2023 \
    --max-concurrent-jobs 4 \
    --min-dates 3