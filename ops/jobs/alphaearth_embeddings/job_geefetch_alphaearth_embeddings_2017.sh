#!/bin/bash

#SBATCH --job-name=gfalpha17
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --hint=multithread
#SBATCH --time=0-20:00:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.err

set -xeuo pipefail

echo "### Running $SLURM_JOB_NAME ###"

module purge
export PYTHONUSERBASE=$ALL_CCFRWORK/envs/gf-0
export PATH=$ALL_CCFRWORK/envs/gf-0/bin:$PATH
module load pytorch-gpu/py3/2.5.0 gdal/3.10.0 google-cloud-cli

DATA_DIR=/lustre/fsn1/projects/rech/ego/uyr48jk/alphaearth_embeddings/2017
CONFIG_PATH=/linkhome/rech/gennjv01/uyr48jk/work/geefetch_configs/alphaearth_embeddings/config_alphaearth_embeddings_2017.yaml

mkdir -p "$DATA_DIR"

geefetch custom alphaearth_embeddings -c "$CONFIG_PATH"

echo "### Finished $SLURM_JOB_NAME ###"
