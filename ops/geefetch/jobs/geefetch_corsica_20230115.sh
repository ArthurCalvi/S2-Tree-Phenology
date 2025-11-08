#!/bin/bash

#SBATCH --job-name=gfc01
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --hint=multithread
#SBATCH --time=0-02:00:00 # total run time limit (HH:MM:SS)
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.err

echo "### Running $SLURM_JOB_NAME ###"

set -x

# Set your environment and load modules
module purge
export PYTHONUSERBASE=$ALL_CCFRWORK/envs/gf-0
PATH=$ALL_CCFRWORK/envs/gf-0/bin:$PATH
module load pytorch-gpu/py3/2.5.0 gdal/3.10.0 google-cloud-cli

# directory must match with 'data_dir' in yaml config
mkdir /lustre/fsn1/projects/rech/ego/uyr48jk/mosaic_corsica_2023/20230115

# Run geefetch
geefetch all -c /linkhome/rech/gennjv01/uyr48jk/work/S2-Tree-Phenology/ops/geefetch/configs/config_corsica_20230115.yaml

echo "### Finished $SLURM_JOB_NAME"