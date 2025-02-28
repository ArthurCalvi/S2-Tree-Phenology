#!/bin/bash
#SBATCH --job-name=qa-raw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.err

echo "### Running QA Stats on raw mosaics ###"
set -x

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdaltorch-gpu/py3/2.2.0  # environment with rasterio

INPUT_DIR="/lustre/fsn1/projects/rech/ego/uyr48jk/mosaic2023"
OUTPUT_JSON="/lustre/fsn1/projects/rech/ego/uyr48jk/mosaic2023/qa_raw_mosaic.json"

python $WORK/S2-Tree-Phenology/src/qa/qa_mosaics.py \
    --input-dir "$INPUT_DIR" \
    --output-json "$OUTPUT_JSON"
