#!/bin/bash
#SBATCH --job-name=qa-stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.err

echo "### Running QA Stats ###"
set -x

source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

INPUT_DIR="/lustre/fswork/projects/rech/ego/uyr48jk/S2-Tree-Phenology/configs"
OUTPUT_JSON="/lustre/fsn1/projects/rech/ego/uyr48jk/features2023/qa_stats_s2_features.json"

python $WORK/S2-Tree-Phenology/src/features/qa_uint16_stats.py \
    --input-dir "$INPUT_DIR" \
    --output-json "$OUTPUT_JSON"