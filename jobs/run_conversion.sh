#!/bin/bash
#SBATCH --job-name=uint16_to_uint8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/ego/uyr48jk/work/slurm_logs/%x_%j.err

echo "### Running UInt16 => UInt8 conversion ###"
set -x

source $HOME/.bashrc
module load gdal  # for Rasterio dependencies
module load pytorch-gpu/py3/2.2.0  # or any environment that has rasterio

# We set the input and output directories
INPUT_DIR="/lustre/fswork/projects/rech/ego/uyr48jk/S2-Tree-Phenology/configs"
OUTPUT_DIR="/lustre/fsn1/projects/rech/ego/uyr48jk/features2023"

python $WORK/S2-Tree-Phenology/src/features/convert_uint16_to_uint8.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
