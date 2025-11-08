#!/bin/bash
#SBATCH --job-name=mosaic_qa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.out
#SBATCH --error=/linkhome/rech/gennjv01/uyr48jk/work/slurm_logs/%x_%j.err

echo '### Running collect_dem_info job ###'
set -x

# Load environment
source $HOME/.bashrc
module purge
module load pytorch-gpu/py3/2.2.0
module load gdal

# Define directories and file paths
BASE_DIR="/lustre/fsn1/projects/rech/ego/uyr48jk/mosaic_corsica_2023"
OUTPUT_JSON="/linkhome/rech/gennjv01/uyr48jk/work/QA/mosaics_corsica_qa.json"

# Create output directory if it doesn't exist
mkdir -p "$(dirname ${OUTPUT_JSON})"
mkdir -p /linkhome/rech/gennjv01/uyr48jk/work/slurm_logs

# Process all monthly folders
for month_dir in ${BASE_DIR}/*/; do
    # Remove trailing slash and get the basename (e.g., 20230115)
    month_name=$(basename ${month_dir})
    
    # Set the input directory to the s2 subdirectory of the current month
    INPUT_DIR="${month_dir}s2"
    
    echo "Processing no-data info collection for month: ${month_name}"
    echo "Input directory: ${INPUT_DIR}"
    echo "Output JSON file: ${OUTPUT_JSON}"
    
    # Run the python script
    python $WORK/python_scripts/get_no_data_perc.py \
        "${INPUT_DIR}" \
        "${OUTPUT_JSON}" \
        --debug
done

echo "All months processed successfully"