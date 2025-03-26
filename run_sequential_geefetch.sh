#!/bin/bash

# Script to run geefetch jobs sequentially with a delay
# Usage: ./run_sequential_geefetch.sh [delay_minutes]

# Default delay of 10 minutes if not specified
DELAY_MINUTES=${1:-10}
DELAY_SECONDS=$((DELAY_MINUTES * 60))

# Directory containing the job scripts
JOB_DIR="/linkhome/rech/gennjv01/uyr48jk/work/S2-Tree-Phenology/geefetch/jobs"

# Find all the geefetch_corsica_*.sh files and sort them
JOB_FILES=($(find ${JOB_DIR} -name "geefetch_corsica_*.sh" | sort))

echo "Found ${#JOB_FILES[@]} geefetch jobs to run"
echo "Will run each job with a ${DELAY_MINUTES} minute delay between submissions"

for ((i=0; i<${#JOB_FILES[@]}; i++)); do
    JOB_FILE="${JOB_FILES[$i]}"
    JOB_NAME=$(basename "${JOB_FILE}")
    
    echo "[$((i+1))/${#JOB_FILES[@]}] Submitting job: ${JOB_NAME}"
    
    # Submit the job
    JOB_ID=$(sbatch "${JOB_FILE}" | awk '{print $4}')
    echo "  Job submitted with ID: ${JOB_ID}"
    
    # If this is not the last job, wait before submitting the next one
    if [ $i -lt $((${#JOB_FILES[@]}-1)) ]; then
        NEXT_JOB=$(basename "${JOB_FILES[$((i+1))]}")
        echo "  Waiting ${DELAY_MINUTES} minutes before submitting ${NEXT_JOB}..."
        sleep ${DELAY_SECONDS}
    fi
done

echo "All jobs have been submitted!" 