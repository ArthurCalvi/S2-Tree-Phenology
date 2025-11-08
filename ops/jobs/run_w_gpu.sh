#!/bin/bash

#SBATCH --job-name=tz-fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=0-01:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/ego/umj78qb/slurmlogs/%x_%A.out
#SBATCH --error=/lustre/fsn1/projects/rech/ego/umj78qb/slurmlogs/%x_%A.err
#SBATCH --mail-user=davejpurnell@gmail.com
#SBATCH --mail-type=BEGIN,END

# Create a monitoring log file
LOG_FILE="/lustre/fsn1/projects/rech/ego/umj78qb/slurmlogs/${SLURM_JOB_ID}.mem"

# Function to log memory usage
log_usage() {
    while true; do
        date >> "$LOG_FILE"
        sstat --format=JobID,AveCPU,AveRSS,AveVMSize -j ${SLURM_JOB_ID} >> "$LOG_FILE"
        echo "---" >> "$LOG_FILE"
        sleep 10  # Log every x seconds
    done
}

# Start monitoring in background
log_usage &
MONITOR_PID=$!

echo "### Running $SLURM_JOB_NAME ###"

export TMPDIR=$JOBSCRATCH

set -x

# Set your environment and load modules
module purge
export PYTHONUSERBASE=$WORK/envs/sprout25-1
PATH=$WORK/envs/sprout25-1/bin:$PATH
module load pytorch-gpu/py3/2.5.0 gdal/3.10.0

# go to sprout-dev directory
cd $WORK/sprout-dev

# Run sprout
srun python -m sprout fit -c _configs/tanzania/tz-fit.yaml

echo "### Finished $SLURM_JOB_NAME"

# Clean up monitoring process
kill $MONITOR_PID