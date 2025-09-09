#!/bin/bash
#SBATCH --job-name=beat_sync
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=1-10
#SBATCH --output=beat_sync.out
#SBATCH --error=beat_sync.err

singularity exec \
    --overlay /scratch/qx244/envs/amd-overlay.ext3:ro \
    /scratch/work/public/singularity/rocm6.1.0-ubuntu22.04.4.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
    conda activate py310; \
    cd /scratch/qx244/code/s3m; \
    python run.py"
