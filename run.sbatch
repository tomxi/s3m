#!/bin/bash
#SBATCH --job-name=yamnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:mi250:1
#SBATCH --array=1-4
#SBATCH --output=openl3arr.out
#SBATCH --error=openl3arr.err

singularity exec \
    --overlay /scratch/qx244/envs/amd-overlay.ext3:ro \
    /scratch/work/public/singularity/rocm6.1.0-ubuntu22.04.4.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
    conda activate py310; \
    cd /scratch/qx244/code/s3m; \
    python run.py --ftype openl3"
