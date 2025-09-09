#!/bin/bash
#SBATCH --job-name=extract_yamnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:mi250:1
#SBATCH --output=extract_yamnet.out
#SBATCH --error=extract_yamnet.err

singularity exec \
    --overlay /scratch/qx244/envs/amd-overlay.ext3:ro \
    /scratch/work/public/singularity/rocm6.1.0-ubuntu22.04.4.sif \
    /bin/bash -c \
    "source /ext3/env.sh; \
    conda activate py310; \
    cd /scratch/qx244/code/s3m; \
    python run.py --ftype yamnet"
