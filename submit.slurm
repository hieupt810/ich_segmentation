#!/bin/bash
#SBATCH --job-name=ich-pipeline
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=ich-%j.out
#SBATCH --error=ich-%j.err

set -euo pipefail

## Loading the required modules
module reset
module load Anaconda3 \
    GCC/14.2.0 \
    CUDA/11.8.0 cuda11.8/toolkit cuDNN/8.9.2.26-CUDA-11.8.0

## Print some information about the job
echo "[$(date '+%F %T')] Host: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
nvidia-smi || true

## Run the pipeline
conda activate ich-segmentation
python main.py all \
    --epochs 50 \
    --clf-epochs 30 \
    --batch-size 32 \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

## Print completion message
echo "[$(date '+%F %T')] Job completed: ${SLURM_JOB_ID:-local}"