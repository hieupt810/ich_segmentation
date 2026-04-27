#!/bin/bash
#SBATCH --job-name=ich-pipeline
#SBATCH --partition=amperenodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ich-%j.out
#SBATCH --error=logs/ich-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output

module reset
module load Anaconda3 GCC/14.2.0 CUDA/11.8.0 cuda11.8/toolkit cuDNN/8.9.2.26-CUDA-11.8.0

conda activate ich-segmentation

echo "[$(date '+%F %T')] Host: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
nvidia-smi || true

python main.py all \
    --epochs 50 \
    --clf-epochs 30 \
    --batch-size 64 \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --max-vis-samples 16

echo "[$(date '+%F %T')] Pipeline finished."
