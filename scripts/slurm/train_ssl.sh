#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --partition=R4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=local/slurm_logs/%x-%j.out
# SSL pretraining on vegi, one torchrun process per allocated GPU; the config's batch_size is per GPU.
# Submit from the repository root (after mkdir -p local/slurm_logs):
#   RUN_NAME=<name> [CONFIG=configs/...yaml] sbatch [--nodelist=vegi-01] scripts/slurm/train_ssl.sh [--set key=value ...]
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate histo-positive-pairs
num_gpus=$(nvidia-smi -L | wc -l)
workers_per_gpu=$((SLURM_CPUS_PER_TASK / num_gpus - 1))
export OMP_NUM_THREADS=1
torchrun --standalone --nproc-per-node "$num_gpus" -m src.train_ssl \
    --config "${CONFIG:-configs/ssl_vicreg_resnet18.yaml}" --output-dir "runs/ssl/${RUN_NAME:?set RUN_NAME}" \
    --set num_workers="$workers_per_gpu" "$@"
