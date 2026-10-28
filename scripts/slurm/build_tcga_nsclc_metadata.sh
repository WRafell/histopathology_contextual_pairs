#!/bin/bash
#SBATCH --job-name=nsclc-metadata
#SBATCH --partition=R3090
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=local/slurm_logs/%x-%j.out
# Build runs/metadata/tcga_nsclc_20x_224/csv/tcga_nsclc_data.csv on soda after scripts/slurm/patch_tcga_nsclc.sh.
# Submit from the repository root (after mkdir -p local/slurm_logs): sbatch scripts/slurm/build_tcga_nsclc_metadata.sh [extra --set overrides]
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate histo-positive-pairs
python -u -m src.build_tcga_nsclc_metadata --config configs/tcga_nsclc_metadata.yaml \
    --output-dir "${OUTPUT_DIR:-runs/metadata/tcga_nsclc_20x_224}" --set num_workers="$SLURM_CPUS_PER_TASK" "$@"
