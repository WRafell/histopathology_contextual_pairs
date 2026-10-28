#!/bin/bash
#SBATCH --job-name=nsclc-patch
#SBATCH --partition=R3090
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-31%16
#SBATCH --output=local/slurm_logs/%x-%A_%a.out
# Extract 20x 224 px TCGA-NSCLC patches (stride 224) with the patcher snapshot that made the Camelyon16
# patches, one shard per array task. Resubmitting the same array resumes from the completed slides.
# Run scripts/link_tcga_nsclc_images.sh first. Submit from the repository root (after mkdir -p local/slurm_logs):
#   sbatch scripts/slurm/patch_tcga_nsclc.sh
set -euo pipefail
PATCHER_DIR="${PATCHER_DIR:-/workspace/wrafell/toposlide_run/patch224_20261010-1732}"
IMAGES_DIR="${IMAGES_DIR:-/nas2/lait/1000_Members/wrafell/TCGA/NSCLC/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas2/lait/1000_Members/wrafell/TCGA/NSCLC/patches_20x_224_png}"
SHARD_COUNT="${SHARD_COUNT:-32}"
mkdir -p "$OUTPUT_DIR"
# The patcher is not under version control; record which code made the patches.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  (cd "$PATCHER_DIR" && find patching histo_tiling_src -name '*.py' | sort | xargs sha256sum) > "$OUTPUT_DIR/patcher_sha256.txt"
fi
export PYTHONPATH="$PATCHER_DIR/histo_tiling_src:$PATCHER_DIR/patching"
"$HOME/miniconda3/envs/wsi_patch/bin/python" -u "$PATCHER_DIR/patching/extract_tcga_patches.py" \
  --images-dir "$IMAGES_DIR" --slide-glob '*.svs' --output-dir "$OUTPUT_DIR" --label TCGA-NSCLC \
  --magnifications 20 --patch-size 224 --stride 224 \
  --shard-count "$SHARD_COUNT" --shard-index "$SLURM_ARRAY_TASK_ID"
