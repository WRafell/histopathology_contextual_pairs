#!/bin/bash
#SBATCH --job-name=nsclc-md5
#SBATCH --partition=R3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=local/slurm_logs/%x-%j.out
# Check the md5 of every TCGA-NSCLC slide in configs/splits/tcga_nsclc_split.csv against its GDC manifest.
# Submit from the repository root (after mkdir -p local/slurm_logs): sbatch scripts/slurm/check_tcga_nsclc_md5.sh
# Prints only failures, then a count; exits non-zero if any slide fails.
set -euo pipefail
export LC_ALL=C  # sort and join must agree on the order
TCGA_DIR="${TCGA_DIR:-/nas2/lait/1000_Members/wrafell/TCGA}"
SPLIT_FILE="${SPLIT_FILE:-configs/splits/tcga_nsclc_split.csv}"
status=0
for project in LUAD LUSC; do
  # GDC manifest columns: id, filename, md5, size, state.
  awk -F, -v project="$project" 'NR > 1 && $3 == project {print $1 ".svs"}' "$SPLIT_FILE" | sort > "/tmp/nsclc-$SLURM_JOB_ID-$project"
  awk -F'\t' 'NR > 1 {print $2 "\t" $3}' "$TCGA_DIR/$project/gdc_manifest.tsv" | sort \
    | join -t $'\t' "/tmp/nsclc-$SLURM_JOB_ID-$project" - | awk -F'\t' '{print $2 "  " $1}' \
    | (cd "$TCGA_DIR/$project/raw" && md5sum --check --quiet -) || status=1
  echo "$project: checked $(wc -l < "/tmp/nsclc-$SLURM_JOB_ID-$project") slides"
  rm -f "/tmp/nsclc-$SLURM_JOB_ID-$project"
done
exit "$status"
