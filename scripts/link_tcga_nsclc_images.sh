#!/bin/bash
# Link the slides of configs/splits/tcga_nsclc_split.csv into one folder for the patcher (run once on soda,
# from the repository root): bash scripts/link_tcga_nsclc_images.sh
# Links point to TCGA/<project>/raw/<slide>.svs; existing links are kept.
set -euo pipefail
TCGA_DIR="${TCGA_DIR:-/nas2/lait/1000_Members/wrafell/TCGA}"
IMAGES_DIR="${IMAGES_DIR:-$TCGA_DIR/NSCLC/images}"
SPLIT_FILE="${SPLIT_FILE:-configs/splits/tcga_nsclc_split.csv}"
mkdir -p "$IMAGES_DIR"
tail -n +2 "$SPLIT_FILE" | while IFS=, read -r slide patient project slide_label subset; do
  source_path="$TCGA_DIR/$project/raw/$slide.svs"
  [ -f "$source_path" ] || { echo "missing: $source_path" >&2; exit 1; }
  ln -sfn "$source_path" "$IMAGES_DIR/$slide.svs"
done
echo "$(find "$IMAGES_DIR" -name '*.svs' | wc -l) slide links in $IMAGES_DIR"
