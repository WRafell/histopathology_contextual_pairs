#!/bin/bash

# Exit script if any command fails
set -e

datasets=("seedp-stomach" "seedp-colon" "tcga" "camelyon16")
pps="preselected_positive_pairs-lonely"
proximity_levels=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# proximity_levels=(7 8 9 10 11 12 13 14 15 20 25 30)
proximity_levels=(0 1 2 3 4 5 10 15 20 25 30)
ssl_methods=("barlow_twins" "byol" "vicreg")

# ssl_methods=("barlow_twins")
# proximity_levels=(0 1)

for dataset in "${datasets[@]}"; do
    for ssl_method in "${ssl_methods[@]}"; do
        for proximity_level in "${proximity_levels[@]}"; do


            echo "SSL Method: $ssl_method , Proximity level: $proximity_level"

            echo "Starting SSL model training with Ablation..."
            python ablation_proximity.py dataset=$dataset positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method 
            echo "SSL model training complete."
            sleep 1

            echo "Generating embeddings..."
            python generate_embeddings.py dataset=$dataset positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method 
            echo "Embeddings generation complete."    
            sleep 1

            echo "Starting MIL model training..."
            python train_mil_model.py dataset=$dataset positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method
            echo "MIL model training complete."
            sleep 1
        done
    done
done

echo "All tasks completed successfully."