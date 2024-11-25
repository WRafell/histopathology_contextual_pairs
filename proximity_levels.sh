#!/bin/bash

# Exit script if any command fails
set -e

pps="preselected_positive_pairs"
proximity_levels=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# proximity_levels=(7 8 9 10 11 12 13 14 15 20 25 30)
proximity_levels=(0 1 2 3 4 5 10 15 20 25 30)
ssl_methods=("barlow_twins" "byol" "vicreg")


for ssl_method in "${ssl_methods[@]}"; do
    for proximity_level in "${proximity_levels[@]}"; do


        echo "SSL Method: $ssl_method , Proximity level: $proximity_level"

        echo "Starting SSL model training..."
        python train_ssl_model.py positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method 
        echo "SSL model training complete."
        sleep 1

        echo "Generating embeddings..."
        python generate_embeddings.py positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method 
        echo "Embeddings generation complete."    
        sleep 1

        echo "Starting MIL model training..."
        python train_mil_model.py positive_pair_sampling=$pps proximity_level=$proximity_level ssl_method=$ssl_method
        echo "MIL model training complete."
        sleep 1
    done
done

echo "All tasks completed successfully."