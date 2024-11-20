#!/bin/bash

# Exit script if any command fails
set -e


# PPS: ('baseline' 'randaugmentna' 'closest_patch' 'random_neighbor' 'region' 
#       'different_normal_slide' 'preselected_positive_pairs')
# SSL Methods: ("barlow_twins" "byol" "simsiam" "vicreg" "dino (soon)")


pps_list=("region" "different_normal_slide")
ssl_methods=("barlow_twins" "byol" "vicreg" "simsiam")


for ssl_method in "${ssl_methods[@]}"; do
    for pps in "${pps_list[@]}"; do

        echo "SSL Method: $ssl_method , PPS: $pps"

        echo "Starting SSL model training..." 
        python train_ssl_model.py positive_pair_sampling=$pps ssl_method=$ssl_method
        echo "SSL model training complete."
        sleep 1

        echo "Generating embeddings..."
        python generate_embeddings.py positive_pair_sampling=$pps ssl_method=$ssl_method
        echo "Embeddings generation complete."
        sleep 1

        echo "Starting MIL model training..."
        python train_mil_model.py positive_pair_sampling=$pps ssl_method=$ssl_method
        echo "MIL model training complete."
        sleep 1
    done
done

echo "All tasks completed successfully."