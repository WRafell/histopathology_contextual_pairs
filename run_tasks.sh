#!/bin/bash

# Exit script if any command fails
set -e

pps_list=("closest_patch")
ssl_methods=("barlow_twins" "byol" "vicreg" "simsiam")

for pps in "${pps_list[@]}"; do
    for ssl_method in "${ssl_methods[@]}"; do

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