#!/bin/bash

# Exit script if any command fails
set -e


# PPS: ('baseline' 'randaugmentna' 'closest_patch' 'random_neighbor' 'region' 
#       'different_normal_slide' 'different_slide_same_label' 'preselected_positive_pairs')
# SSL Methods: ("barlow_twins" "byol" "simsiam" "vicreg" "dino (soon)")


pps="closest_patch"
datasets=("tcga")
ssl_methods=("barlow_twins" "byol" "vicreg")
# ssl_methods=("dino")
minimum_distances=(1 5 10 0)
# minimum_distances=(2 3 4 15 20)


for dataset in "${datasets[@]}"; do
    for ssl_method in "${ssl_methods[@]}"; do
        for minimum_distance in "${minimum_distances[@]}"; do

            # if [ "$ssl_method" = "barlow_twins" ] && [ "$pps" = "different_normal_slide" ]; then
            #     echo "Skipping SSL method: $ssl_method and PPS: $pps"
            #     continue
            # fi

            echo "SSL Method: $ssl_method , PPS: $pps, Minimum Distance: $minimum_distance"

            echo "Starting SSL model training..." 
            python train_ssl_model.py positive_pair_sampling=$pps ssl_method=$ssl_method minimum_distance=$minimum_distance dataset=$dataset
            echo "SSL model training complete."
            sleep 1

            echo "Generating embeddings..."
            python generate_embeddings.py positive_pair_sampling=$pps ssl_method=$ssl_method minimum_distance=$minimum_distance dataset=$dataset
            echo "Embeddings generation complete."
            sleep 1

            echo "Starting MIL model training..."
            python train_mil_model.py positive_pair_sampling=$pps ssl_method=$ssl_method minimum_distance=$minimum_distance dataset=$dataset
            echo "MIL model training complete."
            sleep 1
        done
    done
done

echo "All tasks completed successfully."