#!/bin/bash

# Exit script if any command fails
set -e


# PPS: ('baseline' 'randaugmentna' 'closest_patch' 'random_neighbor' 'region' 
#       'different_normal_slide' 'different_slide_same_label' 'preselected_positive_pairs')
# SSL Methods: ("barlow_twins" "byol" "simsiam" "vicreg" "dino (soon)")


datasets=("camelyon16")
# pps_list=("baseline" "random_neighbor" "different_slide_same_label" "different_normal_slide")
pps_list=("baseline")
ssl_methods=("barlow_twins" "byol" "vicreg")
# pps_list=("baseline")


for dataset in "${datasets[@]}"; do
    for ssl_method in "${ssl_methods[@]}"; do
        for pps in "${pps_list[@]}"; do

            if [ "$dataset" = "tcga" ] && [ "$pps" = "different_normal_slide" ]; then
                echo "Skipping dataset: $dataset and PPS: $pps"
                continue
            fi

            echo "Dataset: $dataset, SSL Method: $ssl_method , PPS: $pps"

            echo "Starting SSL model training..." 
            python train_ssl_model.py positive_pair_sampling=$pps ssl_method=$ssl_method dataset=$dataset
            echo "SSL model training complete."
            sleep 1

            echo "Generating embeddings..."
            python generate_embeddings.py positive_pair_sampling=$pps ssl_method=$ssl_method dataset=$dataset
            echo "Embeddings generation complete."
            sleep 1

            echo "Starting MIL model training..."
            python train_mil_model.py positive_pair_sampling=$pps ssl_method=$ssl_method dataset=$dataset
            echo "MIL model training complete."
            sleep 1
        done
    done
done

echo "All tasks completed successfully."