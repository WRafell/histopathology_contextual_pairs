#!/bin/bash

# Exit script if any command fails
set -e

start_time=$(date +%s)

# PPS: ('baseline' 'randaugmentna' 'closest_patch' 'random_neighbor' 'region' 
#       'different_normal_slide' 'different_slide_same_label' 'preselected_positive_pairs')
# SSL Methods: ("barlow_twins" "byol" "simsiam" "vicreg" "dino (soon)")


datasets=("seedp-stomach")
# pps_list=("baseline" "random_neighbor" "different_slide_same_label" "different_normal_slide")
pps_list=("baseline")
# ssl_methods=("barlow_twins" "byol" "vicreg")
ssl_methods=("barlow_twins")
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

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$(( elapsed_time / 3600 ))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$(( elapsed_time % 60 ))

printf "All tasks completed successfully in %02d:%02d:%02d.\n" $hours $minutes $seconds