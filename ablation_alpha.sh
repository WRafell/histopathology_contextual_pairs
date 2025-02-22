#!/bin/bash

# Exit script if any command fails
set -e

start_time=$(date +%s)

# PPS: ('baseline' 'randaugmentna' 'closest_patch' 'random_neighbor' 'region' 
#       'different_normal_slide' 'different_slide_same_label' 'preselected_positive_pairs')
# SSL Methods: ("barlow_twins" "byol" "simsiam" "vicreg" "dino (soon)")

pps="closest_patch"
datasets=("seedp-stomach" "seedp-colon" "camelyon16" "tcga")
ssl_methods=("barlow_twins" "byol" "vicreg")
maximum_distances=(1)
alphas=(0.25 0.75)

datasets=("seedp-stomach")
ssl_methods=("byol" "vicreg" "barlow_twins")
alphas=(0.25)


for dataset in "${datasets[@]}"; do
    for ssl_method in "${ssl_methods[@]}"; do
        for maximum_distance in "${maximum_distances[@]}"; do
            for alpha in  "${alphas[@]}"; do

                echo "SSL Method: $ssl_method , DT: $dataset, PPS: $pps, Maximum distance: $maximum_distance. Alpha: $alpha"

                echo "Starting SSL model training..." 
                python train_ssl_model.py positive_pair_sampling=$pps ssl_method=$ssl_method maximum_distance=$maximum_distance dataset=$dataset alpha=$alpha
                echo "SSL model training complete."
                sleep 1

                echo "Generating embeddings..."
                python generate_embeddings.py positive_pair_sampling=$pps ssl_method=$ssl_method maximum_distance=$maximum_distance dataset=$dataset alpha=$alpha
                echo "Embeddings generation complete."
                sleep 1

                echo "Starting MIL model training..."
                python train_mil_model.py positive_pair_sampling=$pps ssl_method=$ssl_method maximum_distance=$maximum_distance dataset=$dataset alpha=$alpha
                echo "MIL model training complete."
                sleep 1
            done
        done
    done
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$(( elapsed_time / 3600 ))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$(( elapsed_time % 60 ))

printf "All tasks completed successfully in %02d:%02d:%02d.\n" $hours $minutes $seconds