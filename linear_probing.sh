#!/bin/bash

# Exit script if any command fails
set -e

pps_list=("baseline" "random_neighbor" "different_slide_same_label" "different_normal_slide")
ssl_methods=("barlow_twins" "byol" "vicreg")
dataset="seedp-colon"

# Nested loops for SSL methods and PPS
for ssl_method in "${ssl_methods[@]}"; do
    for pps in "${pps_list[@]}"; do

        # Print current configuration
        echo "Dataset: $dataset, SSL Method: $ssl_method, PPS: $pps"

        # Linear Probing
        echo "Starting Linear Probing model training..."
        python linear_probing.py \
            positive_pair_sampling="$pps" \
            ssl_method="$ssl_method" \
            dataset="$dataset"
        echo "Linear Probing model training complete."

        # Sleep for 1 second
        sleep 1

    done
done

pps="closest_patch"
ssl_methods=("barlow_twins" "byol" "vicreg")
minimum_distances=(1 5 10 0)

for ssl_method in "${ssl_methods[@]}"; do
    for minimum_distance in "${minimum_distances[@]}"; do

        # Print current configuration
        echo "SSL Method: $ssl_method , PPS: $pps, Minimum Distance: $minimum_distance"

        # Linear Probing
        echo "Starting Closest Patch..."
        python linear_probing.py \
            positive_pair_sampling="$pps" \
            ssl_method="$ssl_method" \
            dataset="$dataset" \
            minimum_distance=$minimum_distance
        echo "Closest Patch training complete."

        # Sleep for 1 second
        sleep 1
    done
done



echo "All tasks completed successfully."