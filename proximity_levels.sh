#!/bin/bash

# Exit script if any command fails
set -e

# proximity_levels=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30)
proximity_levels=(8 9 10 11 12 13 14 15 20 25 30)

for proximity_level in "${proximity_levels[@]}"; do

    echo "Proximity level: $proximity_level"

    echo "Starting SSL model training..."
    python train_ssl_model.py proximity_level=$proximity_level
    echo "SSL model training complete."
    sleep 1

    echo "Generating embeddings..."
    python generate_embeddings.py proximity_level=$proximity_level 
    echo "Embeddings generation complete."
    sleep 1

    echo "Starting MIL model training..."
    python train_mil_model.py proximity_level=$proximity_level 
    echo "MIL model training complete."
    sleep 1
done

echo "All tasks completed successfully."