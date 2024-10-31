#!/bin/bash

# Exit script if any command fails
set -e

echo "Starting SSL model training..."
python train_ssl_model.py proximity_level=8
echo "SSL model training complete."

echo "Generating embeddings..."
python generate_embeddings.py proximity_level=8
echo "Embeddings generation complete."

echo "Starting MIL model training..."
python train_mil_model.py proximity_level=8
echo "MIL model training complete."

echo "All tasks completed successfully."