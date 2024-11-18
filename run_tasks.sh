#!/bin/bash

# Exit script if any command fails
set -e

echo "Starting SSL model training..."
python train_ssl_model.py
echo "SSL model training complete."
sleep 1

echo "Generating embeddings..."
python generate_embeddings.py
echo "Embeddings generation complete."
sleep 1

echo "Starting MIL model training..."
python train_mil_model.py 
echo "MIL model training complete."
sleep 1

echo "All tasks completed successfully."