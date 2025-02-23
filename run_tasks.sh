#!/bin/bash

# Exit script if any command fails
set -e

start_time=$(date +%s)

# TASKS
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


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

hours=$(( elapsed_time / 3600 ))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$(( elapsed_time % 60 ))

printf "All tasks completed successfully in %02d:%02d:%02d.\n" $hours $minutes $seconds