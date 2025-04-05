#!/bin/bash

# Script to train a 175B parameter model optimized for a 30-day timeframe
# This script sets up the environment and launches the training with optimal parameters

echo "========================================================"
echo "Setting up environment for 175B parameter model training"
echo "========================================================"

# Install dependencies
pip install -r requirements.txt

# Install compatible version of optax for Python 3.8
pip install 'optax<0.1.7'

# Make the launcher script executable
chmod +x run_all.py

# Set environment variables for TPU
export TPU_NAME=${TPU_NAME:-"v4-32"}
export TPU_ZONE=${TPU_ZONE:-"us-central1-a"}

# Set optimal training parameters for 175B model in 30-day timeframe
echo "========================================================"
echo "Starting 175B parameter model training"
echo "Optimized for 30-day training window"
echo "========================================================"

# Launch training with optimal parameters for 175B model
./run_all.py \
  --model_size 175b \
  --dataset code-mix \
  --batch_size 32 \
  --steps 400000 \
  --learning_rate 0.0001 \
  --max_seq_length 131072 \
  --use_flash_attention \
  --use_reasoning_layer \
  --num_checkpoints 30 \
  --output_dir ./175b_model \
  --debug \
  --force

echo "========================================================"
echo "Training launched successfully"
echo "Monitor progress with: tail -f training.log"
echo "========================================================"
