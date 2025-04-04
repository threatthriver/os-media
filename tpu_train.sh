#!/bin/bash

# tpu_train.sh - Shell script wrapper to run the LLM training with a single command
# This script sets up the environment and runs the training script

set -e  # Exit on any error

# Check if required environment variables are set
if [ -z "$GCS_BUCKET" ]; then
    echo "Please set the GCS_BUCKET environment variable"
    echo "Example: export GCS_BUCKET=gs://your-bucket-name"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo "Please set the PROJECT_ID environment variable"
    echo "Example: export PROJECT_ID=your-gcp-project-id"
    exit 1
fi

# Parse command line arguments
MODEL_SIZE="600b"
DEBUG=""
PROFILE=""
BATCH_SIZE=""
STEPS=""
LEARNING_RATE=""
CHECKPOINT=""
DATASET_PATH=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_size)
            MODEL_SIZE="$2"
            shift
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --profile)
            PROFILE="--profile"
            shift
            ;;
        --batch_size)
            BATCH_SIZE="--batch_size $2"
            shift
            shift
            ;;
        --steps)
            STEPS="--steps $2"
            shift
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="--learning_rate $2"
            shift
            shift
            ;;
        --load_checkpoint)
            CHECKPOINT="--load_checkpoint $2"
            shift
            shift
            ;;
        --dataset_path)
            DATASET_PATH="--dataset_path $2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print banner
echo "=" * 80
echo "LLM Training Script for TPU v4-32"
echo "Model Size: $MODEL_SIZE"
echo "GCS Bucket: $GCS_BUCKET"
echo "Project ID: $PROJECT_ID"
echo "=" * 80

# Make scripts executable
chmod +x setup_environment.sh
chmod +x download_datasets.sh
chmod +x tpu_train.py

# Run the training script
python3 tpu_train.py \
    --config model_config.yml \
    --gcs_bucket $GCS_BUCKET \
    --project_id $PROJECT_ID \
    --model_size $MODEL_SIZE \
    $DEBUG $PROFILE $BATCH_SIZE $STEPS $LEARNING_RATE $CHECKPOINT $DATASET_PATH

echo "Training complete!"
