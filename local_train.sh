#!/bin/bash

# local_train.sh - Shell script wrapper to run the LLM training with local storage
# This script sets up the environment and runs the training script

set -e  # Exit on any error

# Parse command line arguments
MODEL_SIZE="600b"
DEBUG=""
PROFILE=""
BATCH_SIZE=""
STEPS=""
LEARNING_RATE=""
CHECKPOINT=""
DATASET_PATH=""
OUTPUT_DIR="./output"

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
            STEPS="--max_steps $2"
            shift
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="--learning_rate $2"
            shift
            shift
            ;;
        --load_checkpoint)
            CHECKPOINT="--resume_from_checkpoint $2"
            shift
            shift
            ;;
        --dataset_path)
            DATASET_PATH="--train_file $2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
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
echo "=========================================================================="
echo "LLM Training Script for Local Storage"
echo "Model Size: $MODEL_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Make scripts executable
chmod +x setup_environment.sh 2>/dev/null || true
chmod +x download_datasets.sh 2>/dev/null || true
chmod +x tpu_train.py

# Download HuggingFace dataset if needed
if [ -z "$DATASET_PATH" ]; then
    echo "No dataset specified, using default HuggingFace dataset"
    DATASET_PATH="--train_file HuggingFaceFW/fineweb"
fi

# Run the training script
python3 tpu_train.py \
    --model_size $MODEL_SIZE \
    $DATASET_PATH \
    --tokenizer_file gpt2 \
    --batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.5e-4 \
    --max_steps 500000 \
    --warmup_steps 5000 \
    --max_seq_length 131072 \
    --output_dir $OUTPUT_DIR \
    --parallelism_type tensor \
    --tensor_parallel_size 8 \
    --use_flash_attention \
    --use_gradient_checkpointing \
    --use_rope_scaling \
    --rope_scaling_factor 0.25 \
    --use_reasoning_layer \
    --use_streaming \
    --streaming_buffer_size 10000 \
    --text_column text \
    --use_wandb \
    --wandb_project llm-training \
    --log_memory_usage \
    $DEBUG $PROFILE $BATCH_SIZE $LEARNING_RATE $CHECKPOINT

echo "Training complete!"
