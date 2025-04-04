#!/bin/bash

# train_with_local_storage.sh - Script to manage LLM training with local storage
# This script handles the entire training process, including resource checking,
# dataset preparation, and training with local storage.

set -e  # Exit on any error

# Default parameters
MODEL_SIZE="600b"
OUTPUT_DIR="./output"
BATCH_SIZE=32
STEPS=500000
LEARNING_RATE=0.00015
MAX_SEQ_LENGTH=131072
DATASET="HuggingFaceFW/fineweb"
NUM_CHECKPOINTS=5
CHECKPOINT_INTERVAL=1000
LOGGING_INTERVAL=100
EVAL_INTERVAL=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_size)
            MODEL_SIZE="$2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --steps)
            STEPS="$2"
            shift
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --num_checkpoints)
            NUM_CHECKPOINTS="$2"
            shift
            shift
            ;;
        --checkpoint_interval)
            CHECKPOINT_INTERVAL="$2"
            shift
            shift
            ;;
        --logging_interval)
            LOGGING_INTERVAL="$2"
            shift
            shift
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift
            shift
            ;;
        --resume)
            RESUME="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-latest"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_size SIZE           Model size (7b, 13b, 70b, 175b, 600b) [default: 600b]"
            echo "  --output_dir DIR            Output directory [default: ./output]"
            echo "  --batch_size SIZE           Batch size per device [default: 32]"
            echo "  --steps STEPS               Number of training steps [default: 500000]"
            echo "  --learning_rate RATE        Learning rate [default: 0.00015]"
            echo "  --max_seq_length LENGTH     Maximum sequence length [default: 131072]"
            echo "  --dataset DATASET           Dataset to use [default: HuggingFaceFW/fineweb]"
            echo "  --num_checkpoints NUM       Number of checkpoints to keep [default: 5]"
            echo "  --checkpoint_interval INT   Steps between checkpoints [default: 1000]"
            echo "  --logging_interval INT      Steps between logging [default: 100]"
            echo "  --eval_interval INT         Steps between evaluations [default: 5000]"
            echo "  --resume                    Resume training from latest checkpoint"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo "=========================================================================="
echo "LLM Training with Local Storage"
echo "Model Size: $MODEL_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Batch Size: $BATCH_SIZE"
echo "Steps: $STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "=========================================================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python3 -c "import jax, flax, optax, numpy" 2>/dev/null || {
    echo "Error: Required packages are not installed"
    echo "Please install the required packages:"
    echo "pip install jax jaxlib flax optax numpy"
    exit 1
}

# Check resources
echo "Checking resources..."
python3 check_resources.py --model_size $MODEL_SIZE --output_dir $OUTPUT_DIR --num_checkpoints $NUM_CHECKPOINTS || {
    echo "Error: Insufficient resources for training"
    exit 1
}

# Make scripts executable
chmod +x local_train.sh

# Run the training script
echo "Starting training..."
./local_train.sh \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --learning_rate $LEARNING_RATE \
    --dataset_path $DATASET \
    --output_dir $OUTPUT_DIR \
    $RESUME

echo "Training complete!"
