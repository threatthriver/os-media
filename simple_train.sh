#!/bin/bash
# Simplified script to run the LLM training

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the simplified training script
python3 simple_tpu_train.py \
  --model_size 600b \
  --batch_size 32 \
  --learning_rate 1.5e-4 \
  --max_steps 500000 \
  --warmup_steps 5000 \
  --output_dir ./output

echo "Simplified training script completed!"
