#!/bin/bash
# Script to train a 600B parameter LLM with 128K context window on TPU v4-32

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the training with optimized parameters
python3 tpu_train.py \
  --model_size 600b \
  --train_file HuggingFaceFW/fineweb \
  --tokenizer_file gpt2 \
  --batch_size 32 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1.5e-4 \
  --max_steps 500000 \
  --warmup_steps 5000 \
  --max_seq_length 131072 \
  --output_dir ./output \
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
  --log_memory_usage
