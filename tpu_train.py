#!/usr/bin/env python3
"""
Main training script for LLM training on TPU v4-32.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import jax
import jax.numpy as jnp
import flax
import tensorflow as tf
import numpy as np
import sentencepiece as spm
from functools import partial

# Import local modules
from model.llm import LLM, LLMConfig
from data.tokenizer import SentencePieceTokenizer
from data.dataset import TextDataset, load_jsonl_dataset
from data.dataloader import TPUDataLoader
from training.trainer import Trainer, TrainingState, TrainingConfig as TrainerConfig
from training.optimizer import create_adamw_optimizer, create_lion_optimizer
from training.scheduler import create_linear_warmup_cosine_decay_schedule
from parallelism.data_parallel import DataParallel
from parallelism.tensor_parallel import TensorParallel
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logging import setup_logger, log_metrics, create_summary_writer, log_metrics_to_tensorboard
from config import TrainingConfig, get_model_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LLM on TPU v4-32")

    # Model parameters
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "70b", "175b", "600b"],
                        help="Model size")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")

    # Dataset parameters
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training file")
    parser.add_argument("--eval_file", type=str, default="",
                        help="Path to evaluation file")
    parser.add_argument("--tokenizer_file", type=str, required=True,
                        help="Path to tokenizer file")
    parser.add_argument("--max_seq_length", type=int, default=32768,
                        help="Maximum sequence length")

    # Parallelism parameters
    parser.add_argument("--parallelism_type", type=str, default="data", choices=["data", "tensor"],
                        help="Type of parallelism")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="Number of tensor parallel devices")

    # Performance optimization parameters
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                        help="Use flash attention for efficiency")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to save memory")

    # Long context support parameters
    parser.add_argument("--use_rope_scaling", action="store_true", default=True,
                        help="Use RoPE scaling for longer contexts")
    parser.add_argument("--rope_scaling_factor", type=float, default=0.5,
                        help="Scaling factor for RoPE frequencies")

    # Reasoning capabilities parameters
    parser.add_argument("--use_reasoning_layer", action="store_true", default=True,
                        help="Use additional reasoning layers")
    parser.add_argument("--num_reasoning_layers", type=int, default=None,
                        help="Number of additional reasoning layers (overrides model config)")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between checkpoints")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Number of steps between evaluations")

    # Miscellaneous parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default="",
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def create_config(args):
    """Create training configuration."""
    # Get model configuration
    model_config = get_model_config(args.model_size)

    # Override model configuration with command line arguments if provided
    if args.num_reasoning_layers is not None:
        model_config.num_reasoning_layers = args.num_reasoning_layers

    # Update model configuration with command line arguments
    model_config.use_flash_attention = args.use_flash_attention
    model_config.use_gradient_checkpointing = args.use_gradient_checkpointing
    model_config.use_rope_scaling = args.use_rope_scaling
    model_config.rope_scaling_factor = args.rope_scaling_factor
    model_config.use_reasoning_layer = args.use_reasoning_layer

    # Create training configuration
    config = TrainingConfig(
        output_dir=args.output_dir,
        model_config=model_config,

        # Training parameters
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,

        # Dataset parameters
        train_file=args.train_file,
        eval_file=args.eval_file,
        tokenizer_file=args.tokenizer_file,
        max_seq_length=args.max_seq_length,

        # Parallelism parameters
        parallelism_type=args.parallelism_type,
        tensor_parallel_size=args.tensor_parallel_size,

        # Performance optimization parameters
        use_flash_attention=args.use_flash_attention,
        use_gradient_checkpointing=args.use_gradient_checkpointing,

        # Long context support parameters
        use_rope_scaling=args.use_rope_scaling,
        rope_scaling_factor=args.rope_scaling_factor,

        # Reasoning capabilities parameters
        use_reasoning_layer=args.use_reasoning_layer,
        num_reasoning_layers=args.num_reasoning_layers if args.num_reasoning_layers is not None else model_config.num_reasoning_layers,
        reasoning_intermediate_size=model_config.reasoning_intermediate_size,

        # Logging parameters
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,

        # Miscellaneous parameters
        seed=args.seed
    )

    return config


def setup_parallelism(config):
    """Set up parallelism."""
    if config.parallelism_type == "data":
        return DataParallel()
    elif config.parallelism_type == "tensor":
        return TensorParallel(num_tp=config.tensor_parallel_size)
    else:
        raise ValueError(f"Parallelism type {config.parallelism_type} not supported")


def create_model(config):
    """Create model."""
    return LLM(config.model_config)


def create_optimizer(config, num_train_steps):
    """Create optimizer."""
    # Create learning rate schedule
    lr_schedule = create_linear_warmup_cosine_decay_schedule(
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=num_train_steps - config.warmup_steps,
        final_learning_rate_factor=0.1
    )

    # Create optimizer
    if config.optimizer == "adamw":
        return create_adamw_optimizer(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_epsilon
        )
    elif config.optimizer == "lion":
        return create_lion_optimizer(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            b1=config.adam_beta1,
            b2=config.adam_beta2
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported")


def create_train_state(config, model, optimizer, rng):
    """Create training state."""
    # Create dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)

    # Initialize model parameters
    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init(params_rng, dummy_input)

    # Create training state
    return TrainingState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng,
        loss_scale=1.0
    )


def load_tokenizer(config):
    """Load tokenizer."""
    return SentencePieceTokenizer(config.tokenizer_file)


def load_dataset(config, tokenizer):
    """Load dataset."""
    # Load training dataset
    train_dataset = load_jsonl_dataset(
        file_path=config.train_file,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )

    # Load evaluation dataset
    eval_dataset = None
    if config.eval_file:
        eval_dataset = load_jsonl_dataset(
            file_path=config.eval_file,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
        )

    return train_dataset, eval_dataset


def create_data_loaders(config, train_dataset, eval_dataset, tokenizer):
    """Create data loaders."""
    # Create training data loader
    train_loader = TPUDataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pad_token_id=tokenizer.pad_token_id
    )

    # Create evaluation data loader
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = TPUDataLoader(
            dataset=eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return train_loader, eval_loader


def main():
    """Main function optimized for TPU v4-32."""
    # Parse arguments
    args = parse_args()

    # Print TPU configuration information
    print("TPU Configuration:")
    print(f"Number of TPU devices: {jax.device_count()}")
    print(f"TPU devices: {jax.devices()}")
    print(f"JAX process index: {jax.process_index()}")
    print(f"JAX process count: {jax.process_count()}")
    print(f"JAX local devices: {jax.local_devices()}")
    print(f"JAX local device count: {jax.local_device_count()}")

    # Create configuration
    config = create_config(args)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logger(
        name="tpu_train",
        log_file=os.path.join(config.output_dir, "train.log")
    )

    # Log configuration
    logger.info(f"Configuration: {config}")

    # Log hardware information
    logger.info(f"Training on TPU v4-32 with {jax.device_count()} devices")
    logger.info(f"Model size: {args.model_size} ({config.model_config.hidden_size} hidden size, "
               f"{config.model_config.num_hidden_layers} layers)")

    # Calculate approximate model size
    param_count = (
        # Embedding parameters
        config.model_config.vocab_size * config.model_config.hidden_size +
        # Transformer layers
        config.model_config.num_hidden_layers * (
            # Self-attention
            4 * config.model_config.hidden_size * config.model_config.hidden_size +
            # Feed-forward
            2 * config.model_config.hidden_size * config.model_config.intermediate_size +
            # Layer normalization
            4 * config.model_config.hidden_size
        ) +
        # Reasoning layers if enabled
        (config.model_config.use_reasoning_layer and config.model_config.num_reasoning_layers) * (
            # Self-attention
            4 * config.model_config.hidden_size * config.model_config.hidden_size +
            # Feed-forward with larger hidden dimension
            2 * config.model_config.hidden_size * config.model_config.reasoning_intermediate_size +
            # Layer normalization
            4 * config.model_config.hidden_size
        ) +
        # Final layer normalization
        config.model_config.hidden_size +
        # Output projection
        config.model_config.hidden_size * config.model_config.vocab_size
    )

    # Log parameter count
    logger.info(f"Approximate parameter count: {param_count / 1e9:.2f} billion parameters")

    # Calculate memory requirements
    bytes_per_param = 2 if config.dtype == jnp.bfloat16 else 4  # bfloat16 or float32
    model_size_gb = param_count * bytes_per_param / 1e9
    optimizer_size_gb = model_size_gb * 2  # Adam uses 2x model size for optimizer states
    activation_size_gb = model_size_gb * 0.2  # Rough estimate for activations
    total_memory_gb = model_size_gb + optimizer_size_gb + activation_size_gb

    # Log memory requirements
    logger.info(f"Estimated memory requirements:")
    logger.info(f"  Model parameters: {model_size_gb:.2f} GB")
    logger.info(f"  Optimizer states: {optimizer_size_gb:.2f} GB")
    logger.info(f"  Activations: {activation_size_gb:.2f} GB")
    logger.info(f"  Total: {total_memory_gb:.2f} GB")

    # Check if memory requirements exceed available TPU memory
    tpu_memory_gb = 32 * jax.device_count()  # Each TPU v4 has 32GB HBM
    logger.info(f"Available TPU memory: {tpu_memory_gb:.2f} GB")
    if total_memory_gb > tpu_memory_gb * 0.9:  # Leave 10% margin
        logger.warning(f"Memory requirements ({total_memory_gb:.2f} GB) may exceed available TPU memory ({tpu_memory_gb:.2f} GB)")
        logger.warning("Consider enabling gradient checkpointing and using a smaller batch size")

    # Set random seed
    rng = jax.random.PRNGKey(config.seed)

    # Measure initialization time
    start_time = time.time()

    # Set up parallelism with optimized configuration for TPU v4-32
    parallel = setup_parallelism(config)

    # Create model
    model = create_model(config)
    logger.info(f"Model created in {time.time() - start_time:.2f} seconds")

    # Create optimizer with memory-efficient configuration
    optimizer = create_optimizer(config, config.max_steps)

    # Create training state
    state_start_time = time.time()
    state = create_train_state(config, model, optimizer, rng)
    logger.info(f"Training state created in {time.time() - state_start_time:.2f} seconds")

    # Shard parameters with optimized sharding strategy
    shard_start_time = time.time()
    state = state.replace(params=parallel.shard_params(state.params))
    logger.info(f"Parameters sharded in {time.time() - shard_start_time:.2f} seconds")

    # Load checkpoint if requested
    if args.resume_from_checkpoint:
        checkpoint_start_time = time.time()
        state, step = load_checkpoint(args.resume_from_checkpoint, state)
        logger.info(f"Checkpoint loaded in {time.time() - checkpoint_start_time:.2f} seconds")

    # Load tokenizer
    tokenizer_start_time = time.time()
    tokenizer = load_tokenizer(config)
    logger.info(f"Tokenizer loaded in {time.time() - tokenizer_start_time:.2f} seconds")

    # Load dataset with optimized loading
    dataset_start_time = time.time()
    train_dataset, eval_dataset = load_dataset(config, tokenizer)
    logger.info(f"Datasets loaded in {time.time() - dataset_start_time:.2f} seconds")

    # Create data loaders with optimized configuration for TPU v4-32
    dataloader_start_time = time.time()
    train_loader, eval_loader = create_data_loaders(
        config,
        train_dataset,
        eval_dataset,
        tokenizer
    )
    logger.info(f"Data loaders created in {time.time() - dataloader_start_time:.2f} seconds")

    # Create TensorBoard summary writer
    summary_writer = create_summary_writer(
        os.path.join(config.output_dir, "tensorboard")
    )

    # Create trainer configuration with optimized settings for TPU v4-32
    trainer_config = TrainerConfig(
        model_config=config.model_config,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        output_dir=config.output_dir,
        seed=config.seed,
        dtype=config.dtype,
        # Additional optimized settings for TPU v4-32
        use_pjit=True,  # Use pjit for better performance
        use_scan=True,  # Use scan for layer iteration
        use_remat=config.model_config.use_gradient_checkpointing,  # Use rematerialization for memory efficiency
        use_sharded_optim=True,  # Use sharded optimizer states
        profile_steps=100,  # Profile every 100 steps
        async_checkpointing=True,  # Use async checkpointing for better performance
    )

    # Create trainer
    trainer = Trainer(
        config=trainer_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        state=state,
        parallel=parallel,  # Pass parallelism object for optimized training
    )

    # Log total initialization time
    logger.info(f"Total initialization time: {time.time() - start_time:.2f} seconds")

    # Calculate estimated training time
    steps_per_day = 24 * 60 * 60 / (5 * 60)  # Assuming 5 minutes per 100 steps (rough estimate)
    estimated_days = config.max_steps / steps_per_day
    logger.info(f"Estimated training time: {estimated_days:.2f} days for {config.max_steps} steps")

    # Train model with performance monitoring
    try:
        train_start_time = time.time()
        trainer.train()
        train_duration = time.time() - train_start_time
        logger.info(f"Training completed in {train_duration / 3600:.2f} hours")
        logger.info(f"Average training speed: {config.max_steps / train_duration:.2f} steps/second")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
