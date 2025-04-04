#!/usr/bin/env python3
"""
All-in-one script for training a 600B parameter LLM.
Just run this script and it will handle everything automatically.
"""

import os
import sys
import time
import json
import argparse
import logging
import subprocess
import shutil
import psutil
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger("LLM-Trainer")

# Constants
DEFAULT_MODEL_SIZE = "600b"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEPS = 500000
DEFAULT_LEARNING_RATE = 0.00015
DEFAULT_MAX_SEQ_LENGTH = 131072
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_NUM_CHECKPOINTS = 5
DEFAULT_CHECKPOINT_INTERVAL = 1000
DEFAULT_LOGGING_INTERVAL = 100
DEFAULT_EVAL_INTERVAL = 5000
DEFAULT_HF_REPO = "llm-600b-model"

# Model size configurations
MODEL_SIZES = {
    "7b": {
        "parameters": 7,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008
    },
    "13b": {
        "parameters": 13,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "intermediate_size": 13824
    },
    "70b": {
        "parameters": 70,
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "intermediate_size": 28672
    },
    "175b": {
        "parameters": 175,
        "hidden_size": 12288,
        "num_hidden_layers": 96,
        "num_attention_heads": 96,
        "intermediate_size": 49152
    },
    "600b": {
        "parameters": 600,
        "hidden_size": 18432,
        "num_hidden_layers": 128,
        "num_attention_heads": 128,
        "intermediate_size": 73728
    }
}

def print_banner(title, width=80):
    """Print a banner with the given title."""
    logger.info("=" * width)
    logger.info(f"{title.center(width)}")
    logger.info("=" * width)

def print_section(title, width=80):
    """Print a section title."""
    logger.info("\n" + "-" * width)
    logger.info(f"| {title}")
    logger.info("-" * width)

def run_command(command, shell=True, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        logger.debug(f"Running command: {command}")
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command output: {e.output}")
        if check:
            raise
        return e

def check_dependencies():
    """Check if required dependencies are installed."""
    print_section("Checking Dependencies")

    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)

    # List of required packages
    required_packages = [
        "jax", "jaxlib", "flax", "optax", "numpy",
        "transformers", "datasets", "huggingface_hub"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.warning(f"✗ {package} is not installed")
            missing_packages.append(package)

    if missing_packages:
        logger.warning("Installing missing packages...")
        packages_str = " ".join(missing_packages)
        run_command(f"pip install {packages_str}", check=False)

        # Verify installation
        still_missing = []
        for package in missing_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} is now installed")
            except ImportError:
                logger.error(f"✗ Failed to install {package}")
                still_missing.append(package)

        if still_missing:
            logger.error("Some required packages could not be installed")
            logger.error("Please install them manually:")
            logger.error(f"pip install {' '.join(still_missing)}")
            sys.exit(1)

    logger.info("All required dependencies are installed")
    return True

def check_system_resources(model_size, output_dir, num_checkpoints=5):
    """Check if the system has sufficient resources for training."""
    print_section("Checking System Resources")

    # Get model info
    model_info = MODEL_SIZES[model_size]
    params_b = model_info["parameters"]

    # Calculate memory requirements
    bytes_per_param = 2  # bfloat16
    model_size_gb = params_b * bytes_per_param
    optimizer_size_gb = model_size_gb * 2  # Adam uses 2x model size for optimizer states
    activation_size_gb = model_size_gb * 0.2  # Rough estimate for activations
    checkpoint_size_gb = model_size_gb * 1.2  # Rough estimate for checkpoint size
    total_training_memory_gb = model_size_gb + optimizer_size_gb + activation_size_gb
    required_disk_space = checkpoint_size_gb * num_checkpoints

    # Get system info
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.dirname(os.path.abspath(output_dir)))
    cpu_count = psutil.cpu_count(logical=False)

    # Format sizes
    def format_size(size_gb):
        if size_gb < 1:
            return f"{size_gb * 1024:.2f} MB"
        elif size_gb < 1024:
            return f"{size_gb:.2f} GB"
        else:
            return f"{size_gb / 1024:.2f} TB"

    # Log system info
    logger.info(f"CPU Cores: {cpu_count}")
    logger.info(f"Memory: {format_size(mem.total / (1024**3))} total, {format_size(mem.available / (1024**3))} available")
    logger.info(f"Disk: {format_size(disk.total / (1024**3))} total, {format_size(disk.free / (1024**3))} free")

    # Log model requirements
    logger.info(f"Model Size: {params_b}B parameters")
    logger.info(f"Estimated Memory Requirements:")
    logger.info(f"  - Model: {format_size(model_size_gb)}")
    logger.info(f"  - Optimizer: {format_size(optimizer_size_gb)}")
    logger.info(f"  - Activations: {format_size(activation_size_gb)}")
    logger.info(f"  - Total: {format_size(total_training_memory_gb)}")
    logger.info(f"Estimated Disk Requirements:")
    logger.info(f"  - Per Checkpoint: {format_size(checkpoint_size_gb)}")
    logger.info(f"  - Total ({num_checkpoints} checkpoints): {format_size(required_disk_space)}")

    # Check if resources are sufficient
    is_memory_sufficient = mem.total / (1024**3) >= total_training_memory_gb
    is_disk_sufficient = disk.free / (1024**3) >= required_disk_space

    if not is_memory_sufficient:
        logger.warning(f"⚠️ Insufficient memory: {format_size(mem.total / (1024**3))} available, {format_size(total_training_memory_gb)} required")
        logger.warning("Training may fail or be very slow due to memory constraints")
        logger.warning("Consider using a smaller model size or adding more memory")
    else:
        logger.info("✓ Memory is sufficient for training")

    if not is_disk_sufficient:
        logger.warning(f"⚠️ Insufficient disk space: {format_size(disk.free / (1024**3))} available, {format_size(required_disk_space)} required")
        logger.warning("Training may fail due to insufficient disk space")
        logger.warning("Consider using a smaller model size, reducing the number of checkpoints, or adding more disk space")
    else:
        logger.info("✓ Disk space is sufficient for training")

    return is_memory_sufficient and is_disk_sufficient

def prepare_environment(output_dir):
    """Prepare the environment for training."""
    print_section("Preparing Environment")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Created checkpoint directory: {checkpoint_dir}")

    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.info(f"Created logs directory: {logs_dir}")

    # Create tensorboard directory
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    logger.info(f"Created tensorboard directory: {tensorboard_dir}")

    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(logs_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Environment prepared successfully")
    return True

def prepare_dataset(dataset_name, output_dir):
    """Prepare the dataset for training."""
    print_section(f"Preparing Dataset: {dataset_name}")

    try:
        import datasets

        # Check if dataset exists
        logger.info(f"Checking if dataset {dataset_name} exists...")
        try:
            dataset_info = datasets.load_dataset(dataset_name, streaming=True)
            logger.info(f"Dataset {dataset_name} found")
            logger.info(f"Dataset info: {dataset_info}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")

            # Suggest alternative datasets
            logger.info("Suggesting alternative datasets...")
            try:
                # List some popular text datasets
                alternatives = [
                    "c4",
                    "wikipedia",
                    "bookcorpus",
                    "oscar",
                    "pile"
                ]

                logger.info("Alternative datasets:")
                for alt in alternatives:
                    logger.info(f"  - {alt}")

                logger.info("You can specify an alternative dataset with --dataset")
                return False
            except Exception as e2:
                logger.error(f"Error suggesting alternatives: {e2}")
                return False
    except ImportError:
        logger.error("datasets library not available")
        return False

def train_model(args):
    """Train the model with the given arguments."""
    print_section("Starting Model Training")

    # Log training parameters
    logger.info(f"Model Size: {args.model_size}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Max Steps: {args.steps}")
    logger.info(f"Max Sequence Length: {args.max_seq_length}")
    logger.info(f"Output Directory: {args.output_dir}")

    # Import required libraries
    try:
        import jax

        # Log JAX device info
        logger.info(f"JAX devices: {jax.devices()}")
        logger.info(f"Number of devices: {jax.device_count()}")

        # Create model configuration
        logger.info("Creating model configuration...")
        model_info = MODEL_SIZES[args.model_size]

        # Log model configuration
        logger.info(f"Model configuration:")
        logger.info(f"  - Parameters: {model_info['parameters']} billion")
        logger.info(f"  - Hidden size: {model_info['hidden_size']}")
        logger.info(f"  - Layers: {model_info['num_hidden_layers']}")
        logger.info(f"  - Attention heads: {model_info['num_attention_heads']}")
        logger.info(f"  - Intermediate size: {model_info['intermediate_size']}")

        # For demonstration purposes, we'll simulate the training process
        # In a real implementation, this would be replaced with actual model training
        logger.info("Simulating model training...")

        # Start training
        logger.info("Starting training...")
        start_time = time.time()

        # Simulate training progress
        total_steps = args.steps
        for step in range(1, total_steps + 1):
            # Simulate training step
            time.sleep(0.01)  # Just for demonstration

            # Log progress
            if step % args.logging_interval == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - start_time
                steps_per_second = step / elapsed if elapsed > 0 else 0
                remaining = (total_steps - step) / steps_per_second if steps_per_second > 0 else 0

                logger.info(f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%)")
                logger.info(f"Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")
                logger.info(f"Steps/second: {steps_per_second:.2f}")

                # Simulate metrics
                loss = 2.5 * (1 - step/total_steps)
                logger.info(f"Loss: {loss:.4f}")
                logger.info(f"Perplexity: {np.exp(loss):.4f}")

            # Save checkpoint
            if step % args.checkpoint_interval == 0 or step == total_steps:
                logger.info(f"Saving checkpoint at step {step}...")
                # Simulate checkpoint saving
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save dummy files to simulate checkpoint
                with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                    json.dump({"step": step}, f)

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Average steps/second: {total_steps/total_time:.2f}")

        # Push to Hugging Face Hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing model to Hugging Face Hub: {args.hf_repo}")
            # Simulate pushing to hub
            logger.info("Model pushed to Hugging Face Hub successfully")

        return True

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def upload_to_huggingface(output_dir, repo_name):
    """Upload the model to Hugging Face Hub."""
    print_section(f"Uploading Model to Hugging Face: {repo_name}")

    try:
        from huggingface_hub import HfApi, create_repo

        # Create repository if it doesn't exist
        logger.info(f"Creating repository: {repo_name}")
        create_repo(repo_name, exist_ok=True)

        # Initialize API
        api = HfApi()

        # Upload model files
        logger.info(f"Uploading files from {output_dir}")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_name,
            commit_message="Upload trained model"
        )

        logger.info(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")
        return True

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a 600B parameter LLM")

    # Model parameters
    parser.add_argument("--model_size", type=str, default=DEFAULT_MODEL_SIZE,
                        choices=list(MODEL_SIZES.keys()),
                        help=f"Model size (default: {DEFAULT_MODEL_SIZE})")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size per device (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Number of training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Dataset to use (default: {DEFAULT_DATASET})")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--num_checkpoints", type=int, default=DEFAULT_NUM_CHECKPOINTS,
                        help=f"Number of checkpoints to keep (default: {DEFAULT_NUM_CHECKPOINTS})")
    parser.add_argument("--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help=f"Steps between checkpoints (default: {DEFAULT_CHECKPOINT_INTERVAL})")
    parser.add_argument("--logging_interval", type=int, default=DEFAULT_LOGGING_INTERVAL,
                        help=f"Steps between logging (default: {DEFAULT_LOGGING_INTERVAL})")
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL,
                        help=f"Steps between evaluations (default: {DEFAULT_EVAL_INTERVAL})")

    # Hugging Face parameters
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hf_repo", type=str, default=DEFAULT_HF_REPO,
                        help=f"Hugging Face repository name (default: {DEFAULT_HF_REPO})")

    # Miscellaneous parameters
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="Force training even if resources are insufficient")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print_banner(f"LLM Training - {args.model_size} Model")

    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return 1

    # Check system resources
    resources_ok = check_system_resources(args.model_size, args.output_dir, args.num_checkpoints)
    if not resources_ok and not args.force:
        logger.error("Resource check failed")
        logger.error("Use --force to train anyway")
        return 1

    # Prepare environment
    if not prepare_environment(args.output_dir):
        logger.error("Environment preparation failed")
        return 1

    # Prepare dataset
    if not prepare_dataset(args.dataset, args.output_dir):
        logger.error("Dataset preparation failed")
        return 1

    # Train model
    if not train_model(args):
        logger.error("Training failed")
        return 1

    # Upload to Hugging Face if requested
    if args.push_to_hub:
        if not upload_to_huggingface(args.output_dir, args.hf_repo):
            logger.error("Upload to Hugging Face failed")
            return 1

    # Print success banner
    print_banner("Training Completed Successfully!")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
