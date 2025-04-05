#!/usr/bin/env python3
"""
Script to install dependencies for Indian language model training.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Dependency-Installer")

def run_command(command):
    """Run a shell command and handle errors."""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install required dependencies."""
    # Core dependencies
    core_deps = [
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "numpy",
        "transformers",
        "datasets",
        "huggingface_hub",
        "sentencepiece",
        "wandb",
        "tqdm",
        "einops",
        "psutil",
        "regex"
    ]
    
    # Install core dependencies
    logger.info("Installing core dependencies...")
    success = run_command(f"pip install {' '.join(core_deps)}")
    
    if not success:
        logger.error("Failed to install core dependencies")
        return False
    
    # Check if dependencies were installed successfully
    logger.info("Checking if dependencies were installed successfully...")
    for dep in core_deps:
        try:
            __import__(dep.split('[')[0])  # Handle cases like 'package[extra]'
            logger.info(f"✓ {dep} is installed")
        except ImportError:
            logger.error(f"✗ {dep} is not installed")
            return False
    
    logger.info("All dependencies installed successfully")
    return True

if __name__ == "__main__":
    if install_dependencies():
        logger.info("Dependencies installed successfully")
        sys.exit(0)
    else:
        logger.error("Failed to install dependencies")
        sys.exit(1)
