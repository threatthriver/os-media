"""
Logging utilities for LLM training.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    logger: Optional[logging.Logger] = None,
    prefix: str = "",
    log_to_console: bool = True
) -> None:
    """
    Log metrics.
    
    Args:
        metrics: Dictionary of metrics
        step: Training step
        logger: Logger
        prefix: Prefix for metric names
        log_to_console: Whether to log to console
    """
    # Add prefix to metric names
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    # Convert metrics to Python types
    metrics = {
        k: float(v) if isinstance(v, (np.ndarray, jnp.ndarray)) else v
        for k, v in metrics.items()
    }
    
    # Log to console
    if log_to_console:
        print(f"Step {step}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Log to logger
    if logger is not None:
        logger.info(f"Step {step}: {metrics}")


def create_summary_writer(log_dir: str) -> tf.summary.SummaryWriter:
    """
    Create TensorBoard summary writer.
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        TensorBoard summary writer
    """
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create summary writer
    return tf.summary.create_file_writer(log_dir)


def log_metrics_to_tensorboard(
    metrics: Dict[str, Any],
    step: int,
    writer: tf.summary.SummaryWriter,
    prefix: str = ""
) -> None:
    """
    Log metrics to TensorBoard.
    
    Args:
        metrics: Dictionary of metrics
        step: Training step
        writer: TensorBoard summary writer
        prefix: Prefix for metric names
    """
    # Add prefix to metric names
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    # Convert metrics to Python types
    metrics = {
        k: float(v) if isinstance(v, (np.ndarray, jnp.ndarray)) else v
        for k, v in metrics.items()
    }
    
    # Log metrics to TensorBoard
    with writer.as_default():
        for k, v in metrics.items():
            if isinstance(v, float):
                tf.summary.scalar(k, v, step=step)
            elif isinstance(v, (list, tuple)) and all(isinstance(x, float) for x in v):
                tf.summary.histogram(k, v, step=step)
    
    # Flush writer
    writer.flush()


def log_text_to_tensorboard(
    text: str,
    tag: str,
    step: int,
    writer: tf.summary.SummaryWriter
) -> None:
    """
    Log text to TensorBoard.
    
    Args:
        text: Text to log
        tag: Tag for text
        step: Training step
        writer: TensorBoard summary writer
    """
    # Log text to TensorBoard
    with writer.as_default():
        tf.summary.text(tag, text, step=step)
    
    # Flush writer
    writer.flush()


def log_model_summary(
    model: Any,
    input_shape: tuple,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log model summary.
    
    Args:
        model: Model
        input_shape: Input shape
        logger: Logger
    """
    # Create dummy input
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    
    # Initialize model
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # Count parameters
    param_count = sum(
        np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)
    )
    
    # Log model summary
    summary = f"Model summary:\n"
    summary += f"  Input shape: {input_shape}\n"
    summary += f"  Parameter count: {param_count:,}\n"
    
    # Log to console
    print(summary)
    
    # Log to logger
    if logger is not None:
        logger.info(summary)
