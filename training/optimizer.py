"""
Optimizers for LLM training.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict, Optional, Tuple, Union
import flax


def create_adamw_optimizer(
    learning_rate: Union[float, Callable],
    weight_decay: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mask: Optional[Callable] = None
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer.
    
    Args:
        learning_rate: Learning rate or learning rate schedule
        weight_decay: Weight decay coefficient
        b1: First moment decay
        b2: Second moment decay
        eps: Epsilon for numerical stability
        mask: Function to mask parameters from weight decay
        
    Returns:
        AdamW optimizer
    """
    if mask is None:
        # Default mask excludes bias and layer norm parameters from weight decay
        def mask(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            return {
                k: (k[-1] != "bias" and not k[-1].startswith("layer_norm"))
                for k in flat_params.keys()
            }
    
    # Create optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            mask=mask
        )
    )
    
    return optimizer


def create_lion_optimizer(
    learning_rate: Union[float, Callable],
    weight_decay: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.99,
    mask: Optional[Callable] = None
) -> optax.GradientTransformation:
    """
    Create Lion optimizer.
    
    Args:
        learning_rate: Learning rate or learning rate schedule
        weight_decay: Weight decay coefficient
        b1: First moment decay
        b2: Second moment decay
        mask: Function to mask parameters from weight decay
        
    Returns:
        Lion optimizer
    """
    if mask is None:
        # Default mask excludes bias and layer norm parameters from weight decay
        def mask(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            return {
                k: (k[-1] != "bias" and not k[-1].startswith("layer_norm"))
                for k in flat_params.keys()
            }
    
    # Create optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.lion(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            mask=mask
        )
    )
    
    return optimizer
