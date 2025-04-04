"""
Learning rate schedulers for LLM training.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, Tuple, Union
import math


def create_cosine_decay_schedule(
    learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
    alpha: float = 0.0
) -> Callable[[int], float]:
    """
    Create cosine decay learning rate schedule with linear warmup.
    
    Args:
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps
        alpha: Minimum learning rate factor
        
    Returns:
        Learning rate schedule function
    """
    def schedule(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return learning_rate * step / max(1, warmup_steps)
        else:
            # Cosine decay
            decay_ratio = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
            decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(1.0, decay_ratio)))
            return learning_rate * (alpha + (1.0 - alpha) * decay_factor)
    
    return schedule


def create_linear_warmup_cosine_decay_schedule(
    learning_rate: float,
    warmup_steps: int,
    decay_steps: int,
    final_learning_rate_factor: float = 0.1
) -> optax.Schedule:
    """
    Create learning rate schedule with linear warmup and cosine decay.
    
    Args:
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps
        final_learning_rate_factor: Final learning rate as a fraction of peak
        
    Returns:
        Learning rate schedule
    """
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=decay_steps,
        alpha=final_learning_rate_factor
    )
    
    return optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )


def create_warmup_cosine_decay_schedule_with_plateau(
    learning_rate: float,
    warmup_steps: int,
    plateau_steps: int,
    decay_steps: int,
    final_learning_rate_factor: float = 0.1
) -> Callable[[int], float]:
    """
    Create learning rate schedule with linear warmup, plateau, and cosine decay.
    
    Args:
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        plateau_steps: Number of steps to maintain peak learning rate
        decay_steps: Number of decay steps
        final_learning_rate_factor: Final learning rate as a fraction of peak
        
    Returns:
        Learning rate schedule function
    """
    def schedule(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return learning_rate * step / max(1, warmup_steps)
        elif step < warmup_steps + plateau_steps:
            # Plateau at peak learning rate
            return learning_rate
        else:
            # Cosine decay
            decay_step = step - warmup_steps - plateau_steps
            decay_ratio = decay_step / max(1, decay_steps)
            decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(1.0, decay_ratio)))
            return learning_rate * (final_learning_rate_factor + (1.0 - final_learning_rate_factor) * decay_factor)
    
    return schedule
