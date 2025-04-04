"""
Activation functions for the LLM model.
"""

import jax
import jax.numpy as jnp
from typing import Callable


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        GELU activation applied to input
    """
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def swiglu(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    Used in modern LLMs like PaLM and Gemini.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        SwiGLU activation applied to inputs
    """
    return x * jax.nn.sigmoid(y)


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        ReLU activation applied to input
    """
    return jnp.maximum(0, x)


class GELU:
    """GELU activation function class."""
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return gelu(x)


class SwiGLU:
    """SwiGLU activation function class."""
    
    def __call__(self, x: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
        return swiglu(x, gate)


class ReLU:
    """ReLU activation function class."""
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return relu(x)


def get_activation_fn(name: str) -> Callable:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
        
    Raises:
        ValueError: If activation function is not supported
    """
    if name.lower() == 'gelu':
        return gelu
    elif name.lower() == 'swiglu':
        return swiglu
    elif name.lower() == 'relu':
        return relu
    else:
        raise ValueError(f"Activation function {name} not supported")
