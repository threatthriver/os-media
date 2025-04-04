"""
Embedding layers for the LLM model.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Callable
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Attributes:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        dtype: Data type for embeddings
    """
    vocab_size: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.embed_dim),
            self.dtype
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Apply token embedding.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Token embeddings [batch_size, seq_len, embed_dim]
        """
        return jnp.take(self.embedding, input_ids, axis=0)


class PositionalEmbedding(nn.Module):
    """
    Learned positional embedding layer.

    Attributes:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        dtype: Data type for embeddings
    """
    max_seq_len: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.embed_dim),
            self.dtype
        )

    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Apply positional embedding.

        Args:
            positions: Position indices [batch_size, seq_len]

        Returns:
            Positional embeddings [batch_size, seq_len, embed_dim]
        """
        return jnp.take(self.embedding, positions, axis=0)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with support for long sequences.

    Attributes:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        scale: Scaling factor for RoPE frequencies (for longer contexts)
        dtype: Data type for embeddings
        use_dynamic_scaling: Whether to use dynamic scaling for longer contexts
    """
    dim: int
    max_seq_len: int = 32768  # Increased to support longer contexts
    base: int = 10000
    scale: float = 1.0  # Scaling factor for RoPE frequencies
    dtype: jnp.dtype = jnp.float32
    use_dynamic_scaling: bool = True  # Enable dynamic scaling for longer contexts
    original_max_seq_len: int = 4096  # Original max sequence length for scaling

    def setup(self):
        # Apply scaling for longer contexts if enabled
        effective_base = self.base
        if self.use_dynamic_scaling and self.max_seq_len > self.original_max_seq_len:
            # Dynamic NTK-aware scaling for longer contexts
            # This helps maintain the same level of position sensitivity at longer distances
            scaling_factor = math.log(self.max_seq_len / self.original_max_seq_len) / math.log(2)
            effective_base = self.base * (self.scale ** scaling_factor)

        # Compute frequency bands
        freqs = effective_base ** (-jnp.arange(0, self.dim, 2) / self.dim)
        # Compute position encodings
        pos = jnp.arange(self.max_seq_len)
        # Outer product of positions and frequencies
        freqs = jnp.outer(pos, freqs).astype(self.dtype)
        # Cache cos and sin values
        self.cos_cached = jnp.cos(freqs)
        self.sin_cached = jnp.sin(freqs)

    def _rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        """Rotate half of the dimensions."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: jnp.ndarray, positions: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply rotary positional embedding.

        Args:
            x: Input tensor [batch_size, seq_len, ..., dim]
            positions: Optional position indices [batch_size, seq_len]

        Returns:
            Tensor with rotary positional encoding applied
        """
        seq_len = x.shape[1]

        if positions is None:
            positions = jnp.arange(seq_len)

        # Ensure positions are within bounds
        positions = jnp.clip(positions, 0, self.max_seq_len - 1)

        # Get cos and sin values for the positions
        cos = jnp.take(self.cos_cached, positions, axis=0)[:, :seq_len]
        sin = jnp.take(self.sin_cached, positions, axis=0)[:, :seq_len]

        # Reshape for broadcasting
        cos = cos.reshape(cos.shape + (1,) * (x.ndim - 3))
        sin = sin.reshape(sin.shape + (1,) * (x.ndim - 3))

        # Apply rotary embedding
        return x * cos + self._rotate_half(x) * sin


def get_rope_embedding(
    dim: int,
    max_seq_len: int = 32768,
    base: int = 10000,
    scale: float = 1.0,
    use_dynamic_scaling: bool = True,
    original_max_seq_len: int = 4096,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get rotary positional embedding (RoPE) sin and cos values with support for long sequences.

    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length (default: 32768 for long context)
        base: Base for frequency computation
        scale: Scaling factor for RoPE frequencies (for longer contexts)
        use_dynamic_scaling: Whether to use dynamic scaling for longer contexts
        original_max_seq_len: Original max sequence length for scaling
        dtype: Data type for embeddings

    Returns:
        Tuple of (cos, sin) arrays for RoPE
    """
    # Apply scaling for longer contexts if enabled
    effective_base = base
    if use_dynamic_scaling and max_seq_len > original_max_seq_len:
        # Dynamic NTK-aware scaling for longer contexts
        scaling_factor = math.log(max_seq_len / original_max_seq_len) / math.log(2)
        effective_base = base * (scale ** scaling_factor)

    # Compute frequency bands
    freqs = effective_base ** (-jnp.arange(0, dim, 2) / dim)
    # Compute position encodings
    pos = jnp.arange(max_seq_len)
    # Outer product of positions and frequencies
    freqs = jnp.outer(pos, freqs).astype(dtype)
    # Compute cos and sin values
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)

    return cos, sin
