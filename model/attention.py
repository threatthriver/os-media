"""
Attention mechanisms for the LLM model.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, Callable, Union
import math
import functools
from einops import rearrange, repeat

from model.embedding import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Attributes:
        dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout_rate: Dropout probability
        dtype: Data type for computations
    """
    dim: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Determine head dimension if not provided
        self.actual_head_dim = self.head_dim or self.dim // self.num_heads

        # Projection matrices
        self.q_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="q_proj"
        )

        self.k_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="k_proj"
        )

        self.v_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="v_proj"
        )

        self.out_proj = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="out_proj"
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply multi-head attention.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and values
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)

        # Handle cached key and values for incremental decoding
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        # Save key and value for future use if caching
        present_key_value = (k, v) if use_cache else None

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.actual_head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)

        # Compute attention output
        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = jnp.matmul(attention_weights, v)

        # Transpose and reshape to [batch_size, seq_len, dim]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_heads * self.actual_head_dim)

        # Project to output dimension
        output = self.out_proj(attention_output)

        outputs = (output, attention_weights, present_key_value) if output_attentions else (output, None, present_key_value)

        return outputs


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention mechanism.
    Uses a single key and value head for multiple query heads.

    Attributes:
        dim: Hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads (usually 1 or a small number)
        head_dim: Dimension of each attention head
        dropout_rate: Dropout probability
        dtype: Data type for computations
    """
    dim: int
    num_query_heads: int
    num_kv_heads: int = 1
    head_dim: Optional[int] = None
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Determine head dimension if not provided
        self.actual_head_dim = self.head_dim or self.dim // self.num_query_heads

        # Projection matrices
        self.q_proj = nn.Dense(
            features=self.num_query_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="q_proj"
        )

        self.k_proj = nn.Dense(
            features=self.num_kv_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="k_proj"
        )

        self.v_proj = nn.Dense(
            features=self.num_kv_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="v_proj"
        )

        self.out_proj = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="out_proj"
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply multi-query attention.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and values
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, num_query_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.actual_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.actual_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.actual_head_dim)

        # Handle cached key and values for incremental decoding
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        # Save key and value for future use if caching
        present_key_value = (k, v) if use_cache else None

        # Transpose to [batch_size, num_*_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Repeat k and v for each query head
        if self.num_kv_heads < self.num_query_heads:
            # Calculate how many times to repeat
            repeats = self.num_query_heads // self.num_kv_heads
            # Repeat k and v along the head dimension
            k = jnp.repeat(k, repeats, axis=1)
            v = jnp.repeat(v, repeats, axis=1)

        # Compute attention scores
        # [batch_size, num_query_heads, seq_len, seq_len]
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.actual_head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)

        # Compute attention output
        # [batch_size, num_query_heads, seq_len, head_dim]
        attention_output = jnp.matmul(attention_weights, v)

        # Transpose and reshape to [batch_size, seq_len, dim]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_query_heads * self.actual_head_dim)

        # Project to output dimension
        output = self.out_proj(attention_output)

        outputs = (output, attention_weights, present_key_value) if output_attentions else (output, None, present_key_value)

        return outputs


def flash_attention(q, k, v, mask=None, dropout_rate=0.0, deterministic=True, causal=True, block_size=128):
    """
    Implements optimized Flash Attention algorithm for TPU v4-32 with blocked computation.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len, head_dim]
        mask: Attention mask [batch_size, 1, seq_len, seq_len]
        dropout_rate: Dropout probability
        deterministic: Whether to use deterministic operations (no dropout)
        causal: Whether to use causal masking
        block_size: Block size for chunked attention computation

    Returns:
        Output tensor [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Scaled dot-product
    q = q * scale

    # For short sequences, use standard attention
    if seq_len <= block_size:
        # Compute attention scores
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1))

        # Apply causal mask if needed
        if causal:
            causal_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=1)
            causal_mask = jnp.expand_dims(jnp.expand_dims(causal_mask, 0), 0)  # [1, 1, seq_len, seq_len]
            scores = jnp.where(causal_mask, jnp.finfo(scores.dtype).min, scores)

        # Apply attention mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply dropout
        if dropout_rate > 0.0 and not deterministic:
            attention_weights = nn.dropout(attention_weights, rate=dropout_rate, deterministic=deterministic)

        # Compute attention output
        output = jnp.matmul(attention_weights, v)

        return output

    # For long sequences, use blocked attention computation
    # This implementation is optimized for TPU by using blocks that fit in HBM

    # Pad sequence length to multiple of block_size for efficient blocking
    padded_seq_len = ((seq_len + block_size - 1) // block_size) * block_size
    pad_len = padded_seq_len - seq_len

    if pad_len > 0:
        # Pad inputs
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
    else:
        q_padded, k_padded, v_padded = q, k, v

    # Initialize output
    output_padded = jnp.zeros((batch_size, num_heads, padded_seq_len, head_dim), dtype=q.dtype)

    # Define a scan function for processing blocks
    def block_scan_fn(carry, idx):
        block_start = idx * block_size
        block_end = block_start + block_size
        q_block = jax.lax.dynamic_slice(
            q_padded, (0, 0, block_start, 0),
            (batch_size, num_heads, block_size, head_dim)
        )

        # Compute attention for this block
        attn_weights = jnp.matmul(q_block, jnp.swapaxes(k_padded, -2, -1))

        # Apply causal mask if needed
        if causal:
            # Create causal mask for this block
            row_idx = jnp.arange(block_size) + block_start
            col_idx = jnp.arange(padded_seq_len)
            causal_mask = jnp.less(row_idx[:, None], col_idx[None, :])
            causal_mask = jnp.logical_not(causal_mask)
            causal_mask = jnp.expand_dims(jnp.expand_dims(causal_mask, 0), 0)
            attn_weights = jnp.where(causal_mask, jnp.finfo(attn_weights.dtype).min, attn_weights)

        # Apply attention mask if provided
        if mask is not None:
            # Slice the mask for this block
            if mask.shape[-2] == 1:  # Broadcast mask
                mask_block = mask
            else:
                mask_block = jax.lax.dynamic_slice(
                    mask, (0, 0, block_start, 0),
                    (batch_size, 1, block_size, mask.shape[-1])
                )
            attn_weights = attn_weights + mask_block

        # Apply softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply dropout
        if dropout_rate > 0.0 and not deterministic:
            attn_weights = nn.dropout(attn_weights, rate=dropout_rate, deterministic=deterministic)

        # Compute output for this block
        block_output = jnp.matmul(attn_weights, v_padded)

        # Update output
        output_padded_updated = jax.lax.dynamic_update_slice(
            carry, block_output, (0, 0, block_start, 0)
        )

        return output_padded_updated, None

    # Process blocks
    num_blocks = padded_seq_len // block_size
    output_padded, _ = jax.lax.scan(
        block_scan_fn, output_padded, jnp.arange(num_blocks)
    )

    # Slice to get original sequence length
    output = jax.lax.dynamic_slice(
        output_padded, (0, 0, 0, 0),
        (batch_size, num_heads, seq_len, head_dim)
    )

    return output


class FlashAttention(nn.Module):
    """
    Optimized Flash Attention implementation for TPU v4-32 with support for very long sequences.

    Attributes:
        dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout_rate: Dropout probability
        dtype: Data type for computations
        use_causal_mask: Whether to use causal masking
        block_size: Block size for chunked attention computation
        use_fused_attention: Whether to use fused attention operations
    """
    dim: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    use_causal_mask: bool = True
    block_size: int = 128  # Optimal block size for TPU v4-32
    use_fused_attention: bool = True  # Use fused attention operations when available

    def setup(self):
        # Determine head dimension if not provided
        self.actual_head_dim = self.head_dim or self.dim // self.num_heads

        # Round head dimension to multiple of 8 for TPU efficiency
        if self.actual_head_dim % 8 != 0:
            print(f"Warning: Head dimension {self.actual_head_dim} is not a multiple of 8. "
                  f"This may reduce TPU efficiency.")

        # Projection matrices with optimized initialization for stability
        self.q_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='normal'
            ),
            name="q_proj"
        )

        self.k_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='normal'
            ),
            name="k_proj"
        )

        self.v_proj = nn.Dense(
            features=self.num_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='normal'
            ),
            name="v_proj"
        )

        self.out_proj = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode='fan_out', distribution='normal'
            ),
            name="out_proj"
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,  # Unused but kept for API compatibility
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply optimized flash attention for TPU v4-32.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len] (unused but kept for API compatibility)
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and values
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Check if sequence length is compatible with block size
        if seq_len > 32768 and self.block_size < 256:
            # Adjust block size for very long sequences
            adjusted_block_size = min(512, ((seq_len + 511) // 512) * 512)
            print(f"Adjusting block size to {adjusted_block_size} for sequence length {seq_len}")
            block_size = adjusted_block_size
        else:
            block_size = self.block_size

        # Project inputs to queries, keys, and values with optimized memory layout
        # Use jit to optimize the projection operations
        @jax.jit
        def project_qkv(states):
            q = self.q_proj(states)
            k = self.k_proj(states)
            v = self.v_proj(states)
            return q, k, v

        q, k, v = project_qkv(hidden_states)

        # Reshape to [batch_size, seq_len, num_heads, head_dim] with optimized memory layout
        # This reshaping is optimized for TPU memory access patterns
        q = q.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.actual_head_dim)

        # Handle cached key and values for incremental decoding
        key_seq_len = seq_len
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
            key_seq_len = k.shape[1]  # Update key sequence length

        # Save key and value for future use if caching
        present_key_value = (k, v) if use_cache else None

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        # This transpose is optimized for TPU memory access patterns
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Try to use JAX's built-in optimized attention if available and enabled
        use_jax_attention = self.use_fused_attention and hasattr(jax.lax, 'dot_general_attention')

        if use_jax_attention and not output_attentions and seq_len <= 4096:
            # Use JAX's built-in optimized attention for shorter sequences
            try:
                # Prepare mask for JAX attention
                if attention_mask is not None:
                    # Convert mask to the format expected by dot_general_attention
                    bias = attention_mask
                else:
                    bias = None

                # Use JAX's optimized attention
                attention_output = jax.lax.dot_general_attention(
                    q, k, v, bias=bias, precision=jax.lax.Precision.DEFAULT
                )
            except (AttributeError, TypeError) as e:
                # Fall back to custom implementation if JAX's optimized attention fails
                print(f"Warning: JAX optimized attention failed, falling back to custom implementation: {e}")
                use_jax_attention = False
        else:
            use_jax_attention = False

        if not use_jax_attention:
            # Apply our optimized flash attention implementation
            attention_output = flash_attention(
                q=q,
                k=k,
                v=v,
                mask=attention_mask,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                causal=self.use_causal_mask,
                block_size=block_size
            )

        # Transpose and reshape to [batch_size, seq_len, dim] with optimized memory layout
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_heads * self.actual_head_dim)

        # Project to output dimension with optimized memory layout
        output = self.out_proj(attention_output)

        # For compatibility with other attention implementations
        attention_weights = None
        if output_attentions:
            # Compute attention weights for visualization
            # Note: This is not efficient and should only be used for debugging
            attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.actual_head_dim)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        outputs = (output, attention_weights, present_key_value) if output_attentions else (output, None, present_key_value)

        return outputs


class RotaryMultiQueryAttention(nn.Module):
    """
    Multi-Query Attention with Rotary Position Embeddings (RoPE).

    Attributes:
        dim: Hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length for RoPE
        rope_base: Base for RoPE frequency computation
        dropout_rate: Dropout probability
        dtype: Data type for computations
    """
    dim: int
    num_query_heads: int
    num_kv_heads: int = 1
    head_dim: Optional[int] = None
    max_seq_len: int = 4096
    rope_base: int = 10000
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Determine head dimension if not provided
        self.actual_head_dim = self.head_dim or self.dim // self.num_query_heads

        # Projection matrices
        self.q_proj = nn.Dense(
            features=self.num_query_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="q_proj"
        )

        self.k_proj = nn.Dense(
            features=self.num_kv_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="k_proj"
        )

        self.v_proj = nn.Dense(
            features=self.num_kv_heads * self.actual_head_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="v_proj"
        )

        self.out_proj = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="out_proj"
        )

        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Rotary position embeddings
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=self.actual_head_dim,
            max_seq_len=self.max_seq_len,
            base=self.rope_base,
            dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Apply rotary multi-query attention.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and values
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, num_query_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.actual_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.actual_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.actual_head_dim)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]

        # Apply rotary embeddings to q and k
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # Handle cached key and values for incremental decoding
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        # Save key and value for future use if caching
        present_key_value = (k, v) if use_cache else None

        # Transpose to [batch_size, num_*_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Repeat k and v for each query head
        if self.num_kv_heads < self.num_query_heads:
            # Calculate how many times to repeat
            repeats = self.num_query_heads // self.num_kv_heads
            # Repeat k and v along the head dimension
            k = jnp.repeat(k, repeats, axis=1)
            v = jnp.repeat(v, repeats, axis=1)

        # Compute attention scores
        # [batch_size, num_query_heads, seq_len, seq_len]
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.actual_head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)

        # Compute attention output
        # [batch_size, num_query_heads, seq_len, head_dim]
        attention_output = jnp.matmul(attention_weights, v)

        # Transpose and reshape to [batch_size, seq_len, dim]
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, self.num_query_heads * self.actual_head_dim)

        # Project to output dimension
        output = self.out_proj(attention_output)

        outputs = (output, attention_weights, present_key_value) if output_attentions else (output, None, present_key_value)

        return outputs
