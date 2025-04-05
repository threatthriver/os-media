"""
TPU-optimized model architecture for a 600B parameter LLM designed to excel at coding and reasoning tasks.
This implementation includes specialized optimizations for Google TPU v4-32 hardware.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Any, Callable, Optional, Tuple, Dict
import math
from einops import rearrange

# TPU-specific optimizations
# Enable bfloat16 precision by default for TPU
DEFAULT_DTYPE = jnp.bfloat16
# Optimal shard size for TPU v4-32
OPTIMAL_SHARD_SIZE = 8

class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    epsilon: float = 1e-5
    dtype: Any = jnp.float32
    bias: bool = True
    scale: bool = True

    @nn.compact
    def __call__(self, x):
        """Apply layer normalization."""
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)

        y = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        if self.scale:
            gamma = self.param('gamma', nn.initializers.ones, (features,), self.dtype)
            y = y * gamma

        if self.bias:
            beta = self.param('beta', nn.initializers.zeros, (features,), self.dtype)
            y = y + beta

        return y

class MLP(nn.Module):
    """MLP with SwiGLU activation for improved reasoning capabilities."""
    dim: int
    hidden_dim: int
    dtype: Any = DEFAULT_DTYPE
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Apply MLP."""
        # SwiGLU activation - better for reasoning tasks
        x = nn.Dense(features=self.hidden_dim * 2, dtype=self.dtype, name='fc_up')(x)
        x_gate, x_val = jnp.split(x, 2, axis=-1)

        # SwiGLU activation function
        x = jax.nn.swish(x_gate) * x_val

        # Apply dropout for regularization
        if not deterministic and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        # Project back to input dimension
        x = nn.Dense(features=self.dim, dtype=self.dtype, name='fc_down')(x)

        return x

class FlashAttention(nn.Module):
    """
    Flash Attention implementation optimized for TPU.
    This implementation uses block-sparse attention patterns for efficient computation.
    """
    dim: int
    num_heads: int
    head_dim: int = 128
    dropout_rate: float = 0.0
    dtype: Any = DEFAULT_DTYPE
    causal: bool = True
    block_size: int = 128  # Optimal block size for TPU

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply flash attention."""
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        qkv = nn.Dense(features=3 * self.num_heads * self.head_dim,
                       dtype=self.dtype,
                       kernel_init=nn.initializers.normal(0.02),
                       name='qkv')(x)

        # Split into queries, keys, values and reshape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale queries
        scale = math.sqrt(self.head_dim)
        q = q / scale

        # Compute attention using blocks for memory efficiency
        # This is a simplified version of flash attention
        # In a real implementation, we would use a more optimized kernel

        # Reshape to blocks
        blocks = seq_len // self.block_size
        if blocks == 0:
            blocks = 1

        # Pad sequence length to multiple of block_size
        pad_len = blocks * self.block_size - seq_len
        if pad_len > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

        # Reshape to blocks
        q_blocks = q.reshape(batch_size, self.num_heads, blocks, self.block_size, self.head_dim)
        k_blocks = k.reshape(batch_size, self.num_heads, blocks, self.block_size, self.head_dim)
        v_blocks = v.reshape(batch_size, self.num_heads, blocks, self.block_size, self.head_dim)

        # Compute attention scores and apply causal mask
        output_blocks = []

        for i in range(blocks):
            q_block = q_blocks[:, :, i:i+1]  # (batch_size, num_heads, 1, block_size, head_dim)

            # Compute attention scores for this block with all previous blocks
            attn_blocks = []

            for j in range(i + 1):  # Only attend to current and previous blocks (causal)
                k_block = k_blocks[:, :, j:j+1]  # (batch_size, num_heads, 1, block_size, head_dim)
                v_block = v_blocks[:, :, j:j+1]  # (batch_size, num_heads, 1, block_size, head_dim)

                # Compute attention scores
                scores = jnp.matmul(q_block, jnp.transpose(k_block, (0, 1, 2, 4, 3)))
                # scores: (batch_size, num_heads, 1, block_size, block_size)

                # Apply causal mask within the last block
                if self.causal and i == j:
                    causal_mask = jnp.triu(jnp.ones((self.block_size, self.block_size), dtype=jnp.bool_), k=1)
                    scores = jnp.where(causal_mask, -1e10, scores)

                # Apply softmax
                attn_weights = jax.nn.softmax(scores, axis=-1)

                # Apply dropout
                if not deterministic and self.dropout_rate > 0:
                    attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=False)

                # Apply attention to values
                block_output = jnp.matmul(attn_weights, v_block)
                attn_blocks.append(block_output)

            # Combine attention outputs
            output_block = sum(attn_blocks)
            output_blocks.append(output_block)

        # Combine all blocks
        output = jnp.concatenate(output_blocks, axis=2)
        # Reshape back to original shape
        output = output.reshape(batch_size, self.num_heads, blocks * self.block_size, self.head_dim)

        # Remove padding if necessary
        if pad_len > 0:
            output = output[:, :, :seq_len, :]

        # Reshape to original dimensions
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Project to output dimension
        output = nn.Dense(features=self.dim,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.normal(0.02),
                          name='output')(output)

        return output

class RoPEAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings for better long-context handling."""
    dim: int
    num_heads: int
    head_dim: int = 128
    dropout_rate: float = 0.0
    dtype: Any = DEFAULT_DTYPE
    max_positions: int = 131072  # Support for 128K context window
    base_scaling: float = 10000.0
    scaling_factor: float = 0.25  # RoPE scaling for extended context

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply attention with rotary positional embeddings."""
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        qkv = nn.Dense(features=3 * self.num_heads * self.head_dim,
                       dtype=self.dtype,
                       kernel_init=nn.initializers.normal(0.02),
                       name='qkv')(x)

        # Split into queries, keys, values
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional embeddings
        q, k = self._apply_rotary_embeddings(q, k, seq_len)

        # Compute attention
        scale = math.sqrt(self.head_dim)
        attention = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale

        # Apply mask if provided
        if mask is not None:
            attention = jnp.where(mask, attention, -1e10)

        # Apply softmax
        attention = jax.nn.softmax(attention, axis=-1)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            attention = nn.Dropout(rate=self.dropout_rate)(attention, deterministic=False)

        # Compute output
        output = jnp.matmul(attention, v)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Project to output dimension
        output = nn.Dense(features=self.dim,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.normal(0.02),
                          name='output')(output)

        return output

    def _apply_rotary_embeddings(self, q, k, seq_len):
        """Apply rotary positional embeddings to queries and keys."""
        # Create position indices
        position = jnp.arange(seq_len)[None, None, :, None]  # (1, 1, seq_len, 1)

        # Create frequency bands
        dim = self.head_dim // 2
        freq = jnp.arange(0, dim, 1.0)[None, None, None, :]
        freq = 1.0 / (10000 ** (freq / dim))

        # Compute angles
        angles = position * freq

        # Compute sin and cos
        sin = jnp.sin(angles)
        cos = jnp.cos(angles)

        # Reshape queries and keys
        q_reshaped = rearrange(q, 'b h s (d r) -> b h s d r', r=2)
        k_reshaped = rearrange(k, 'b h s (d r) -> b h s d r', r=2)

        # Apply rotation
        q_out1 = q_reshaped[:, :, :, :, 0] * cos - q_reshaped[:, :, :, :, 1] * sin
        q_out2 = q_reshaped[:, :, :, :, 1] * cos + q_reshaped[:, :, :, :, 0] * sin
        k_out1 = k_reshaped[:, :, :, :, 0] * cos - k_reshaped[:, :, :, :, 1] * sin
        k_out2 = k_reshaped[:, :, :, :, 1] * cos + k_reshaped[:, :, :, :, 0] * sin

        # Concatenate
        q_out = jnp.stack([q_out1, q_out2], axis=-1)
        k_out = jnp.stack([k_out1, k_out2], axis=-1)

        # Reshape back
        q_out = rearrange(q_out, 'b h s d r -> b h s (d r)')
        k_out = rearrange(k_out, 'b h s d r -> b h s (d r)')

        return q_out, k_out

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    dtype: Any = DEFAULT_DTYPE
    use_flash_attention: bool = True

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply transformer block with pre-normalization for better training stability."""
        # Pre-normalization for better training stability
        h = LayerNorm(dtype=self.dtype)(x)

        # Choose attention implementation based on configuration
        if self.use_flash_attention:
            h = FlashAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype
            )(h, mask, deterministic)
        else:
            h = RoPEAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype
            )(h, mask, deterministic)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)

        # Residual connection
        x = x + h

        # MLP with pre-normalization
        h = LayerNorm(dtype=self.dtype)(x)
        h = MLP(dim=self.dim, hidden_dim=self.mlp_dim, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, deterministic)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)

        # Residual connection
        x = x + h

        return x

class ReasoningLayer(nn.Module):
    """
    Special reasoning layer for enhanced logical reasoning and coding capabilities.
    This layer uses a mixture of experts approach to specialize in different reasoning patterns.
    """
    dim: int
    num_experts: int = 8
    expert_dim: int = 1024
    dropout_rate: float = 0.0
    dtype: Any = DEFAULT_DTYPE

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Apply reasoning layer with mixture of experts."""
        batch_size, seq_len, _ = x.shape

        # Router network to determine expert weights
        router_logits = nn.Dense(
            features=self.num_experts,
            dtype=self.dtype,
            name='router'
        )(x)

        # Apply softmax to get expert weights
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        # Create expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # Each expert is a small MLP
            h = nn.Dense(
                features=self.expert_dim,
                dtype=self.dtype,
                name=f'expert_{i}_up'
            )(x)
            h = jax.nn.gelu(h)
            h = nn.Dense(
                features=self.dim,
                dtype=self.dtype,
                name=f'expert_{i}_down'
            )(h)
            expert_outputs.append(h)

        # Stack expert outputs
        expert_outputs = jnp.stack(expert_outputs, axis=-2)  # (batch_size, seq_len, num_experts, dim)

        # Weight and combine expert outputs
        router_probs = router_probs[..., None]  # (batch_size, seq_len, num_experts, 1)
        output = jnp.sum(expert_outputs * router_probs, axis=-2)  # (batch_size, seq_len, dim)

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=False)

        # Residual connection
        output = output + x

        return output

class LLM(nn.Module):
    """Large Language Model with transformer architecture."""
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int = 131072
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    dtype: Any = DEFAULT_DTYPE
    use_flash_attention: bool = True
    use_reasoning_layer: bool = True
    reasoning_layer_interval: int = 4  # Add reasoning layer every N transformer blocks

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, deterministic=True, return_dict=True):
        """Apply LLM to input tokens."""
        batch_size, seq_length = input_ids.shape

        # Token embeddings
        token_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(0.02),
            dtype=self.dtype
        )(input_ids)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        # Convert attention mask to attention bias
        attention_bias = (1.0 - attention_mask[:, None, None, :]) * -1e10

        # Apply transformer blocks with reasoning layers
        x = token_embeddings
        for i in range(self.num_hidden_layers):
            x = TransformerBlock(
                dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                mlp_dim=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                name=f'layer_{i}'
            )(x, attention_bias, deterministic)

            # Add reasoning layer at specified intervals
            if self.use_reasoning_layer and (i + 1) % self.reasoning_layer_interval == 0:
                x = ReasoningLayer(
                    dim=self.hidden_size,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    name=f'reasoning_layer_{i // self.reasoning_layer_interval}'
                )(x, deterministic)

        # Final layer normalization
        x = LayerNorm(dtype=self.dtype)(x)

        # Language modeling head
        logits = nn.Dense(
            features=self.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02),
            name='lm_head'
        )(x)

        if return_dict:
            return {'logits': logits}
        else:
            return logits

def create_model(model_size='600b', max_seq_length=131072, use_flash_attention=True, use_reasoning_layer=True):
    """Create a model with the specified size."""
    # Model size configurations
    model_sizes = {
        "7b": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008
        },
        "13b": {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "intermediate_size": 13824
        },
        "70b": {
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "intermediate_size": 28672
        },
        "175b": {
            "hidden_size": 12288,
            "num_hidden_layers": 96,
            "num_attention_heads": 96,
            "intermediate_size": 49152
        },
        "600b": {
            "hidden_size": 18432,
            "num_hidden_layers": 128,
            "num_attention_heads": 128,
            "intermediate_size": 73728
        }
    }

    # Get model configuration
    config = model_sizes[model_size]

    # Create model
    model = LLM(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=max_seq_length,
        use_flash_attention=use_flash_attention,
        use_reasoning_layer=use_reasoning_layer,
        dtype=jnp.bfloat16  # Use bfloat16 for better performance
    )

    return model
