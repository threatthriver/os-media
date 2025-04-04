"""
Model architecture for a 600B parameter LLM optimized for high-performance hardware.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Any, Callable, Optional, Tuple, Dict
import math
from einops import rearrange

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
    """MLP with SwiGLU activation."""
    dim: int
    hidden_dim: int
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        """Apply MLP."""
        # SwiGLU activation
        w1 = self.param('w1', nn.initializers.normal(0.02), (self.dim, self.hidden_dim), self.dtype)
        w2 = self.param('w2', nn.initializers.normal(0.02), (self.dim, self.hidden_dim), self.dtype)
        w3 = self.param('w3', nn.initializers.normal(0.02), (self.hidden_dim, self.dim), self.dtype)
        
        # Project to hidden dimension
        h1 = jnp.dot(x, w1)
        h2 = jnp.dot(x, w2)
        
        # SwiGLU activation
        h = jnp.multiply(h1, jax.nn.swish(h2))
        
        # Project back to input dimension
        return jnp.dot(h, w3)

class RoPEAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings."""
    dim: int
    num_heads: int
    head_dim: int = 64
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
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
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply transformer block."""
        # Attention with residual connection and layer normalization
        h = LayerNorm(dtype=self.dtype)(x)
        h = RoPEAttention(dim=self.dim, 
                          num_heads=self.num_heads, 
                          dropout_rate=self.dropout_rate, 
                          dtype=self.dtype)(h, mask, deterministic)
        x = x + h
        
        # MLP with residual connection and layer normalization
        h = LayerNorm(dtype=self.dtype)(x)
        h = MLP(dim=self.dim, hidden_dim=self.mlp_dim, dtype=self.dtype)(h)
        
        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)
            
        x = x + h
        
        return x

class LLM(nn.Module):
    """Large Language Model with transformer architecture."""
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int = 131072
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    
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
        
        # Apply transformer blocks
        x = token_embeddings
        for i in range(self.num_hidden_layers):
            x = TransformerBlock(
                dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                mlp_dim=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name=f'layer_{i}'
            )(x, attention_bias, deterministic)
        
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

def create_model(model_size='600b', max_seq_length=131072):
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
        dtype=jnp.bfloat16  # Use bfloat16 for better performance
    )
    
    return model
