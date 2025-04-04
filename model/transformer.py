"""
Transformer blocks for the LLM model.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, Callable, Union
import math

from model.attention import MultiHeadAttention, MultiQueryAttention, RotaryMultiQueryAttention


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    Attributes:
        dim: Input and output dimension
        hidden_dim: Hidden dimension
        dropout_rate: Dropout probability
        dtype: Data type for computations
    """
    dim: int
    hidden_dim: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.gate_proj = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="gate_proj"
        )
        
        self.up_proj = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="up_proj"
        )
        
        self.down_proj = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name="down_proj"
        )
        
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            deterministic: Whether to use deterministic operations (no dropout)
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # SwiGLU activation
        gate = self.gate_proj(x)
        gate = jax.nn.silu(gate)
        
        up = self.up_proj(x)
        
        # Element-wise multiplication
        hidden = gate * up
        
        # Project back to input dimension
        output = self.down_proj(hidden)
        
        # Apply dropout
        output = self.dropout(output, deterministic=deterministic)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward network.
    
    Attributes:
        dim: Hidden dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in feed-forward network
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
        layer_norm_epsilon: Epsilon for layer normalization
        dtype: Data type for computations
    """
    dim: int
    num_heads: int
    hidden_dim: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="input_layernorm"
        )
        
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="post_attention_layernorm"
        )
        
        # Attention
        self.attention = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            name="attention"
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="feed_forward"
        )
        
        # Dropout
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
        Apply transformer block.
        
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
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        hidden_states = attention_outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.feed_forward(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attention_outputs[1:]
        
        return outputs


class TransformerLayer(nn.Module):
    """
    Transformer layer with multi-query attention and feed-forward network.
    
    Attributes:
        dim: Hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: Hidden dimension in feed-forward network
        max_seq_len: Maximum sequence length for RoPE
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
        layer_norm_epsilon: Epsilon for layer normalization
        use_rope: Whether to use rotary position embeddings
        dtype: Data type for computations
    """
    dim: int
    num_query_heads: int
    num_kv_heads: int = 1
    hidden_dim: int = None
    max_seq_len: int = 4096
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_rope: bool = True
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # Set hidden dimension if not provided
        if self.hidden_dim is None:
            self.actual_hidden_dim = 4 * self.dim
        else:
            self.actual_hidden_dim = self.hidden_dim
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="input_layernorm"
        )
        
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="post_attention_layernorm"
        )
        
        # Attention
        if self.use_rope:
            self.attention = RotaryMultiQueryAttention(
                dim=self.dim,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                max_seq_len=self.max_seq_len,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name="attention"
            )
        else:
            self.attention = MultiQueryAttention(
                dim=self.dim,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name="attention"
            )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=self.actual_hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="feed_forward"
        )
        
        # Dropout
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
        Apply transformer layer.
        
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
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        hidden_states = attention_outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.feed_forward(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,) + attention_outputs[1:]
        
        return outputs
