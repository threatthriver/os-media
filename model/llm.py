"""
LLM model implementation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any, Callable, Union, List
import math
import time
from dataclasses import dataclass

from model.embedding import TokenEmbedding, RotaryPositionalEmbedding
from model.transformer import TransformerLayer


@dataclass
class LLMConfig:
    """
    Configuration for LLM model.

    Attributes:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        intermediate_size: Hidden dimension in feed-forward network
        hidden_act: Activation function
        max_position_embeddings: Maximum sequence length
        initializer_range: Standard deviation for initializers
        rms_norm_eps: Epsilon for RMSNorm
        use_cache: Whether to use cached key and values
        pad_token_id: ID of padding token
        bos_token_id: ID of beginning of sequence token
        eos_token_id: ID of end of sequence token
        tie_word_embeddings: Whether to tie input and output embeddings
        rope_theta: Base for RoPE frequency computation
        attention_dropout: Dropout probability for attention
        hidden_dropout: Dropout probability for hidden states
        dtype: Data type for computations
        use_flash_attention: Whether to use flash attention for efficiency
        use_gradient_checkpointing: Whether to use gradient checkpointing to save memory
        use_rope_scaling: Whether to use RoPE scaling for longer contexts
        rope_scaling_factor: Scaling factor for RoPE frequencies
        use_parallel_residual: Whether to use parallel residual connections
        use_reasoning_layer: Whether to use additional reasoning layers
        num_reasoning_layers: Number of additional reasoning layers
        reasoning_intermediate_size: Hidden dimension in reasoning feed-forward network
    """
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_query_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 11008
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768  # Increased to support longer contexts
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    # Performance optimizations
    use_flash_attention: bool = True  # Use flash attention for efficiency
    use_gradient_checkpointing: bool = True  # Use gradient checkpointing to save memory

    # Long context support
    use_rope_scaling: bool = True  # Use RoPE scaling for longer contexts
    rope_scaling_factor: float = 0.5  # Scaling factor for RoPE frequencies

    # Architecture enhancements
    use_parallel_residual: bool = True  # Use parallel residual connections

    # Reasoning capabilities
    use_reasoning_layer: bool = True  # Use additional reasoning layers
    num_reasoning_layers: int = 2  # Number of additional reasoning layers
    reasoning_intermediate_size: int = 16384  # Hidden dimension in reasoning feed-forward network


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Attributes:
        dim: Hidden dimension
        eps: Epsilon for numerical stability
        dtype: Data type for computations
    """
    dim: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            'weight',
            nn.initializers.ones,
            (self.dim,),
            self.dtype
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [batch_size, seq_len, dim]

        Returns:
            Normalized tensor [batch_size, seq_len, dim]
        """
        # Calculate RMS
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)

        # Scale with learned parameters
        return x * self.weight


class ReasoningLayer(nn.Module):
    """
    Reasoning layer for enhanced reasoning capabilities.

    This layer adds additional processing to enhance the model's reasoning abilities.
    It consists of a self-attention layer followed by a larger feed-forward network.

    Attributes:
        dim: Hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        hidden_dim: Hidden dimension in feed-forward network
        max_seq_len: Maximum sequence length
        dropout_rate: Dropout probability
        attention_dropout_rate: Dropout probability for attention
        layer_norm_epsilon: Epsilon for layer normalization
        use_flash_attention: Whether to use flash attention
        dtype: Data type for computations
    """
    dim: int
    num_query_heads: int
    num_kv_heads: int
    hidden_dim: int
    max_seq_len: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_flash_attention: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        from model.attention import FlashAttention, RotaryMultiQueryAttention
        from model.transformer import FeedForward

        # Layer normalization
        self.input_layernorm = RMSNorm(
            dim=self.dim,
            eps=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="input_layernorm"
        )

        self.post_attention_layernorm = RMSNorm(
            dim=self.dim,
            eps=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="post_attention_layernorm"
        )

        # Attention
        if self.use_flash_attention:
            self.attention = FlashAttention(
                dim=self.dim,
                num_heads=self.num_query_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name="attention"
            )
        else:
            self.attention = RotaryMultiQueryAttention(
                dim=self.dim,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                max_seq_len=self.max_seq_len,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name="attention"
            )

        # Feed-forward network with larger hidden dimension for reasoning
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
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Apply reasoning layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, dim]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
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

        return hidden_states


class LLM(nn.Module):
    """
    Large Language Model implementation with enhanced reasoning capabilities and support for longer contexts.

    Attributes:
        config: Model configuration
    """
    config: LLMConfig

    def setup(self):
        config = self.config
        from model.attention import FlashAttention

        # Token embeddings
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.dtype,
            name="embed_tokens"
        )

        # Transformer layers
        self.layers = [
            TransformerLayer(
                dim=config.hidden_size,
                num_query_heads=config.num_query_heads,
                num_kv_heads=config.num_kv_heads,
                hidden_dim=config.intermediate_size,
                max_seq_len=config.max_position_embeddings,
                dropout_rate=config.hidden_dropout,
                attention_dropout_rate=config.attention_dropout,
                layer_norm_epsilon=config.rms_norm_eps,
                use_rope=True,
                dtype=config.dtype,
                name=f"layers_{i}"
            )
            for i in range(config.num_hidden_layers)
        ]

        # Reasoning layers for enhanced reasoning capabilities
        self.reasoning_layers = []
        if config.use_reasoning_layer:
            self.reasoning_layers = [
                ReasoningLayer(
                    dim=config.hidden_size,
                    num_query_heads=config.num_query_heads,
                    num_kv_heads=config.num_kv_heads,
                    hidden_dim=config.reasoning_intermediate_size,
                    max_seq_len=config.max_position_embeddings,
                    dropout_rate=config.hidden_dropout,
                    attention_dropout_rate=config.attention_dropout,
                    layer_norm_epsilon=config.rms_norm_eps,
                    use_flash_attention=config.use_flash_attention,
                    dtype=config.dtype,
                    name=f"reasoning_layers_{i}"
                )
                for i in range(config.num_reasoning_layers)
            ]

        # Final layer normalization
        self.norm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            name="norm"
        )

        # Output projection
        if not config.tie_word_embeddings:
            self.lm_head = nn.Dense(
                features=config.vocab_size,
                use_bias=False,
                dtype=config.dtype,
                kernel_init=nn.initializers.normal(stddev=config.initializer_range),
                name="lm_head"
            )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        enable_reasoning: bool = True,  # Whether to use reasoning layers
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply LLM model with enhanced reasoning capabilities.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_values: Cached key and value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            deterministic: Whether to use deterministic operations (no dropout)
            enable_reasoning: Whether to use reasoning layers

        Returns:
            Dictionary of model outputs
        """
        batch_size, seq_length = input_ids.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]

        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = nn.make_causal_mask(input_ids)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past_key_values if None
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        # Initialize lists for storing outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_present_key_values = () if self.config.use_cache else None

        # Apply transformer layers with gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing and not self.config.use_cache and not output_attentions:
            # Define a custom layer application function for gradient checkpointing
            def apply_layer(layer_idx, h, mask, pos_ids, past_kv):
                layer = self.layers[layer_idx]
                outputs = layer(
                    hidden_states=h,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    past_key_value=past_kv,
                    output_attentions=False,
                    use_cache=False,
                    deterministic=deterministic,
                )
                return outputs[0]

            # Apply layers with gradient checkpointing
            for i in range(self.config.num_hidden_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                # Apply gradient checkpointing
                hidden_states = jax.checkpoint(
                    apply_layer,
                    static_argnums=(0, 4),  # layer_idx and deterministic are static
                )(i, hidden_states, attention_mask, position_ids, None)
        else:
            # Standard layer application without gradient checkpointing
            for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=self.config.use_cache,
                    deterministic=deterministic,
                )

                hidden_states = layer_outputs[0]

                if self.config.use_cache:
                    all_present_key_values += (layer_outputs[2],)

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

        # Apply reasoning layers if enabled and available
        if enable_reasoning and self.config.use_reasoning_layer and self.reasoning_layers and not past_key_values[0]:
            # Only apply reasoning layers during full-context processing (not during generation)
            for reasoning_layer in self.reasoning_layers:
                hidden_states = reasoning_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                )

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Apply output projection
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            # Tie weights with input embeddings
            logits = jnp.matmul(hidden_states, self.embed_tokens.embedding.T)

        if not return_dict:
            return (logits, all_present_key_values, all_hidden_states, all_attentions)

        return {
            'logits': logits,
            'past_key_values': all_present_key_values,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Generate text using the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: ID of padding token
            eos_token_id: ID of end of sequence token
            deterministic: Whether to use deterministic operations (no dropout)

        Returns:
            Generated token IDs [batch_size, max_length]
        """
        batch_size, seq_length = input_ids.shape

        # Use model's token IDs if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Initialize generated sequences with input IDs
        generated_ids = input_ids

        # Initialize past key values
        past_key_values = None

        # Generate tokens up to max_length
        for i in range(max_length - seq_length):
            # Forward pass
            outputs = self(
                input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                past_key_values=past_key_values,
                deterministic=deterministic,
            )

            # Get logits and past key values
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float('-inf'))
                logits = logits.at[jnp.arange(batch_size)[:, None], top_k_indices].set(top_k_logits)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = jax.lax.sort(logits, dimension=-1, is_stable=True)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove = jnp.concatenate([
                    jnp.zeros_like(sorted_indices_to_remove[:, :1]),
                    sorted_indices_to_remove[:, :-1]
                ], axis=-1)

                # Scatter sorted indices to original logits
                indices_to_remove = jnp.zeros_like(sorted_indices_to_remove)
                indices_to_remove = indices_to_remove.at[jnp.arange(batch_size)[:, None], sorted_indices].set(sorted_indices_to_remove)
                logits = jnp.where(indices_to_remove, float('-inf'), logits)

            # Sample or greedy decoding
            if do_sample:
                # Sample from the distribution
                next_token_ids = jax.random.categorical(
                    jax.random.PRNGKey(int(time.time())), logits, axis=-1
                )
            else:
                # Greedy decoding
                next_token_ids = jnp.argmax(logits, axis=-1)

            # Concatenate new tokens to generated IDs
            generated_ids = jnp.concatenate([generated_ids, next_token_ids[:, None]], axis=1)

            # Check if all sequences have reached EOS
            if jnp.all(next_token_ids == eos_token_id):
                break

        return generated_ids
