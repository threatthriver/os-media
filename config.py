"""
Configuration for LLM training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import jax.numpy as jnp

from model.llm import LLMConfig


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Attributes:
        output_dir: Output directory
        model_config: Model configuration

        # Training parameters
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of training steps
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping

        # Optimizer parameters
        optimizer: Optimizer type
        adam_beta1: Beta1 for Adam optimizer
        adam_beta2: Beta2 for Adam optimizer
        adam_epsilon: Epsilon for Adam optimizer

        # Logging parameters
        logging_steps: Number of steps between logging
        save_steps: Number of steps between checkpoints
        eval_steps: Number of steps between evaluations

        # Dataset parameters
        train_file: Path to training file
        eval_file: Path to evaluation file
        max_seq_length: Maximum sequence length

        # Tokenizer parameters
        tokenizer_file: Path to tokenizer file

        # Parallelism parameters
        parallelism_type: Type of parallelism
        tensor_parallel_size: Number of tensor parallel devices
        pipeline_parallel_size: Number of pipeline parallel devices

        # Performance optimization parameters
        use_flash_attention: Whether to use flash attention for efficiency
        use_gradient_checkpointing: Whether to use gradient checkpointing to save memory

        # Long context support parameters
        use_rope_scaling: Whether to use RoPE scaling for longer contexts
        rope_scaling_factor: Scaling factor for RoPE frequencies

        # Reasoning capabilities parameters
        use_reasoning_layer: Whether to use additional reasoning layers
        num_reasoning_layers: Number of additional reasoning layers
        reasoning_intermediate_size: Hidden dimension in reasoning feed-forward network

        # Miscellaneous parameters
        seed: Random seed
        dtype: Data type for computations
        mixed_precision: Whether to use mixed precision training
    """
    # Output directory
    output_dir: str = "output"

    # Model configuration
    model_config: LLMConfig = field(default_factory=LLMConfig)

    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimizer parameters
    optimizer: str = "adamw"  # "adamw", "lion"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging parameters
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000

    # Dataset parameters
    train_file: str = ""
    eval_file: str = ""
    max_seq_length: int = 32768  # Increased to support longer contexts

    # Tokenizer parameters
    tokenizer_file: str = ""

    # Parallelism parameters
    parallelism_type: str = "data"  # "data", "tensor", "pipeline"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Performance optimization parameters
    use_flash_attention: bool = True  # Use flash attention for efficiency
    use_gradient_checkpointing: bool = True  # Use gradient checkpointing to save memory

    # Long context support parameters
    use_rope_scaling: bool = True  # Use RoPE scaling for longer contexts
    rope_scaling_factor: float = 0.5  # Scaling factor for RoPE frequencies

    # Reasoning capabilities parameters
    use_reasoning_layer: bool = True  # Use additional reasoning layers
    num_reasoning_layers: int = 2  # Number of additional reasoning layers
    reasoning_intermediate_size: int = 16384  # Hidden dimension in reasoning feed-forward network

    # Miscellaneous parameters
    seed: int = 42
    dtype: jnp.dtype = jnp.bfloat16
    mixed_precision: bool = True


@dataclass
class ModelSizeConfig:
    """
    Configuration for different model sizes.

    Attributes:
        name: Model size name
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        intermediate_size: Hidden dimension in feed-forward network
        max_position_embeddings: Maximum sequence length
        vocab_size: Size of vocabulary
        reasoning_intermediate_size: Hidden dimension in reasoning feed-forward network
        num_reasoning_layers: Number of additional reasoning layers
    """
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_query_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_position_embeddings: int = 32768  # Increased to support longer contexts
    vocab_size: int = 32000
    reasoning_intermediate_size: int = 16384  # Hidden dimension for reasoning layers
    num_reasoning_layers: int = 2  # Number of reasoning layers


# Model size configurations
MODEL_SIZES = {
    "7b": ModelSizeConfig(
        name="7b",
        hidden_size=4096,
        num_hidden_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        intermediate_size=11008,
        max_position_embeddings=32768,  # Increased to support longer contexts
        vocab_size=32000,
        reasoning_intermediate_size=16384,
        num_reasoning_layers=1
    ),
    "13b": ModelSizeConfig(
        name="13b",
        hidden_size=5120,
        num_hidden_layers=40,
        num_query_heads=40,
        num_kv_heads=10,
        intermediate_size=13824,
        max_position_embeddings=32768,  # Increased to support longer contexts
        vocab_size=32000,
        reasoning_intermediate_size=20480,
        num_reasoning_layers=1
    ),
    "70b": ModelSizeConfig(
        name="70b",
        hidden_size=8192,
        num_hidden_layers=80,
        num_query_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        max_position_embeddings=32768,  # Increased to support longer contexts
        vocab_size=32000,
        reasoning_intermediate_size=32768,
        num_reasoning_layers=2
    ),
    "175b": ModelSizeConfig(
        name="175b",
        hidden_size=12288,
        num_hidden_layers=96,
        num_query_heads=96,
        num_kv_heads=12,
        intermediate_size=49152,
        max_position_embeddings=32768,  # Increased to support longer contexts
        vocab_size=32000,
        reasoning_intermediate_size=49152,
        num_reasoning_layers=2
    ),
    "600b": ModelSizeConfig(
        name="600b",
        hidden_size=18432,
        num_hidden_layers=128,
        num_query_heads=128,
        num_kv_heads=16,
        intermediate_size=73728,
        max_position_embeddings=32768,  # Increased to support longer contexts
        vocab_size=32000,
        reasoning_intermediate_size=73728,
        num_reasoning_layers=3
    )
}


def get_model_config(model_size: str) -> LLMConfig:
    """
    Get model configuration for a specific model size.

    Args:
        model_size: Model size name

    Returns:
        Model configuration
    """
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Model size {model_size} not supported")

    size_config = MODEL_SIZES[model_size]

    return LLMConfig(
        vocab_size=size_config.vocab_size,
        hidden_size=size_config.hidden_size,
        num_hidden_layers=size_config.num_hidden_layers,
        num_query_heads=size_config.num_query_heads,
        num_kv_heads=size_config.num_kv_heads,
        intermediate_size=size_config.intermediate_size,
        max_position_embeddings=size_config.max_position_embeddings,

        # Performance optimizations
        use_flash_attention=True,
        use_gradient_checkpointing=True,

        # Long context support
        use_rope_scaling=True,
        rope_scaling_factor=0.5,

        # Reasoning capabilities
        use_reasoning_layer=True,
        num_reasoning_layers=size_config.num_reasoning_layers,
        reasoning_intermediate_size=size_config.reasoning_intermediate_size
    )
