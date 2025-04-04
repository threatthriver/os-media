"""
Tensor parallelism for LLM training on TPU v4-32.
Optimized for training a 600B parameter model efficiently.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import flax.linen as nn
from functools import partial
import numpy as np
import time

from parallelism.sharding import ShardingStrategy, ParameterSharding, create_device_mesh


class TensorParallel:
    """
    Tensor parallelism for distributed training on TPU v4-32.

    Attributes:
        num_devices: Number of devices
        num_tp: Number of tensor parallel devices
        mesh: Device mesh
        dp_size: Number of data parallel devices
    """

    def __init__(
        self,
        num_devices: Optional[int] = None,
        num_tp: int = 8,
        use_2d_sharding: bool = True
    ):
        """
        Initialize tensor parallelism optimized for TPU v4-32.

        Args:
            num_devices: Number of devices (defaults to all available devices)
            num_tp: Number of tensor parallel devices
            use_2d_sharding: Whether to use 2D sharding for better efficiency
        """
        # Get number of devices
        self.num_devices = num_devices or jax.device_count()
        self.num_tp = min(num_tp, self.num_devices)

        # Calculate optimal data parallelism size
        self.dp_size = self.num_devices // self.num_tp

        # Log device configuration
        print(f"TPU configuration: {self.num_devices} total devices")
        print(f"Tensor parallelism: {self.num_tp} devices")
        print(f"Data parallelism: {self.dp_size} devices")

        # Create device mesh
        self.mesh = create_device_mesh(self.num_devices, num_tp=self.num_tp)

        # Create parameter sharding
        self.param_sharding = ParameterSharding(
            self.mesh,
            ShardingStrategy.TENSOR_PARALLEL
        )

        # Store 2D sharding preference
        self.use_2d_sharding = use_2d_sharding

    def shard_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shard parameters across devices optimized for TPU v4-32.

        Args:
            params: Model parameters

        Returns:
            Sharded parameters
        """
        # Create sharding rules
        rules = self.param_sharding.create_sharding_rules(params)

        # Measure sharding time for performance monitoring
        start_time = time.time()

        # Apply 2D sharding for better TPU utilization if enabled
        if self.use_2d_sharding:
            # Modify rules for 2D sharding of large matrices
            rules = self._apply_2d_sharding(rules, params)

        # Shard parameters
        with self.mesh:
            sharded_params = jax.tree_map(
                lambda p, r: jax.lax.with_sharding_constraint(p, r),
                params,
                rules
            )

        # Log sharding time
        sharding_time = time.time() - start_time
        print(f"Parameter sharding completed in {sharding_time:.2f} seconds")

        return sharded_params

    def _apply_2d_sharding(self, rules: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply 2D sharding to large matrices for better TPU utilization.

        Args:
            rules: Sharding rules
            params: Model parameters

        Returns:
            Modified sharding rules
        """
        # Flatten parameter tree
        flat_params = jax.tree_util.tree_flatten(params)[0]
        flat_paths = jax.tree_util.tree_flatten_with_path(params)[0]
        paths = ['/'.join(str(p) for p in path) for path, _ in flat_paths]

        # Create modified rules
        modified_rules = dict(rules)

        # Apply 2D sharding to large matrices
        for path, param in zip(paths, flat_params):
            # Check if parameter is a large matrix (>= 4096 in both dimensions)
            if len(param.shape) == 2 and param.shape[0] >= 4096 and param.shape[1] >= 4096:
                # Apply 2D sharding
                modified_rules[path] = P('dp', 'tp')

        return modified_rules

    def parallelize(self, fn: Callable, donate_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
        """
        Parallelize function across devices with optimizations for TPU v4-32.

        Args:
            fn: Function to parallelize
            donate_argnums: Indices of arguments to donate (for memory optimization)

        Returns:
            Parallelized function
        """
        # Use cached computation for better performance
        fn = jax.jit(fn, donate_argnums=donate_argnums)

        # Parallelize function with optimized device mapping
        return jax.pmap(
            fn,
            axis_name="batch",
            devices=self.mesh.devices.reshape(self.dp_size, -1)[:, 0],
            donate_argnums=donate_argnums
        )

    def gather_outputs(self, outputs: Any) -> Any:
        """
        Gather outputs from devices with optimized communication.

        Args:
            outputs: Outputs from parallelized function

        Returns:
            Gathered outputs
        """
        # Measure gathering time for performance monitoring
        start_time = time.time()

        # Gather outputs
        gathered = jax.tree_map(lambda x: x[0], outputs)

        # Log gathering time for large outputs
        if isinstance(outputs, dict) and 'logits' in outputs:
            gather_time = time.time() - start_time
            if gather_time > 0.1:  # Only log if significant
                print(f"Output gathering completed in {gather_time:.2f} seconds")

        return gathered

    def all_reduce(self, values: Any, reduce_type: str = "mean") -> Any:
        """
        Perform all-reduce operation across devices with optimized communication.

        Args:
            values: Values to reduce
            reduce_type: Type of reduction ("mean" or "sum")

        Returns:
            Reduced values
        """
        if reduce_type == "mean":
            return jax.lax.pmean(values, axis_name="batch")
        elif reduce_type == "sum":
            return jax.lax.psum(values, axis_name="batch")
        else:
            raise ValueError(f"Unsupported reduce type: {reduce_type}")

    def replicate(self, values: Any) -> Any:
        """
        Replicate values across devices with optimized memory usage.

        Args:
            values: Values to replicate

        Returns:
            Replicated values
        """
        # Use broadcast instead of replication for better performance
        return jax.tree_map(lambda x: jnp.broadcast_to(x, (self.dp_size,) + x.shape), values)

    def split_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Split batch across devices with optimized memory layout for TPU.

        Args:
            batch: Batch of data

        Returns:
            Split batch
        """
        # Compute batch size per device
        batch_size = batch["input_ids"].shape[0]
        per_device_batch_size = batch_size // self.dp_size

        # Check if batch size is divisible by number of devices
        if batch_size % self.dp_size != 0:
            print(f"Warning: Batch size {batch_size} is not divisible by number of data parallel devices {self.dp_size}")
            # Adjust batch size to be divisible
            new_batch_size = per_device_batch_size * self.dp_size
            batch = jax.tree_map(lambda x: x[:new_batch_size], batch)

        # Split batch with optimized memory layout
        return jax.tree_map(
            lambda x: x.reshape(self.dp_size, per_device_batch_size, *x.shape[1:]),
            batch
        )
