"""
Data parallelism for LLM training.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import flax.linen as nn
from functools import partial

from parallelism.sharding import ShardingStrategy, ParameterSharding, create_device_mesh


class DataParallel:
    """
    Data parallelism for distributed training.
    
    Attributes:
        num_devices: Number of devices
        mesh: Device mesh
    """
    
    def __init__(self, num_devices: Optional[int] = None):
        """
        Initialize data parallelism.
        
        Args:
            num_devices: Number of devices (defaults to all available devices)
        """
        # Get number of devices
        self.num_devices = num_devices or jax.device_count()
        
        # Create device mesh
        self.mesh = create_device_mesh(self.num_devices)
    
    def shard_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shard parameters across devices.
        
        Args:
            params: Model parameters
            
        Returns:
            Sharded parameters
        """
        return params
    
    def parallelize(self, fn: Callable) -> Callable:
        """
        Parallelize function across devices.
        
        Args:
            fn: Function to parallelize
            
        Returns:
            Parallelized function
        """
        return jax.pmap(fn, axis_name="batch")
    
    def gather_outputs(self, outputs: Any) -> Any:
        """
        Gather outputs from devices.
        
        Args:
            outputs: Outputs from parallelized function
            
        Returns:
            Gathered outputs
        """
        return jax.tree_map(lambda x: x[0], outputs)
    
    def all_reduce(self, values: Any) -> Any:
        """
        Perform all-reduce operation across devices.
        
        Args:
            values: Values to reduce
            
        Returns:
            Reduced values
        """
        return jax.lax.pmean(values, axis_name="batch")
    
    def replicate(self, values: Any) -> Any:
        """
        Replicate values across devices.
        
        Args:
            values: Values to replicate
            
        Returns:
            Replicated values
        """
        return jax.tree_map(lambda x: jnp.array([x] * self.num_devices), values)
    
    def split_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Split batch across devices.
        
        Args:
            batch: Batch of data
            
        Returns:
            Split batch
        """
        # Compute batch size per device
        batch_size = batch["input_ids"].shape[0]
        per_device_batch_size = batch_size // self.num_devices
        
        # Split batch
        return jax.tree_map(
            lambda x: x.reshape(self.num_devices, per_device_batch_size, *x.shape[1:]),
            batch
        )
