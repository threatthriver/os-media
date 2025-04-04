"""
Parameter sharding strategies for LLM training.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from typing import Dict, List, Optional, Tuple, Union, Any
import flax.linen as nn
from enum import Enum, auto


class ShardingStrategy(Enum):
    """
    Sharding strategies for model parameters.
    """
    FULLY_REPLICATED = auto()  # Replicate parameters across all devices
    DATA_PARALLEL = auto()     # Shard data across devices, replicate parameters
    TENSOR_PARALLEL = auto()   # Shard parameters across devices
    PIPELINE_PARALLEL = auto() # Shard layers across devices
    FSDP = auto()              # Fully Sharded Data Parallel


class ParameterSharding:
    """
    Parameter sharding for distributed training.
    
    Attributes:
        mesh: Device mesh
        strategy: Sharding strategy
    """
    
    def __init__(
        self,
        mesh: Mesh,
        strategy: ShardingStrategy = ShardingStrategy.DATA_PARALLEL
    ):
        """
        Initialize parameter sharding.
        
        Args:
            mesh: Device mesh
            strategy: Sharding strategy
        """
        self.mesh = mesh
        self.strategy = strategy
    
    def create_sharding_rules(self, params: Dict[str, Any]) -> Dict[str, P]:
        """
        Create sharding rules for parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Dictionary of parameter sharding rules
        """
        rules = {}
        
        if self.strategy == ShardingStrategy.FULLY_REPLICATED:
            # Replicate all parameters
            rules = jax.tree_map(lambda _: P(), params)
        
        elif self.strategy == ShardingStrategy.DATA_PARALLEL:
            # Replicate all parameters
            rules = jax.tree_map(lambda _: P(), params)
        
        elif self.strategy == ShardingStrategy.TENSOR_PARALLEL:
            # Shard attention and feed-forward parameters
            rules = self._create_tensor_parallel_rules(params)
        
        elif self.strategy == ShardingStrategy.PIPELINE_PARALLEL:
            # Shard layers across devices
            rules = self._create_pipeline_parallel_rules(params)
        
        elif self.strategy == ShardingStrategy.FSDP:
            # Shard all parameters
            rules = self._create_fsdp_rules(params)
        
        return rules
    
    def _create_tensor_parallel_rules(self, params: Dict[str, Any]) -> Dict[str, P]:
        """
        Create tensor parallel sharding rules.
        
        Args:
            params: Model parameters
            
        Returns:
            Dictionary of parameter sharding rules
        """
        rules = {}
        
        # Flatten parameter tree
        flat_params = jax.tree_util.tree_flatten(params)[0]
        flat_paths = jax.tree_util.tree_flatten_with_path(params)[0]
        paths = ["/".join(str(p) for p in path) for path, _ in flat_paths]
        
        # Create rules for each parameter
        for path, param in zip(paths, flat_params):
            if "attention" in path and "q_proj" in path:
                # Shard query projection along head dimension
                rules[path] = P("tp", None)
            elif "attention" in path and "k_proj" in path:
                # Shard key projection along head dimension
                rules[path] = P("tp", None)
            elif "attention" in path and "v_proj" in path:
                # Shard value projection along head dimension
                rules[path] = P("tp", None)
            elif "attention" in path and "out_proj" in path:
                # Shard output projection along input dimension
                rules[path] = P(None, "tp")
            elif "feed_forward" in path and "gate_proj" in path:
                # Shard gate projection along output dimension
                rules[path] = P(None, "tp")
            elif "feed_forward" in path and "up_proj" in path:
                # Shard up projection along output dimension
                rules[path] = P(None, "tp")
            elif "feed_forward" in path and "down_proj" in path:
                # Shard down projection along input dimension
                rules[path] = P("tp", None)
            else:
                # Replicate other parameters
                rules[path] = P()
        
        return rules
    
    def _create_pipeline_parallel_rules(self, params: Dict[str, Any]) -> Dict[str, P]:
        """
        Create pipeline parallel sharding rules.
        
        Args:
            params: Model parameters
            
        Returns:
            Dictionary of parameter sharding rules
        """
        rules = {}
        
        # Flatten parameter tree
        flat_params = jax.tree_util.tree_flatten(params)[0]
        flat_paths = jax.tree_util.tree_flatten_with_path(params)[0]
        paths = ["/".join(str(p) for p in path) for path, _ in flat_paths]
        
        # Create rules for each parameter
        for path, param in zip(paths, flat_params):
            if "layers" in path:
                # Extract layer index
                layer_idx = int(path.split("layers_")[1].split("/")[0])
                
                # Shard layers across pipeline stages
                rules[path] = P("pp")
            else:
                # Replicate other parameters
                rules[path] = P()
        
        return rules
    
    def _create_fsdp_rules(self, params: Dict[str, Any]) -> Dict[str, P]:
        """
        Create fully sharded data parallel sharding rules.
        
        Args:
            params: Model parameters
            
        Returns:
            Dictionary of parameter sharding rules
        """
        # Shard all parameters along the first dimension
        return jax.tree_map(lambda p: P("fsdp", None), params)


def create_device_mesh(
    num_devices: int,
    num_tp: int = 1,
    num_pp: int = 1
) -> Mesh:
    """
    Create device mesh for distributed training.
    
    Args:
        num_devices: Number of devices
        num_tp: Number of tensor parallel devices
        num_pp: Number of pipeline parallel devices
        
    Returns:
        Device mesh
    """
    # Compute number of data parallel devices
    num_dp = num_devices // (num_tp * num_pp)
    
    # Create device mesh
    devices = jnp.array(jax.devices()).reshape(num_dp, num_pp, num_tp)
    
    # Create mesh
    return Mesh(devices, ("dp", "pp", "tp"))


def shard_params(
    params: Dict[str, Any],
    mesh: Mesh,
    strategy: ShardingStrategy = ShardingStrategy.DATA_PARALLEL
) -> Dict[str, Any]:
    """
    Shard parameters across devices.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        strategy: Sharding strategy
        
    Returns:
        Sharded parameters
    """
    # Create parameter sharding
    param_sharding = ParameterSharding(mesh, strategy)
    
    # Create sharding rules
    rules = param_sharding.create_sharding_rules(params)
    
    # Shard parameters
    with mesh:
        sharded_params = jax.tree_map(
            lambda p, r: jax.lax.with_sharding_constraint(p, r),
            params,
            rules
        )
    
    return sharded_params
