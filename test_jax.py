#!/usr/bin/env python3
"""
Simple JAX test script to check if JAX is working correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np

def main():
    # Print JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")
    
    # Simple matrix multiplication
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))
    
    # Time the computation
    import time
    start_time = time.time()
    result = jnp.dot(x, y)
    end_time = time.time()
    
    print(f"Matrix multiplication result shape: {result.shape}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # Test JIT compilation
    @jax.jit
    def jitted_matmul(a, b):
        return jnp.dot(a, b)
    
    # Warm up JIT
    _ = jitted_matmul(x, y)
    
    # Time JIT computation
    start_time = time.time()
    result_jit = jitted_matmul(x, y)
    end_time = time.time()
    
    print(f"JIT matrix multiplication result shape: {result_jit.shape}")
    print(f"JIT time taken: {end_time - start_time:.4f} seconds")
    
    print("JAX test completed successfully!")

if __name__ == "__main__":
    main()
