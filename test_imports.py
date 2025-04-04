#!/usr/bin/env python3
"""
Test script to check which imports are working.
"""

import sys
print(f"Python version: {sys.version}")

# Try importing core libraries
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"Error importing JAX: {e}")

try:
    import flax
    print(f"Flax version: {flax.__version__}")
except ImportError as e:
    print(f"Error importing Flax: {e}")

try:
    import optax
    print(f"Optax version: {optax.__version__}")
except ImportError as e:
    print(f"Error importing Optax: {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Error importing Transformers: {e}")

try:
    import datasets
    print(f"Datasets version: {datasets.__version__}")
except ImportError as e:
    print(f"Error importing Datasets: {e}")

try:
    import sentencepiece as spm
    print(f"SentencePiece version: {spm.__version__}")
except ImportError as e:
    print(f"Error importing SentencePiece: {e}")

print("Import test complete")
