"""
Data loaders for LLM training.
"""

import random
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union, Callable, Iterator, Any
from data.dataset import Dataset


def pad_batch(
    examples: List[Dict[str, np.ndarray]],
    pad_token_id: int = 0
) -> Dict[str, np.ndarray]:
    """
    Pad batch of examples to the same length.

    Args:
        examples: List of examples
        pad_token_id: Padding token ID

    Returns:
        Padded batch
    """
    # Get maximum length
    max_length = max(example["input_ids"].shape[0] for example in examples)

    # Initialize batch
    batch = {
        "input_ids": np.full((len(examples), max_length), pad_token_id, dtype=np.int32),
        "attention_mask": np.zeros((len(examples), max_length), dtype=np.int32),
        "position_ids": np.zeros((len(examples), max_length), dtype=np.int32),
    }

    # Fill batch
    for i, example in enumerate(examples):
        length = example["input_ids"].shape[0]
        batch["input_ids"][i, :length] = example["input_ids"]
        batch["attention_mask"][i, :length] = example["attention_mask"]
        batch["position_ids"][i, :length] = example["position_ids"]

    return batch


def create_masks(
    input_ids: np.ndarray,
    pad_token_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create attention mask and padding mask.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        pad_token_id: Padding token ID

    Returns:
        Tuple of (attention_mask, padding_mask)
    """
    # Create padding mask
    padding_mask = (input_ids != pad_token_id).astype(np.int32)

    # Create causal attention mask
    seq_len = input_ids.shape[1]
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.int32))

    # Combine padding mask and causal mask
    batch_size = input_ids.shape[0]
    attention_mask = padding_mask[:, None, None, :] * causal_mask[None, None, :, :]

    # Convert to float and apply large negative value to masked positions
    attention_mask = (1.0 - attention_mask.astype(np.float32)) * -1e9

    return attention_mask, padding_mask


class DataLoader:
    """
    Data loader for LLM training.

    Attributes:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pad_token_id: Padding token ID
        collate_fn: Function to collate examples into batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        pad_token_id: int = 0,
        collate_fn: Optional[Callable] = None
    ):
        """
        Initialize data loader.

        Args:
            dataset: Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            pad_token_id: Padding token ID
            collate_fn: Function to collate examples into batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_token_id = pad_token_id

        # Set collate function
        if collate_fn is None:
            self.collate_fn = lambda examples: pad_batch(examples, pad_token_id)
        else:
            self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Iterate over batches.

        Returns:
            Iterator over batches
        """
        # Get indices
        indices = list(range(len(self.dataset)))

        # Shuffle indices if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        batch_indices = []
        for idx in indices:
            batch_indices.append(idx)

            if len(batch_indices) == self.batch_size:
                # Get examples
                examples = [self.dataset[i] for i in batch_indices]

                # Collate examples into batch
                batch = self.collate_fn(examples)

                # Create masks
                attention_mask, padding_mask = create_masks(
                    batch["input_ids"],
                    self.pad_token_id
                )

                # Update batch
                batch["attention_mask"] = attention_mask
                batch["padding_mask"] = padding_mask

                # Convert to JAX arrays
                batch = {k: jnp.array(v) for k, v in batch.items()}

                yield batch

                # Reset batch indices
                batch_indices = []

        # Yield last batch if not empty and not dropping last
        if batch_indices and not self.drop_last:
            # Get examples
            examples = [self.dataset[i] for i in batch_indices]

            # Collate examples into batch
            batch = self.collate_fn(examples)

            # Create masks
            attention_mask, padding_mask = create_masks(
                batch["input_ids"],
                self.pad_token_id
            )

            # Update batch
            batch["attention_mask"] = attention_mask
            batch["padding_mask"] = padding_mask

            # Convert to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}

            yield batch


class TPUDataLoader:
    """
    Data loader optimized for TPU v4-32 with high-performance data loading.

    Attributes:
        dataset: Dataset
        batch_size: Batch size per device
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pad_token_id: Padding token ID
        collate_fn: Function to collate examples into batch
        prefetch_size: Number of batches to prefetch
        use_pjit: Whether to use pjit for data loading
        use_circular_buffer: Whether to use circular buffer for data loading
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,  # Default to True for TPU efficiency
        pad_token_id: int = 0,
        collate_fn: Optional[Callable] = None,
        prefetch_size: int = 2,
        use_pjit: bool = True,
        use_circular_buffer: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize data loader optimized for TPU v4-32.

        Args:
            dataset: Dataset
            batch_size: Batch size per device
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            pad_token_id: Padding token ID
            collate_fn: Function to collate examples into batch
            prefetch_size: Number of batches to prefetch
            use_pjit: Whether to use pjit for data loading
            use_circular_buffer: Whether to use circular buffer for data loading
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_token_id = pad_token_id
        self.prefetch_size = prefetch_size
        self.use_pjit = use_pjit
        self.use_circular_buffer = use_circular_buffer
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        # Set collate function with optimized padding
        if collate_fn is None:
            self.collate_fn = lambda examples: self._optimized_pad_batch(examples, pad_token_id)
        else:
            self.collate_fn = collate_fn

        # Get number of devices
        self.num_devices = jax.device_count()
        print(f"TPUDataLoader: Using {self.num_devices} devices")

        # Compute global batch size
        self.global_batch_size = self.batch_size * self.num_devices
        print(f"TPUDataLoader: Global batch size: {self.global_batch_size}")

        # Create prefetch buffer
        self.prefetch_buffer = []

    def _optimized_pad_batch(self, examples: List[Dict[str, np.ndarray]], pad_token_id: int) -> Dict[str, np.ndarray]:
        """
        Optimized padding function for TPU v4-32.

        Args:
            examples: List of examples
            pad_token_id: Padding token ID

        Returns:
            Padded batch
        """
        # Get maximum length
        max_length = max(example["input_ids"].shape[0] for example in examples)

        # Round max_length to multiple of 128 for TPU efficiency
        max_length = ((max_length + 127) // 128) * 128

        # Initialize batch with preallocated arrays
        batch_size = len(examples)
        batch = {
            "input_ids": np.full((batch_size, max_length), pad_token_id, dtype=np.int32),
            "attention_mask": np.zeros((batch_size, max_length), dtype=np.int32),
            "position_ids": np.zeros((batch_size, max_length), dtype=np.int32),
        }

        # Fill batch with vectorized operations where possible
        for i, example in enumerate(examples):
            length = example["input_ids"].shape[0]
            batch["input_ids"][i, :length] = example["input_ids"]
            batch["attention_mask"][i, :length] = 1  # Simplified mask creation
            batch["position_ids"][i, :length] = np.arange(length, dtype=np.int32)

        return batch

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Iterate over batches with optimized data loading for TPU v4-32.

        Returns:
            Iterator over batches
        """
        # Get indices
        indices = np.array(range(len(self.dataset)), dtype=np.int32)

        # Shuffle indices if requested
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)

        # Create batches
        num_batches = len(indices) // self.global_batch_size
        if not self.drop_last and len(indices) % self.global_batch_size > 0:
            num_batches += 1

        # Prefetch batches in background if enabled
        if self.use_circular_buffer:
            import threading
            import queue

            # Create queue for prefetched batches
            batch_queue = queue.Queue(maxsize=self.prefetch_size)

            # Define batch loading function
            def load_batches():
                for batch_idx in range(num_batches):
                    # Get batch indices
                    start_idx = batch_idx * self.global_batch_size
                    end_idx = min(start_idx + self.global_batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]

                    # Pad batch if necessary
                    if len(batch_indices) < self.global_batch_size and not self.drop_last:
                        # Pad with repeated indices
                        pad_indices = batch_indices[:self.global_batch_size - len(batch_indices)]
                        batch_indices = np.concatenate([batch_indices, pad_indices])
                    elif len(batch_indices) < self.global_batch_size and self.drop_last:
                        # Skip incomplete batch
                        continue

                    # Get examples
                    examples = [self.dataset[int(i)] for i in batch_indices]

                    # Collate examples into batch
                    batch = self.collate_fn(examples)

                    # Create masks
                    attention_mask, padding_mask = create_masks(
                        batch["input_ids"],
                        self.pad_token_id
                    )

                    # Update batch
                    batch["attention_mask"] = attention_mask
                    batch["padding_mask"] = padding_mask

                    # Reshape batch for devices
                    batch = {
                        k: v.reshape(self.num_devices, self.batch_size, *v.shape[1:])
                        for k, v in batch.items()
                    }

                    # Convert to JAX arrays with optimized memory layout
                    batch = {k: jnp.asarray(v, dtype=jnp.dtype(v.dtype)) for k, v in batch.items()}

                    # Add batch to queue
                    batch_queue.put(batch)

                # Signal end of batches
                batch_queue.put(None)

            # Start batch loading thread
            thread = threading.Thread(target=load_batches)
            thread.daemon = True
            thread.start()

            # Yield batches from queue
            while True:
                batch = batch_queue.get()
                if batch is None:
                    break
                yield batch
        else:
            # Standard batch loading
            for batch_idx in range(num_batches):
                # Get batch indices
                start_idx = batch_idx * self.global_batch_size
                end_idx = min(start_idx + self.global_batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # Skip incomplete batch if dropping last
                if len(batch_indices) < self.global_batch_size and self.drop_last:
                    continue

                # Pad batch if necessary
                if len(batch_indices) < self.global_batch_size:
                    # Pad with repeated indices
                    pad_indices = batch_indices[:self.global_batch_size - len(batch_indices)]
                    batch_indices = np.concatenate([batch_indices, pad_indices])

                # Get examples
                examples = [self.dataset[int(i)] for i in batch_indices]

                # Collate examples into batch
                batch = self.collate_fn(examples)

                # Create masks
                attention_mask, padding_mask = create_masks(
                    batch["input_ids"],
                    self.pad_token_id
                )

                # Update batch
                batch["attention_mask"] = attention_mask
                batch["padding_mask"] = padding_mask

                # Reshape batch for devices
                batch = {
                    k: v.reshape(self.num_devices, self.batch_size, *v.shape[1:])
                    for k, v in batch.items()
                }

                # Convert to JAX arrays with optimized memory layout
                batch = {k: jnp.asarray(v, dtype=jnp.dtype(v.dtype)) for k, v in batch.items()}

                yield batch
