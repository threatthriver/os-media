"""
Dataset classes for LLM training.
"""

import os
import json
import random
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable, Iterator, Any
import jax.numpy as jnp
from data.tokenizer import Tokenizer

# Set up logging
logger = logging.getLogger(__name__)

# Try to import datasets library for streaming
try:
    import datasets
    from datasets import load_dataset, Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFace datasets library not available. Streaming datasets will be disabled.")
    DATASETS_AVAILABLE = False


class Dataset:
    """
    Base dataset class.
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize dataset.

        Args:
            tokenizer: Tokenizer for encoding/decoding text
        """
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary of tensors
        """
        raise NotImplementedError


class StreamingDataset(Dataset):
    """
    Streaming dataset for efficient training with large datasets.

    This dataset streams data from disk or remote sources, minimizing memory usage.
    It supports HuggingFace datasets streaming mode for efficient processing.

    Attributes:
        tokenizer: Tokenizer for encoding/decoding text
        dataset_path: Path to dataset file or HuggingFace dataset name
        max_seq_length: Maximum sequence length
        streaming: Whether to use streaming mode
        buffer_size: Size of buffer for streaming
        seed: Random seed for shuffling
        hf_dataset: HuggingFace dataset object
        text_column: Name of text column in dataset
        buffer: Buffer of processed examples
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        max_seq_length: int = 131072,  # Support for 128K tokens
        streaming: bool = True,
        buffer_size: int = 1000,
        seed: int = 42,
        text_column: str = "text",
        preprocessing_num_workers: int = 16,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize streaming dataset.

        Args:
            tokenizer: Tokenizer for encoding/decoding text
            dataset_path: Path to dataset file or HuggingFace dataset name
            max_seq_length: Maximum sequence length
            streaming: Whether to use streaming mode
            buffer_size: Size of buffer for streaming
            seed: Random seed for shuffling
            text_column: Name of text column in dataset
            preprocessing_num_workers: Number of workers for preprocessing
            use_auth_token: HuggingFace auth token for private datasets
        """
        super().__init__(tokenizer)

        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length
        self.streaming = streaming and DATASETS_AVAILABLE
        self.buffer_size = buffer_size
        self.seed = seed
        self.text_column = text_column
        self.preprocessing_num_workers = preprocessing_num_workers

        # Initialize buffer
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_ready = threading.Event()
        self.buffer_idx = 0
        self.dataset_exhausted = False

        # Load dataset
        self._load_dataset(use_auth_token)

        # Start buffer filling thread
        if self.streaming:
            self.buffer_thread = threading.Thread(target=self._fill_buffer)
            self.buffer_thread.daemon = True
            self.buffer_thread.start()

    def _load_dataset(self, use_auth_token: Optional[str] = None):
        """
        Load dataset from file or HuggingFace.

        Args:
            use_auth_token: HuggingFace auth token for private datasets
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets library is required for streaming datasets")

        logger.info(f"Loading dataset from {self.dataset_path}")
        start_time = time.time()

        # Check if dataset_path is a file or HuggingFace dataset
        if os.path.exists(self.dataset_path):
            # Load from file
            file_extension = os.path.splitext(self.dataset_path)[1]
            if file_extension == ".jsonl" or file_extension == ".json":
                self.hf_dataset = load_dataset(
                    "json",
                    data_files=self.dataset_path,
                    streaming=self.streaming,
                    use_auth_token=use_auth_token
                )["train"]
            elif file_extension == ".txt":
                self.hf_dataset = load_dataset(
                    "text",
                    data_files=self.dataset_path,
                    streaming=self.streaming,
                    use_auth_token=use_auth_token
                )["train"]
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        else:
            # Load from HuggingFace
            self.hf_dataset = load_dataset(
                self.dataset_path,
                streaming=self.streaming,
                use_auth_token=use_auth_token
            )["train"]

        # Shuffle dataset if streaming
        if self.streaming:
            self.hf_dataset = self.hf_dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

        # Get dataset length if not streaming
        if not self.streaming:
            self.dataset_length = len(self.hf_dataset)
            logger.info(f"Dataset length: {self.dataset_length}")

    def _fill_buffer(self):
        """
        Fill buffer with processed examples in background thread.
        """
        try:
            # Create iterator
            dataset_iter = iter(self.hf_dataset)

            while True:
                # Check if buffer needs filling
                with self.buffer_lock:
                    if len(self.buffer) >= self.buffer_size:
                        # Buffer is full, wait
                        self.buffer_ready.set()
                        time.sleep(0.1)
                        continue

                # Get next example
                try:
                    example = next(dataset_iter)
                except StopIteration:
                    # Dataset exhausted
                    self.dataset_exhausted = True
                    self.buffer_ready.set()
                    break

                # Process example
                processed = self._process_example(example)

                # Add to buffer
                with self.buffer_lock:
                    self.buffer.append(processed)

                    # Signal that buffer has items
                    if len(self.buffer) > 0:
                        self.buffer_ready.set()
        except Exception as e:
            logger.error(f"Error in buffer filling thread: {e}")
            self.dataset_exhausted = True
            self.buffer_ready.set()

    def _process_example(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process example from dataset.

        Args:
            example: Example from dataset

        Returns:
            Processed example
        """
        # Get text from example
        if self.text_column in example:
            text = example[self.text_column]
        else:
            # Try to find text column
            text_columns = ["text", "content", "document", "input_text"]
            for col in text_columns:
                if col in example:
                    text = example[col]
                    break
            else:
                # Use first string column
                for key, value in example.items():
                    if isinstance(value, str):
                        text = value
                        break
                else:
                    raise ValueError(f"No text column found in example: {example}")

        # Tokenize text
        input_ids = self.tokenizer.encode(text)

        # Truncate if necessary
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]

        # Create numpy arrays
        input_ids = np.array(input_ids, dtype=np.int32)

        return {"input_ids": input_ids}

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Dataset length
        """
        if self.streaming:
            # For streaming datasets, return a large number
            return 1_000_000_000  # Effectively infinite
        else:
            return self.dataset_length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Args:
            idx: Item index (ignored in streaming mode)

        Returns:
            Dictionary of tensors
        """
        if self.streaming:
            # In streaming mode, get item from buffer
            self.buffer_ready.wait()  # Wait for buffer to have items

            with self.buffer_lock:
                if len(self.buffer) == 0:
                    if self.dataset_exhausted:
                        # Reset buffer index and raise StopIteration
                        self.buffer_idx = 0
                        raise StopIteration("Dataset exhausted")
                    else:
                        # Wait for buffer to be filled
                        self.buffer_ready.clear()
                        return self.__getitem__(idx)  # Retry

                # Get item from buffer
                item = self.buffer[self.buffer_idx]
                self.buffer_idx += 1

                # If we've consumed all items in buffer, clear it
                if self.buffer_idx >= len(self.buffer):
                    self.buffer = []
                    self.buffer_idx = 0
                    self.buffer_ready.clear()

                return item
        else:
            # In non-streaming mode, get item directly
            example = self.hf_dataset[idx]
            return self._process_example(example)


class TextDataset(Dataset):
    """
    Text dataset.

    Attributes:
        texts: List of texts
        tokenizer: Tokenizer for encoding/decoding text
        max_length: Maximum sequence length
        add_bos: Whether to add beginning of sequence token
        add_eos: Whether to add end of sequence token
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        max_length: int = 1024,
        add_bos: bool = True,
        add_eos: bool = False
    ):
        """
        Initialize dataset.

        Args:
            texts: List of texts
            tokenizer: Tokenizer for encoding/decoding text
            max_length: Maximum sequence length
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token
        """
        super().__init__(tokenizer)
        self.texts = texts
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Dataset length
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary of tensors
        """
        # Get text
        text = self.texts[idx]

        # Encode text
        token_ids = self.tokenizer.encode(
            text,
            add_bos=self.add_bos,
            add_eos=self.add_eos
        )

        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create attention mask
        attention_mask = np.ones(len(token_ids), dtype=np.int32)

        # Create position IDs
        position_ids = np.arange(len(token_ids), dtype=np.int32)

        return {
            "input_ids": np.array(token_ids, dtype=np.int32),
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }


class TokenizedDataset(Dataset):
    """
    Pre-tokenized dataset.

    Attributes:
        token_ids: List of token ID sequences
        tokenizer: Tokenizer for encoding/decoding text
        max_length: Maximum sequence length
        add_bos: Whether to add beginning of sequence token
        add_eos: Whether to add end of sequence token
    """

    def __init__(
        self,
        token_ids: List[List[int]],
        tokenizer: Tokenizer,
        max_length: int = 1024,
        add_bos: bool = True,
        add_eos: bool = False
    ):
        """
        Initialize dataset.

        Args:
            token_ids: List of token ID sequences
            tokenizer: Tokenizer for encoding/decoding text
            max_length: Maximum sequence length
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token
        """
        super().__init__(tokenizer)
        self.token_ids = token_ids
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Dataset length
        """
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary of tensors
        """
        # Get token IDs
        ids = self.token_ids[idx].copy()

        # Add special tokens
        if self.add_bos:
            ids = [self.tokenizer.bos_token_id] + ids

        if self.add_eos:
            ids = ids + [self.tokenizer.eos_token_id]

        # Truncate if necessary
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        # Create attention mask
        attention_mask = np.ones(len(ids), dtype=np.int32)

        # Create position IDs
        position_ids = np.arange(len(ids), dtype=np.int32)

        return {
            "input_ids": np.array(ids, dtype=np.int32),
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }


class ConcatDataset(Dataset):
    """
    Concatenated dataset.

    Attributes:
        datasets: List of datasets
        tokenizer: Tokenizer for encoding/decoding text
        weights: Weights for sampling from datasets
    """

    def __init__(
        self,
        datasets: List[Dataset],
        tokenizer: Tokenizer,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize dataset.

        Args:
            datasets: List of datasets
            tokenizer: Tokenizer for encoding/decoding text
            weights: Weights for sampling from datasets
        """
        super().__init__(tokenizer)
        self.datasets = datasets

        # Set weights if not provided
        if weights is None:
            weights = [1.0] * len(datasets)

        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

        # Compute cumulative weights
        self.cumulative_weights = np.cumsum(self.weights)

        # Compute dataset lengths
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Dataset length
        """
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary of tensors
        """
        # Sample dataset according to weights
        r = random.random()
        dataset_idx = 0
        for i, cw in enumerate(self.cumulative_weights):
            if r <= cw:
                dataset_idx = i
                break

        # Sample item from selected dataset
        item_idx = random.randint(0, self.lengths[dataset_idx] - 1)

        return self.datasets[dataset_idx][item_idx]


def load_jsonl_dataset(
    file_path: str,
    tokenizer: Tokenizer,
    text_key: str = "text",
    max_length: int = 1024,
    add_bos: bool = True,
    add_eos: bool = False,
    max_samples: Optional[int] = None
) -> TextDataset:
    """
    Load dataset from JSONL file.

    Args:
        file_path: Path to JSONL file
        tokenizer: Tokenizer for encoding/decoding text
        text_key: Key for text field in JSON objects
        max_length: Maximum sequence length
        add_bos: Whether to add beginning of sequence token
        add_eos: Whether to add end of sequence token
        max_samples: Maximum number of samples to load

    Returns:
        Text dataset
    """
    # Load texts from JSONL file
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break

            data = json.loads(line)
            texts.append(data[text_key])

    # Create dataset
    return TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=add_bos,
        add_eos=add_eos
    )


def load_text_dataset(
    file_path: str,
    tokenizer: Tokenizer,
    max_length: int = 1024,
    add_bos: bool = True,
    add_eos: bool = False,
    max_samples: Optional[int] = None
) -> TextDataset:
    """
    Load dataset from text file.

    Args:
        file_path: Path to text file
        tokenizer: Tokenizer for encoding/decoding text
        max_length: Maximum sequence length
        add_bos: Whether to add beginning of sequence token
        add_eos: Whether to add end of sequence token
        max_samples: Maximum number of samples to load

    Returns:
        Text dataset
    """
    # Load texts from text file
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break

            texts.append(line.strip())

    # Create dataset
    return TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=add_bos,
        add_eos=add_eos
    )
