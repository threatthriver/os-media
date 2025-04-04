"""
Dataset classes for LLM training.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Iterator
import jax.numpy as jnp
from data.tokenizer import Tokenizer


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
