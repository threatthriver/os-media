"""
Dataset utilities for Indian language datasets.
This module provides functions to load and process Indian language datasets.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets, interleave_datasets, Dataset, IterableDataset

logger = logging.getLogger("LLM-Trainer.IndianDatasets")

# List of Indian language datasets available on Hugging Face
INDIAN_DATASETS = {
    # General Indian language datasets
    "ai4bharat/indic-corp": {
        "description": "Large-scale Indic text corpus with 12 Indian languages",
        "languages": ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
        "size": "9B tokens",
        "weight": 0.25
    },
    "ai4bharat/IndicGLUE": {
        "description": "Benchmark for Indian language NLU tasks",
        "languages": ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
        "size": "100M tokens",
        "weight": 0.05
    },
    "ai4bharat/samanantar": {
        "description": "Parallel corpus for Indian languages",
        "languages": ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
        "size": "46M parallel sentences",
        "weight": 0.10
    },
    "flax-community/oscar-indic": {
        "description": "Web data in Indian languages from OSCAR corpus",
        "languages": ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or"],
        "size": "5B tokens",
        "weight": 0.15
    },
    "neuralspace-reverie/indic-corpusv2": {
        "description": "Cleaned web data in Indian languages",
        "languages": ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or"],
        "size": "3B tokens",
        "weight": 0.15
    },
    # Hindi-specific datasets
    "hindi-llm/hindi-corpus": {
        "description": "Large Hindi text corpus",
        "languages": ["hi"],
        "size": "2B tokens",
        "weight": 0.10
    },
    # English-Indian code-mixed datasets
    "ai4bharat/indiccorp-codemixed": {
        "description": "Code-mixed text in Indian languages and English",
        "languages": ["hi-en", "bn-en", "te-en", "ta-en", "mr-en"],
        "size": "500M tokens",
        "weight": 0.10
    },
    # Global datasets for balance
    "HuggingFaceFW/fineweb": {
        "description": "High-quality web data",
        "languages": ["en", "multilingual"],
        "size": "1.5T tokens",
        "weight": 0.05
    },
    "EleutherAI/pile": {
        "description": "Diverse English text corpus",
        "languages": ["en"],
        "size": "825B tokens",
        "weight": 0.05
    }
}

def load_indian_dataset_mix(streaming=True) -> Union[Dataset, IterableDataset]:
    """
    Load a mix of Indian language datasets with appropriate weights.
    
    Args:
        streaming: Whether to stream the datasets (recommended for large datasets)
        
    Returns:
        A dataset or iterable dataset with a mix of Indian language data
    """
    logger.info("Loading Indian language dataset mix")
    
    # Datasets to load with their weights
    datasets_to_load = []
    
    # Add all Indian datasets with their weights
    for ds_name, ds_info in INDIAN_DATASETS.items():
        datasets_to_load.append((ds_name, ds_info["weight"]))
    
    loaded_datasets = []
    dataset_weights = []
    
    # Load each dataset
    for ds_name, weight in datasets_to_load:
        try:
            logger.info(f"Loading dataset {ds_name}...")
            ds = load_dataset(ds_name, streaming=streaming)
            
            # Most datasets have a 'train' split
            if 'train' in ds:
                loaded_datasets.append(ds['train'])
                dataset_weights.append(weight)
                logger.info(f"Successfully loaded {ds_name} (train split)")
            # Some datasets might have different split names
            elif 'text' in ds:
                loaded_datasets.append(ds['text'])
                dataset_weights.append(weight)
                logger.info(f"Successfully loaded {ds_name} (text split)")
            # If no standard split is found, use the first available split
            else:
                first_split = list(ds.keys())[0]
                loaded_datasets.append(ds[first_split])
                dataset_weights.append(weight)
                logger.info(f"Successfully loaded {ds_name} ({first_split} split)")
                
        except Exception as e:
            logger.warning(f"Could not load {ds_name}: {e}")
    
    if not loaded_datasets:
        logger.error("Failed to load any datasets")
        raise ValueError("No datasets could be loaded")
    
    # Normalize weights
    total_weight = sum(dataset_weights)
    normalized_weights = [w / total_weight for w in dataset_weights]
    
    # Interleave datasets with weights
    logger.info(f"Interleaving {len(loaded_datasets)} datasets with normalized weights")
    
    combined_dataset = interleave_datasets(
        loaded_datasets,
        probabilities=normalized_weights,
        stopping_strategy='first_exhausted'
    )
    
    logger.info("Indian language dataset mix created successfully")
    return combined_dataset

def prepare_training_sample(example, tokenizer, max_length=2048):
    """
    Prepare a training sample by tokenizing the text.
    
    Args:
        example: The example from the dataset
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized example
    """
    # Extract text field (different datasets might have different field names)
    if 'text' in example:
        text = example['text']
    elif 'content' in example:
        text = example['content']
    elif 'sentence' in example:
        text = example['sentence']
    else:
        # Try to find a field that might contain text
        text_fields = [k for k, v in example.items() if isinstance(v, str) and len(v) > 20]
        if text_fields:
            text = example[text_fields[0]]
        else:
            # If no suitable field is found, use an empty string
            text = ""
    
    # Tokenize the text
    tokenized = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
        max_length=max_length
    )
    
    # Create model inputs
    inputs = {
        "input_ids": tokenized,
        "labels": tokenized.copy(),  # For language modeling, labels are the same as inputs
    }
    
    return inputs

def create_indian_data_loader(tokenizer, batch_size=32, max_length=2048, streaming=True):
    """
    Create a data loader for Indian language datasets.
    
    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        streaming: Whether to stream the datasets
        
    Returns:
        A data loader for Indian language datasets
    """
    # Load the dataset mix
    dataset = load_indian_dataset_mix(streaming=streaming)
    
    # Define the preprocessing function
    def preprocess_function(examples):
        # Process each example in the batch
        processed = []
        for i in range(len(examples['text']) if 'text' in examples else 1):
            # Extract the example
            example = {k: examples[k][i] if i < len(examples[k]) else examples[k][0] for k in examples}
            # Process the example
            processed.append(prepare_training_sample(example, tokenizer, max_length))
        
        # Combine the processed examples
        result = {
            "input_ids": [p["input_ids"] for p in processed],
            "labels": [p["labels"] for p in processed],
        }
        
        return result
    
    # Apply preprocessing
    preprocessed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    return preprocessed_dataset
