#!/usr/bin/env python3
"""
Test script for Indian language model training components.
This script tests the tokenizer, dataset loading, and model creation.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test.log')
    ]
)
logger = logging.getLogger("Indian-LLM-Test")

def test_tokenizer(sample_text=None):
    """Test the Indian language tokenizer."""
    logger.info("Testing Indian language tokenizer...")
    
    try:
        from tokenizer import create_tokenizer
        
        # Sample text in multiple Indian languages
        if sample_text is None:
            sample_text = """
            English: Hello world, this is a test.
            Hindi: नमस्ते दुनिया, यह एक परीक्षण है।
            Tamil: வணக்கம் உலகம், இது ஒரு சோதனை.
            Telugu: హలో వరల్డ్, ఇది ఒక పరీక్ష.
            Bengali: হ্যালো ওয়ার্ল্ড, এটি একটি পরীক্ষা।
            Code: def hello_world(): print("Hello, World!")
            """
        
        # Create a simple tokenizer for testing
        tokenizer = create_tokenizer(
            train_files=None,
            pretrained_path="./tokenizer/indian_tokenizer" if os.path.exists("./tokenizer/indian_tokenizer.model") else None
        )
        
        # If no pretrained tokenizer exists, train a new one
        if tokenizer is None:
            logger.info("No pretrained tokenizer found. Creating a temporary one for testing...")
            
            # Create a temporary file with sample text
            with open("sample_text.txt", "w", encoding="utf-8") as f:
                f.write(sample_text)
            
            # Train tokenizer on sample text
            tokenizer = create_tokenizer(
                train_files=["sample_text.txt"],
                output_dir="./tokenizer",
                model_prefix="indian_tokenizer",
                vocab_size=1000  # Small vocab for testing
            )
        
        # Test tokenization
        tokens = tokenizer.tokenize(sample_text)
        logger.info(f"Tokenized sample text into {len(tokens)} tokens")
        logger.info(f"First 10 tokens: {tokens[:10]}")
        
        # Test encoding and decoding
        encoded = tokenizer.encode(sample_text)
        logger.info(f"Encoded sample text into {len(encoded)} token ids")
        logger.info(f"First 10 token ids: {encoded[:10]}")
        
        decoded = tokenizer.decode(encoded)
        logger.info(f"Decoded text length: {len(decoded)}")
        
        # Check if decoding preserves the text
        if len(decoded) > len(sample_text) * 0.8:  # Allow some loss due to normalization
            logger.info("✓ Tokenizer encoding/decoding test passed")
        else:
            logger.warning("⚠️ Tokenizer encoding/decoding test failed - significant text loss")
        
        return tokenizer
    
    except Exception as e:
        logger.error(f"Tokenizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_indian_datasets():
    """Test loading Indian language datasets."""
    logger.info("Testing Indian language datasets...")
    
    try:
        from indian_datasets import load_indian_dataset_mix
        
        # Try to load datasets in streaming mode
        logger.info("Loading Indian language datasets in streaming mode...")
        dataset = load_indian_dataset_mix(streaming=True)
        
        # Test by getting a few samples
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 5:  # Get 5 samples
                break
        
        logger.info(f"Successfully loaded {len(samples)} samples from the dataset")
        
        # Check if samples contain text
        text_fields = []
        for sample in samples:
            for key in sample:
                if isinstance(sample[key], str) and len(sample[key]) > 20:
                    text_fields.append(key)
                    break
        
        if text_fields:
            logger.info(f"✓ Dataset contains text fields: {set(text_fields)}")
            # Print a sample
            sample_idx = 0
            field = text_fields[0]
            logger.info(f"Sample text: {samples[sample_idx][field][:100]}...")
        else:
            logger.warning("⚠️ No text fields found in dataset samples")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_model():
    """Test model creation and forward pass."""
    logger.info("Testing model creation...")
    
    try:
        import jax
        import jax.numpy as jnp
        from model import create_model
        
        # Check for TPU availability
        devices = jax.devices()
        logger.info(f"Available devices: {len(devices)}")
        
        tpu_available = any('TPU' in device.platform for device in devices)
        if tpu_available:
            logger.info("✓ TPU devices detected")
        else:
            logger.warning("⚠️ No TPU devices found, using available devices")
        
        # Create a small model for testing
        logger.info("Creating a small model for testing...")
        test_model = create_model(
            model_size="7b",  # Use smallest model for testing
            max_seq_length=1024,
            use_flash_attention=True,
            use_reasoning_layer=True
        )
        
        # Initialize model with random key
        logger.info("Initializing model parameters...")
        rng = jax.random.PRNGKey(0)
        input_shape = (1, 16)  # Batch size 1, sequence length 16
        input_ids = jnp.ones(input_shape, dtype=jnp.int32)
        
        # Initialize parameters (this might take some time)
        params = test_model.init(rng, input_ids)
        
        # Get parameter count
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        logger.info(f"Model initialized with {param_count:,} parameters")
        
        # Try a forward pass
        logger.info("Testing forward pass...")
        outputs = test_model.apply(params, input_ids)
        
        # Check outputs
        if "logits" in outputs:
            logits = outputs["logits"]
            logger.info(f"Forward pass successful. Output shape: {logits.shape}")
            logger.info("✓ Model test passed")
        else:
            logger.warning("⚠️ Forward pass returned unexpected output format")
        
        return test_model
    
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_trainer(tokenizer=None, dataset=None, model=None):
    """Test trainer setup."""
    logger.info("Testing trainer setup...")
    
    try:
        from trainer import TPUTrainer
        
        # Use provided components or create new ones
        if tokenizer is None:
            tokenizer = test_tokenizer()
        
        if dataset is None:
            dataset = test_indian_datasets()
        
        if model is None:
            model = test_model()
        
        # Skip if any component failed
        if None in (tokenizer, dataset, model):
            logger.error("Cannot test trainer because one or more components failed")
            return None
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = TPUTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            output_dir="./test_output",
            max_steps=10,  # Just a few steps for testing
            logging_steps=1,
            save_steps=5,
            use_wandb=False,  # Disable wandb for testing
            tensor_parallel_size=1,  # Use minimal parallelism for testing
            pipeline_parallel_size=1
        )
        
        logger.info("✓ Trainer created successfully")
        
        # Test training setup (but don't actually train)
        logger.info("Testing training setup...")
        trainer.setup_training()
        
        logger.info("✓ Training setup successful")
        
        return trainer
    
    except Exception as e:
        logger.error(f"Trainer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Indian language model components")
    parser.add_argument("--tokenizer", action="store_true", help="Test tokenizer only")
    parser.add_argument("--datasets", action="store_true", help="Test datasets only")
    parser.add_argument("--model", action="store_true", help="Test model only")
    parser.add_argument("--trainer", action="store_true", help="Test trainer only")
    parser.add_argument("--all", action="store_true", help="Test all components")
    
    args = parser.parse_args()
    
    # If no specific test is requested, test all
    if not any([args.tokenizer, args.datasets, args.model, args.trainer, args.all]):
        args.all = True
    
    # Run tests
    tokenizer = None
    dataset = None
    model = None
    trainer = None
    
    if args.tokenizer or args.all:
        tokenizer = test_tokenizer()
    
    if args.datasets or args.all:
        dataset = test_indian_datasets()
    
    if args.model or args.all:
        model = test_model()
    
    if args.trainer or args.all:
        trainer = test_trainer(tokenizer, dataset, model)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    if args.tokenizer or args.all:
        logger.info(f"Tokenizer: {'✓ PASSED' if tokenizer is not None else '✗ FAILED'}")
    
    if args.datasets or args.all:
        logger.info(f"Datasets: {'✓ PASSED' if dataset is not None else '✗ FAILED'}")
    
    if args.model or args.all:
        logger.info(f"Model: {'✓ PASSED' if model is not None else '✗ FAILED'}")
    
    if args.trainer or args.all:
        logger.info(f"Trainer: {'✓ PASSED' if trainer is not None else '✗ FAILED'}")
    
    # Overall result
    if args.all:
        all_passed = all(x is not None for x in [tokenizer, dataset, model, trainer])
        logger.info("\nOverall: " + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
