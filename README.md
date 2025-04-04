# LLM Training - TPU-Optimized for Coding Excellence

This repository contains a streamlined solution for training a Large Language Model (LLM) with up to 600 billion parameters, specifically optimized for TPU hardware and designed to excel at coding tasks. The trained model aims to outperform GPT and Gemini models in coding and reasoning benchmarks.

## Quick Start

Just run the script and it will handle everything automatically:

```bash
python run.py --dataset code-mix
```

That's it! The script will:
- Check system resources and TPU availability
- Install required dependencies
- Prepare the optimized dataset mix for coding tasks
- Train the model with specialized reasoning layers
- Save checkpoints
- Upload the model to Hugging Face (if requested)

## Files

This repository contains just 3 essential files:

1. **run.py** - The main script that handles everything
2. **model.py** - Contains the model architecture
3. **requirements.txt** - Lists all dependencies

## Available Model Sizes

| Size | Parameters | Hidden Size | Layers | Attention Heads |
|------|------------|-------------|--------|----------------|
| 7b   | 7 billion  | 4096        | 32     | 32             |
| 13b  | 13 billion | 5120        | 40     | 40             |
| 70b  | 70 billion | 8192        | 80     | 64             |
| 175b | 175 billion| 12288       | 96     | 96             |
| 600b | 600 billion| 18432       | 128    | 128            |

## Command-Line Options

```bash
# Train with optimized dataset mix for coding tasks
python run.py --dataset code-mix --model_size 600b

# Use a smaller model for testing
python run.py --model_size 7b --dataset code-mix

# Customize TPU parallelism
python run.py --tensor_parallel_size 8 --dataset code-mix

# Disable specialized reasoning layers (not recommended for coding tasks)
python run.py --use_reasoning_layer false --dataset code-mix

# Push to Hugging Face
python run.py --dataset code-mix --push_to_hub --hf_repo "your-username/your-model-name"

# Resume training from the latest checkpoint
python run.py --dataset code-mix --resume
```

## TPU Optimization Options

```bash
# Full optimization for TPU v4-32
python run.py \
  --dataset code-mix \
  --model_size 600b \
  --batch_size 32 \
  --steps 500000 \
  --learning_rate 0.00015 \
  --max_seq_length 131072 \
  --use_flash_attention \
  --use_reasoning_layer \
  --tensor_parallel_size 8 \
  --gradient_checkpointing \
  --precision bfloat16
```

## Model Features

- **128K token context window** for handling large code repositories and documentation
- **Specialized reasoning layers** designed specifically for coding tasks
- **Flash attention** for efficient computation on TPU hardware
- **Gradient checkpointing** for memory efficiency with large models
- **Rotary positional embeddings (RoPE)** with scaling for extended context
- **Mixture of experts** approach in reasoning layers for specialized code understanding
- **TPU-optimized architecture** with tensor parallelism for maximum performance
- **Code-focused dataset mix** combining GitHub code, The Stack, and high-quality text data
- **SwiGLU activation** for improved reasoning capabilities
- **Cosine learning rate schedule** with warmup for stable training

## License

MIT
