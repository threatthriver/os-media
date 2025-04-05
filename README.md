# LLM Training - TPU-Optimized for 600B Parameter Model

This repository contains a streamlined solution for training a Large Language Model (LLM) with up to 600 billion parameters, specifically optimized for Google TPU v4-32 hardware. The implementation is designed to maximize performance on TPU hardware and includes specialized optimizations for coding and reasoning tasks.

## Quick Start

Just run the unified launcher script and it will handle everything automatically:

```bash
python3 run_all.py
```

That's it! The script will:
- Check system resources and TPU availability
- Install required dependencies
- Prepare the optimized dataset mix for coding tasks
- Train the model with specialized reasoning layers
- Save checkpoints
- Upload the model to Hugging Face (if requested)

## Available Model Sizes

| Size | Parameters | Hidden Size | Layers | Attention Heads |
|------|------------|-------------|--------|----------------|
| 7b   | 7 billion  | 4096        | 32     | 32             |
| 13b  | 13 billion | 5120        | 40     | 40             |
| 70b  | 70 billion | 8192        | 80     | 64             |
| 175b | 175 billion| 12288       | 96     | 96             |
| 600b | 600 billion| 18432       | 128    | 128            |

## Running in Ubuntu Terminal

To run the script in an Ubuntu terminal:

1. Make sure you have Python 3.8+ installed:
   ```bash
   python3 --version
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Make the script executable:
   ```bash
   chmod +x run_all.py
   ```

4. Run the script:
   ```bash
   ./run_all.py
   ```

5. For specific configurations:
   ```bash
   ./run_all.py --model_size 70b --dataset indian-mix --steps 100000
   ```

## Available Modes

The unified launcher supports multiple operation modes:

| Mode      | Description                                      | Command                           |
|-----------|--------------------------------------------------|-----------------------------------|
| train     | Train the model (default)                        | `./run_all.py --mode train`       |
| test      | Run tests on the model and components            | `./run_all.py --mode test`        |
| tokenize  | Create or load a tokenizer                       | `./run_all.py --mode tokenize`    |
| all       | Run all operations (tokenize, train, test)       | `./run_all.py --mode all`         |

## Command-Line Options

```
usage: run_all.py [-h] [--mode {train,test,tokenize,all}]
                  [--model_size {7b,13b,70b,175b,600b}] [--output_dir OUTPUT_DIR]
                  [--batch_size BATCH_SIZE] [--steps STEPS]
                  [--learning_rate LEARNING_RATE] [--max_seq_length MAX_SEQ_LENGTH]
                  [--use_flash_attention] [--use_reasoning_layer]
                  [--num_checkpoints NUM_CHECKPOINTS]
                  [--dataset {code-mix,indian-mix,HuggingFaceFW/fineweb,codeparrot/github-code,bigcode/the-stack,togethercomputer/RedPajama-Data-1T,EleutherAI/pile}]
                  [--tokenizer_path TOKENIZER_PATH] [--push_to_hub] [--hf_repo HF_REPO]
                  [--resume] [--force] [--debug] [--skip_dependency_check]
                  [--skip_resource_check]
```

### Key Options

- `--mode`: Operation mode (train, test, tokenize, all)
- `--model_size`: Model size (7b, 13b, 70b, 175b, 600b)
- `--dataset`: Dataset to use (code-mix, indian-mix, or specific datasets)
- `--steps`: Number of training steps
- `--learning_rate`: Learning rate for training
- `--max_seq_length`: Maximum sequence length
- `--push_to_hub`: Push model to Hugging Face Hub
- `--debug`: Enable debug logging

## Examples

### Train a 70B model with Indian language focus

```bash
./run_all.py --model_size 70b --dataset indian-mix --steps 100000
```

### Test model components

```bash
./run_all.py --mode test
```

### Train with custom parameters

```bash
./run_all.py --model_size 13b --batch_size 64 --learning_rate 0.0001 --steps 50000
```

### Push trained model to Hugging Face

```bash
./run_all.py --push_to_hub --hf_repo "your-username/your-model-name"
```

## TPU Optimization

This implementation includes specialized optimizations for Google TPU v4-32 hardware:

- Uses bfloat16 precision for optimal TPU performance
- Implements efficient tensor parallelism across TPU cores
- Includes pipeline parallelism for large model training
- Optimizes memory usage with gradient checkpointing
- Implements flash attention for faster training
- Uses specialized reasoning layers for improved model capabilities

## Dataset Options

The launcher supports multiple dataset configurations:

- `code-mix`: A mix of coding-focused datasets (GitHub code, The Stack, etc.)
- `indian-mix`: A mix of Indian language datasets
- Individual datasets from Hugging Face

## Model Architecture

The model architecture includes:

- Transformer-based architecture with up to 128 layers
- Flash attention for efficient sequence processing
- Specialized reasoning layers for improved capabilities
- RoPE positional embeddings for better long-context understanding
- SwiGLU activation functions for improved reasoning

## Requirements

- Python 3.8+
- JAX/Flax for TPU optimization
- Hugging Face libraries (transformers, datasets)
- Weights & Biases for experiment tracking
