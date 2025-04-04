# TPU-Optimized LLM Training

This repository contains a highly optimized implementation for training Large Language Models (LLMs) on TPU v4-32 hardware. The code is specifically designed to efficiently train a 600 billion parameter model within a 30-day timeframe.

## Features

- **TPU v4-32 Optimizations**: Specialized code for TPU v4-32 hardware with efficient parallelism strategies
- **Memory Efficiency**: Optimized memory usage with gradient checkpointing and efficient attention mechanisms
- **Performance Monitoring**: Comprehensive logging and performance tracking
- **Long Context Support**: Support for very long sequences (up to 128K tokens)
- **Enhanced Reasoning**: Additional reasoning layers for improved model capabilities
- **One-Command Training**: Simple `run.py` script that handles the entire training process
- **Hugging Face Integration**: Automatic model upload to Hugging Face Hub
- **Resource Checking**: Automatic verification of system resources before training

## Requirements

See `requirements.txt` for the full list of dependencies. Key requirements:

```
jax[tpu]>=0.4.20
jaxlib>=0.4.20
libtpu-nightly  # Only required on TPU hardware
flax>=0.7.5
tensorflow>=2.15.0  # Required for full training (Python <= 3.11)
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/threatthriver/train_llm.git
cd train_llm
```

### Option 1: For Local Testing (Python 3.13+)

```bash
# Install minimal dependencies
pip install jax jaxlib numpy

# Make scripts executable
chmod +x simple_train.sh

# Run the simplified script
./simple_train.sh
```

### Option 2: For Full Training (Python 3.11)

```bash
# Create a virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x train_llm.sh

# Run the training script
./train_llm.sh
```

### Option 3: For TPU Training (on Google Cloud)

```bash
# Create a TPU VM with v4-32 configuration
gcloud compute tpus tpu-vm create llm-training \
  --zone=us-central2-b \
  --accelerator-type=v4-32 \
  --version=tpu-vm-base

# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh llm-training --zone=us-central2-b

# Clone the repository and install dependencies
git clone https://github.com/threatthriver/train_llm.git
cd train_llm
pip install -r requirements.txt

# Set required environment variables
export GCS_BUCKET=gs://your-bucket-name
export PROJECT_ID=your-gcp-project-id

# Run the training script
chmod +x tpu_train.sh
./tpu_train.sh
```

## Usage

### One-Command Training (Recommended)

The easiest way to train the model is to use the all-in-one `run.py` script:

```bash
# Just run this single command
python run.py
```

This script will automatically:
- Check system resources
- Install required dependencies
- Prepare the dataset
- Train the model
- Save checkpoints
- Upload the model to Hugging Face (if requested)

You can customize the training with various options:

```bash
# Train with custom parameters
python run.py --model_size 600b --batch_size 32 --steps 500000 --learning_rate 0.00015

# Use a smaller model for testing
python run.py --model_size 7b

# Push to Hugging Face
python run.py --push_to_hub --hf_repo "your-username/your-model-name"

# Resume training from the latest checkpoint
python run.py --resume
```

### Full Training on TPU Hardware

Alternatively, you can use the `tpu_train.py` script or the provided shell scripts:

```bash
# Option 1: Using the Python script directly
python tpu_train.py \
  --model_size 600b \
  --train_file HuggingFaceFW/fineweb \
  --tokenizer_file gpt2 \
  --batch_size 32 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1.5e-4 \
  --max_steps 500000 \
  --warmup_steps 5000 \
  --max_seq_length 131072 \
  --output_dir ./output \
  --parallelism_type tensor \
  --tensor_parallel_size 8 \
  --use_flash_attention \
  --use_gradient_checkpointing \
  --use_rope_scaling \
  --use_reasoning_layer

# Option 2: Using the provided shell script
./train_llm.sh
```

### Local Testing with Simplified Scripts

For local testing without TPU hardware, use the simplified scripts:

```bash
# Run the simplified training script
./simple_train.sh
```

The simplified script will verify that JAX is working correctly on your system and simulate the training process without requiring TPU hardware or all dependencies.

## Architecture

The implementation includes:

- **Optimized Flash Attention**: Blocked implementation for efficient memory usage
- **Tensor Parallelism**: Efficient parameter sharding across TPU devices
- **Data Parallelism**: Optimized data loading and processing
- **Mixed Precision Training**: BFloat16 support for TPU
- **Gradient Checkpointing**: Memory-efficient backpropagation

## Available Model Sizes

| Size | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size |
|------|------------|-------------|--------|----------------|------------------|
| 7b   | 7 billion  | 4096        | 32     | 32             | 11008            |
| 13b  | 13 billion | 5120        | 40     | 40             | 13824            |
| 70b  | 70 billion | 8192        | 80     | 64             | 28672            |
| 175b | 175 billion| 12288       | 96     | 96             | 49152            |
| 600b | 600 billion| 18432       | 128    | 128            | 73728            |

## Performance

On TPU v4-32 hardware, this implementation achieves:

- Efficient training of 600B parameter models
- Support for sequence lengths up to 32K tokens
- Memory-efficient operation with gradient checkpointing
- Optimized communication patterns for TPU pods

## Troubleshooting

### Common Issues

1. **Python Version Compatibility**
   - The full training script requires Python 3.11 or lower due to TensorFlow compatibility
   - If using Python 3.13+, use the simplified script (`simple_train.sh`)

2. **Missing Dependencies**
   - If you encounter `ModuleNotFoundError`, install the missing package:
     ```bash
     pip install <package-name>
     ```
   - For SentencePiece build issues, you may need to install system dependencies:
     ```bash
     # On Ubuntu/Debian
     apt-get install cmake build-essential pkg-config

     # On macOS
     brew install cmake pkg-config
     ```

3. **TPU Support**
   - The full training script requires TPU hardware
   - You'll see errors like `Unable to initialize backend 'tpu'` when running on non-TPU hardware
   - Use the simplified script for local testing

4. **Permission Issues**
   - Make sure scripts are executable:
     ```bash
     chmod +x *.sh
     ```

5. **Memory Issues on TPU**
   - If you encounter OOM errors, try reducing batch size or enabling gradient checkpointing

## License

MIT
