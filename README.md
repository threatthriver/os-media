# LLM Training - Simple Solution

This repository contains a streamlined solution for training a Large Language Model (LLM) with up to 600 billion parameters.

## Quick Start

Just run the script and it will handle everything automatically:

```bash
python run.py
```

That's it! The script will:
- Check system resources
- Install required dependencies
- Prepare the dataset
- Train the model
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
# Train with custom parameters
python run.py --model_size 600b --batch_size 32 --steps 500000 --learning_rate 0.00015

# Use a smaller model for testing
python run.py --model_size 7b

# Push to Hugging Face
python run.py --push_to_hub --hf_repo "your-username/your-model-name"

# Resume training from the latest checkpoint
python run.py --resume
```

## Model Features

- 128K token context window
- Enhanced reasoning capabilities
- Flash attention for efficient computation
- Gradient checkpointing for memory efficiency
- Rotary positional embeddings (RoPE)

## License

MIT
