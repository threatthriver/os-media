# Training LLM with Local Storage

This guide provides instructions for training a Large Language Model (LLM) using local storage instead of Google Cloud Storage (GCS) buckets.

## System Requirements

Based on your system configuration:
- CPU: 240 cores (AMD EPYC 7B12)
- RAM: Significant amount required (at least 1.5TB recommended for 600B model)
- Storage: At least 5TB of free space for checkpoints and datasets

## Quick Start

To start training with local storage:

```bash
# Check resources
python3 check_resources.py --model_size 600b

# Start training
./train_with_local_storage.sh
```

## Configuration Options

You can customize the training with various options:

```bash
./train_with_local_storage.sh \
  --model_size 600b \
  --output_dir ./output \
  --batch_size 32 \
  --steps 500000 \
  --learning_rate 0.00015 \
  --max_seq_length 131072 \
  --dataset HuggingFaceFW/fineweb \
  --num_checkpoints 5 \
  --checkpoint_interval 1000 \
  --logging_interval 100 \
  --eval_interval 5000
```

## Available Model Sizes

- `7b`: 7 billion parameters
- `13b`: 13 billion parameters
- `70b`: 70 billion parameters
- `175b`: 175 billion parameters
- `600b`: 600 billion parameters

## Memory Requirements

| Model Size | Parameters | Model Memory | Optimizer Memory | Total Training Memory |
|------------|------------|--------------|------------------|------------------------|
| 7b         | 7 billion  | 14 GB        | 28 GB            | ~50 GB                 |
| 13b        | 13 billion | 26 GB        | 52 GB            | ~90 GB                 |
| 70b        | 70 billion | 140 GB       | 280 GB           | ~450 GB                |
| 175b       | 175 billion| 350 GB       | 700 GB           | ~1.1 TB                |
| 600b       | 600 billion| 1.2 TB       | 2.4 TB           | ~3.8 TB                |

## Disk Space Requirements

Each checkpoint requires approximately 1.2x the model size in disk space. With 5 checkpoints for a 600B model, you'll need about 3.6TB of free disk space.

## Resuming Training

To resume training from the latest checkpoint:

```bash
./train_with_local_storage.sh --resume
```

## Monitoring Training

Training logs are saved to the output directory. You can monitor the training progress with:

```bash
tail -f ./output/train.log
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the batch size:
   ```bash
   ./train_with_local_storage.sh --batch_size 16
   ```

2. Enable more aggressive gradient checkpointing:
   ```bash
   ./local_train.sh --use_gradient_checkpointing
   ```

3. Try a smaller model size:
   ```bash
   ./train_with_local_storage.sh --model_size 175b
   ```

### Disk Space Issues

If you run out of disk space:

1. Reduce the number of checkpoints:
   ```bash
   ./train_with_local_storage.sh --num_checkpoints 3
   ```

2. Increase the checkpoint interval:
   ```bash
   ./train_with_local_storage.sh --checkpoint_interval 5000
   ```

## Advanced Configuration

For advanced configuration, you can modify the `local_model_config.yml` file directly.
