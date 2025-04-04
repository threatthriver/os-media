#!/bin/bash

# setup_environment.sh - Script to set up the environment for training a powerful LLM on TPU v4-32
# This script will clone MaxText, install dependencies, and configure the environment

set -e  # Exit on any error

echo "Setting up environment for LLM training on TPU v4-32..."

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Clone MaxText repository
if [ ! -d "maxtext" ]; then
    echo "Cloning MaxText repository..."
    git clone https://github.com/AI-Hypercomputer/maxtext.git
    cd maxtext
else
    echo "MaxText repository already exists, updating..."
    cd maxtext
    git pull
fi

# Install dependencies
echo "Installing dependencies..."
bash setup.sh
pre-commit install

# Set up environment variables
echo "Setting up environment variables..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
export JAX_PLATFORMS="tpu"

# Check TPU configuration
echo "Checking TPU configuration..."
python3 -c "import jax; print(f'TPU devices: {jax.device_count()}')"
python3 -c "import jax; print(f'TPU type: {jax.devices()[0].platform}')"

# Create a GCS bucket for checkpoints and logs if it doesn't exist
# Note: You need to replace YOUR_PROJECT_ID with your actual GCP project ID
if [ -z "$GCS_BUCKET" ]; then
    echo "Please set the GCS_BUCKET environment variable"
    echo "Example: export GCS_BUCKET=gs://your-bucket-name"
    exit 1
fi

echo "Checking GCS bucket access..."
gsutil ls $GCS_BUCKET > /dev/null || (echo "Creating GCS bucket $GCS_BUCKET..." && gsutil mb -p $PROJECT_ID $GCS_BUCKET)

# Set up Cloud Storage FUSE for efficient data access
echo "Setting up Cloud Storage FUSE..."
BUCKET_NAME=$(echo $GCS_BUCKET | sed 's/gs:\/\///')
MOUNT_PATH="/tmp/gcsfuse"
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$BUCKET_NAME MOUNT_PATH=$MOUNT_PATH

echo "Environment setup complete!"
cd ..

# Return to the original directory
echo "You can now run the training script with: bash tpu_train.sh"
