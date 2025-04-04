#!/bin/bash

# download_datasets.sh - Script to download and prepare datasets for LLM training
# This script will download and process high-quality datasets for training a powerful LLM

set -e  # Exit on any error

if [ -z "$GCS_BUCKET" ]; then
    echo "Please set the GCS_BUCKET environment variable"
    echo "Example: export GCS_BUCKET=gs://your-bucket-name"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo "Please set the PROJECT_ID environment variable"
    echo "Example: export PROJECT_ID=your-gcp-project-id"
    exit 1
fi

echo "Downloading and preparing datasets for LLM training..."

# Create a temporary directory for dataset processing
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

cd maxtext

# Download C4 dataset (Common Crawl)
echo "Downloading C4 dataset..."
bash download_dataset.sh $PROJECT_ID $GCS_BUCKET

# Download RedPajama dataset
echo "Downloading RedPajama dataset..."
mkdir -p $TEMP_DIR/redpajama
gsutil -m cp -r gs://redpajama-data-1/redpajama-v1/* $GCS_BUCKET/redpajama/

# Download The Pile dataset
echo "Downloading The Pile dataset..."
mkdir -p $TEMP_DIR/pile
gsutil -m cp -r gs://the-pile-v1/* $GCS_BUCKET/pile/

# Download SlimPajama dataset (cleaned version of RedPajama)
echo "Downloading SlimPajama dataset..."
mkdir -p $TEMP_DIR/slimpajama
gsutil -m cp -r gs://cerebras-slimpajama/* $GCS_BUCKET/slimpajama/

# Create a combined dataset index file
echo "Creating dataset index file..."
cat > $TEMP_DIR/dataset_index.json << EOL
{
  "datasets": [
    {
      "name": "c4",
      "path": "$GCS_BUCKET/c4/en/3.0.1/",
      "weight": 0.3
    },
    {
      "name": "redpajama",
      "path": "$GCS_BUCKET/redpajama/",
      "weight": 0.2
    },
    {
      "name": "pile",
      "path": "$GCS_BUCKET/pile/",
      "weight": 0.2
    },
    {
      "name": "slimpajama",
      "path": "$GCS_BUCKET/slimpajama/",
      "weight": 0.3
    }
  ]
}
EOL

# Upload the dataset index file to GCS
gsutil cp $TEMP_DIR/dataset_index.json $GCS_BUCKET/dataset_index.json

echo "Dataset preparation complete!"
echo "Datasets are available at: $GCS_BUCKET"
echo "Dataset index file: $GCS_BUCKET/dataset_index.json"

# Clean up
rm -rf $TEMP_DIR
cd ..
