#!/bin/bash
# Script to upload the repository to Hugging Face using git

# Set the Hugging Face token
HF_TOKEN="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"
REPO_NAME="Threatthriver/tpu-optimized-llm"

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
fi

# Configure git
git config user.email "user@example.com"
git config user.name "HF Uploader"

# Add files
git add .

# Commit
git commit -m "Initial commit of TPU-optimized LLM training code"

# Add Hugging Face as remote
git remote add hf "https://huggingface.co/datasets/${REPO_NAME}"

# Push to Hugging Face
git push -f "https://${HF_TOKEN}@huggingface.co/datasets/${REPO_NAME}" main

echo "Repository uploaded to: https://huggingface.co/datasets/${REPO_NAME}"
