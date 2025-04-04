#!/usr/bin/env python3
"""
Script to upload the repository to Hugging Face using the Hugging Face CLI.
"""

import os
import sys
import subprocess
from huggingface_hub import create_repo, upload_folder

def upload_to_hugging_face(repo_name="Threatthriver/tpu-optimized-llm"):
    """
    Upload the repository to Hugging Face using the Hugging Face CLI.

    Args:
        repo_name: Repository name
    """
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, repo_type="dataset")
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository already exists or error: {e}")

    # Upload the folder
    upload_folder(
        folder_path=".",
        repo_id=repo_name,
        repo_type="dataset",
        ignore_patterns=[".git/*", "__pycache__/*", "*.pyc", ".DS_Store", "upload_to_hf.py", "upload_to_hf_cli.py"]
    )

    print(f"Repository uploaded to: https://huggingface.co/datasets/{repo_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
    else:
        repo_name = "tpu-optimized-llm"

    upload_to_hugging_face(repo_name)
