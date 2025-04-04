#!/usr/bin/env python3
"""
Script to upload the model to Hugging Face.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")

    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face token")
    parser.add_argument("--repo_name", type=str, default="threatthriver/tpu-optimized-llm",
                        help="Repository name")
    parser.add_argument("--repo_type", type=str, default="model",
                        choices=["model", "dataset", "space"],
                        help="Repository type")
    parser.add_argument("--model_path", type=str, default=".",
                        help="Path to model files")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Whether to create a private repository")

    return parser.parse_args()

def upload_to_hugging_face(token, repo_name, repo_type, model_path, private):
    """
    Upload the model to Hugging Face using the Hub API.

    Args:
        token: Hugging Face token
        repo_name: Repository name
        repo_type: Repository type
        model_path: Path to model files
        private: Whether to create a private repository
    """
    # Create API client
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, repo_type=repo_type, token=token, private=private, exist_ok=True)
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository already exists or error: {e}")

    # Upload the folder
    print(f"Uploading files from {model_path} to {repo_name}...")
    upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type=repo_type,
        token=token,
        ignore_patterns=[".git/*", "__pycache__/*", "*.pyc", ".DS_Store", "upload_to_hf.py", "upload_to_hf_cli.py", "upload_files.py", "upload_with_git.sh"]
    )

    print(f"Model uploaded to: https://huggingface.co/{repo_type}s/{repo_name}")

def main():
    """Main function."""
    args = parse_args()

    # Upload to Hugging Face
    upload_to_hugging_face(
        token=args.token,
        repo_name=args.repo_name,
        repo_type=args.repo_type,
        model_path=args.model_path,
        private=args.private
    )

if __name__ == "__main__":
    main()
