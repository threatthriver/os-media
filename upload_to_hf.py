#!/usr/bin/env python3
"""
Script to upload the repository to Hugging Face.
"""

import os
import sys
import subprocess
from huggingface_hub import HfApi, create_repo

def upload_to_hugging_face(token, repo_name="tpu-optimized-llm"):
    """
    Upload the repository to Hugging Face.
    
    Args:
        token: Hugging Face token
        repo_name: Repository name
    """
    # Create API client
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, token=token, private=False)
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository already exists or error: {e}")
    
    # Initialize git if not already initialized
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
    
    # Configure git
    subprocess.run(["git", "config", "user.email", "user@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "HF Uploader"], check=True)
    
    # Add files
    subprocess.run(["git", "add", "."], check=True)
    
    # Commit
    subprocess.run(["git", "commit", "-m", "Initial commit of TPU-optimized LLM training code"], check=True)
    
    # Add Hugging Face as remote
    remote_url = f"https://huggingface.co/datasets/{repo_name}"
    try:
        subprocess.run(["git", "remote", "add", "hf", remote_url], check=True)
    except:
        # Remote might already exist
        pass
    
    # Push to Hugging Face
    push_command = f"git push -f https://{token}@huggingface.co/datasets/{repo_name} main"
    os.system(push_command)  # Using os.system to avoid showing the token in process list
    
    print(f"Repository uploaded to: {remote_url}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_to_hf.py <huggingface_token>")
        sys.exit(1)
    
    token = sys.argv[1]
    upload_to_hugging_face(token)
