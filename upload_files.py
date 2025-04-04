#!/usr/bin/env python3
"""
Script to upload individual files to Hugging Face.
"""

import os
import glob
from huggingface_hub import HfApi

def upload_files(token, repo_name="Threatthriver/tpu-optimized-llm"):
    """
    Upload individual files to Hugging Face.
    
    Args:
        token: Hugging Face token
        repo_name: Repository name
    """
    # Create API client
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_name, repo_type="dataset", exist_ok=True)
        print(f"Repository created or already exists: {repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
    
    # Upload README.md first
    try:
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset"
        )
        print("Uploaded README.md")
    except Exception as e:
        print(f"Error uploading README.md: {e}")
    
    # Upload requirements.txt
    try:
        api.upload_file(
            path_or_fileobj="requirements.txt",
            path_in_repo="requirements.txt",
            repo_id=repo_name,
            repo_type="dataset"
        )
        print("Uploaded requirements.txt")
    except Exception as e:
        print(f"Error uploading requirements.txt: {e}")
    
    # Upload Python files
    for py_file in glob.glob("**/*.py", recursive=True):
        if py_file not in ["upload_to_hf.py", "upload_to_hf_cli.py", "upload_files.py"]:
            try:
                api.upload_file(
                    path_or_fileobj=py_file,
                    path_in_repo=py_file,
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print(f"Uploaded {py_file}")
            except Exception as e:
                print(f"Error uploading {py_file}: {e}")
    
    # Upload shell scripts
    for sh_file in glob.glob("**/*.sh", recursive=True):
        if sh_file != "upload_with_git.sh":
            try:
                api.upload_file(
                    path_or_fileobj=sh_file,
                    path_in_repo=sh_file,
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print(f"Uploaded {sh_file}")
            except Exception as e:
                print(f"Error uploading {sh_file}: {e}")
    
    # Upload YAML files
    for yml_file in glob.glob("**/*.yml", recursive=True):
        try:
            api.upload_file(
                path_or_fileobj=yml_file,
                path_in_repo=yml_file,
                repo_id=repo_name,
                repo_type="dataset"
            )
            print(f"Uploaded {yml_file}")
        except Exception as e:
            print(f"Error uploading {yml_file}: {e}")
    
    print(f"Files uploaded to: https://huggingface.co/datasets/{repo_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python upload_files.py <huggingface_token>")
        sys.exit(1)
    
    token = sys.argv[1]
    upload_files(token)
