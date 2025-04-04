#!/usr/bin/env python3
"""
Check available resources for LLM training.
"""

import os
import sys
import psutil
import argparse
import math
import json
from typing import Dict, Any, Optional, Tuple

def get_memory_info() -> Dict[str, float]:
    """
    Get memory information.
    
    Returns:
        Dictionary with memory information in GB
    """
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "total": mem.total / (1024 ** 3),
        "available": mem.available / (1024 ** 3),
        "used": mem.used / (1024 ** 3),
        "free": mem.free / (1024 ** 3),
        "percent": mem.percent,
        "swap_total": swap.total / (1024 ** 3),
        "swap_used": swap.used / (1024 ** 3),
        "swap_free": swap.free / (1024 ** 3),
        "swap_percent": swap.percent
    }

def get_disk_info(path: str = ".") -> Dict[str, float]:
    """
    Get disk information.
    
    Args:
        path: Path to check
        
    Returns:
        Dictionary with disk information in GB
    """
    disk = psutil.disk_usage(path)
    
    return {
        "total": disk.total / (1024 ** 3),
        "used": disk.used / (1024 ** 3),
        "free": disk.free / (1024 ** 3),
        "percent": disk.percent
    }

def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information.
    
    Returns:
        Dictionary with CPU information
    """
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "percent": psutil.cpu_percent(interval=1),
        "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }

def estimate_model_size(model_size: str) -> Dict[str, float]:
    """
    Estimate model size and memory requirements.
    
    Args:
        model_size: Model size (7b, 13b, 70b, 175b, 600b)
        
    Returns:
        Dictionary with model size information in GB
    """
    # Model sizes in billions of parameters
    sizes = {
        "7b": 7,
        "13b": 13,
        "70b": 70,
        "175b": 175,
        "600b": 600
    }
    
    if model_size not in sizes:
        raise ValueError(f"Invalid model size: {model_size}")
    
    # Get number of parameters in billions
    params_b = sizes[model_size]
    
    # Calculate memory requirements
    bytes_per_param = 2  # bfloat16
    model_size_gb = params_b * bytes_per_param
    optimizer_size_gb = model_size_gb * 2  # Adam uses 2x model size for optimizer states
    activation_size_gb = model_size_gb * 0.2  # Rough estimate for activations
    checkpoint_size_gb = model_size_gb * 1.2  # Rough estimate for checkpoint size
    
    return {
        "parameters": params_b,
        "model_size_gb": model_size_gb,
        "optimizer_size_gb": optimizer_size_gb,
        "activation_size_gb": activation_size_gb,
        "total_training_memory_gb": model_size_gb + optimizer_size_gb + activation_size_gb,
        "checkpoint_size_gb": checkpoint_size_gb
    }

def check_resources(model_size: str, output_dir: str, num_checkpoints: int = 5) -> Dict[str, Any]:
    """
    Check if resources are sufficient for training.
    
    Args:
        model_size: Model size (7b, 13b, 70b, 175b, 600b)
        output_dir: Output directory
        num_checkpoints: Number of checkpoints to keep
        
    Returns:
        Dictionary with resource information
    """
    # Get resource information
    memory_info = get_memory_info()
    disk_info = get_disk_info(output_dir)
    cpu_info = get_cpu_info()
    model_info = estimate_model_size(model_size)
    
    # Calculate required disk space
    required_disk_space = model_info["checkpoint_size_gb"] * num_checkpoints
    
    # Check if resources are sufficient
    is_memory_sufficient = memory_info["total"] >= model_info["total_training_memory_gb"]
    is_disk_sufficient = disk_info["free"] >= required_disk_space
    
    return {
        "memory": memory_info,
        "disk": disk_info,
        "cpu": cpu_info,
        "model": model_info,
        "required_disk_space": required_disk_space,
        "is_memory_sufficient": is_memory_sufficient,
        "is_disk_sufficient": is_disk_sufficient,
        "is_sufficient": is_memory_sufficient and is_disk_sufficient
    }

def format_size(size_gb: float) -> str:
    """
    Format size in GB.
    
    Args:
        size_gb: Size in GB
        
    Returns:
        Formatted size string
    """
    if size_gb < 1:
        return f"{size_gb * 1024:.2f} MB"
    elif size_gb < 1024:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_gb / 1024:.2f} TB"

def print_resource_check(resource_check: Dict[str, Any]) -> None:
    """
    Print resource check information.
    
    Args:
        resource_check: Resource check information
    """
    print("=" * 80)
    print("Resource Check for LLM Training")
    print("=" * 80)
    
    # Print memory information
    print("\nMemory Information:")
    print(f"  Total Memory: {format_size(resource_check['memory']['total'])}")
    print(f"  Available Memory: {format_size(resource_check['memory']['available'])}")
    print(f"  Used Memory: {format_size(resource_check['memory']['used'])} ({resource_check['memory']['percent']}%)")
    
    if resource_check['memory']['swap_total'] > 0:
        print(f"  Swap Memory: {format_size(resource_check['memory']['swap_total'])}")
        print(f"  Used Swap: {format_size(resource_check['memory']['swap_used'])} ({resource_check['memory']['swap_percent']}%)")
    
    # Print disk information
    print("\nDisk Information:")
    print(f"  Total Disk Space: {format_size(resource_check['disk']['total'])}")
    print(f"  Free Disk Space: {format_size(resource_check['disk']['free'])}")
    print(f"  Used Disk Space: {format_size(resource_check['disk']['used'])} ({resource_check['disk']['percent']}%)")
    
    # Print CPU information
    print("\nCPU Information:")
    print(f"  Physical Cores: {resource_check['cpu']['physical_cores']}")
    print(f"  Logical Cores: {resource_check['cpu']['logical_cores']}")
    print(f"  CPU Usage: {resource_check['cpu']['percent']}%")
    
    if resource_check['cpu']['frequency']:
        print(f"  CPU Frequency: {resource_check['cpu']['frequency']} MHz")
    
    # Print model information
    print("\nModel Information:")
    print(f"  Model Size: {resource_check['model']['parameters']} billion parameters")
    print(f"  Model Memory: {format_size(resource_check['model']['model_size_gb'])}")
    print(f"  Optimizer Memory: {format_size(resource_check['model']['optimizer_size_gb'])}")
    print(f"  Activation Memory: {format_size(resource_check['model']['activation_size_gb'])}")
    print(f"  Total Training Memory: {format_size(resource_check['model']['total_training_memory_gb'])}")
    print(f"  Checkpoint Size: {format_size(resource_check['model']['checkpoint_size_gb'])}")
    
    # Print required resources
    print("\nRequired Resources:")
    print(f"  Required Disk Space: {format_size(resource_check['required_disk_space'])}")
    
    # Print resource check result
    print("\nResource Check Result:")
    print(f"  Memory: {'SUFFICIENT' if resource_check['is_memory_sufficient'] else 'INSUFFICIENT'}")
    print(f"  Disk: {'SUFFICIENT' if resource_check['is_disk_sufficient'] else 'INSUFFICIENT'}")
    print(f"  Overall: {'SUFFICIENT' if resource_check['is_sufficient'] else 'INSUFFICIENT'}")
    
    print("\nRecommendations:")
    if not resource_check['is_memory_sufficient']:
        print("  - Insufficient memory for training. Consider using a smaller model or adding more memory.")
        print(f"    Need {format_size(resource_check['model']['total_training_memory_gb'])} but only have {format_size(resource_check['memory']['total'])}")
    
    if not resource_check['is_disk_sufficient']:
        print("  - Insufficient disk space for checkpoints. Consider using a smaller model or adding more disk space.")
        print(f"    Need {format_size(resource_check['required_disk_space'])} but only have {format_size(resource_check['disk']['free'])}")
    
    print("=" * 80)

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Check resources for LLM training")
    parser.add_argument("--model_size", type=str, default="600b", choices=["7b", "13b", "70b", "175b", "600b"],
                        help="Model size")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of checkpoints to keep")
    parser.add_argument("--json", action="store_true",
                        help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Check resources
    resource_check = check_resources(args.model_size, args.output_dir, args.num_checkpoints)
    
    # Print resource check information
    if args.json:
        print(json.dumps(resource_check, indent=2))
    else:
        print_resource_check(resource_check)
    
    # Exit with error if resources are insufficient
    if not resource_check["is_sufficient"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
