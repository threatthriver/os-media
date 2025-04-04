"""
Checkpointing utilities for LLM training.
"""

import os
import time
import json
import flax
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
import msgpack
import orbax.checkpoint as ocp
from flax.training import train_state


def save_checkpoint(
    state: Any,
    path: str,
    keep: int = 5,
    overwrite: bool = False,
    save_optimizer_state: bool = True
) -> None:
    """
    Save checkpoint.
    
    Args:
        state: Training state
        path: Path to save checkpoint
        keep: Number of checkpoints to keep
        overwrite: Whether to overwrite existing checkpoint
        save_optimizer_state: Whether to save optimizer state
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save step
    step = int(state.step)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(path, f"checkpoint-{step}")
    
    if os.path.exists(checkpoint_dir) and not overwrite:
        print(f"Checkpoint {checkpoint_dir} already exists, skipping...")
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model parameters
    with open(os.path.join(checkpoint_dir, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(state.params))
    
    # Save optimizer state if requested
    if save_optimizer_state:
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(state.opt_state))
    
    # Save training state
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump({
            "step": int(state.step),
            "timestamp": time.time()
        }, f)
    
    # Remove old checkpoints if needed
    if keep > 0:
        checkpoints = [
            d for d in os.listdir(path)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
        ]
        
        # Sort checkpoints by step
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[1]),
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep:]:
            checkpoint_dir = os.path.join(path, checkpoint)
            print(f"Removing old checkpoint {checkpoint_dir}")
            
            # Remove files
            for file in os.listdir(checkpoint_dir):
                os.remove(os.path.join(checkpoint_dir, file))
            
            # Remove directory
            os.rmdir(checkpoint_dir)
    
    print(f"Checkpoint saved at {checkpoint_dir}")


def load_checkpoint(
    path: str,
    state: Optional[Any] = None,
    step: Optional[int] = None
) -> Tuple[Any, int]:
    """
    Load checkpoint.
    
    Args:
        path: Path to load checkpoint from
        state: Training state to load parameters into
        step: Step to load (defaults to latest)
        
    Returns:
        Tuple of (state, step)
    """
    # Find checkpoint directory
    if step is None:
        # Find latest checkpoint
        checkpoints = [
            d for d in os.listdir(path)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
        ]
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {path}")
        
        # Sort checkpoints by step
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[1]),
            reverse=True
        )
        
        # Get latest checkpoint
        checkpoint_dir = os.path.join(path, checkpoints[0])
        step = int(checkpoints[0].split("-")[1])
    else:
        # Get specific checkpoint
        checkpoint_dir = os.path.join(path, f"checkpoint-{step}")
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint {checkpoint_dir} not found")
    
    # Load model parameters
    with open(os.path.join(checkpoint_dir, "params.msgpack"), "rb") as f:
        params = flax.serialization.from_bytes(
            {} if state is None else state.params,
            f.read()
        )
    
    # Load optimizer state if available
    opt_state = None
    if os.path.exists(os.path.join(checkpoint_dir, "opt_state.msgpack")):
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "rb") as f:
            opt_state = flax.serialization.from_bytes(
                {} if state is None else state.opt_state,
                f.read()
            )
    
    # Load training state
    with open(os.path.join(checkpoint_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    
    # Update state
    if state is not None:
        state = state.replace(
            params=params,
            step=training_state["step"]
        )
        
        if opt_state is not None:
            state = state.replace(opt_state=opt_state)
    else:
        state = {
            "params": params,
            "opt_state": opt_state,
            "step": training_state["step"]
        }
    
    print(f"Checkpoint loaded from {checkpoint_dir}")
    
    return state, step


def save_checkpoint_orbax(
    state: Any,
    path: str,
    keep: int = 5,
    overwrite: bool = False
) -> None:
    """
    Save checkpoint using Orbax.
    
    Args:
        state: Training state
        path: Path to save checkpoint
        keep: Number of checkpoints to keep
        overwrite: Whether to overwrite existing checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save step
    step = int(state.step)
    
    # Create checkpoint manager
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=keep,
        create=True
    )
    manager = ocp.CheckpointManager(
        path,
        checkpointer,
        options
    )
    
    # Save checkpoint
    save_args = ocp.SaveArgs(
        overwrite=overwrite
    )
    manager.save(
        step,
        state,
        save_kwargs={"save_args": save_args}
    )
    
    print(f"Checkpoint saved at {path}/checkpoint-{step}")


def load_checkpoint_orbax(
    path: str,
    state: Optional[Any] = None,
    step: Optional[int] = None
) -> Tuple[Any, int]:
    """
    Load checkpoint using Orbax.
    
    Args:
        path: Path to load checkpoint from
        state: Training state to load parameters into
        step: Step to load (defaults to latest)
        
    Returns:
        Tuple of (state, step)
    """
    # Create checkpoint manager
    checkpointer = ocp.PyTreeCheckpointer()
    manager = ocp.CheckpointManager(
        path,
        checkpointer
    )
    
    # Find step to load
    if step is None:
        step = manager.latest_step()
        
        if step is None:
            raise ValueError(f"No checkpoints found in {path}")
    
    # Load checkpoint
    state = manager.restore(
        step,
        items=state
    )
    
    print(f"Checkpoint loaded from {path}/checkpoint-{step}")
    
    return state, step
