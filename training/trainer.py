"""
Trainer for LLM model.
"""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import math
import os
from dataclasses import dataclass
from functools import partial

from model.llm import LLM, LLMConfig
from training.optimizer import create_adamw_optimizer
from training.scheduler import create_linear_warmup_cosine_decay_schedule


@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    Attributes:
        model_config: Model configuration
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of training steps
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        adam_beta1: Beta1 for Adam optimizer
        adam_beta2: Beta2 for Adam optimizer
        adam_epsilon: Epsilon for Adam optimizer
        logging_steps: Number of steps between logging
        save_steps: Number of steps between checkpoints
        eval_steps: Number of steps between evaluations
        output_dir: Directory for saving checkpoints and logs
        seed: Random seed
        dtype: Data type for computations
    """
    model_config: LLMConfig
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: str = "output"
    seed: int = 42
    dtype: jnp.dtype = jnp.bfloat16


class TrainingState(train_state.TrainState):
    """
    Training state for LLM model.
    
    Attributes:
        step: Current training step
        apply_fn: Model apply function
        params: Model parameters
        tx: Optimizer
        opt_state: Optimizer state
        dropout_rng: RNG for dropout
        loss_scale: Loss scale for mixed precision training
        grad_accum: Accumulated gradients
        accum_steps: Number of steps accumulated
    """
    dropout_rng: jnp.ndarray
    loss_scale: float
    grad_accum: Optional[Dict[str, jnp.ndarray]] = None
    accum_steps: int = 0


def create_train_state(
    config: TrainingConfig,
    model: LLM,
    rng: jnp.ndarray
) -> TrainingState:
    """
    Create training state.
    
    Args:
        config: Training configuration
        model: LLM model
        rng: Random number generator
        
    Returns:
        Training state
    """
    # Create learning rate schedule
    lr_schedule = create_linear_warmup_cosine_decay_schedule(
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps - config.warmup_steps,
        final_learning_rate_factor=0.1
    )
    
    # Create optimizer
    optimizer = create_adamw_optimizer(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon
    )
    
    # Initialize model parameters
    params_rng, dropout_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    params = model.init(params_rng, dummy_input)
    
    # Create training state
    return TrainingState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng,
        loss_scale=1.0
    )


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        mask: Mask for padding [batch_size, seq_len]
        
    Returns:
        Loss value
    """
    vocab_size = logits.shape[-1]
    
    # Convert labels to one-hot
    labels_onehot = jax.nn.one_hot(labels, vocab_size)
    
    # Compute per-token loss
    loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
        return jnp.sum(loss) / jnp.sum(mask)
    
    return jnp.mean(loss)


def train_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    config: TrainingConfig
) -> Tuple[TrainingState, Dict[str, float]]:
    """
    Perform a single training step.
    
    Args:
        state: Training state
        batch: Batch of data
        config: Training configuration
        
    Returns:
        Updated training state and metrics
    """
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    
    # Define loss function
    def loss_fn(params):
        outputs = state.apply_fn(
            params,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )
        
        logits = outputs["logits"]
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]
        
        # Create mask for padding
        mask = None
        if "attention_mask" in batch:
            mask = batch["attention_mask"][:, 1:]
        
        # Compute loss
        loss = cross_entropy_loss(shift_logits, shift_labels, mask)
        
        # Scale loss for mixed precision training
        return loss * state.loss_scale, (loss, logits)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (scaled_loss, (loss, logits)), grads = grad_fn(state.params)
    
    # Unscale gradients
    grads = jax.tree_map(lambda g: g / state.loss_scale, grads)
    
    # Accumulate gradients
    if state.grad_accum is None:
        grad_accum = grads
    else:
        grad_accum = jax.tree_map(
            lambda g1, g2: g1 + g2, state.grad_accum, grads
        )
    
    accum_steps = state.accum_steps + 1
    
    # Update parameters if we have accumulated enough gradients
    if accum_steps == config.gradient_accumulation_steps:
        # Compute average gradients
        grad_accum = jax.tree_map(
            lambda g: g / config.gradient_accumulation_steps, grad_accum
        )
        
        # Clip gradients
        grad_norm = optax.global_norm(grad_accum)
        grad_accum = jax.tree_map(
            lambda g: g * jnp.minimum(1.0, config.max_grad_norm / (grad_norm + 1e-8)),
            grad_accum
        )
        
        # Update parameters
        new_state = state.apply_gradients(
            grads=grad_accum,
            dropout_rng=new_dropout_rng
        )
        
        # Reset gradient accumulation
        new_state = new_state.replace(grad_accum=None, accum_steps=0)
    else:
        # Continue accumulating gradients
        new_state = state.replace(
            dropout_rng=new_dropout_rng,
            grad_accum=grad_accum,
            accum_steps=accum_steps
        )
    
    # Compute metrics
    metrics = {
        "loss": loss,
        "learning_rate": state.tx.learning_rate(state.step),
        "perplexity": jnp.exp(loss),
    }
    
    if accum_steps == config.gradient_accumulation_steps:
        metrics["grad_norm"] = grad_norm
    
    return new_state, metrics


def eval_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, float]:
    """
    Perform a single evaluation step.
    
    Args:
        state: Training state
        batch: Batch of data
        
    Returns:
        Evaluation metrics
    """
    # Forward pass
    outputs = state.apply_fn(
        state.params,
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        deterministic=True
    )
    
    logits = outputs["logits"]
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    
    # Create mask for padding
    mask = None
    if "attention_mask" in batch:
        mask = batch["attention_mask"][:, 1:]
    
    # Compute loss
    loss = cross_entropy_loss(shift_logits, shift_labels, mask)
    
    # Compute metrics
    metrics = {
        "eval_loss": loss,
        "eval_perplexity": jnp.exp(loss),
    }
    
    return metrics


class Trainer:
    """
    Trainer for LLM model.
    
    Attributes:
        config: Training configuration
        model: LLM model
        state: Training state
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
    """
    def __init__(
        self,
        config: TrainingConfig,
        model: LLM,
        train_dataloader: Callable,
        eval_dataloader: Optional[Callable] = None,
        state: Optional[TrainingState] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model: LLM model
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            state: Training state (if resuming training)
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Create training state if not provided
        if state is None:
            rng = jax.random.PRNGKey(config.seed)
            self.state = create_train_state(config, model, rng)
        else:
            self.state = state
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Parallelize training step
        self.p_train_step = jax.pmap(
            partial(train_step, config=config),
            axis_name="batch"
        )
        
        # Parallelize evaluation step
        self.p_eval_step = jax.pmap(eval_step, axis_name="batch")
    
    def train(self):
        """
        Train the model.
        """
        # Initialize metrics
        train_metrics = []
        
        # Start training
        print(f"Starting training for {self.config.max_steps} steps...")
        start_time = time.time()
        
        # Training loop
        for step, batch in enumerate(self.train_dataloader(), start=1):
            # Perform training step
            self.state, metrics = self.p_train_step(self.state, batch)
            
            # Collect metrics
            train_metrics.append(metrics)
            
            # Log metrics
            if step % self.config.logging_steps == 0:
                # Compute average metrics
                avg_metrics = {
                    k: jnp.mean([m[k] for m in train_metrics])
                    for k in train_metrics[0]
                }
                
                # Log metrics
                print(f"Step {step}/{self.config.max_steps}:")
                for k, v in avg_metrics.items():
                    print(f"  {k}: {v:.4f}")
                
                # Reset metrics
                train_metrics = []
                
                # Log elapsed time
                elapsed = time.time() - start_time
                print(f"  Time elapsed: {elapsed:.2f}s")
                print(f"  Steps per second: {step / elapsed:.2f}")
            
            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.save_checkpoint(step)
            
            # Evaluate
            if self.eval_dataloader is not None and step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                print(f"Evaluation at step {step}:")
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Check if we've reached the maximum number of steps
            if step >= self.config.max_steps:
                break
        
        # Save final checkpoint
        self.save_checkpoint(self.config.max_steps)
        
        # Final evaluation
        if self.eval_dataloader is not None:
            eval_metrics = self.evaluate()
            print(f"Final evaluation:")
            for k, v in eval_metrics.items():
                print(f"  {k}: {v:.4f}")
        
        # Log total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Average steps per second: {self.config.max_steps / total_time:.2f}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            raise ValueError("Evaluation data loader is not provided")
        
        # Initialize metrics
        eval_metrics = []
        
        # Evaluation loop
        for batch in self.eval_dataloader():
            # Perform evaluation step
            metrics = self.p_eval_step(self.state, batch)
            
            # Collect metrics
            eval_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            k: jnp.mean([m[k] for m in eval_metrics])
            for k in eval_metrics[0]
        }
        
        return avg_metrics
    
    def save_checkpoint(self, step: int):
        """
        Save checkpoint.
        
        Args:
            step: Current training step
        """
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model parameters
        with open(os.path.join(checkpoint_dir, "params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self.state.params))
        
        # Save optimizer state
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self.state.opt_state))
        
        # Save training state
        with open(os.path.join(checkpoint_dir, "training_state.msgpack"), "wb") as f:
            state_dict = {
                "step": self.state.step,
                "loss_scale": self.state.loss_scale,
                "accum_steps": self.state.accum_steps
            }
            f.write(flax.serialization.to_bytes(state_dict))
        
        print(f"Checkpoint saved at {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> TrainingState:
        """
        Load checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory
            
        Returns:
            Training state
        """
        # Load model parameters
        with open(os.path.join(checkpoint_dir, "params.msgpack"), "rb") as f:
            params = flax.serialization.from_bytes(self.state.params, f.read())
        
        # Load optimizer state
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "rb") as f:
            opt_state = flax.serialization.from_bytes(self.state.opt_state, f.read())
        
        # Load training state
        with open(os.path.join(checkpoint_dir, "training_state.msgpack"), "rb") as f:
            state_dict = flax.serialization.from_bytes({}, f.read())
        
        # Create new training state
        new_state = self.state.replace(
            params=params,
            opt_state=opt_state,
            step=state_dict["step"],
            loss_scale=state_dict["loss_scale"],
            accum_steps=state_dict["accum_steps"]
        )
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
        
        return new_state
