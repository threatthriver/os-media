"""
TPU-optimized trainer for a 600B parameter LLM with Indian language focus.
This implementation includes specialized optimizations for Google TPU v4-32 hardware.
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
import flax
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
import optax
import wandb
from tqdm import tqdm

logger = logging.getLogger("LLM-Trainer.Trainer")

class TPUTrainer:
    """
    TPU-optimized trainer for large language models.
    Designed specifically for TPU v4-32 hardware with optimizations for 600B parameter models.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir="./output",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=500000,
        warmup_steps=2000,
        logging_steps=100,
        save_steps=1000,
        eval_steps=5000,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_wandb=True,
        wandb_project="indian-llm-600b",
        wandb_run_name=None,
        precision="bfloat16",
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        data_parallel_size=None,  # Auto-calculated if None
        seed=42,
        use_gradient_checkpointing=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or f"indian-llm-600b-{time.strftime('%Y%m%d-%H%M%S')}"
        self.precision = precision
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.seed = seed
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Set up TPU devices
        self.setup_tpu_devices()
        
        # Set up precision
        self.dtype = self.get_precision_dtype()
        
        # Set up optimizer
        self.optimizer = self.create_optimizer()
        
        # Set up training state
        self.state = None  # Will be initialized in setup_training
        
        # Set up metrics
        self.metrics = {
            "train_loss": [],
            "train_perplexity": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "learning_rate": [],
            "grad_norm": [],
            "tokens_per_second": [],
        }
        
        # Set up wandb
        if self.use_wandb:
            self.setup_wandb()
    
    def setup_tpu_devices(self):
        """Set up TPU devices and mesh."""
        # Get all TPU devices
        devices = jax.devices()
        num_devices = len(devices)
        
        logger.info(f"Found {num_devices} TPU devices")
        
        # Calculate data parallel size if not provided
        if self.data_parallel_size is None:
            self.data_parallel_size = num_devices // (self.tensor_parallel_size * self.pipeline_parallel_size)
        
        # Ensure we're using all devices
        assert num_devices == self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size, \
            f"Device count mismatch: {num_devices} != {self.tensor_parallel_size} * {self.pipeline_parallel_size} * {self.data_parallel_size}"
        
        # Create device mesh
        self.device_mesh = mesh_utils.create_device_mesh(
            (self.data_parallel_size, self.tensor_parallel_size, self.pipeline_parallel_size)
        )
        
        logger.info(f"Created device mesh with shape {self.device_mesh.shape}")
        
        # Define partition specs for model parameters and optimizer states
        self.param_spec = P('data', 'tensor', 'pipeline')
        self.data_spec = P('data', None, None)
        
        logger.info("TPU devices and mesh set up successfully")
    
    def get_precision_dtype(self):
        """Get JAX dtype based on precision setting."""
        if self.precision == "float32":
            return jnp.float32
        elif self.precision == "bfloat16":
            return jnp.bfloat16
        elif self.precision == "float16":
            return jnp.float16
        else:
            logger.warning(f"Unknown precision {self.precision}, defaulting to bfloat16")
            return jnp.bfloat16
    
    def create_optimizer(self):
        """Create optimizer with learning rate schedule."""
        # Create learning rate schedule with warmup and cosine decay
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.learning_rate,
            transition_steps=self.warmup_steps
        )
        
        cosine_fn = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.max_steps - self.warmup_steps,
            alpha=0.1  # Final learning rate will be 10% of peak
        )
        
        lr_schedule = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[self.warmup_steps]
        )
        
        # Create optimizer with weight decay
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),  # Gradient clipping
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=self.weight_decay
            )
        )
        
        return optimizer
    
    def setup_wandb(self):
        """Set up Weights & Biases for experiment tracking."""
        try:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "max_steps": self.max_steps,
                    "warmup_steps": self.warmup_steps,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "precision": self.precision,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "pipeline_parallel_size": self.pipeline_parallel_size,
                    "data_parallel_size": self.data_parallel_size,
                    "seed": self.seed,
                    "use_gradient_checkpointing": self.use_gradient_checkpointing,
                    "model_config": {
                        "hidden_size": self.model.hidden_size,
                        "num_hidden_layers": self.model.num_hidden_layers,
                        "num_attention_heads": self.model.num_attention_heads,
                        "intermediate_size": self.model.intermediate_size,
                        "max_position_embeddings": self.model.max_position_embeddings,
                        "vocab_size": self.tokenizer.vocab_size,
                    }
                }
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.use_wandb = False
    
    def setup_training(self):
        """Set up training state and functions."""
        # Create training state
        def create_train_state(rng):
            """Create initial training state."""
            params = self.model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))
            return train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.optimizer
            )
        
        # Initialize parameters with a PRNG key
        rng = jax.random.PRNGKey(self.seed)
        self.state = create_train_state(rng)
        
        # Define loss function
        def loss_fn(logits, labels):
            """Compute cross entropy loss."""
            # Shift logits and labels for next-token prediction
            logits = logits[:, :-1, :]  # Remove last token prediction
            labels = labels[:, 1:]      # Remove first token (usually BOS)
            
            # Create one-hot encoded labels
            one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
            
            # Compute cross entropy loss
            loss = optax.softmax_cross_entropy(logits, one_hot_labels)
            
            # Create mask for padding tokens
            mask = (labels != self.tokenizer.pad_token_id).astype(jnp.float32)
            
            # Apply mask and compute mean
            loss = (loss * mask).sum() / mask.sum()
            
            return loss
        
        # Define training step
        def train_step(state, batch):
            """Perform a single training step."""
            def compute_loss(params):
                logits = state.apply_fn(params, batch["input_ids"])
                loss = loss_fn(logits, batch["labels"])
                return loss, logits
            
            # Compute loss and gradients
            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            (loss, logits), grads = grad_fn(state.params)
            
            # Update state
            new_state = state.apply_gradients(grads=grads)
            
            # Compute metrics
            metrics = {
                "loss": loss,
                "perplexity": jnp.exp(loss),
                "grad_norm": optax.global_norm(grads),
            }
            
            return new_state, metrics
        
        # Define evaluation step
        def eval_step(state, batch):
            """Perform a single evaluation step."""
            logits = state.apply_fn(state.params, batch["input_ids"])
            loss = loss_fn(logits, batch["labels"])
            
            # Compute metrics
            metrics = {
                "loss": loss,
                "perplexity": jnp.exp(loss),
            }
            
            return metrics
        
        # Parallelize training step across devices
        self.p_train_step = pjit(
            train_step,
            in_axis_resources=(self.param_spec, self.data_spec),
            out_axis_resources=(self.param_spec, None),
            donate_argnums=(0,)
        )
        
        # Parallelize evaluation step across devices
        self.p_eval_step = pjit(
            eval_step,
            in_axis_resources=(self.param_spec, self.data_spec),
            out_axis_resources=None
        )
        
        logger.info("Training setup completed successfully")
    
    def train(self):
        """Train the model."""
        logger.info("Starting training")
        
        # Set up training
        self.setup_training()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create data iterator
        train_iter = iter(self.train_dataset)
        
        # Training loop
        start_time = time.time()
        global_step = 0
        
        with tqdm(total=self.max_steps, desc="Training") as pbar:
            while global_step < self.max_steps:
                # Get batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataset)
                    batch = next(train_iter)
                
                # Shard batch across devices
                sharded_batch = jax_utils.replicate(batch)
                
                # Perform training step
                self.state, metrics = self.p_train_step(self.state, sharded_batch)
                
                # Update metrics
                self.metrics["train_loss"].append(metrics["loss"])
                self.metrics["train_perplexity"].append(metrics["perplexity"])
                self.metrics["grad_norm"].append(metrics["grad_norm"])
                
                # Calculate tokens per second
                elapsed = time.time() - start_time
                tokens_per_second = (global_step + 1) * batch["input_ids"].shape[0] * batch["input_ids"].shape[1] / elapsed
                self.metrics["tokens_per_second"].append(tokens_per_second)
                
                # Log metrics
                if (global_step + 1) % self.logging_steps == 0:
                    # Calculate average metrics
                    avg_loss = np.mean(self.metrics["train_loss"][-self.logging_steps:])
                    avg_perplexity = np.mean(self.metrics["train_perplexity"][-self.logging_steps:])
                    avg_grad_norm = np.mean(self.metrics["grad_norm"][-self.logging_steps:])
                    avg_tokens_per_second = np.mean(self.metrics["tokens_per_second"][-self.logging_steps:])
                    
                    # Log to console
                    logger.info(f"Step {global_step + 1}/{self.max_steps} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"Perplexity: {avg_perplexity:.4f} | "
                                f"Grad Norm: {avg_grad_norm:.4f} | "
                                f"Tokens/s: {avg_tokens_per_second:.2f}")
                    
                    # Log to wandb
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/perplexity": avg_perplexity,
                            "train/grad_norm": avg_grad_norm,
                            "train/tokens_per_second": avg_tokens_per_second,
                            "train/learning_rate": self.optimizer.learning_rate(global_step),
                        }, step=global_step + 1)
                
                # Evaluate
                if self.eval_dataset is not None and (global_step + 1) % self.eval_steps == 0:
                    self.evaluate()
                
                # Save checkpoint
                if (global_step + 1) % self.save_steps == 0:
                    self.save_checkpoint(global_step + 1)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "loss": metrics["loss"],
                    "ppl": metrics["perplexity"],
                    "tok/s": tokens_per_second
                })
                
                global_step += 1
        
        # Final evaluation
        if self.eval_dataset is not None:
            self.evaluate()
        
        # Save final checkpoint
        self.save_checkpoint(global_step)
        
        # Log training completion
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        return self.state
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        logger.info("Starting evaluation")
        
        # Create data iterator
        eval_iter = iter(self.eval_dataset)
        
        # Evaluation loop
        eval_losses = []
        eval_perplexities = []
        
        for _ in tqdm(range(100), desc="Evaluating"):  # Evaluate on 100 batches
            # Get batch
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(self.eval_dataset)
                batch = next(eval_iter)
            
            # Shard batch across devices
            sharded_batch = jax_utils.replicate(batch)
            
            # Perform evaluation step
            metrics = self.p_eval_step(self.state, sharded_batch)
            
            # Update metrics
            eval_losses.append(metrics["loss"])
            eval_perplexities.append(metrics["perplexity"])
        
        # Calculate average metrics
        avg_loss = np.mean(eval_losses)
        avg_perplexity = np.mean(eval_perplexities)
        
        # Update metrics
        self.metrics["eval_loss"].append(avg_loss)
        self.metrics["eval_perplexity"].append(avg_perplexity)
        
        # Log to console
        logger.info(f"Evaluation | Loss: {avg_loss:.4f} | Perplexity: {avg_perplexity:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "eval/loss": avg_loss,
                "eval/perplexity": avg_perplexity,
            })
        
        return avg_loss, avg_perplexity
    
    def save_checkpoint(self, step):
        """Save a checkpoint of the model."""
        logger.info(f"Saving checkpoint at step {step}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model parameters
        with open(os.path.join(checkpoint_dir, "params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self.state.params))
        
        # Save optimizer state
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self.state.opt_state))
        
        # Save training state
        with open(os.path.join(checkpoint_dir, "train_state.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(self.state))
        
        # Save configuration
        config = {
            "step": step,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "precision": self.precision,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "seed": self.seed,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "model_config": {
                "hidden_size": self.model.hidden_size,
                "num_hidden_layers": self.model.num_hidden_layers,
                "num_attention_heads": self.model.num_attention_heads,
                "intermediate_size": self.model.intermediate_size,
                "max_position_embeddings": self.model.max_position_embeddings,
                "vocab_size": self.tokenizer.vocab_size,
            },
            "metrics": {
                "train_loss": float(np.mean(self.metrics["train_loss"][-self.logging_steps:])),
                "train_perplexity": float(np.mean(self.metrics["train_perplexity"][-self.logging_steps:])),
                "eval_loss": float(np.mean(self.metrics["eval_loss"])) if self.metrics["eval_loss"] else None,
                "eval_perplexity": float(np.mean(self.metrics["eval_perplexity"])) if self.metrics["eval_perplexity"] else None,
            },
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        return checkpoint_dir
    
    def load_checkpoint(self, checkpoint_dir):
        """Load a checkpoint of the model."""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model parameters
        with open(os.path.join(checkpoint_dir, "params.msgpack"), "rb") as f:
            params = flax.serialization.from_bytes(self.state.params, f.read())
        
        # Load optimizer state
        with open(os.path.join(checkpoint_dir, "opt_state.msgpack"), "rb") as f:
            opt_state = flax.serialization.from_bytes(self.state.opt_state, f.read())
        
        # Update state
        self.state = self.state.replace(
            params=params,
            opt_state=opt_state
        )
        
        logger.info(f"Checkpoint loaded successfully")
        
        return self.state
