"""
Mini-BERT Training Loop with MLM, Gradient Accumulation, and Diagnostics.

Features:
- Masked Language Model (MLM) training
- Gradient accumulation for large effective batch sizes
- Comprehensive gradient checking and validation
- Memory monitoring and optimization
- Learning rate scheduling
- Checkpointing and resuming
"""

import numpy as np
import time
import os
from typing import Dict, Optional, Tuple
import json

from model import MiniBERT
from gradients import MiniBERTGradients
from data import prepare_data_loaders, TextDataLoader
from utils import (GradientChecker, TrainingDiagnostics, get_memory_usage, 
                   save_checkpoint, load_checkpoint, estimate_training_time)
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, SYSTEM_CONFIG

class AdamOptimizer:
    """
    Adam optimizer implementation in pure NumPy.
    
    Mathematical formulation:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)  
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    """
    
    def __init__(self, learning_rate: float = 1e-4, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8, 
                 weight_decay: float = 0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State variables
        self.step_count = 0
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def initialize_state(self, params: Dict[str, np.ndarray]):
        """Initialize optimizer state for all parameters."""
        for param_name, param in params.items():
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
    
    def step(self, params: Dict[str, np.ndarray], 
             gradients: Dict[str, np.ndarray]):
        """
        Perform one optimization step.
        
        Args:
            params: Model parameters to update
            gradients: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name not in gradients:
                continue
                
            param = params[param_name]
            grad = gradients[param_name]
            
            # Skip if gradient is all zeros
            if not np.any(grad):
                continue
            
            # Initialize state if needed
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)
            
            # Add weight decay to gradient
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update biased first and second moment estimates
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.step_count)
            
            # Update parameters
            params[param_name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def get_state(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'step_count': self.step_count,
            'learning_rate': self.learning_rate,
            'm': self.m,
            'v': self.v
        }
    
    def load_state(self, state: Dict):
        """Load optimizer state from checkpoint."""
        self.step_count = state['step_count']
        self.learning_rate = state['learning_rate']
        self.m = state['m']
        self.v = state['v']

class LearningRateScheduler:
    """
    Learning rate scheduler with linear warmup and decay.
    
    Schedule:
    - Linear warmup from 0 to peak_lr over warmup_steps
    - Linear decay from peak_lr to 0 over remaining steps
    """
    
    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / self.warmup_steps
        else:
            # Linear decay
            remaining_steps = self.total_steps - self.warmup_steps
            decay_steps = step - self.warmup_steps
            decay_ratio = max(0.0, 1.0 - decay_steps / remaining_steps)
            return self.peak_lr * decay_ratio

def clip_gradients(gradients: Dict[str, np.ndarray], max_norm: float) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: Dictionary of parameter gradients
        max_norm: Maximum allowed gradient norm
        
    Returns:
        actual_norm: Global gradient norm before clipping
    """
    # Compute global gradient norm
    total_norm = 0.0
    for grad in gradients.values():
        if np.any(grad):  # Skip zero gradients
            total_norm += np.sum(grad ** 2)
    
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_ratio = max_norm / total_norm
        for param_name in gradients:
            gradients[param_name] *= clip_ratio
    
    return total_norm

def train_mini_bert(
    resume_from_checkpoint: Optional[str] = None,
    run_gradient_check: bool = True,
    save_freq: int = 10000,
    log_freq: int = 100
) -> MiniBERT:
    """
    Complete Mini-BERT training pipeline.
    
    Args:
        resume_from_checkpoint: Path to checkpoint to resume from
        run_gradient_check: Whether to run gradient checking
        save_freq: Frequency of checkpoint saving
        log_freq: Frequency of logging
        
    Returns:
        Trained model
    """
    print("=" * 60)
    print("Mini-BERT Training Pipeline")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(SYSTEM_CONFIG.random_seed)
    
    # Initialize model and optimizer
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    optimizer = AdamOptimizer(
        learning_rate=TRAINING_CONFIG.learning_rate,
        beta1=TRAINING_CONFIG.beta1,
        beta2=TRAINING_CONFIG.beta2,
        eps=TRAINING_CONFIG.eps,
        weight_decay=TRAINING_CONFIG.weight_decay
    )
    
    # Initialize optimizer state
    optimizer.initialize_state(model.params)
    
    # Set up learning rate scheduler
    scheduler = LearningRateScheduler(
        peak_lr=TRAINING_CONFIG.learning_rate,
        warmup_steps=TRAINING_CONFIG.warmup_steps,
        total_steps=TRAINING_CONFIG.num_train_steps
    )
    
    # Initialize diagnostics
    os.makedirs(SYSTEM_CONFIG.log_dir, exist_ok=True)
    diagnostics = TrainingDiagnostics(
        log_file=os.path.join(SYSTEM_CONFIG.log_dir, "training.log")
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = load_checkpoint(resume_from_checkpoint)
        
        # Restore model parameters
        model.params = checkpoint['model_params']
        
        # Restore optimizer state
        optimizer.load_state(checkpoint['optimizer_state'])
        
        start_step = checkpoint['step']
        print(f"Resumed from step {start_step}")
    
    # Prepare data loaders
    print("Preparing data loaders...")
    data_loader, tokenizer = prepare_data_loaders(DATA_CONFIG, TRAINING_CONFIG)
    
    # Run gradient checking if requested
    if run_gradient_check and start_step == 0:
        print("\nRunning gradient checks...")
        checker = GradientChecker(epsilon=1e-5, tolerance=1e-3)
        
        # Create small test batch
        test_batch_size = 2
        test_seq_len = 16
        test_input_ids = np.random.randint(0, model.V, (test_batch_size, test_seq_len))
        test_labels = np.random.randint(0, model.V, (test_batch_size, test_seq_len))
        test_mask = np.ones((test_batch_size, test_seq_len))
        
        results = checker.check_gradients(
            model, grad_computer, test_input_ids, test_labels, test_mask,
            param_subset=['W_Q_0', 'W1_0', 'ln1_gamma_0', 'mlm_head_W']
        )
        
        # Check if gradients passed
        all_passed = all(result['passed'] for result in results.values())
        if all_passed:
            print("✓ All gradient checks passed!")
        else:
            print("✗ Some gradient checks failed - check implementation")
            for param, result in results.items():
                if not result['passed']:
                    print(f"  {param}: max_rel_diff = {result['max_rel_diff']:.2e}")
    
    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print(f"Target: {TRAINING_CONFIG.num_train_steps:,} steps")
    print(f"Batch size: {TRAINING_CONFIG.train_batch_size} "
          f"(accumulate {TRAINING_CONFIG.gradient_accumulation_steps})")
    
    training_start_time = time.time()
    step = start_step
    
    # Initialize gradient accumulation
    accumulated_gradients = {}
    accumulation_count = 0
    running_loss = 0.0
    
    try:
        # Create batch iterator
        batch_iterator = data_loader.create_batches(TRAINING_CONFIG.train_micro_batch_size)
        
        for batch in batch_iterator:
            if step >= TRAINING_CONFIG.num_train_steps:
                break
            
            # Extract batch data
            input_ids = batch['input_ids']  # [micro_batch_size, seq_len]
            labels = batch['labels']        # [micro_batch_size, seq_len]
            mlm_mask = batch['mlm_mask']    # [micro_batch_size, seq_len]
            
            # Forward pass
            logits, cache = model.forward(input_ids)
            
            # Compute loss and gradients
            loss, grad_logits = grad_computer.compute_mlm_loss_and_gradients(
                logits, labels, mlm_mask
            )
            
            # Backward pass
            grad_computer.zero_gradients()
            grad_computer.backward_from_logits(grad_logits, cache)
            
            # Accumulate gradients
            if not accumulated_gradients:
                # Initialize accumulated gradients
                for param_name, grad in grad_computer.gradients.items():
                    accumulated_gradients[param_name] = grad.copy()
            else:
                # Add to accumulated gradients
                for param_name, grad in grad_computer.gradients.items():
                    accumulated_gradients[param_name] += grad
            
            accumulation_count += 1
            running_loss += loss
            
            # Update parameters when accumulation is complete
            if accumulation_count >= TRAINING_CONFIG.gradient_accumulation_steps:
                # Average accumulated gradients
                for param_name in accumulated_gradients:
                    accumulated_gradients[param_name] /= TRAINING_CONFIG.gradient_accumulation_steps
                
                # Clip gradients
                grad_norm = clip_gradients(accumulated_gradients, TRAINING_CONFIG.max_grad_norm)
                
                # Update learning rate
                current_lr = scheduler.get_lr(step)
                optimizer.learning_rate = current_lr
                
                # Optimizer step
                optimizer.step(model.params, accumulated_gradients)
                
                # Compute average loss
                avg_loss = running_loss / TRAINING_CONFIG.gradient_accumulation_steps
                
                # Log metrics
                metrics = {
                    'loss': avg_loss,
                    'learning_rate': current_lr,
                    'grad_norm': grad_norm,
                    **diagnostics.compute_gradient_stats(accumulated_gradients)
                }
                
                diagnostics.log_step(step, metrics)
                diagnostics.print_metrics(step, metrics, print_freq=log_freq)
                
                # Memory monitoring
                if step % SYSTEM_CONFIG.memory_check_freq == 0:
                    memory = get_memory_usage()
                    if memory['rss_mb'] > SYSTEM_CONFIG.max_memory_gb * 1024:
                        print(f"Warning: Memory usage ({memory['rss_mb']:.0f}MB) "
                              f"exceeds limit ({SYSTEM_CONFIG.max_memory_gb*1024:.0f}MB)")
                
                # Save checkpoint
                if step % save_freq == 0 and step > 0:
                    checkpoint_path = os.path.join(
                        SYSTEM_CONFIG.checkpoint_dir, 
                        f"checkpoint_step_{step}.pkl"
                    )
                    save_checkpoint(
                        model, optimizer.get_state(), step, avg_loss, checkpoint_path
                    )
                
                # Estimate remaining time
                if step % (log_freq * 10) == 0 and step > start_step:
                    time_est = estimate_training_time(
                        step - start_step, TRAINING_CONFIG.num_train_steps - start_step,
                        training_start_time
                    )
                    print(f"Progress: {step}/{TRAINING_CONFIG.num_train_steps} "
                          f"({100*step/TRAINING_CONFIG.num_train_steps:.1f}%) | "
                          f"ETA: {time_est['eta']} | Speed: {time_est['speed']}")
                
                # Reset accumulation
                accumulated_gradients = {}
                accumulation_count = 0
                running_loss = 0.0
                step += 1
                
                # Additional gradient check during training
                if (run_gradient_check and 
                    step % TRAINING_CONFIG.gradient_check_freq == 0 and 
                    step > start_step):
                    print(f"\nRunning gradient check at step {step}...")
                    checker = GradientChecker(epsilon=1e-5, tolerance=1e-3)
                    check_results = checker.check_gradients(
                        model, grad_computer, input_ids[:2], labels[:2], mlm_mask[:2],
                        param_subset=['W_Q_0', 'ln1_gamma_0']
                    )
                    
                    all_passed = all(r['passed'] for r in check_results.values())
                    status = "✓ PASS" if all_passed else "✗ FAIL"
                    print(f"Gradient check: {status}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint
        final_checkpoint_path = os.path.join(
            SYSTEM_CONFIG.checkpoint_dir, 
            "final_checkpoint.pkl"
        )
        save_checkpoint(
            model, optimizer.get_state(), step, 
            running_loss / max(1, accumulation_count), final_checkpoint_path
        )
        
        # Print training summary
        total_time = time.time() - training_start_time
        print(f"\nTraining Summary:")
        print(f"  Steps completed: {step:,}")
        print(f"  Total time: {total_time/3600:.1f} hours")
        print(f"  Average speed: {step/total_time:.1f} steps/second")
        
        final_memory = get_memory_usage()
        print(f"  Peak memory: {final_memory['rss_mb']:.0f} MB")
        
        # Save training plots if possible
        try:
            plot_path = os.path.join(SYSTEM_CONFIG.log_dir, "loss_curve.png")
            diagnostics.plot_loss_curve(save_path=plot_path)
        except Exception as e:
            print(f"Could not save plots: {e}")
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists(DATA_CONFIG.train_data_path):
        print(f"Training data not found: {DATA_CONFIG.train_data_path}")
        print("Please ensure training data is available before running.")
        exit(1)
    
    # Create necessary directories
    os.makedirs(SYSTEM_CONFIG.log_dir, exist_ok=True)
    os.makedirs(SYSTEM_CONFIG.checkpoint_dir, exist_ok=True)
    
    # Print configuration summary
    print("Mini-BERT Training Configuration:")
    print(f"  Model: L={MODEL_CONFIG.num_layers}, H={MODEL_CONFIG.hidden_size}, "
          f"A={MODEL_CONFIG.num_attention_heads}")
    print(f"  Data: {DATA_CONFIG.train_data_path}")
    print(f"  Steps: {TRAINING_CONFIG.num_train_steps:,}")
    print(f"  Batch size: {TRAINING_CONFIG.train_batch_size} "
          f"(micro: {TRAINING_CONFIG.train_micro_batch_size})")
    print(f"  Learning rate: {TRAINING_CONFIG.learning_rate:.2e}")
    print(f"  Memory limit: {SYSTEM_CONFIG.max_memory_gb:.1f} GB")
    
    # Ask for confirmation
    try:
        response = input("\nProceed with training? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            exit(0)
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        exit(0)
    
    # Start training
    trained_model = train_mini_bert(
        resume_from_checkpoint=None,  # Set to checkpoint path to resume
        run_gradient_check=True,
        save_freq=TRAINING_CONFIG.save_steps,
        log_freq=TRAINING_CONFIG.logging_steps
    )