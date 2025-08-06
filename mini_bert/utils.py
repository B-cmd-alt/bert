"""
Utility functions for Mini-BERT training and debugging.

Includes:
- Gradient checking via finite differences
- Memory profiling and monitoring  
- Training diagnostics and visualization
- Shape validation utilities
"""

import numpy as np
import time
import pickle
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024**2),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024**2),  # Virtual Memory Size
        'percent': process.memory_percent(),     # Percentage of system RAM
        'available_mb': psutil.virtual_memory().available / (1024**2)
    }

def profile_memory(func):
    """
    Decorator to profile memory usage of a function.
    
    Usage:
        @profile_memory
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        mem_before = get_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        mem_delta = mem_after['rss_mb'] - mem_before['rss_mb']
        elapsed = end_time - start_time
        
        print(f"Function {func.__name__}:")
        print(f"  Memory: {mem_before['rss_mb']:.1f} -> {mem_after['rss_mb']:.1f} MB (Δ{mem_delta:+.1f})")
        print(f"  Time: {elapsed:.3f}s")
        
        return result
    return wrapper

class GradientChecker:
    """
    Numerical gradient checking using finite differences.
    
    Compares analytical gradients against numerical approximations:
    f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    """
    
    def __init__(self, epsilon: float = 1e-5, tolerance: float = 1e-4):
        self.epsilon = epsilon
        self.tolerance = tolerance
    
    def check_gradients(self, model, grad_computer, input_ids: np.ndarray, 
                       labels: np.ndarray, mask: np.ndarray, 
                       param_subset: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Check gradients for a subset of parameters using finite differences.
        
        Args:
            model: Mini-BERT model
            grad_computer: Gradient computation object
            input_ids: Input token IDs [B, T]
            labels: Target labels [B, T]
            mask: MLM mask [B, T]
            param_subset: List of parameter names to check (None = check all)
            
        Returns:
            Dictionary with gradient check results for each parameter
        """
        print(f"Checking gradients with epsilon={self.epsilon}, tolerance={self.tolerance}")
        
        # Get analytical gradients
        logits, cache = model.forward(input_ids)
        loss_analytical, grad_logits = grad_computer.compute_mlm_loss_and_gradients(
            logits, labels, mask
        )
        
        grad_computer.zero_gradients()
        grad_computer.backward_from_logits(grad_logits, cache)
        analytical_grads = {name: grad.copy() for name, grad in grad_computer.gradients.items()}
        
        # Select parameters to check
        if param_subset is None:
            # Check a representative subset to avoid excessive computation
            param_subset = [
                'token_embeddings', 'W_Q_0', 'W1_0', 'ln1_gamma_0', 'mlm_head_W'
            ]
        
        results = {}
        
        for param_name in param_subset:
            if param_name not in model.params:
                print(f"Warning: Parameter {param_name} not found, skipping")
                continue
                
            print(f"  Checking {param_name}...")
            
            param = model.params[param_name]
            analytical_grad = analytical_grads[param_name]
            
            # Sample a few elements for checking (to avoid excessive computation)
            flat_param = param.flatten()
            flat_grad = analytical_grad.flatten()
            
            # Sample up to 20 elements randomly
            n_samples = min(20, len(flat_param))
            indices = np.random.choice(len(flat_param), n_samples, replace=False)
            
            numerical_grads = []
            analytical_vals = []
            
            for idx in indices:
                # Compute numerical gradient at this index
                original_val = flat_param[idx]
                
                # Forward perturbation
                flat_param[idx] = original_val + self.epsilon
                model.params[param_name] = flat_param.reshape(param.shape)
                logits_plus, _ = model.forward(input_ids)
                loss_plus, _ = grad_computer.compute_mlm_loss_and_gradients(
                    logits_plus, labels, mask
                )
                
                # Backward perturbation  
                flat_param[idx] = original_val - self.epsilon
                model.params[param_name] = flat_param.reshape(param.shape)
                logits_minus, _ = model.forward(input_ids)
                loss_minus, _ = grad_computer.compute_mlm_loss_and_gradients(
                    logits_minus, labels, mask
                )
                
                # Restore original value
                flat_param[idx] = original_val
                model.params[param_name] = flat_param.reshape(param.shape)
                
                # Compute numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2 * self.epsilon)
                numerical_grads.append(numerical_grad)
                analytical_vals.append(flat_grad[idx])
            
            # Compute statistics
            numerical_grads = np.array(numerical_grads)
            analytical_vals = np.array(analytical_vals)
            
            abs_diff = np.abs(numerical_grads - analytical_vals)
            rel_diff = abs_diff / (np.abs(numerical_grads) + np.abs(analytical_vals) + 1e-8)
            
            max_abs_diff = np.max(abs_diff)
            max_rel_diff = np.max(rel_diff)
            mean_abs_diff = np.mean(abs_diff)
            mean_rel_diff = np.mean(rel_diff)
            
            # Check if gradients match within tolerance
            passed = max_rel_diff < self.tolerance
            
            results[param_name] = {
                'passed': passed,
                'max_abs_diff': max_abs_diff,
                'max_rel_diff': max_rel_diff,
                'mean_abs_diff': mean_abs_diff,
                'mean_rel_diff': mean_rel_diff,
                'n_samples': n_samples,
                'numerical_grads': numerical_grads[:5],  # First 5 for inspection
                'analytical_grads': analytical_vals[:5]
            }
            
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {status}: max_rel_diff={max_rel_diff:.2e}, mean_rel_diff={mean_rel_diff:.2e}")
        
        return results

class TrainingDiagnostics:
    """
    Training diagnostics and monitoring utilities.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.metrics = defaultdict(list)
        self.log_file = log_file
        self.start_time = time.time()
        
        if log_file:
            # Create log directory if needed
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a training step."""
        # Add timestamp and step
        metrics['step'] = step
        metrics['elapsed_time'] = time.time() - self.start_time
        metrics['memory_mb'] = get_memory_usage()['rss_mb']
        
        # Store in memory
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        # Write to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                metric_strs = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                              for k, v in metrics.items()]
                f.write(f"Step {step}: {', '.join(metric_strs)}\n")
    
    def print_metrics(self, step: int, metrics: Dict[str, float], 
                     print_freq: int = 100):
        """Print training metrics to console."""
        if step % print_freq == 0:
            elapsed = time.time() - self.start_time
            memory = get_memory_usage()['rss_mb']
            
            print(f"Step {step:6d} | "
                  f"Loss: {metrics.get('loss', 0):.4f} | "
                  f"LR: {metrics.get('learning_rate', 0):.2e} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Mem: {memory:.0f}MB")
    
    def compute_gradient_stats(self, gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute gradient statistics for monitoring."""
        all_grads = []
        stats = {}
        
        for param_name, grad in gradients.items():
            if np.any(grad != 0):  # Skip zero gradients
                grad_flat = grad.flatten()
                all_grads.append(grad_flat)
                
                # Per-parameter stats
                stats[f'{param_name}_grad_norm'] = np.linalg.norm(grad_flat)
                stats[f'{param_name}_grad_max'] = np.max(np.abs(grad_flat))
        
        # Global gradient stats
        if all_grads:
            all_grads_flat = np.concatenate(all_grads)
            stats['global_grad_norm'] = np.linalg.norm(all_grads_flat)
            stats['global_grad_max'] = np.max(np.abs(all_grads_flat))
            stats['global_grad_mean'] = np.mean(np.abs(all_grads_flat))
            stats['global_grad_std'] = np.std(all_grads_flat)
        
        return stats
    
    def plot_loss_curve(self, save_path: Optional[str] = None):
        """Plot training loss curve (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            steps = self.metrics['step']
            losses = self.metrics['loss']
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Mini-BERT Training Loss')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Loss curve saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

def validate_shapes(tensors: Dict[str, np.ndarray], expected_shapes: Dict[str, Tuple]):
    """
    Validate tensor shapes match expected shapes.
    
    Args:
        tensors: Dictionary of tensor_name -> tensor
        expected_shapes: Dictionary of tensor_name -> expected_shape_tuple
        
    Raises:
        AssertionError if shapes don't match
    """
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape
            assert actual == expected, f"{name}: expected {expected}, got {actual}"

def save_checkpoint(model, optimizer_state: Dict, step: int, 
                   loss: float, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: Mini-BERT model
        optimizer_state: Optimizer state dictionary  
        step: Current training step
        loss: Current loss value
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'model_params': model.params,
        'optimizer_state': optimizer_state,
        'step': step,
        'loss': loss,
        'config': model.config.__dict__,
        'timestamp': time.time()
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {filepath} (step {step}, loss {loss:.4f})")

def load_checkpoint(filepath: str) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint data
    """
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded: {filepath} (step {checkpoint['step']}, "
          f"loss {checkpoint['loss']:.4f})")
    
    return checkpoint

def estimate_training_time(steps_completed: int, total_steps: int, 
                          start_time: float) -> Dict[str, str]:
    """
    Estimate remaining training time.
    
    Args:
        steps_completed: Number of steps completed
        total_steps: Total number of training steps
        start_time: Training start timestamp
        
    Returns:
        Dictionary with time estimates
    """
    if steps_completed == 0:
        return {'eta': 'unknown', 'elapsed': '0s', 'speed': '0 steps/s'}
    
    elapsed = time.time() - start_time
    steps_per_second = steps_completed / elapsed
    remaining_steps = total_steps - steps_completed
    eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    return {
        'eta': format_time(eta_seconds),
        'elapsed': format_time(elapsed),
        'speed': f"{steps_per_second:.1f} steps/s"
    }

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test memory monitoring
    print("\nMemory usage:")
    memory = get_memory_usage()
    for key, value in memory.items():
        print(f"  {key}: {value}")
    
    # Test gradient checker
    print("\nTesting gradient checker...")
    from model import MiniBERT
    from gradients import MiniBERTGradients
    
    model = MiniBERT()
    grad_computer = MiniBERTGradients(model)
    checker = GradientChecker(epsilon=1e-5, tolerance=1e-3)
    
    # Small test case
    B, T = 2, 4
    input_ids = np.random.randint(0, model.V, (B, T))
    labels = np.random.randint(0, model.V, (B, T))
    mask = np.ones((B, T))  # Mask all positions for testing
    
    results = checker.check_gradients(
        model, grad_computer, input_ids, labels, mask,
        param_subset=['W_Q_0', 'ln1_gamma_0']  # Just check a couple
    )
    
    print("Gradient check results:")
    for param_name, result in results.items():
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {param_name}: {status} (rel_diff: {result['max_rel_diff']:.2e})")
    
    # Test diagnostics
    print("\nTesting diagnostics...")
    diagnostics = TrainingDiagnostics()
    
    # Simulate some training steps
    for step in range(5):
        metrics = {
            'loss': 5.0 - step * 0.5,
            'learning_rate': 1e-4,
            'grad_norm': 1.0 + np.random.randn() * 0.1
        }
        diagnostics.log_step(step, metrics)
        diagnostics.print_metrics(step, metrics, print_freq=2)
    
    print("✓ Utilities testing completed!")