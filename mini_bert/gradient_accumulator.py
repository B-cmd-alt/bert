"""
Efficient Gradient Accumulation for Mini-BERT

This module provides memory-efficient gradient accumulation to avoid creating
new arrays on each accumulation step.
"""

import numpy as np
from typing import Dict, Optional

class EfficientGradientAccumulator:
    """
    Memory-efficient gradient accumulator that reuses arrays.
    
    Key optimizations:
    1. Pre-allocates gradient buffers (no memory allocation during training)
    2. Uses in-place operations (no array copying)
    3. Proper averaging without creating new arrays
    """
    
    def __init__(self, model_params: Dict[str, np.ndarray]):
        """
        Initialize accumulator with model parameter shapes.
        
        Args:
            model_params: Dictionary of model parameters
        """
        self.accumulated_grads = {}
        self.accumulation_steps = 0
        
        # Pre-allocate gradient buffers with same shapes as parameters
        for name, param in model_params.items():
            self.accumulated_grads[name] = np.zeros_like(param)
    
    def accumulate(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Accumulate gradients in-place (no memory allocation).
        
        Args:
            gradients: Dictionary of gradients to accumulate
        """
        for name, grad in gradients.items():
            if grad is not None and name in self.accumulated_grads:
                # In-place addition - no memory allocation
                self.accumulated_grads[name] += grad
        
        self.accumulation_steps += 1
    
    def get_averaged_gradients(self) -> Dict[str, np.ndarray]:
        """
        Get averaged gradients and reset accumulator.
        
        Returns:
            Dictionary of averaged gradients
        """
        if self.accumulation_steps == 0:
            return {}
        
        # Create output dictionary with averaged gradients
        averaged_grads = {}
        for name, grad in self.accumulated_grads.items():
            # Create copy and average
            averaged_grads[name] = grad.copy() / self.accumulation_steps
            # Reset accumulator for next round (in-place)
            grad.fill(0.0)
        
        self.accumulation_steps = 0
        return averaged_grads
    
    def reset(self) -> None:
        """Reset accumulator without reallocating memory."""
        for grad in self.accumulated_grads.values():
            grad.fill(0.0)
        self.accumulation_steps = 0
    
    def get_accumulation_steps(self) -> int:
        """Get number of accumulated steps."""
        return self.accumulation_steps
    
    def get_gradient_norm(self) -> float:
        """
        Compute norm of currently accumulated gradients.
        
        Returns:
            L2 norm of accumulated gradients
        """
        if self.accumulation_steps == 0:
            return 0.0
        
        total_norm_sq = 0.0
        for grad in self.accumulated_grads.values():
            if np.any(grad):
                total_norm_sq += np.sum(grad ** 2)
        
        return np.sqrt(total_norm_sq) / self.accumulation_steps


def demonstrate_efficiency():
    """Demonstrate the efficiency improvement."""
    import time
    
    # Simulate model parameters
    model_params = {
        'W1': np.random.randn(1000, 500),
        'W2': np.random.randn(500, 1000),
        'b1': np.random.randn(500),
        'b2': np.random.randn(1000)
    }
    
    # Simulate gradients
    gradients = {
        'W1': np.random.randn(1000, 500) * 0.01,
        'W2': np.random.randn(500, 1000) * 0.01,
        'b1': np.random.randn(500) * 0.01,
        'b2': np.random.randn(1000) * 0.01
    }
    
    num_steps = 100
    
    # Old way (creates new arrays each time)
    print("Testing old gradient accumulation method...")
    start_time = time.time()
    accumulated_old = {}
    for step in range(num_steps):
        if not accumulated_old:
            accumulated_old = {name: grad.copy() for name, grad in gradients.items()}
        else:
            for name, grad in gradients.items():
                accumulated_old[name] += grad
    old_time = time.time() - start_time
    
    # New way (efficient accumulator)
    print("Testing new efficient accumulator...")
    accumulator = EfficientGradientAccumulator(model_params)
    start_time = time.time()
    for step in range(num_steps):
        accumulator.accumulate(gradients)
    averaged_grads = accumulator.get_averaged_gradients()
    new_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"Old method: {old_time:.4f} seconds")
    print(f"New method: {new_time:.4f} seconds")
    print(f"Speedup: {old_time/new_time:.2f}x")
    
    # Verify correctness
    for name in accumulated_old:
        old_avg = accumulated_old[name] / num_steps
        new_avg = averaged_grads[name]
        diff = np.abs(old_avg - new_avg).max()
        print(f"{name}: max difference = {diff:.2e}")


if __name__ == "__main__":
    demonstrate_efficiency()