"""
Pure NumPy AdamW Optimizer Implementation.

Mathematical Formulation:
AdamW (Adam with decoupled weight decay):

1. Initialize: m₀ = 0, v₀ = 0, step = 0
2. For each optimization step:
   a) step = step + 1
   b) Apply weight decay: θ = θ - lr * weight_decay * θ
   c) Compute biased moments:
      m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
      v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
   d) Bias correction:
      m̂_t = m_t / (1 - β₁^t)
      v̂_t = v_t / (1 - β₂^t)
   e) Parameter update:
      θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

Where:
- lr: learning rate
- β₁ = 0.9: exponential decay rate for first moment estimates
- β₂ = 0.999: exponential decay rate for second moment estimates  
- ε = 1e-8: small constant for numerical stability
- weight_decay: L2 regularization coefficient
"""

import numpy as np
from typing import Dict, Any, Optional

class AdamW:
    """
    AdamW optimizer implemented in pure NumPy.
    
    AdamW differs from Adam by applying weight decay directly to parameters
    rather than adding it to gradients, which provides better generalization.
    """
    
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate (α)
            beta1: Exponential decay rate for first moment estimates (β₁)
            beta2: Exponential decay rate for second moment estimates (β₂)  
            eps: Small constant for numerical stability (ε)
            weight_decay: Weight decay coefficient (λ)
        """
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Optimizer state
        self.step_count = 0
        self.m: Dict[str, np.ndarray] = {}  # First moment estimates
        self.v: Dict[str, np.ndarray] = {}  # Second moment estimates
        
        # Precompute commonly used values
        self._sqrt_eps = np.sqrt(eps)
        
        print(f"AdamW optimizer initialized:")
        print(f"  learning_rate: {learning_rate}")
        print(f"  beta1: {beta1}, beta2: {beta2}")
        print(f"  eps: {eps}, weight_decay: {weight_decay}")
    
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], step_num: Optional[int] = None) -> Dict[str, float]:
        """
        Perform one optimization step.
        
        Mathematical Steps:
        1. Increment step counter: t = t + 1
        2. Apply weight decay: θ = θ * (1 - lr * weight_decay)
        3. Update biased moments:
           m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
           v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        4. Compute bias-corrected moments:
           m̂_t = m_t / (1 - β₁^t)
           v̂_t = v_t / (1 - β₂^t)
        5. Update parameters:
           θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
        
        Args:
            params: Dictionary of parameter name -> parameter array
            grads: Dictionary of parameter name -> gradient array
            step_num: Optional step number (uses internal counter if None)
            
        Returns:
            Dictionary with optimization statistics
        """
        # Update step count
        if step_num is not None:
            self.step_count = step_num
        else:
            self.step_count += 1
        
        # Bias correction factors
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
        
        # Statistics for monitoring
        total_param_norm = 0.0
        total_grad_norm = 0.0
        total_update_norm = 0.0
        params_updated = 0
        
        # Process each parameter
        for param_name in params:
            if param_name not in grads:
                continue  # Skip parameters without gradients
                
            param = params[param_name]
            grad = grads[param_name]
            
            # Skip if gradient is all zeros
            if not np.any(grad):
                continue
                
            # Initialize momentum buffers if needed
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)
            
            m_t = self.m[param_name]
            v_t = self.v[param_name]
            
            # Apply weight decay directly to parameters (AdamW style)
            # θ = θ * (1 - lr * weight_decay)
            if self.weight_decay > 0:
                param *= (1.0 - self.learning_rate * self.weight_decay)
            
            # Update biased first and second moment estimates
            # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            m_t *= self.beta1
            m_t += (1.0 - self.beta1) * grad
            
            # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            v_t *= self.beta2
            v_t += (1.0 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moments
            # m̂_t = m_t / (1 - β₁^t)
            m_hat = m_t / bias_correction1
            
            # v̂_t = v_t / (1 - β₂^t)  
            v_hat = v_t / bias_correction2
            
            # Compute parameter update
            # Δθ = lr * m̂_t / (√v̂_t + ε)
            denominator = np.sqrt(v_hat) + self.eps
            update = self.learning_rate * m_hat / denominator
            
            # Apply update
            # θ_t = θ_{t-1} - Δθ
            param -= update
            
            # Update statistics
            total_param_norm += np.sum(param ** 2)
            total_grad_norm += np.sum(grad ** 2)
            total_update_norm += np.sum(update ** 2)
            params_updated += 1
        
        # Compute global norms
        param_norm = np.sqrt(total_param_norm)
        grad_norm = np.sqrt(total_grad_norm)
        update_norm = np.sqrt(total_update_norm)
        
        # Return optimization statistics
        stats = {
            'step': self.step_count,
            'learning_rate': self.learning_rate,
            'param_norm': float(param_norm),
            'grad_norm': float(grad_norm),
            'update_norm': float(update_norm),
            'bias_correction1': float(bias_correction1),
            'bias_correction2': float(bias_correction2),
            'params_updated': params_updated
        }
        
        return stats
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate
    
    def set_lr(self, learning_rate: float):
        """Set learning rate."""
        self.learning_rate = learning_rate
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            'step_count': self.step_count,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': {k: v.copy() for k, v in self.m.items()},
            'v': {k: v.copy() for k, v in self.v.items()}
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self.step_count = state_dict['step_count']
        self.learning_rate = state_dict['learning_rate']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.m = {k: v.copy() for k, v in state_dict['m'].items()}
        self.v = {k: v.copy() for k, v in state_dict['v'].items()}


class LRScheduler:
    """
    Learning rate scheduler with linear warmup and decay.
    
    Mathematical Formulation:
    1. Warmup phase (steps 0 to warmup_steps):
       lr(t) = base_lr * (t / warmup_steps)
    
    2. Decay phase (steps warmup_steps to total_steps):
       lr(t) = base_lr * (1 - (t - warmup_steps) / (total_steps - warmup_steps))
    """
    
    def __init__(self, 
                 optimizer: AdamW,
                 warmup_steps: int,
                 total_steps: int,
                 base_lr: Optional[float] = None):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: AdamW optimizer instance
            warmup_steps: Number of warmup steps (typically 10% of total)
            total_steps: Total number of training steps
            base_lr: Base learning rate (uses optimizer's LR if None)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr if base_lr is not None else optimizer.get_lr()
        
        print(f"LR Scheduler initialized:")
        print(f"  base_lr: {self.base_lr}")
        print(f"  warmup_steps: {warmup_steps} ({100*warmup_steps/total_steps:.1f}%)")
        print(f"  total_steps: {total_steps}")
    
    def get_lr(self, step: int) -> float:
        """
        Compute learning rate for given step.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            Learning rate for this step
        """
        if step < self.warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            lr_scale = step / self.warmup_steps
        else:
            # Linear decay: lr = base_lr * (1 - progress)
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = max(0.0, 1.0 - progress)
        
        return self.base_lr * lr_scale
    
    def step(self, step: int):
        """Update optimizer learning rate for given step."""
        new_lr = self.get_lr(step)
        self.optimizer.set_lr(new_lr)


def test_adamw():
    """Test AdamW optimizer functionality."""
    print("Testing AdamW optimizer...")
    
    # Create test parameters
    np.random.seed(42)
    params = {
        'weight': np.random.randn(3, 4) * 0.1,
        'bias': np.random.randn(4) * 0.1
    }
    
    # Create test gradients
    grads = {
        'weight': np.random.randn(3, 4) * 0.01,
        'bias': np.random.randn(4) * 0.01
    }
    
    # Initialize optimizer
    optimizer = AdamW(learning_rate=0.01, weight_decay=0.1)
    
    # Store initial parameters
    initial_weight = params['weight'].copy()
    initial_bias = params['bias'].copy()
    
    # Perform optimization step
    stats = optimizer.step(params, grads)
    
    # Check that parameters changed
    weight_change = np.linalg.norm(params['weight'] - initial_weight)
    bias_change = np.linalg.norm(params['bias'] - initial_bias)
    
    assert weight_change > 0, "Weight parameters should have changed"
    assert bias_change > 0, "Bias parameters should have changed"
    
    print(f"  Step: {stats['step']}")
    print(f"  Parameter change: weight={weight_change:.6f}, bias={bias_change:.6f}")
    print(f"  Gradient norm: {stats['grad_norm']:.6f}")
    print(f"  Update norm: {stats['update_norm']:.6f}")
    print("  [PASS] AdamW basic functionality")
    
    # Test multiple steps
    for step in range(2, 6):
        stats = optimizer.step(params, grads)
        assert stats['step'] == step, f"Step counter should be {step}, got {stats['step']}"
    
    print("  [PASS] AdamW multiple steps")
    
    # Test state saving/loading
    state = optimizer.state_dict()
    
    # Create new optimizer and load state
    new_optimizer = AdamW()
    new_optimizer.load_state_dict(state)
    
    assert new_optimizer.step_count == optimizer.step_count, "Step count should match"
    assert new_optimizer.learning_rate == optimizer.learning_rate, "Learning rate should match"
    
    print("  [PASS] AdamW state save/load")


def test_lr_scheduler():
    """Test learning rate scheduler."""
    print("Testing LR scheduler...")
    
    # Create optimizer and scheduler
    optimizer = AdamW(learning_rate=1e-3)
    scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    # Test warmup phase
    lr_0 = scheduler.get_lr(0)
    lr_50 = scheduler.get_lr(50)
    lr_100 = scheduler.get_lr(100)
    
    assert lr_0 == 0.0, f"LR at step 0 should be 0, got {lr_0}"
    assert lr_50 == 5e-4, f"LR at step 50 should be 5e-4, got {lr_50}"
    assert abs(lr_100 - 1e-3) < 1e-6, f"LR at step 100 should be 1e-3, got {lr_100}"
    
    print(f"  Warmup: step 0={lr_0:.2e}, step 50={lr_50:.2e}, step 100={lr_100:.2e}")
    
    # Test decay phase
    lr_550 = scheduler.get_lr(550)  # Middle of decay
    lr_1000 = scheduler.get_lr(1000)  # End of decay
    
    assert abs(lr_550 - 5e-4) < 1e-6, f"LR at step 550 should be 5e-4, got {lr_550}"
    assert lr_1000 == 0.0, f"LR at step 1000 should be 0, got {lr_1000}"
    
    print(f"  Decay: step 550={lr_550:.2e}, step 1000={lr_1000:.2e}")
    print("  [PASS] LR scheduler")


if __name__ == "__main__":
    print("=" * 50)
    print("AdamW Optimizer Test Suite")
    print("=" * 50)
    
    test_adamw()
    print()
    test_lr_scheduler()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All optimizer tests passed!")
    print("=" * 50)