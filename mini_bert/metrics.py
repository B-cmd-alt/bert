"""
Evaluation Metrics for Mini-BERT Training.

Purpose:
- Masked token accuracy: Percentage correct predictions on masked positions only
- Perplexity proxy: exp(cross_entropy_loss) for language modeling quality
- Gradient sanity checks: Global grad norm, parameter RMS, NaN detection

Mathematical Formulations:
1. Masked Accuracy = (1/N) Σ 𝟙[argmax(logits[i]) = target[i]] for masked positions
2. Perplexity = exp(cross_entropy_loss) 
3. Gradient Norm = √(Σᵢ ||∇θᵢ||₂²) across all parameters

CLI Usage:
    python metrics.py --test  # Run unit tests
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings

def masked_accuracy(logits: np.ndarray, target_ids: np.ndarray, ignore_index: int = -100) -> float:
    """
    Compute accuracy only over masked positions (non-ignore positions).
    
    Mathematical Formula:
    accuracy = (1/N) Σ 𝟙[argmax(logits[i,j]) = target_ids[i,j]] 
               for (i,j) where target_ids[i,j] ≠ ignore_index
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore (typically -100 for non-masked positions)
        
    Returns:
        accuracy: Fraction of correct predictions at masked positions (0.0 to 1.0)
        
    Shape Assertions:
        logits.shape = (B, T, V)
        target_ids.shape = (B, T)
    """
    # Input validation and shape assertions
    assert isinstance(logits, np.ndarray), f"logits must be numpy array, got {type(logits)}"
    assert isinstance(target_ids, np.ndarray), f"target_ids must be numpy array, got {type(target_ids)}"
    assert logits.ndim == 3, f"logits must be 3D [B, T, V], got shape {logits.shape}"
    assert target_ids.ndim == 2, f"target_ids must be 2D [B, T], got shape {target_ids.shape}"
    assert logits.shape[:2] == target_ids.shape, f"Shape mismatch: logits {logits.shape[:2]} vs targets {target_ids.shape}"
    
    # Find valid (masked) positions
    valid_mask = (target_ids != ignore_index)
    num_valid = np.sum(valid_mask)
    
    if num_valid == 0:
        # No valid positions to evaluate
        return 0.0
    
    # Get predicted token IDs (argmax over vocabulary dimension)
    predicted_ids = np.argmax(logits, axis=-1)  # [B, T]
    
    # Check predictions only at valid positions
    valid_predictions = predicted_ids[valid_mask]  # [num_valid]
    valid_targets = target_ids[valid_mask]        # [num_valid]
    
    # Compute accuracy
    correct = (valid_predictions == valid_targets)
    accuracy = np.mean(correct.astype(np.float32))
    
    return float(accuracy)


def compute_perplexity(logits: np.ndarray, target_ids: np.ndarray, ignore_index: int = -100) -> float:
    """
    Compute perplexity as exp(cross_entropy_loss).
    
    Mathematical Formula:
    1. Cross-entropy: CE = -(1/N) Σ log(p_target[i]) for valid positions
    2. Perplexity: PPL = exp(CE)
    
    Lower perplexity indicates better language modeling performance.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        perplexity: exp(cross_entropy_loss), lower is better
        
    Shape Assertions:
        Same as masked_accuracy()
    """
    # Input validation and shape assertions
    assert isinstance(logits, np.ndarray), f"logits must be numpy array, got {type(logits)}"
    assert isinstance(target_ids, np.ndarray), f"target_ids must be numpy array, got {type(target_ids)}"
    assert logits.ndim == 3, f"logits must be 3D [B, T, V], got shape {logits.shape}"
    assert target_ids.ndim == 2, f"target_ids must be 2D [B, T], got shape {target_ids.shape}"
    assert logits.shape[:2] == target_ids.shape, f"Shape mismatch: logits {logits.shape[:2]} vs targets {target_ids.shape}"
    
    # Find valid positions
    valid_mask = (target_ids != ignore_index)
    num_valid = np.sum(valid_mask)
    
    if num_valid == 0:
        # No valid positions - return high perplexity
        return float('inf')
    
    # Compute log probabilities using numerically stable log-softmax
    # log_softmax(x) = x - log_sum_exp(x)
    logits_max = np.max(logits, axis=-1, keepdims=True)  # [B, T, 1]
    logits_shifted = logits - logits_max                 # [B, T, V]
    exp_logits = np.exp(logits_shifted)                  # [B, T, V]
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True) # [B, T, 1]
    log_probs = logits_shifted - np.log(sum_exp)         # [B, T, V]
    
    # Extract log probabilities for target tokens at valid positions
    valid_positions = np.where(valid_mask)
    valid_batch_indices = valid_positions[0]  # [num_valid]
    valid_seq_indices = valid_positions[1]    # [num_valid]
    valid_target_ids = target_ids[valid_mask] # [num_valid]
    
    # Get log probabilities for valid targets
    valid_log_probs = log_probs[valid_batch_indices, valid_seq_indices, valid_target_ids]
    
    # Cross-entropy loss (negative log likelihood)
    cross_entropy = -np.mean(valid_log_probs)
    
    # Perplexity = exp(cross_entropy)
    perplexity = np.exp(cross_entropy)
    
    return float(perplexity)


def compute_gradient_norm(gradients: Dict[str, np.ndarray]) -> float:
    """
    Compute global L2 gradient norm across all parameters.
    
    Mathematical Formula:
    ||∇||₂ = √(Σᵢ ||∇θᵢ||₂²) where θᵢ are individual parameters
    
    Args:
        gradients: Dictionary of parameter_name -> gradient_array
        
    Returns:
        grad_norm: Global L2 norm of all gradients
    """
    total_norm_squared = 0.0
    
    for param_name, grad in gradients.items():
        if grad is not None and np.any(grad != 0):
            # Add squared L2 norm of this parameter's gradient
            param_norm_squared = np.sum(grad ** 2)
            total_norm_squared += param_norm_squared
    
    global_norm = np.sqrt(total_norm_squared)
    return float(global_norm)


def compute_parameter_rms(parameters: Dict[str, np.ndarray]) -> float:
    """
    Compute RMS (root mean square) of all parameters.
    
    Mathematical Formula:
    RMS = √((1/N) Σᵢ θᵢ²) where N is total number of parameters
    
    Args:
        parameters: Dictionary of parameter_name -> parameter_array
        
    Returns:
        param_rms: RMS value of all parameters
    """
    total_squared = 0.0
    total_count = 0
    
    for param_name, param in parameters.items():
        if param is not None:
            total_squared += np.sum(param ** 2)
            total_count += param.size
    
    if total_count == 0:
        return 0.0
    
    rms = np.sqrt(total_squared / total_count)
    return float(rms)


def check_gradient_health(gradients: Dict[str, np.ndarray], 
                         step: int, 
                         max_norm: float = 1e3,
                         print_freq: int = 100) -> Dict[str, Any]:
    """
    Perform gradient sanity checks and print diagnostics.
    
    Checks for:
    - NaN gradients (critical error)
    - Exploding gradients (norm > max_norm)
    - Zero gradients (potential training issue)
    
    Args:
        gradients: Dictionary of gradients
        step: Current training step
        max_norm: Maximum allowed gradient norm
        print_freq: Print diagnostics every N steps
        
    Returns:
        health_info: Dictionary with gradient health metrics
    """
    grad_norm = compute_gradient_norm(gradients)
    
    # Check for NaN gradients
    has_nan = False
    nan_params = []
    for param_name, grad in gradients.items():
        if grad is not None and np.any(np.isnan(grad)):
            has_nan = True
            nan_params.append(param_name)
    
    # Check for exploding gradients
    is_exploding = grad_norm > max_norm
    
    # Check for zero gradients
    is_zero = grad_norm == 0.0
    
    # Health status
    if has_nan:
        status = "CRITICAL"
        message = f"NaN gradients detected in: {nan_params}"
    elif is_exploding:
        status = "WARNING"
        message = f"Exploding gradients: norm={grad_norm:.2e} > {max_norm:.2e}"
    elif is_zero:
        status = "WARNING"
        message = "All gradients are zero"
    else:
        status = "HEALTHY"
        message = f"Gradient norm: {grad_norm:.2e}"
    
    # Print diagnostics
    if step % print_freq == 0 or status != "HEALTHY":
        print(f"Step {step:6d} | Grad Health: {status:8s} | {message}")
        
        if status == "CRITICAL":
            print("  [CRITICAL] Training may fail due to NaN gradients!")
        elif status == "WARNING" and is_exploding:
            print("  [WARNING] Consider gradient clipping or lower learning rate")
    
    health_info = {
        'step': step,
        'grad_norm': grad_norm,
        'has_nan': has_nan,
        'is_exploding': is_exploding,
        'is_zero': is_zero,
        'status': status,
        'message': message,
        'nan_params': nan_params
    }
    
    return health_info


def evaluate_model_on_batch(model, grad_computer, batch_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Evaluate model on a single batch and return comprehensive metrics.
    
    Args:
        model: Mini-BERT model instance
        grad_computer: Gradient computer instance
        batch_data: Dictionary with 'input_ids', 'target_ids', 'attention_mask'
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    input_ids = batch_data['input_ids']      # [B, T]
    target_ids = batch_data['target_ids']    # [B, T]
    attention_mask = batch_data.get('attention_mask', None)  # [B, T] or None
    
    # Forward pass
    logits, cache = model.forward(input_ids, attention_mask)  # [B, T, V]
    
    # Compute metrics
    masked_acc = masked_accuracy(logits, target_ids, ignore_index=-100)
    perplexity = compute_perplexity(logits, target_ids, ignore_index=-100)
    
    # Cross-entropy loss (for completeness)
    from mlm import mlm_cross_entropy
    loss, _ = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
    
    metrics = {
        'masked_accuracy': masked_acc,
        'perplexity': perplexity,
        'cross_entropy_loss': loss,
        'num_masked_tokens': int(np.sum(target_ids != -100))
    }
    
    return metrics


def print_evaluation_table(results: Dict[str, Dict[str, float]], step: Optional[int] = None):
    """
    Print evaluation results in a tidy table format.
    
    Args:
        results: Dictionary of dataset_name -> metrics
        step: Optional training step number
    """
    print("\n" + "=" * 60)
    if step is not None:
        print(f"EVALUATION RESULTS - Step {step}")
    else:
        print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Table header
    print(f"{'Step':<6} {'Eval-set':<15} {'Metric':<12} {'Value':<10}")
    print("-" * 60)
    
    # Print results
    step_str = str(step) if step is not None else "–"
    
    for dataset_name, metrics in results.items():
        first_metric = True
        for metric_name, value in metrics.items():
            if first_metric:
                dataset_display = dataset_name
                step_display = step_str
                first_metric = False
            else:
                dataset_display = ""
                step_display = ""
            
            # Format value based on type
            if metric_name.endswith('_accuracy') or metric_name.endswith('Acc (%)'):
                value_str = f"{value*100:.1f}%" if value <= 1.0 else f"{value:.1f}%"
            elif metric_name.lower() == 'perplexity' or metric_name == 'PPL':
                value_str = f"{value:.1f}"
            elif 'loss' in metric_name.lower():
                value_str = f"{value:.3f}"
            else:
                value_str = f"{value:.3f}"
            
            print(f"{step_display:<6} {dataset_display:<15} {metric_name:<12} {value_str:<10}")


# Unit tests and demonstrations
def test_masked_accuracy():
    """Test masked accuracy computation."""
    print("Testing masked_accuracy...")
    
    # Create test data
    batch_size, seq_len, vocab_size = 2, 4, 10
    
    # Perfect predictions
    logits = np.zeros((batch_size, seq_len, vocab_size))
    target_ids = np.array([
        [5, -100, 7, -100],  # Only positions 0,2 are masked
        [-100, 6, -100, 8]   # Only positions 1,3 are masked
    ])
    
    # Set logits to give correct predictions
    logits[0, 0, 5] = 10.0  # Correct prediction
    logits[0, 2, 7] = 10.0  # Correct prediction  
    logits[1, 1, 6] = 10.0  # Correct prediction
    logits[1, 3, 8] = 10.0  # Correct prediction
    
    accuracy = masked_accuracy(logits, target_ids)
    assert accuracy == 1.0, f"Perfect predictions should give 100% accuracy, got {accuracy}"
    
    # Half correct predictions
    logits[0, 0, 4] = 15.0  # Wrong prediction (was 5, now predicts 4)
    accuracy = masked_accuracy(logits, target_ids)
    assert accuracy == 0.75, f"3/4 correct should give 75% accuracy, got {accuracy}"
    
    print("  [PASS] masked_accuracy")


def test_perplexity():
    """Test perplexity computation."""
    print("Testing compute_perplexity...")
    
    # Create test data with known entropy
    batch_size, seq_len, vocab_size = 1, 2, 4
    
    # Uniform distribution over 4 tokens -> entropy = log(4) -> perplexity = 4
    logits = np.zeros((batch_size, seq_len, vocab_size))  # Uniform after softmax
    target_ids = np.array([[0, 1]])  # Targets at positions 0,1
    
    ppl = compute_perplexity(logits, target_ids)
    expected_ppl = 4.0  # exp(log(4))
    
    assert abs(ppl - expected_ppl) < 0.1, f"Uniform distribution should give PPL≈4, got {ppl}"
    
    # Perfect predictions -> very low perplexity
    logits[0, 0, 0] = 10.0  # Strong prediction for target 0
    logits[0, 1, 1] = 10.0  # Strong prediction for target 1
    
    ppl_perfect = compute_perplexity(logits, target_ids)
    assert ppl_perfect < 1.1, f"Perfect predictions should give PPL≈1, got {ppl_perfect}"
    
    print("  [PASS] compute_perplexity")


def test_gradient_checks():
    """Test gradient norm and health checks."""
    print("Testing gradient checks...")
    
    # Create test gradients
    gradients = {
        'param1': np.array([[1.0, 2.0], [3.0, 4.0]]),  # norm = sqrt(30)
        'param2': np.array([0.5, 1.5])                  # norm = sqrt(2.5)
    }
    
    grad_norm = compute_gradient_norm(gradients)
    expected_norm = np.sqrt(30 + 2.5)  # sqrt(32.5) ≈ 5.7
    
    assert abs(grad_norm - expected_norm) < 0.1, f"Expected norm≈5.7, got {grad_norm}"
    
    # Test health check
    health = check_gradient_health(gradients, step=100, print_freq=1000)  # Won't print
    assert health['status'] == 'HEALTHY', f"Normal gradients should be healthy"
    assert not health['has_nan'], "Should not have NaN"
    assert not health['is_exploding'], "Should not be exploding"
    
    print("  [PASS] gradient checks")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running metrics unit tests...")
        test_masked_accuracy()
        test_perplexity()
        test_gradient_checks()
        print("\n[SUCCESS] All metrics tests passed!")
    else:
        print(__doc__)
        print("\nAvailable functions:")
        print("- masked_accuracy(logits, target_ids, ignore_index=-100)")
        print("- compute_perplexity(logits, target_ids, ignore_index=-100)")
        print("- compute_gradient_norm(gradients)")
        print("- check_gradient_health(gradients, step, max_norm=1e3)")
        print("- evaluate_model_on_batch(model, grad_computer, batch_data)")
        print("- print_evaluation_table(results, step=None)")
        print("\nRun 'python metrics.py --test' to run unit tests.")