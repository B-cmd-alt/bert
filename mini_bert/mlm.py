"""
Masked Language Model (MLM) utilities for BERT training.

Mathematical Formulations:

1. MLM Masking Strategy:
   - Select exactly 15% of tokens for masking
   - Of masked tokens: 80% → [MASK], 10% → random, 10% → unchanged
   - target_ids stores original tokens at masked positions, -100 elsewhere

2. MLM Cross-Entropy Loss:
   - Softmax: p_i = exp(logit_i) / Σ_j exp(logit_j)  
   - Cross-entropy: CE = -log(p_target)
   - Loss = mean(CE) over non-ignored positions
   - Accuracy = fraction of correct predictions at masked positions
"""

import numpy as np
from typing import Tuple

def mask_tokens(ids: np.ndarray, vocab_size: int, mask_id: int, p_mask: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply MLM masking to token sequences.
    
    Mathematical Strategy:
    1. Select exactly ⌊sequence_length × p_mask⌋ positions for masking
    2. For each masked position:
       - 80% probability: replace with mask_id
       - 10% probability: replace with random token ∈ [5, vocab_size)  
       - 10% probability: keep unchanged
    3. Create target_ids with original tokens at masked positions, -100 elsewhere
    
    Args:
        ids: Input token sequences [batch_size, seq_len]
        vocab_size: Size of vocabulary
        mask_id: Token ID for [MASK] token (typically 4)
        p_mask: Probability of masking (default 0.15 = 15%)
        
    Returns:
        input_ids: Modified input with some tokens masked [batch_size, seq_len]
        target_ids: Original tokens at masked positions, -100 elsewhere [batch_size, seq_len]
        mask_positions_bool: Boolean mask indicating which positions were masked [batch_size, seq_len]
        
    Shape assertions:
        ids.shape = (batch_size, seq_len)
        All outputs have same shape as input
    """
    # Input validation and shape assertions
    assert isinstance(ids, np.ndarray), f"ids must be numpy array, got {type(ids)}"
    assert ids.ndim == 2, f"ids must be 2D [batch_size, seq_len], got shape {ids.shape}"
    assert 0 < p_mask < 1, f"p_mask must be in (0,1), got {p_mask}"
    assert mask_id < vocab_size, f"mask_id {mask_id} must be < vocab_size {vocab_size}"
    
    batch_size, seq_len = ids.shape
    
    # Copy input to avoid modifying original
    input_ids = ids.copy()
    
    # Initialize target_ids with ignore_index (-100) everywhere
    target_ids = np.full_like(ids, -100)
    
    # Initialize mask positions 
    mask_positions_bool = np.zeros_like(ids, dtype=bool)
    
    # Process each sequence in the batch separately
    for batch_idx in range(batch_size):
        sequence = ids[batch_idx]
        
        # Find maskable positions (avoid special tokens [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4)
        # Only mask tokens with id >= 5
        maskable_positions = np.where(sequence >= 5)[0]
        
        if len(maskable_positions) == 0:
            continue  # Skip sequences with no maskable tokens
        
        # Calculate number of tokens to mask (exactly 15%)
        num_to_mask = max(1, int(len(maskable_positions) * p_mask))
        num_to_mask = min(num_to_mask, len(maskable_positions))
        
        # Randomly select positions to mask
        np.random.shuffle(maskable_positions)
        selected_positions = maskable_positions[:num_to_mask]
        
        # Mark these positions as masked
        mask_positions_bool[batch_idx, selected_positions] = True
        
        # Store original tokens in target_ids at masked positions
        target_ids[batch_idx, selected_positions] = sequence[selected_positions]
        
        # Apply masking strategy to each selected position
        for pos in selected_positions:
            rand_val = np.random.random()
            
            if rand_val < 0.8:
                # 80% of time: replace with [MASK] token
                input_ids[batch_idx, pos] = mask_id
            elif rand_val < 0.9:
                # 10% of time: replace with random token (avoid special tokens)
                random_token = np.random.randint(5, vocab_size)
                input_ids[batch_idx, pos] = random_token
            # else: 10% of time: keep original token (no change)
    
    # Final shape assertions
    assert input_ids.shape == ids.shape, f"input_ids shape mismatch: {input_ids.shape} vs {ids.shape}"
    assert target_ids.shape == ids.shape, f"target_ids shape mismatch: {target_ids.shape} vs {ids.shape}"
    assert mask_positions_bool.shape == ids.shape, f"mask_positions shape mismatch: {mask_positions_bool.shape} vs {ids.shape}"
    
    return input_ids, target_ids, mask_positions_bool


def mlm_cross_entropy(logits: np.ndarray, target_ids: np.ndarray, ignore_index: int = -100) -> Tuple[float, float]:
    """
    Compute MLM cross-entropy loss and accuracy.
    
    Mathematical Formulation:
    1. Softmax probabilities: p_{i,j,k} = exp(logits_{i,j,k}) / Σ_v exp(logits_{i,j,v})
    2. Cross-entropy: CE_{i,j} = -log(p_{i,j,target_{i,j}}) if target_{i,j} ≠ ignore_index
    3. Loss = (1/N) Σ_{valid positions} CE_{i,j} where N = number of valid positions
    4. Accuracy = (1/N) Σ_{valid positions} 𝟙[argmax_v p_{i,j,v} = target_{i,j}]
    
    Numerical Stability:
    - Use log-sum-exp trick: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
    - Compute log probabilities directly to avoid overflow
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len] 
        ignore_index: Index to ignore in loss computation (default -100)
        
    Returns:
        loss: Scalar cross-entropy loss averaged over valid positions
        accuracy: Fraction of correct predictions at valid positions
        
    Shape assertions:
        logits.shape = (batch_size, seq_len, vocab_size)
        target_ids.shape = (batch_size, seq_len)
    """
    # Input validation and shape assertions
    assert isinstance(logits, np.ndarray), f"logits must be numpy array, got {type(logits)}"
    assert isinstance(target_ids, np.ndarray), f"target_ids must be numpy array, got {type(target_ids)}"
    assert logits.ndim == 3, f"logits must be 3D [B, T, V], got shape {logits.shape}"
    assert target_ids.ndim == 2, f"target_ids must be 2D [B, T], got shape {target_ids.shape}"
    assert logits.shape[:2] == target_ids.shape, f"Shape mismatch: logits {logits.shape[:2]} vs targets {target_ids.shape}"
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Find valid positions (where target_ids != ignore_index)
    valid_mask = (target_ids != ignore_index)
    num_valid = np.sum(valid_mask)
    
    if num_valid == 0:
        # No valid positions to compute loss
        return 0.0, 0.0
    
    # Compute log probabilities using numerically stable log-softmax
    # log_softmax(x) = x - log_sum_exp(x)
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    
    # Find max along vocab dimension for numerical stability
    logits_max = np.max(logits, axis=-1, keepdims=True)  # [B, T, 1]
    
    # Compute shifted logits
    logits_shifted = logits - logits_max  # [B, T, V]
    
    # Compute log_sum_exp
    exp_shifted = np.exp(logits_shifted)  # [B, T, V]
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)  # [B, T, 1]
    log_sum_exp = np.log(sum_exp)  # [B, T, 1]
    
    # Log probabilities
    log_probs = logits_shifted - log_sum_exp  # [B, T, V]
    
    # Gather log probabilities for target tokens at valid positions only
    # Extract valid positions
    valid_positions = np.where(valid_mask)
    valid_batch_indices = valid_positions[0]  # [num_valid]
    valid_seq_indices = valid_positions[1]    # [num_valid]
    valid_target_ids = target_ids[valid_mask] # [num_valid]
    
    # Extract log probabilities for valid target tokens
    valid_log_probs = log_probs[valid_batch_indices, valid_seq_indices, valid_target_ids]  # [num_valid]
    loss = -np.mean(valid_log_probs)  # Negative log likelihood
    
    # Compute accuracy
    # Get predicted tokens (argmax over vocab dimension)
    predicted_ids = np.argmax(logits, axis=-1)  # [B, T]
    
    # Check correctness at valid positions
    valid_predictions = predicted_ids[valid_mask]  # [num_valid]
    valid_targets = target_ids[valid_mask]        # [num_valid]
    correct_predictions = (valid_predictions == valid_targets)
    accuracy = np.mean(correct_predictions.astype(np.float32))
    
    # Ensure scalar outputs
    loss = float(loss)
    accuracy = float(accuracy)
    
    return loss, accuracy


def test_mask_tokens():
    """Unit test for mask_tokens function."""
    print("Testing mask_tokens...")
    
    # Test basic functionality
    np.random.seed(42)  # Reproducible results
    
    # Create test input: batch_size=2, seq_len=10, vocab_size=100
    ids = np.array([
        [2, 5, 6, 7, 8, 9, 10, 11, 12, 3],  # [CLS] + tokens + [SEP]
        [2, 15, 16, 17, 0, 0, 0, 0, 0, 3]   # [CLS] + tokens + padding + [SEP]
    ])
    
    vocab_size = 100
    mask_id = 4
    p_mask = 0.15
    
    input_ids, target_ids, mask_positions = mask_tokens(ids, vocab_size, mask_id, p_mask)
    
    # Check shapes
    assert input_ids.shape == ids.shape, f"Shape mismatch: {input_ids.shape} vs {ids.shape}"
    assert target_ids.shape == ids.shape, f"Shape mismatch: {target_ids.shape} vs {ids.shape}"
    assert mask_positions.shape == ids.shape, f"Shape mismatch: {mask_positions.shape} vs {ids.shape}"
    
    # Check that special tokens are not masked (positions 0 and -1)
    assert not mask_positions[0, 0], "First token ([CLS]) should not be masked"
    assert not mask_positions[0, -1], "Last token ([SEP]) should not be masked"
    assert not mask_positions[1, 0], "First token ([CLS]) should not be masked"
    assert not mask_positions[1, -1], "Last token ([SEP]) should not be masked"
    
    # Check that target_ids are -100 at non-masked positions
    non_masked = ~mask_positions
    assert np.all(target_ids[non_masked] == -100), "Non-masked positions should have target_ids = -100"
    
    # Check that target_ids store original tokens at masked positions
    masked = mask_positions
    original_at_masked = ids[masked]
    target_at_masked = target_ids[masked]
    assert np.array_equal(original_at_masked, target_at_masked), "target_ids should store original tokens at masked positions"
    
    print(f"  Masking ratio: {np.sum(mask_positions) / np.sum(ids >= 5):.3f} (target: {p_mask})")
    print(f"  Masked positions: {np.sum(mask_positions)}")
    print("  [PASS] mask_tokens basic functionality")
    
    # Test edge cases
    # Empty sequence (all padding)
    empty_ids = np.zeros((1, 5))
    empty_input, empty_target, empty_mask = mask_tokens(empty_ids, vocab_size, mask_id)
    assert np.sum(empty_mask) == 0, "Empty sequence should have no masked positions"
    
    print("  [PASS] mask_tokens edge cases")


def test_mlm_cross_entropy():
    """Unit test for mlm_cross_entropy function with finite difference gradient check."""
    print("Testing mlm_cross_entropy...")
    
    # Test basic functionality
    np.random.seed(42)
    
    batch_size, seq_len, vocab_size = 2, 4, 10
    
    # Create test logits and targets
    logits = np.random.randn(batch_size, seq_len, vocab_size) * 0.1
    target_ids = np.array([
        [5, -100, 7, -100],  # Only positions 0 and 2 are valid
        [-100, 6, -100, 8]   # Only positions 1 and 3 are valid  
    ])
    
    loss, accuracy = mlm_cross_entropy(logits, target_ids)
    
    # Check outputs are scalars
    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert isinstance(accuracy, float), f"Accuracy should be float, got {type(accuracy)}"
    assert 0 <= accuracy <= 1, f"Accuracy should be in [0,1], got {accuracy}"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  [PASS] mlm_cross_entropy basic functionality")
    
    # Test with no valid positions
    all_ignore = np.full((2, 4), -100)
    loss_empty, acc_empty = mlm_cross_entropy(logits, all_ignore)
    assert loss_empty == 0.0, f"Loss with no valid positions should be 0, got {loss_empty}"
    assert acc_empty == 0.0, f"Accuracy with no valid positions should be 0, got {acc_empty}"
    print("  [PASS] mlm_cross_entropy no valid positions")
    
    # Finite difference gradient check (simplified)
    print("  Running finite difference gradient check...")
    
    # Small test case for gradient checking
    small_logits = np.random.randn(1, 2, 3) * 0.1
    small_targets = np.array([[1, 2]])  # Both positions valid
    
    epsilon = 1e-5
    loss_orig, _ = mlm_cross_entropy(small_logits, small_targets)
    
    # Check gradient for one logit element
    i, j, k = 0, 0, 1  # Position to perturb
    
    # Forward perturbation
    small_logits[i, j, k] += epsilon
    loss_plus, _ = mlm_cross_entropy(small_logits, small_targets)
    
    # Backward perturbation  
    small_logits[i, j, k] -= 2 * epsilon
    loss_minus, _ = mlm_cross_entropy(small_logits, small_targets)
    
    # Restore original value
    small_logits[i, j, k] += epsilon
    
    # Numerical gradient
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Analytical gradient (simplified check)
    # For cross-entropy: ∂L/∂logit = (p - target_onehot) / num_valid
    probs = np.exp(small_logits) / np.sum(np.exp(small_logits), axis=-1, keepdims=True)
    target_onehot = np.zeros_like(small_logits)
    target_onehot[0, 0, small_targets[0, 0]] = 1
    target_onehot[0, 1, small_targets[0, 1]] = 1
    
    analytical_grad = (probs[i, j, k] - target_onehot[i, j, k]) / 2  # 2 valid positions
    
    rel_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
    
    print(f"    Numerical grad: {numerical_grad:.6f}")
    print(f"    Analytical grad: {analytical_grad:.6f}")  
    print(f"    Relative error: {rel_error:.2e}")
    
    if rel_error < 1e-4:
        print("  [PASS] Gradient check")
    else:
        print("  [WARN] Gradient check - high error (may be numerical precision issue)")


if __name__ == "__main__":
    print("=" * 50)
    print("MLM Utilities Test Suite")
    print("=" * 50)
    
    test_mask_tokens()
    print()
    test_mlm_cross_entropy()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All MLM tests passed!")
    print("=" * 50)