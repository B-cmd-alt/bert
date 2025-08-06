"""
Test script to verify critical bug fixes in Mini-BERT

This script tests:
1. Attention mask is applied correctly to attention scores
2. Token embedding gradients are computed efficiently  
3. Numerical stability improvements
4. Gradient accumulation efficiency
"""

import numpy as np
import time
from model import MiniBERT, stable_softmax
from gradients import MiniBERTGradients
from tokenizer import WordPieceTokenizer
from gradient_accumulator import EfficientGradientAccumulator


def test_attention_mask_fix():
    """Test that attention mask is applied to scores, not embeddings."""
    print("=" * 60)
    print("TESTING ATTENTION MASK FIX")
    print("=" * 60)
    
    model = MiniBERT()
    
    # Create test input
    batch_size = 2
    seq_len = 8
    input_ids = np.random.randint(5, 1000, (batch_size, seq_len))
    
    # Create attention mask (mask out last 2 positions)
    attention_mask = np.ones((batch_size, seq_len))
    attention_mask[:, -2:] = 0  # Mask last 2 positions
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask: {attention_mask[0]}")
    
    # Forward pass with attention mask
    logits, cache = model.forward(input_ids, attention_mask)
    
    print(f"Output shape: {logits.shape}")
    
    # Check that attention weights are zero for masked positions
    # Look for attention weights in the cache
    layer_caches = cache.get('layer_caches', {})
    
    if 'layer_0' in layer_caches and 'attn_cache' in layer_caches['layer_0']:
        attn_cache = layer_caches['layer_0']['attn_cache']
        if 'attention_weights' in attn_cache:
            attention_weights = attn_cache['attention_weights']  # [B, A, T, T]
            
            # For masked positions (last 2), attention weights to them should be ~0
            masked_attention = attention_weights[0, 0, :, -2:]  # First batch, first head, all queries, last 2 keys
        else:
            print(f"Attention cache keys: {list(attn_cache.keys())}")
            print("FAIL: Could not find attention weights in attention cache")
            return False
    else:
        print(f"Available cache keys: {list(cache.keys())}")
        if layer_caches:
            print(f"Layer cache keys: {list(layer_caches.keys())}")
            if 'layer_0' in layer_caches:
                print(f"Layer 0 keys: {list(layer_caches['layer_0'].keys())}")
        print("FAIL: Could not find layer_0 attention cache")
        return False
    
    print(f"Attention to masked positions (should be ~0): {masked_attention.max():.6f}")
    
    # Verify attention weights sum to 1 (excluding numerical precision)
    attention_sums = attention_weights.sum(axis=-1)  # Sum over keys
    print(f"Attention weights sum (should be ~1): {attention_sums[0, 0, 0]:.6f}")
    
    if masked_attention.max() < 1e-6:
        print("PASS: Attention mask correctly applied to attention scores")
    else:
        print("FAIL: Attention mask not working correctly")
    
    return masked_attention.max() < 1e-6


def test_token_embedding_gradients_speed():
    """Test that token embedding gradients are computed efficiently."""
    print("\n" + "=" * 60)
    print("TESTING TOKEN EMBEDDING GRADIENT SPEED")
    print("=" * 60)
    
    model = MiniBERT()
    grad_computer = MiniBERTGradients(model)
    
    # Create larger test case to see speed difference
    batch_size = 16
    seq_len = 64
    input_ids = np.random.randint(5, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, cache = model.forward(input_ids)
    
    # Create dummy gradient
    grad_logits = np.random.randn(*logits.shape) * 0.01
    
    # Time the gradient computation
    print(f"Computing gradients for batch_size={batch_size}, seq_len={seq_len}")
    
    start_time = time.time()
    grad_computer.zero_gradients()
    grad_computer.backward_from_logits(grad_logits, cache)
    end_time = time.time()
    
    gradient_time = end_time - start_time
    print(f"Gradient computation time: {gradient_time:.4f} seconds")
    
    # Verify gradients are computed
    token_grad_norm = np.linalg.norm(grad_computer.gradients['token_embeddings'])
    print(f"Token embedding gradient norm: {token_grad_norm:.6f}")
    
    if gradient_time < 1.0:  # Should be fast with vectorized implementation
        print("PASS: Token embedding gradients computed efficiently")
        return True
    else:
        print("FAIL: Token embedding gradients too slow")
        return False


def test_numerical_stability():
    """Test numerical stability improvements."""
    print("\n" + "=" * 60)
    print("TESTING NUMERICAL STABILITY")
    print("=" * 60)
    
    # Test stable softmax with extreme values
    extreme_logits = np.array([
        [100, 200, 150],  # Very large values
        [-100, -50, -200],  # Very negative values
        [1e-10, 2e-10, 3e-10]  # Very small values
    ])
    
    print("Testing stable softmax with extreme values...")
    
    # Apply stable softmax
    probs = stable_softmax(extreme_logits, axis=-1)
    
    print(f"Input logits shape: {extreme_logits.shape}")
    print(f"Softmax probabilities:")
    for i, prob_row in enumerate(probs):
        print(f"  Row {i}: {prob_row}")
        print(f"    Sum: {prob_row.sum():.6f}")
    
    # Check if probabilities sum to 1 and contain no NaN/inf
    sums_close_to_1 = np.allclose(probs.sum(axis=-1), 1.0)
    no_nan_inf = np.all(np.isfinite(probs))
    
    if sums_close_to_1 and no_nan_inf:
        print("PASS: Stable softmax handles extreme values correctly")
        return True
    else:
        print("FAIL: Numerical stability issues detected")
        return False


def test_gradient_accumulator_efficiency():
    """Test efficient gradient accumulation."""
    print("\n" + "=" * 60)
    print("TESTING GRADIENT ACCUMULATION EFFICIENCY")
    print("=" * 60)
    
    # Create mock model parameters
    model_params = {
        'W1': np.random.randn(500, 200),
        'W2': np.random.randn(200, 500),
        'b1': np.random.randn(200),
        'b2': np.random.randn(500)
    }
    
    # Create mock gradients
    gradients = {
        'W1': np.random.randn(500, 200) * 0.01,
        'W2': np.random.randn(200, 500) * 0.01,
        'b1': np.random.randn(200) * 0.01,
        'b2': np.random.randn(500) * 0.01
    }
    
    num_accumulation_steps = 50
    
    # Test old method (creates new arrays)
    print("Testing old gradient accumulation method...")
    start_time = time.time()
    accumulated_old = {}
    for step in range(num_accumulation_steps):
        if not accumulated_old:
            accumulated_old = {name: grad.copy() for name, grad in gradients.items()}
        else:
            for name, grad in gradients.items():
                accumulated_old[name] += grad
    
    # Average
    for name in accumulated_old:
        accumulated_old[name] /= num_accumulation_steps
    
    old_time = time.time() - start_time
    
    # Test new efficient method
    print("Testing efficient gradient accumulator...")
    accumulator = EfficientGradientAccumulator(model_params)
    start_time = time.time()
    
    for step in range(num_accumulation_steps):
        accumulator.accumulate(gradients)
    
    averaged_grads = accumulator.get_averaged_gradients()
    new_time = time.time() - start_time
    
    print(f"Old method time: {old_time:.4f} seconds")
    print(f"New method time: {new_time:.4f} seconds")
    if new_time > 0:
        print(f"Speedup: {old_time/new_time:.2f}x")
    else:
        print("Speedup: Very fast (new method completed in <0.0001s)")
    
    # Verify correctness
    max_diff = 0.0
    for name in accumulated_old:
        diff = np.abs(accumulated_old[name] - averaged_grads[name]).max()
        max_diff = max(max_diff, diff)
    
    print(f"Maximum difference between methods: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("PASS: Efficient gradient accumulation works correctly")
        return True
    else:
        print("FAIL: Gradient accumulation issues detected")
        return False


def test_end_to_end_training_step():
    """Test a complete training step with all fixes."""
    print("\n" + "=" * 60)
    print("TESTING END-TO-END TRAINING STEP")
    print("=" * 60)
    
    try:
        # Load tokenizer
        tokenizer = WordPieceTokenizer.load('tokenizer_8k.pkl')
        
        # Create model and gradient computer
        model = MiniBERT()
        grad_computer = MiniBERTGradients(model)
        
        # Create some text data
        text = "The quick brown fox jumps over the lazy dog"
        input_ids = tokenizer.encode(text)
        input_batch = np.array([input_ids])
        
        print(f"Text: '{text}'")
        print(f"Token IDs: {input_ids}")
        print(f"Batch shape: {input_batch.shape}")
        
        # Forward pass
        start_time = time.time()
        logits, cache = model.forward(input_batch)
        forward_time = time.time() - start_time
        
        print(f"Forward pass time: {forward_time:.4f} seconds")
        print(f"Output logits shape: {logits.shape}")
        
        # Create dummy loss gradient
        grad_logits = np.random.randn(*logits.shape) * 0.01
        
        # Backward pass
        start_time = time.time()
        grad_computer.zero_gradients()
        grad_computer.backward_from_logits(grad_logits, cache)
        backward_time = time.time() - start_time
        
        print(f"Backward pass time: {backward_time:.4f} seconds")
        
        # Check gradient norms
        grad_norms = {}
        for name, grad in grad_computer.gradients.items():
            if grad is not None and np.any(grad):
                grad_norms[name] = np.linalg.norm(grad)
        
        print(f"Gradient norms computed for {len(grad_norms)} parameters")
        
        # Check for NaN or inf gradients
        has_nan_inf = any(not np.all(np.isfinite(grad)) 
                         for grad in grad_computer.gradients.values() 
                         if grad is not None)
        
        if not has_nan_inf and len(grad_norms) > 0:
            print("PASS: End-to-end training step completed successfully")
            return True
        else:
            print("FAIL: Issues detected in end-to-end training")
            return False
            
    except Exception as e:
        print(f"FAIL: Exception in end-to-end test: {e}")
        return False


def run_all_tests():
    """Run all critical fix tests."""
    print("RUNNING ALL CRITICAL BUG FIX TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(test_attention_mask_fix())
    test_results.append(test_token_embedding_gradients_speed())
    test_results.append(test_numerical_stability())
    test_results.append(test_gradient_accumulator_efficiency())
    test_results.append(test_end_to_end_training_step())
    
    # Summary
    print("\n" + "=" * 60) 
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Attention Mask Fix",
        "Token Embedding Gradient Speed",
        "Numerical Stability", 
        "Gradient Accumulation Efficiency",
        "End-to-End Training Step"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("*** ALL CRITICAL FIXES WORKING CORRECTLY! ***")
    else:
        print("*** Some fixes need attention ***")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)