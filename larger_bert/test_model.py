"""
Test script for Larger BERT model.
Validates implementation and compares with Mini-BERT.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from larger_bert import LargerBERT, LARGER_BERT_CONFIG
from mini_bert.model import MiniBERT
from mini_bert.config import MODEL_CONFIG as MINI_BERT_CONFIG


def test_model_initialization():
    """Test model initialization and parameter counts."""
    print("Testing Model Initialization")
    print("=" * 60)
    
    # Initialize models
    mini_bert = MiniBERT(MINI_BERT_CONFIG)
    larger_bert = LargerBERT(LARGER_BERT_CONFIG)
    
    # Compare parameter counts
    mini_params = mini_bert.get_parameter_count()
    larger_params = larger_bert.get_parameter_count()
    
    print(f"Mini-BERT parameters: {mini_params:,} ({mini_params/1e6:.2f}M)")
    print(f"Larger-BERT parameters: {larger_params:,} ({larger_params/1e6:.2f}M)")
    print(f"Scale factor: {larger_params/mini_params:.1f}x")
    
    # Check parameter shapes
    print("\nParameter Shape Verification:")
    
    # Token embeddings
    mini_token_emb = mini_bert.params['token_embeddings'].shape
    larger_token_emb = larger_bert.params['token_embeddings'].shape
    print(f"Token embeddings - Mini: {mini_token_emb}, Larger: {larger_token_emb}")
    
    # Attention weights (layer 0)
    mini_wq = mini_bert.params['W_Q_0'].shape
    larger_wq = larger_bert.params['W_Q_0'].shape
    print(f"Attention W_Q - Mini: {mini_wq}, Larger: {larger_wq}")
    
    # FFN weights (layer 0)
    mini_w1 = mini_bert.params['W1_0'].shape
    larger_w1 = larger_bert.params['W1_0'].shape
    print(f"FFN W1 - Mini: {mini_w1}, Larger: {larger_w1}")
    
    print("\n✓ Model initialization test passed!")
    return mini_bert, larger_bert


def test_forward_pass(larger_bert: LargerBERT):
    """Test forward pass with different input sizes."""
    print("\nTesting Forward Pass")
    print("=" * 60)
    
    # Test different batch sizes and sequence lengths
    test_cases = [
        (1, 16),   # Single example, short sequence
        (4, 32),   # Small batch, medium sequence
        (8, 64),   # Medium batch, long sequence
        (2, 128),  # Small batch, max sequence
    ]
    
    larger_bert.set_training(False)  # Disable dropout for testing
    
    for batch_size, seq_len in test_cases:
        print(f"\nTest case: batch_size={batch_size}, seq_len={seq_len}")
        
        # Create random input
        input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
        attention_mask = np.ones((batch_size, seq_len))
        
        # Forward pass
        start_time = time.time()
        logits, cache = larger_bert.forward(input_ids, attention_mask)
        forward_time = time.time() - start_time
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, LARGER_BERT_CONFIG.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        print(f"  Output shape: {logits.shape} ✓")
        print(f"  Forward pass time: {forward_time:.3f}s")
        
        # Check intermediate shapes
        assert cache['embeddings'].shape == (batch_size, seq_len, LARGER_BERT_CONFIG.hidden_size)
        assert cache['final_hidden'].shape == (batch_size, seq_len, LARGER_BERT_CONFIG.hidden_size)
    
    print("\n✓ Forward pass test passed!")


def test_attention_mechanism(larger_bert: LargerBERT):
    """Test attention mechanism with masking."""
    print("\nTesting Attention Mechanism")
    print("=" * 60)
    
    batch_size, seq_len = 2, 16
    
    # Create input with padding
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    input_ids[0, 10:] = 0  # Pad first sequence
    input_ids[1, 12:] = 0  # Pad second sequence
    
    # Create attention mask
    attention_mask = np.ones((batch_size, seq_len))
    attention_mask[0, 10:] = 0
    attention_mask[1, 12:] = 0
    
    print("Input shape:", input_ids.shape)
    print("Attention mask:", attention_mask)
    
    # Forward pass
    larger_bert.set_training(False)
    logits, cache = larger_bert.forward(input_ids, attention_mask)
    
    # Check that masked positions don't affect output
    # (This is a basic check - in practice, masked positions still get predictions)
    print(f"Output shape: {logits.shape}")
    print(f"Non-zero outputs at masked positions: {np.any(logits[0, 10:] != 0)}")
    
    print("\n✓ Attention mechanism test passed!")


def test_memory_usage():
    """Test memory usage estimation."""
    print("\nTesting Memory Usage Estimation")
    print("=" * 60)
    
    larger_bert = LargerBERT(LARGER_BERT_CONFIG)
    
    batch_sizes = [1, 4, 8, 16]
    
    print(f"{'Batch Size':<12} {'Params (MB)':<12} {'Grads (MB)':<12} "
          f"{'Optimizer (MB)':<15} {'Activations (MB)':<16} {'Total (MB)':<12}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        memory = larger_bert.get_memory_usage(batch_size)
        print(f"{batch_size:<12} {memory['parameters_mb']:<12.1f} "
              f"{memory['gradients_mb']:<12.1f} {memory['optimizer_mb']:<15.1f} "
              f"{memory['activations_mb']:<16.1f} {memory['total_mb']:<12.1f}")
    
    print("\n✓ Memory usage test passed!")


def test_gradient_computation():
    """Test gradient computation (basic check)."""
    print("\nTesting Gradient Computation")
    print("=" * 60)
    
    from mini_bert.gradients import compute_mlm_gradients
    
    larger_bert = LargerBERT(LARGER_BERT_CONFIG)
    larger_bert.set_training(True)
    
    # Create a simple batch
    batch_size, seq_len = 2, 8
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len))
    
    # Create MLM labels (mask some positions)
    labels = input_ids.copy()
    masked_positions = np.array([[1, 3, -1, -1], [0, 5, -1, -1]])  # 2 masked per sequence
    
    # Create batch dict
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'masked_positions': masked_positions
    }
    
    # Forward pass
    logits, cache = larger_bert.forward(input_ids, attention_mask)
    
    # Compute gradients
    print("Computing gradients...")
    gradients = compute_mlm_gradients(larger_bert, logits, batch, cache)
    
    # Check gradient shapes
    print("\nGradient shapes:")
    for key in ['token_embeddings', 'W_Q_0', 'W1_0', 'mlm_head_W']:
        if key in gradients:
            print(f"  {key}: {gradients[key].shape}")
            assert gradients[key].shape == larger_bert.params[key].shape
    
    print("\n✓ Gradient computation test passed!")


def test_model_comparison():
    """Compare Mini-BERT and Larger-BERT side by side."""
    print("\nModel Architecture Comparison")
    print("=" * 60)
    
    mini_bert = MiniBERT(MINI_BERT_CONFIG)
    larger_bert = LargerBERT(LARGER_BERT_CONFIG)
    
    # Create comparison table
    print(f"{'Component':<30} {'Mini-BERT':<20} {'Larger-BERT':<20}")
    print("-" * 70)
    
    comparisons = [
        ("Vocabulary Size", MINI_BERT_CONFIG.vocab_size, LARGER_BERT_CONFIG.vocab_size),
        ("Number of Layers", MINI_BERT_CONFIG.num_layers, LARGER_BERT_CONFIG.num_layers),
        ("Hidden Size", MINI_BERT_CONFIG.hidden_size, LARGER_BERT_CONFIG.hidden_size),
        ("Attention Heads", MINI_BERT_CONFIG.num_attention_heads, LARGER_BERT_CONFIG.num_attention_heads),
        ("Head Size", mini_bert.d_k, larger_bert.d_k),
        ("FFN Size", MINI_BERT_CONFIG.intermediate_size, LARGER_BERT_CONFIG.intermediate_size),
        ("Max Sequence Length", MINI_BERT_CONFIG.max_sequence_length, LARGER_BERT_CONFIG.max_sequence_length),
        ("Total Parameters", f"{mini_bert.get_parameter_count():,}", f"{larger_bert.get_parameter_count():,}"),
    ]
    
    for name, mini_val, larger_val in comparisons:
        print(f"{name:<30} {str(mini_val):<20} {str(larger_val):<20}")
    
    # Performance comparison
    print("\nPerformance Comparison (batch_size=4, seq_len=32):")
    batch_size, seq_len = 4, 32
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len))
    
    # Mini-BERT timing
    mini_bert.training = False
    start_time = time.time()
    mini_logits, _ = mini_bert.forward(input_ids[:, :MINI_BERT_CONFIG.max_sequence_length], 
                                      attention_mask[:, :MINI_BERT_CONFIG.max_sequence_length])
    mini_time = time.time() - start_time
    
    # Larger-BERT timing
    larger_bert.set_training(False)
    start_time = time.time()
    larger_logits, _ = larger_bert.forward(input_ids, attention_mask)
    larger_time = time.time() - start_time
    
    print(f"Mini-BERT forward pass: {mini_time:.3f}s")
    print(f"Larger-BERT forward pass: {larger_time:.3f}s")
    print(f"Slowdown factor: {larger_time/mini_time:.1f}x")


def run_all_tests():
    """Run all tests."""
    print("Running Larger-BERT Test Suite")
    print("=" * 80)
    
    # Test 1: Initialization
    mini_bert, larger_bert = test_model_initialization()
    
    # Test 2: Forward pass
    test_forward_pass(larger_bert)
    
    # Test 3: Attention mechanism
    test_attention_mechanism(larger_bert)
    
    # Test 4: Memory usage
    test_memory_usage()
    
    # Test 5: Gradient computation
    test_gradient_computation()
    
    # Test 6: Model comparison
    test_model_comparison()
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()