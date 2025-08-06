"""
Integration Test for Complete Mini-BERT Training Pipeline.

Tests the integration of all components:
- MLM masking and loss functions
- AdamW optimizer with learning rate scheduling
- Training step with gradient accumulation
- Memory usage within budget
- Performance characteristics
"""

import numpy as np
import time
import tempfile
import os
from typing import Dict

# Import all our modules
from model import MiniBERT
from gradients import MiniBERTGradients
from tokenizer import WordPieceTokenizer
from mlm import mask_tokens, mlm_cross_entropy
from optimizer import AdamW, LRScheduler
from config import MODEL_CONFIG

def test_end_to_end_training_step():
    """Test complete training step with all components integrated."""
    print("Testing end-to-end training step...")
    
    # Initialize all components
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    
    # Create simple tokenizer for testing
    tokenizer = WordPieceTokenizer(vocab_size=1000)
    
    # Add basic vocabulary
    test_words = ["hello", "world", "this", "is", "a", "test", "the", "and", "of", "to", "in", "for"]
    vocab_id = len(tokenizer.special_tokens)
    for word in test_words:
        tokenizer.vocab[word] = vocab_id
        tokenizer.inverse_vocab[vocab_id] = word
        vocab_id += 1
    
    # Add characters for fallback
    for char in "abcdefghijklmnopqrstuvwxyz .,!?":
        if char not in tokenizer.vocab:
            tokenizer.vocab[char] = vocab_id
            tokenizer.inverse_vocab[vocab_id] = char
            vocab_id += 1
    
    # Initialize optimizer
    optimizer = AdamW(learning_rate=1e-4, weight_decay=0.01)
    scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    # Create test data
    test_sentences = [
        "hello world this is a test",
        "the test is working well and good",
        "this is another test sentence for training",
        "hello world and test again with more words"
    ]
    
    print(f"  Created tokenizer with {len(tokenizer.vocab)} tokens")
    print(f"  Test sentences: {len(test_sentences)}")
    
    # Training loop simulation
    total_loss = 0.0
    total_accuracy = 0.0
    step_times = []
    
    for step in range(10):  # Test 10 steps
        step_start = time.time()
        
        # Create batch from test sentences
        batch_sequences = []
        for sentence in test_sentences:
            token_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=32)
            # Pad to 32 tokens
            if len(token_ids) < 32:
                token_ids.extend([0] * (32 - len(token_ids)))
            batch_sequences.append(token_ids)
        
        batch = np.array(batch_sequences)  # [4, 32]
        
        # Apply MLM masking
        input_ids, target_ids, mask_positions = mask_tokens(
            batch, vocab_size=len(tokenizer.vocab), mask_id=4, p_mask=0.15
        )
        
        # Forward pass
        logits, cache = model.forward(input_ids)
        
        # Compute loss
        loss, accuracy = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
        
        # Backward pass
        grad_computer.zero_gradients()
        
        # Compute gradients (simplified for test)
        B, T, V = logits.shape
        valid_mask = (target_ids != -100)
        num_valid = np.sum(valid_mask)
        
        if num_valid > 0:
            # Softmax gradient
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            grad_logits = softmax_probs.copy()
            valid_positions = np.where(valid_mask)
            valid_targets = target_ids[valid_mask]
            grad_logits[valid_positions[0], valid_positions[1], valid_targets] -= 1.0
            grad_logits /= num_valid
            
            # Backward through model
            grad_computer.backward_from_logits(grad_logits, cache)
        
        # Update learning rate
        scheduler.step(step)
        
        # Optimizer step
        optimizer.step(model.params, grad_computer.gradients)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        total_loss += loss
        total_accuracy += accuracy
        
        if step % 5 == 0:
            print(f"    Step {step}: Loss={loss:.4f}, Acc={accuracy:.3f}, LR={scheduler.get_lr(step):.2e}, Time={step_time*1000:.0f}ms")
    
    # Results
    avg_loss = total_loss / 10
    avg_accuracy = total_accuracy / 10
    avg_step_time = np.mean(step_times) * 1000
    
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average accuracy: {avg_accuracy:.3f}")
    print(f"  Average step time: {avg_step_time:.0f}ms")
    
    # Verify training is working
    final_loss = loss
    initial_loss = total_loss / 10  # Rough approximation
    
    success = True
    if np.isnan(final_loss):
        print("  [FAIL] Loss is NaN")
        success = False
    elif final_loss > 10.0:
        print("  [FAIL] Loss too high (>10)")
        success = False
    elif avg_step_time > 5000:  # 5 seconds per step is too slow
        print("  [FAIL] Step time too slow (>5s)")
        success = False
    else:
        print("  [PASS] End-to-end training step")
    
    return success

def test_gradient_accumulation():
    """Test gradient accumulation functionality."""
    print("Testing gradient accumulation...")
    
    # Initialize components
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    optimizer = AdamW(learning_rate=1e-4)
    
    # Create test batch
    batch_size, seq_len = 4, 16
    input_ids = np.random.randint(5, 1000, (batch_size, seq_len))
    
    # Simulate gradient accumulation over 4 micro-batches
    accumulated_gradients = {}
    accumulation_steps = 4
    
    for micro_step in range(accumulation_steps):
        # Forward pass
        logits, cache = model.forward(input_ids)
        
        # Create dummy targets for loss
        target_ids = np.random.randint(5, 1000, (batch_size, seq_len))
        mask = np.random.rand(batch_size, seq_len) < 0.15
        target_ids[~mask] = -100  # Only compute loss on masked positions
        
        # Compute loss and gradients
        loss, accuracy = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
        
        grad_computer.zero_gradients()
        
        # Simplified gradient computation
        valid_mask = (target_ids != -100)
        num_valid = np.sum(valid_mask)
        
        if num_valid > 0:
            B, T, V = logits.shape
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            grad_logits = softmax_probs.copy()
            valid_positions = np.where(valid_mask)
            valid_targets = target_ids[valid_mask]
            grad_logits[valid_positions[0], valid_positions[1], valid_targets] -= 1.0
            grad_logits /= num_valid
            
            grad_computer.backward_from_logits(grad_logits, cache)
        
        # Accumulate gradients
        if not accumulated_gradients:
            accumulated_gradients = {name: grad.copy() for name, grad in grad_computer.gradients.items()}
        else:
            for name, grad in grad_computer.gradients.items():
                accumulated_gradients[name] += grad
    
    # Average accumulated gradients
    for name in accumulated_gradients:
        accumulated_gradients[name] /= accumulation_steps
    
    # Check gradient magnitudes
    total_grad_norm = 0.0
    for grad in accumulated_gradients.values():
        if np.any(grad):
            total_grad_norm += np.sum(grad ** 2)
    
    total_grad_norm = np.sqrt(total_grad_norm)
    
    print(f"  Accumulated over {accumulation_steps} micro-batches")
    print(f"  Total gradient norm: {total_grad_norm:.2e}")
    
    success = True
    if np.isnan(total_grad_norm):
        print("  [FAIL] Gradient norm is NaN")
        success = False
    elif total_grad_norm == 0.0:
        print("  [FAIL] All gradients are zero")
        success = False
    else:
        print("  [PASS] Gradient accumulation")
    
    return success

def test_memory_efficiency():
    """Test memory usage is within budget."""
    print("Testing memory efficiency...")
    
    # Get initial memory
    import psutil
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**2)
    
    # Initialize model and components
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    optimizer = AdamW()
    
    # Memory after initialization
    after_init_memory = process.memory_info().rss / (1024**2)
    init_overhead = after_init_memory - initial_memory
    
    # Run forward/backward pass
    batch_size, seq_len = 8, 32
    input_ids = np.random.randint(5, 1000, (batch_size, seq_len))
    
    logits, cache = model.forward(input_ids)
    
    # Memory after forward
    after_forward_memory = process.memory_info().rss / (1024**2)
    forward_overhead = after_forward_memory - after_init_memory
    
    # Backward pass
    target_ids = np.random.randint(5, 1000, (batch_size, seq_len))
    mask = np.random.rand(batch_size, seq_len) < 0.15
    target_ids[~mask] = -100
    
    loss, accuracy = mlm_cross_entropy(logits, target_ids)
    grad_computer.zero_gradients()
    
    # Simplified backward
    valid_mask = (target_ids != -100)
    num_valid = np.sum(valid_mask)
    
    if num_valid > 0:
        B, T, V = logits.shape
        grad_logits = np.random.randn(B, T, V) * 0.01  # Dummy gradients
        grad_computer.backward_from_logits(grad_logits, cache)
    
    # Memory after backward
    peak_memory = process.memory_info().rss / (1024**2)
    total_overhead = peak_memory - initial_memory
    
    print(f"  Initial memory: {initial_memory:.1f} MB")
    print(f"  After initialization: {after_init_memory:.1f} MB (+{init_overhead:.1f} MB)")
    print(f"  After forward pass: {after_forward_memory:.1f} MB (+{forward_overhead:.1f} MB)")
    print(f"  Peak memory: {peak_memory:.1f} MB")
    print(f"  Total overhead: {total_overhead:.1f} MB")
    
    # Check against target (2GB = 2048 MB)
    target_memory_mb = 2048
    within_budget = peak_memory < target_memory_mb
    utilization = 100 * peak_memory / target_memory_mb
    
    print(f"  Target: <{target_memory_mb} MB")
    print(f"  Utilization: {utilization:.1f}%")
    
    if within_budget:
        print("  [PASS] Memory within budget")
        return True
    else:
        print("  [FAIL] Memory exceeds budget")
        return False

def test_overfit_capability():
    """Test that model can overfit a single batch (debug functionality)."""
    print("Testing overfitting capability...")
    
    # Initialize components
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    optimizer = AdamW(learning_rate=1e-3)  # Higher LR for faster overfitting
    
    # Create small fixed batch
    batch_size, seq_len = 2, 16
    fixed_batch = np.random.randint(5, 100, (batch_size, seq_len))
    
    print(f"  Fixed batch shape: {fixed_batch.shape}")
    
    losses = []
    
    # Train on fixed batch for multiple steps
    for step in range(50):  # 50 steps should be enough to overfit
        # Apply MLM masking (consistent masking for overfitting)
        np.random.seed(42)  # Fixed seed for consistent masking
        input_ids, target_ids, mask_positions = mask_tokens(
            fixed_batch, vocab_size=1000, mask_id=4, p_mask=0.15
        )
        
        # Forward pass
        logits, cache = model.forward(input_ids)
        
        # Compute loss
        loss, accuracy = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
        losses.append(loss)
        
        # Backward pass
        grad_computer.zero_gradients()
        
        valid_mask = (target_ids != -100)
        num_valid = np.sum(valid_mask)
        
        if num_valid > 0:
            B, T, V = logits.shape
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            grad_logits = softmax_probs.copy()
            valid_positions = np.where(valid_mask)
            valid_targets = target_ids[valid_mask]
            grad_logits[valid_positions[0], valid_positions[1], valid_targets] -= 1.0
            grad_logits /= num_valid
            
            grad_computer.backward_from_logits(grad_logits, cache)
        
        # Optimizer step
        optimizer.step(model.params, grad_computer.gradients)
        
        if step % 10 == 0:
            print(f"    Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.3f}")
    
    # Check if loss decreased significantly
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.4f}")
    
    # For overfitting, we expect significant loss reduction
    success = loss_reduction > 1.0 and final_loss < 2.0
    
    if success:
        print("  [PASS] Model can overfit (loss reduced significantly)")
    else:
        print("  [FAIL] Model failed to overfit")
    
    return success

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Mini-BERT Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("End-to-End Training Step", test_end_to_end_training_step),
        ("Gradient Accumulation", test_gradient_accumulation),
        ("Memory Efficiency", test_memory_efficiency),
        ("Overfitting Capability", test_overfit_capability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  [ERROR] Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] All integration tests passed!")
        print("Mini-BERT training pipeline is ready for production use.")
    else:
        print(f"\n[WARNING] {total-passed} test(s) failed.")
        print("Please review failing tests before production use.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)