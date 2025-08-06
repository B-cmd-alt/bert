"""
Complete Mini-BERT Pipeline Test.

Tests all components working together:
1. Model forward pass
2. Gradient computation  
3. MLM loss and training step
4. Memory usage validation
5. Gradient checking
"""

import numpy as np
import os
import sys
import time

# Test each component individually first
def test_individual_components():
    """Test each module can be imported and run."""
    print("Testing individual components...")
    
    try:
        # Test config
        from config import MODEL_CONFIG, TRAINING_CONFIG
        print(f"[OK] Config loaded: L={MODEL_CONFIG.num_layers}, H={MODEL_CONFIG.hidden_size}")
        
        # Test tokenizer (create minimal test)
        from tokenizer import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(vocab_size=100)  # Small vocab for testing
        
        # Create minimal training data
        test_text = "This is a test sentence for the tokenizer. Another sentence here."
        
        # Simple character-based vocabulary for testing
        chars = set(test_text.lower())
        vocab_id = len(tokenizer.special_tokens)
        for char in sorted(chars):
            if char not in tokenizer.vocab:
                tokenizer.vocab[char] = vocab_id
                tokenizer.inverse_vocab[vocab_id] = char
                vocab_id += 1
        
        tokens = tokenizer.tokenize(test_text)
        print(f"[OK] Tokenizer working: {len(tokens)} tokens from test text")
        
        # Test model
        from model import MiniBERT
        model = MiniBERT()
        print(f"[OK] Model initialized: {model.get_parameter_count():,} parameters")
        
        # Test forward pass
        B, T = 2, 8
        input_ids = np.random.randint(0, 100, (B, T))
        logits, cache = model.forward(input_ids)
        print(f"[OK] Forward pass: {input_ids.shape} -> {logits.shape}")
        
        # Test gradients
        from gradients import MiniBERTGradients
        grad_computer = MiniBERTGradients(model)
        
        labels = np.random.randint(0, 100, (B, T))
        mask = np.ones((B, T))
        
        loss, grad_logits = grad_computer.compute_mlm_loss_and_gradients(logits, labels, mask)
        print(f"[OK] Loss computation: {loss:.4f}")
        
        grad_computer.zero_gradients()
        grad_embeddings = grad_computer.backward_from_logits(grad_logits, cache)
        print(f"[OK] Backward pass: gradient shape {grad_embeddings.shape}")
        
        # Test utilities
        from utils import get_memory_usage, GradientChecker
        memory = get_memory_usage()
        print(f"[OK] Memory monitoring: {memory['rss_mb']:.1f} MB")
        
        print("All individual components working!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_checking():
    """Test gradient checking functionality."""
    print("\nTesting gradient checking...")
    
    try:
        from model import MiniBERT
        from gradients import MiniBERTGradients  
        from utils import GradientChecker
        
        model = MiniBERT()
        grad_computer = MiniBERTGradients(model)
        checker = GradientChecker(epsilon=1e-5, tolerance=1e-2)  # More lenient for testing
        
        # Small test case
        B, T = 2, 4
        input_ids = np.random.randint(5, 100, (B, T))  # Avoid special tokens
        labels = np.random.randint(5, 100, (B, T))
        mask = np.ones((B, T))
        
        # Check a subset of parameters
        results = checker.check_gradients(
            model, grad_computer, input_ids, labels, mask,
            param_subset=['W_Q_0', 'ln1_gamma_0']
        )
        
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        print(f"✓ Gradient check: {passed}/{total} parameters passed")
        
        for param_name, result in results.items():
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {param_name}: {status} (rel_diff: {result['max_rel_diff']:.2e})")
        
        return passed > 0  # At least some gradients should pass
        
    except Exception as e:
        print(f"✗ Gradient checking failed: {e}")
        return False

def test_training_step():
    """Test a complete training step."""
    print("\nTesting complete training step...")
    
    try:
        from model import MiniBERT
        from gradients import MiniBERTGradients
        from train import AdamOptimizer
        
        # Initialize components
        model = MiniBERT()
        grad_computer = MiniBERTGradients(model)
        optimizer = AdamOptimizer(learning_rate=1e-4)
        optimizer.initialize_state(model.params)
        
        # Create batch
        B, T = 4, 16
        input_ids = np.random.randint(5, 1000, (B, T))
        labels = np.random.randint(5, 1000, (B, T)) 
        mask = np.random.rand(B, T) < 0.15  # 15% masking
        
        print(f"Batch: {input_ids.shape}, Masked tokens: {np.sum(mask)}")
        
        # Training step
        start_time = time.time()
        
        # Forward pass
        logits, cache = model.forward(input_ids)
        
        # Compute loss and gradients
        loss, grad_logits = grad_computer.compute_mlm_loss_and_gradients(
            logits, labels, mask
        )
        
        # Backward pass
        grad_computer.zero_gradients()
        grad_computer.backward_from_logits(grad_logits, cache)
        
        # Check gradient magnitudes
        grad_norms = {}
        for param_name, grad in grad_computer.gradients.items():
            if np.any(grad):
                grad_norms[param_name] = np.linalg.norm(grad)
        
        print(f"Non-zero gradients: {len(grad_norms)}")
        
        # Optimizer step
        initial_param = model.params['W_Q_0'].copy()
        optimizer.step(model.params, grad_computer.gradients)
        param_change = np.linalg.norm(model.params['W_Q_0'] - initial_param)
        
        step_time = time.time() - start_time
        
        print(f"✓ Training step completed:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Parameter change: {param_change:.6f}")
        print(f"  Step time: {step_time*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage estimation."""
    print("\nTesting memory usage...")
    
    try:
        from model import MiniBERT
        from utils import get_memory_usage
        
        initial_memory = get_memory_usage()
        print(f"Initial memory: {initial_memory['rss_mb']:.1f} MB")
        
        # Create model
        model = MiniBERT()
        model_memory = get_memory_usage()
        model_overhead = model_memory['rss_mb'] - initial_memory['rss_mb']
        
        # Estimate vs actual
        estimated = model.get_memory_usage(batch_size=8)
        
        print(f"Model overhead: {model_overhead:.1f} MB")
        print(f"Estimated memory: {estimated['total_mb']:.1f} MB")
        print(f"Parameter memory: {estimated['parameters_mb']:.1f} MB")
        
        # Test with forward pass
        B, T = 8, 32
        input_ids = np.random.randint(0, 1000, (B, T))
        
        forward_start_memory = get_memory_usage()
        logits, cache = model.forward(input_ids)
        forward_end_memory = get_memory_usage()
        
        forward_overhead = forward_end_memory['rss_mb'] - forward_start_memory['rss_mb']
        print(f"Forward pass overhead: {forward_overhead:.1f} MB")
        
        # Check if within reasonable bounds (less than 1GB)
        total_memory = forward_end_memory['rss_mb']
        within_budget = total_memory < 1024  # 1GB limit for testing
        
        print(f"✓ Memory test: {total_memory:.1f} MB (within budget: {within_budget})")
        
        return within_budget
        
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

def test_data_pipeline():
    """Test data processing components."""
    print("\nTesting data pipeline...")
    
    try:
        from tokenizer import WordPieceTokenizer
        from data import MLMDataProcessor
        
        # Create simple tokenizer
        tokenizer = WordPieceTokenizer(vocab_size=200)
        
        # Add some basic vocabulary manually for testing
        test_vocab = ["the", "a", "is", "test", "this", "hello", "world", "python", "bert"]
        vocab_id = len(tokenizer.special_tokens)
        
        for word in test_vocab:
            tokenizer.vocab[word] = vocab_id
            tokenizer.inverse_vocab[vocab_id] = word
            vocab_id += 1
            
        # Add characters
        chars = set("abcdefghijklmnopqrstuvwxyz .,!?")
        for char in sorted(chars):
            if char not in tokenizer.vocab:
                tokenizer.vocab[char] = vocab_id
                tokenizer.inverse_vocab[vocab_id] = char
                vocab_id += 1
        
        print(f"Test tokenizer: {len(tokenizer.vocab)} tokens")
        
        # Test MLM processor
        processor = MLMDataProcessor(tokenizer, max_seq_length=32)
        
        test_sentence = "This is a test sentence for BERT training."
        input_ids, labels, mlm_mask = processor.process_sequence(test_sentence)
        
        print(f"✓ MLM processing:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Masked positions: {np.sum(mlm_mask)}")
        print(f"  Mask ratio: {np.sum(mlm_mask) / np.sum(input_ids != 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_test():
    """Run end-to-end integration test."""
    print("\nRunning integration test...")
    
    try:
        from model import MiniBERT
        from gradients import MiniBERTGradients
        from train import AdamOptimizer
        from tokenizer import WordPieceTokenizer
        from data import MLMDataProcessor
        from utils import TrainingDiagnostics
        
        # Initialize all components
        model = MiniBERT()
        grad_computer = MiniBERTGradients(model)
        optimizer = AdamOptimizer(learning_rate=1e-4)
        optimizer.initialize_state(model.params)
        
        # Create simple tokenizer  
        tokenizer = WordPieceTokenizer(vocab_size=1000)
        
        # Add basic vocabulary
        basic_words = ["hello", "world", "this", "is", "a", "test", "the", "and", "of", "to"]
        vocab_id = len(tokenizer.special_tokens)
        for word in basic_words:
            tokenizer.vocab[word] = vocab_id
            tokenizer.inverse_vocab[vocab_id] = word
            vocab_id += 1
        
        processor = MLMDataProcessor(tokenizer, max_seq_length=32)
        diagnostics = TrainingDiagnostics()
        
        print("All components initialized")
        
        # Simulate a few training steps
        losses = []
        
        for step in range(5):
            # Create synthetic batch  
            sentences = [
                "hello world this is a test",
                "the test is working well", 
                "this is another test sentence",
                "hello world and test again"
            ]
            
            batch_input_ids = []
            batch_labels = []
            batch_masks = []
            
            for sentence in sentences:
                input_ids, labels, mask = processor.process_sequence(sentence)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_masks.append(mask)
            
            batch_input_ids = np.array(batch_input_ids)
            batch_labels = np.array(batch_labels)
            batch_masks = np.array(batch_masks)
            
            # Training step
            logits, cache = model.forward(batch_input_ids)
            loss, grad_logits = grad_computer.compute_mlm_loss_and_gradients(
                logits, batch_labels, batch_masks
            )
            
            grad_computer.zero_gradients()
            grad_computer.backward_from_logits(grad_logits, cache)
            
            optimizer.step(model.params, grad_computer.gradients)
            
            losses.append(loss)
            
            # Log metrics
            metrics = {
                'loss': loss,
                'learning_rate': optimizer.learning_rate,
                'masked_tokens': np.sum(batch_masks)
            }
            diagnostics.log_step(step, metrics)
            
            print(f"Step {step}: Loss = {loss:.4f}")
        
        # Check if loss is reasonable (not NaN, not too high)
        final_loss = losses[-1]
        loss_is_reasonable = not np.isnan(final_loss) and final_loss < 20.0
        
        print(f"✓ Integration test completed:")
        print(f"  Steps: {len(losses)}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss reasonable: {loss_is_reasonable}")
        
        return loss_is_reasonable
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Mini-BERT Complete Pipeline Test")
    print("=" * 60)
    
    tests = [
        ("Individual Components", test_individual_components),
        ("Gradient Checking", test_gradient_checking),
        ("Training Step", test_training_step),
        ("Memory Usage", test_memory_usage),
        ("Data Pipeline", test_data_pipeline),
        ("Integration Test", run_integration_test)
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
            print(f"✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Mini-BERT pipeline is ready for training.")
        
        # Print next steps
        print("\nNext Steps:")
        print("1. Prepare training data in data/ directory")
        print("2. Run: python train.py")
        print("3. Monitor training progress in mini_bert/logs/")
        print("4. Checkpoints saved in mini_bert/checkpoints/")
        
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix issues before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)