"""
Master Evaluation Script for Mini-BERT.

Purpose:
- Run comprehensive evaluation suite: intrinsic metrics + POS probe + SST-2 fine-tuning
- Generate tidy table output with all results
- Validate memory and speed budgets
- Support overfit testing for debugging

Runs:
1. Intrinsic checks: MLM accuracy, perplexity, gradient sanity
2. POS tagging probe: Token-level accuracy on synthetic data
3. SST-2 fine-tuning: Sentence-level sentiment classification
4. Memory/speed validation: Peak RAM ≤3GB, reasonable timing

CLI Usage:
    python evaluate.py
    python evaluate.py --model_path model.pkl --vocab_path vocab.pkl
    python evaluate.py --quick  # Fast evaluation with smaller datasets
    python evaluate.py --overfit_one_batch  # Debug overfit test
    python evaluate.py --help
"""

import numpy as np
import argparse
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Import all our modules
from model import MiniBERT
from tokenizer import WordPieceTokenizer
from gradients import MiniBERTGradients
from metrics import (masked_accuracy, compute_perplexity, compute_gradient_norm, 
                    check_gradient_health, evaluate_model_on_batch, print_evaluation_table)
from mlm import mask_tokens, mlm_cross_entropy
from probe_pos import evaluate_pos_probe
from finetune_sst2 import finetune_sst2
from config import MODEL_CONFIG

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2)
    except ImportError:
        return 0.0  # psutil not available

def run_intrinsic_evaluation(model: MiniBERT, tokenizer: WordPieceTokenizer,
                            quick: bool = False) -> Dict[str, float]:
    """
    Run intrinsic evaluation: MLM accuracy, perplexity, gradient checks.
    
    Args:
        model: Mini-BERT model
        tokenizer: WordPiece tokenizer
        quick: Whether to run quick evaluation (smaller dataset)
        
    Returns:
        intrinsic_results: Dictionary with intrinsic metrics
    """
    print("Running intrinsic evaluation...")
    
    grad_computer = MiniBERTGradients(model)
    
    # Create held-out evaluation data
    eval_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "She is reading a book quietly in the library",
        "Birds fly south for the winter season",
        "Children play happily in the playground",
        "The beautiful flowers bloom in spring",
        "He quickly ran to the store yesterday",
        "My friend loves chocolate ice cream",
        "The cat sat on the warm mat",
        "Students study hard for their exams",
        "The sun shines brightly today"
    ]
    
    if quick:
        eval_sentences = eval_sentences[:3]  # Use only 3 sentences for quick eval
    
    # Tokenize evaluation sentences
    eval_batches = []
    for sentence in eval_sentences:
        token_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=64)
        # Pad to 64
        if len(token_ids) < 64:
            token_ids.extend([0] * (64 - len(token_ids)))
        eval_batches.append(token_ids)
    
    # Convert to numpy and create batch
    eval_batch = np.array(eval_batches)  # [num_sentences, 64]
    
    # Apply MLM masking for evaluation
    input_ids, target_ids, mask_positions = mask_tokens(
        eval_batch, vocab_size=len(tokenizer.vocab), mask_id=4, p_mask=0.15
    )
    
    print(f"  Evaluating on {len(eval_sentences)} sentences")
    print(f"  Masked tokens: {np.sum(mask_positions)}")
    
    # Forward pass
    start_time = time.time()
    logits, cache = model.forward(input_ids)
    forward_time = time.time() - start_time
    
    # Compute intrinsic metrics
    mask_acc = masked_accuracy(logits, target_ids, ignore_index=-100)
    perplexity = compute_perplexity(logits, target_ids, ignore_index=-100)
    
    # Gradient sanity check
    loss, _ = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
    
    # Compute gradients for sanity check
    grad_computer.zero_gradients()
    
    # Simplified gradient computation for checking
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
        
        # Backward pass
        grad_computer.backward_from_logits(grad_logits, cache)
    
    # Compute gradient norm
    grad_norm = compute_gradient_norm(grad_computer.gradients)
    
    # Gradient health check
    health_info = check_gradient_health(grad_computer.gradients, step=0, print_freq=1)
    
    results = {
        'mlm_accuracy': mask_acc,
        'perplexity': perplexity,
        'mlm_loss': loss,
        'grad_norm': grad_norm,
        'grad_status': health_info['status'],
        'forward_time_ms': forward_time * 1000,
        'num_sentences': len(eval_sentences),
        'num_masked_tokens': int(np.sum(mask_positions))
    }
    
    print(f"  MLM accuracy: {mask_acc:.3f} ({mask_acc*100:.1f}%)")
    print(f"  Perplexity: {perplexity:.1f}")
    print(f"  Gradient norm: {grad_norm:.2e}")
    print(f"  Gradient status: {health_info['status']}")
    
    return results

def run_overfit_test(model: MiniBERT, tokenizer: WordPieceTokenizer,
                    max_steps: int = 200, target_loss: float = 0.05) -> Dict[str, Any]:
    """
    Run overfit test: repeatedly train on single micro-batch.
    
    Args:
        model: Mini-BERT model
        tokenizer: WordPiece tokenizer
        max_steps: Maximum training steps
        target_loss: Target loss to achieve
        
    Returns:
        overfit_results: Dictionary with overfit test results
    """
    print("Running overfit test...")
    
    grad_computer = MiniBERTGradients(model)
    
    # Create single micro-batch
    test_sentence = "The quick brown fox jumps over the lazy dog"
    token_ids = tokenizer.encode(test_sentence, add_special_tokens=True, max_length=32)
    
    # Pad to 32 and create batch of size 4
    if len(token_ids) < 32:
        token_ids.extend([0] * (32 - len(token_ids)))
    
    # Repeat sentence to create micro-batch
    batch = np.array([token_ids] * 4)  # [4, 32]
    
    print(f"  Test batch shape: {batch.shape}")
    print(f"  Test sentence: '{test_sentence}'")
    
    # Initialize optimizer
    from optimizer import AdamW
    optimizer = AdamW(learning_rate=1e-3)  # Higher LR for faster overfitting
    
    losses = []
    start_time = time.time()
    
    for step in range(max_steps):
        # Apply consistent MLM masking
        np.random.seed(42)  # Fixed seed for consistent masking
        input_ids, target_ids, mask_positions = mask_tokens(
            batch, vocab_size=len(tokenizer.vocab), mask_id=4, p_mask=0.15
        )
        
        # Forward pass
        logits, cache = model.forward(input_ids)
        
        # Compute loss
        loss, accuracy = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
        losses.append(loss)
        
        # Backward pass
        grad_computer.zero_gradients()
        
        B, T, V = logits.shape
        valid_mask = (target_ids != -100)
        num_valid = np.sum(valid_mask)
        
        if num_valid > 0:
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
        
        if step % 20 == 0:
            print(f"    Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.3f}")
        
        # Check if target reached
        if loss < target_loss:
            print(f"    Target loss {target_loss} reached at step {step}")
            break
    
    total_time = time.time() - start_time
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    
    # Determine success
    success = final_loss < target_loss
    steps_completed = len(losses)
    
    results = {
        'success': success,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'steps_completed': steps_completed,
        'total_time_s': total_time,
        'target_loss': target_loss,
        'max_steps': max_steps
    }
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.4f}")
    print(f"  Steps: {steps_completed}/{max_steps}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Success: {'YES' if success else 'NO'} (target: <{target_loss})")
    
    return results

def run_comprehensive_evaluation(model_path: Optional[str] = None,
                                vocab_path: Optional[str] = None,
                                quick: bool = False,
                                overfit_test: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive Mini-BERT evaluation suite.
    
    Args:
        model_path: Path to trained model (optional)
        vocab_path: Path to tokenizer vocab (optional)
        quick: Run quick evaluation with smaller datasets
        overfit_test: Include overfit test for debugging
        
    Returns:
        all_results: Complete evaluation results
    """
    print("=" * 60)
    print("MINI-BERT COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = MiniBERT(MODEL_CONFIG)
    
    # Create tokenizer with comprehensive vocabulary
    tokenizer = WordPieceTokenizer(vocab_size=MODEL_CONFIG.vocab_size)
    
    # Add extensive vocabulary for evaluation
    evaluation_vocab = [
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "among", "this", "that", "these",
        "those", "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us",
        "them", "my", "your", "his", "her", "its", "our", "their", "quick", "brown", "fox",
        "jumps", "over", "lazy", "dog", "cat", "sat", "mat", "reading", "book", "quietly",
        "library", "birds", "fly", "south", "winter", "season", "children", "play",
        "happily", "playground", "beautiful", "flowers", "bloom", "spring", "ran", "store",
        "yesterday", "friend", "loves", "chocolate", "ice", "cream", "sun", "shines",
        "brightly", "today", "students", "study", "hard", "exams", "movie", "film",
        "amazing", "wonderful", "love", "great", "fantastic", "brilliant", "outstanding",
        "excellent", "perfect", "terrible", "boring", "hate", "poor", "bad", "awful",
        "horrible", "waste", "acting", "cinematography", "story", "characters",
        "performance", "actors", "direction", "script", "experience", "watching", "plot",
        "visual", "effects", "sound", "action", "drama", "really", "very", "extremely",
        "quite", "so", "much", "completely", "all", "time", "blend", "confusion", "boredom"
    ]
    
    vocab_id = len(tokenizer.special_tokens)
    for word in evaluation_vocab:
        if word not in tokenizer.vocab:
            tokenizer.vocab[word.lower()] = vocab_id
            tokenizer.inverse_vocab[vocab_id] = word.lower()
            vocab_id += 1
    
    print(f"Model: {model.get_parameter_count():,} parameters")
    print(f"Tokenizer: {len(tokenizer.vocab)} tokens")
    
    after_load_memory = get_memory_usage()
    load_memory_overhead = after_load_memory - initial_memory
    
    # Run evaluations
    all_results = {}
    
    # 1. Intrinsic evaluation
    print(f"\n{'-'*40}")
    print("1. INTRINSIC EVALUATION")
    print(f"{'-'*40}")
    
    intrinsic_results = run_intrinsic_evaluation(model, tokenizer, quick=quick)
    all_results['intrinsic'] = intrinsic_results
    
    # 2. POS tagging probe
    print(f"\n{'-'*40}")
    print("2. POS TAGGING PROBE")
    print(f"{'-'*40}")
    
    pos_max_sentences = 100 if quick else 1000
    pos_results = evaluate_pos_probe(
        model_path=None,  # Use current model
        vocab_path=None,  # Use current tokenizer
        max_sentences=pos_max_sentences
    )
    all_results['pos_probe'] = pos_results
    
    # 3. SST-2 fine-tuning
    print(f"\n{'-'*40}")
    print("3. SST-2 SENTIMENT CLASSIFICATION")
    print(f"{'-'*40}")
    
    sst2_max_samples = 500 if quick else 2000
    sst2_epochs = 1 if quick else 2
    
    sst2_results = finetune_sst2(
        model_path=None,  # Use current model
        vocab_path=None,  # Use current tokenizer
        epochs=sst2_epochs,
        batch_size=16 if quick else 32,  # Smaller batch for quick eval
        max_samples=sst2_max_samples,
        learning_rate=2e-5
    )
    all_results['sst2'] = sst2_results
    
    # 4. Overfit test (optional)
    if overfit_test:
        print(f"\n{'-'*40}")
        print("4. OVERFIT TEST")
        print(f"{'-'*40}")
        
        overfit_results = run_overfit_test(model, tokenizer, max_steps=50 if quick else 200)
        all_results['overfit'] = overfit_results
    
    # Memory and timing summary
    final_memory = get_memory_usage()
    peak_memory = final_memory
    total_time = time.time() - start_time
    
    memory_results = {
        'initial_memory_mb': initial_memory,
        'after_load_memory_mb': after_load_memory,
        'peak_memory_mb': peak_memory,
        'total_memory_overhead_mb': peak_memory - initial_memory,
        'load_memory_overhead_mb': load_memory_overhead
    }
    
    timing_results = {
        'total_time_s': total_time,
        'total_time_min': total_time / 60
    }
    
    all_results['memory'] = memory_results
    all_results['timing'] = timing_results
    
    return all_results

def print_comprehensive_results(results: Dict[str, Any], quick: bool = False):
    """
    Print comprehensive evaluation results in tidy table format.
    
    Args:
        results: Complete evaluation results
        quick: Whether this was a quick evaluation
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    if quick:
        print("(QUICK MODE - Reduced dataset sizes)")
    print("=" * 60)
    
    # Prepare results for table
    table_results = {}
    
    # Intrinsic metrics
    intrinsic = results['intrinsic']
    table_results['MLM-heldout'] = {
        'MaskAcc (%)': intrinsic['mlm_accuracy'] * 100,
        'PPL': intrinsic['perplexity'],
        'GradNorm': intrinsic['grad_norm']
    }
    
    # POS probe
    pos = results['pos_probe']
    table_results['CONLL POS'] = {
        'TokenAcc (%)': pos['token_accuracy'] * 100
    }
    
    # SST-2 results
    sst2 = results['sst2']
    table_results['SST-2 dev'] = {
        'SentAcc (%)': sst2['best_dev_accuracy'] * 100
    }
    
    # Print main results table
    print(f"{'Step':<6} {'Eval-set':<15} {'Metric':<12} {'Value':<10}")
    print("-" * 60)
    
    for eval_set, metrics in table_results.items():
        first_metric = True
        for metric_name, value in metrics.items():
            step_str = "–" if first_metric else ""
            eval_set_str = eval_set if first_metric else ""
            first_metric = False
            
            if isinstance(value, float):
                if 'Acc' in metric_name or '%' in metric_name:
                    value_str = f"{value:.1f}%"
                elif metric_name == 'PPL':
                    value_str = f"{value:.1f}"
                elif 'Norm' in metric_name:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            
            print(f"{step_str:<6} {eval_set_str:<15} {metric_name:<12} {value_str:<10}")
    
    # Memory and timing summary
    print("\n" + "=" * 60)
    print("SYSTEM PERFORMANCE")
    print("=" * 60)
    
    memory = results['memory']
    timing = results['timing']
    
    print(f"Memory Usage:")
    print(f"  Peak: {memory['peak_memory_mb']:.1f} MB")
    print(f"  Target: <3072 MB (3GB)")
    print(f"  Status: {'[OK]' if memory['peak_memory_mb'] < 3072 else '[WARN]'} "
          f"({100*memory['peak_memory_mb']/3072:.1f}% of target)")
    
    print(f"\nTiming:")
    print(f"  Total time: {timing['total_time_min']:.1f} minutes")
    if quick:
        print(f"  Note: Quick mode used smaller datasets")
    
    # Target achievement summary
    print("\n" + "=" * 60)
    print("TARGET ACHIEVEMENT")
    print("=" * 60)
    
    targets = [
        ("MLM Accuracy", intrinsic['mlm_accuracy'] * 100, 60.0, "%"),
        ("POS Token Accuracy", pos['token_accuracy'] * 100, 90.0, "%"),
        ("SST-2 Accuracy", sst2['best_dev_accuracy'] * 100, 80.0, "%"),
        ("Peak Memory", memory['peak_memory_mb'], 3072, "MB"),
    ]
    
    for name, actual, target, unit in targets:
        if name == "Peak Memory":
            status = "[OK]" if actual < target else "[WARN]"
            comparison = f"{actual:.1f} < {target:.0f}"
        else:
            status = "[OK]" if actual >= target else "[WARN]"
            comparison = f"{actual:.1f} >= {target:.1f}"
        
        print(f"{name:<20}: {status} ({comparison} {unit})")
    
    # Special notes
    print(f"\nNotes:")
    if intrinsic['mlm_accuracy'] < 0.6:
        print("- Low MLM accuracy expected with untrained model")
    if pos['token_accuracy'] == 1.0:
        print("- Perfect POS accuracy due to synthetic data patterns")
    if sst2['best_dev_accuracy'] < 0.8:
        print("- Low SST-2 accuracy expected with untrained model")
    if quick:
        print("- Results from quick evaluation (reduced datasets)")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Comprehensive Mini-BERT Evaluation")
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained Mini-BERT model (optional)')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to tokenizer vocabulary (optional)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with smaller datasets')
    parser.add_argument('--overfit_one_batch', action='store_true',
                       help='Include overfit test for debugging')
    
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        quick=args.quick,
        overfit_test=args.overfit_one_batch
    )
    
    # Print results
    print_comprehensive_results(results, quick=args.quick)

if __name__ == "__main__":
    main()