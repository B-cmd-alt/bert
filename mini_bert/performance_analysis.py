"""
Performance Analysis and Memory Estimation for Mini-BERT Training.

Provides detailed analysis of:
- Memory usage breakdown
- Computational complexity
- Performance benchmarks on target hardware (Dell XPS i7)
- Scaling analysis
"""

import numpy as np
import time
import os
from typing import Dict, Tuple

# Import our modules for analysis
from model import MiniBERT
from gradients import MiniBERTGradients
from mlm import mask_tokens, mlm_cross_entropy
from optimizer import AdamW
from config import MODEL_CONFIG

def analyze_memory_usage() -> Dict[str, float]:
    """
    Detailed memory analysis for Mini-BERT training.
    
    Returns:
        Memory breakdown in MB
    """
    print("=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    # Model parameters
    model = MiniBERT(MODEL_CONFIG)
    param_count = model.get_parameter_count()
    param_memory_mb = param_count * 4 / (1024**2)  # Float32
    
    print(f"Model Parameters:")
    print(f"  Total parameters: {param_count:,}")
    print(f"  Memory (float32): {param_memory_mb:.1f} MB")
    
    # Parameter breakdown
    print(f"\nParameter Breakdown:")
    for name, param in model.params.items():
        param_mb = param.size * 4 / (1024**2)
        print(f"  {name:20s}: {param.shape} = {param.size:,} params = {param_mb:.1f} MB")
    
    # Gradients (same size as parameters)
    grad_memory_mb = param_memory_mb
    print(f"\nGradients: {grad_memory_mb:.1f} MB")
    
    # Optimizer state (Adam: m + v vectors)
    optimizer_memory_mb = param_memory_mb * 2  # m and v states
    print(f"Optimizer state (AdamW): {optimizer_memory_mb:.1f} MB")
    
    # Activations per micro-batch
    micro_batch_size = 8
    seq_len = 64  # Match model's actual max sequence length
    hidden_size = MODEL_CONFIG.hidden_size
    vocab_size = MODEL_CONFIG.vocab_size
    num_layers = MODEL_CONFIG.num_layers
    intermediate_size = MODEL_CONFIG.intermediate_size
    num_heads = MODEL_CONFIG.num_attention_heads
    
    print(f"\nActivations per Micro-batch (batch_size={micro_batch_size}, seq_len={seq_len}):")
    
    # Input embeddings
    input_emb_size = micro_batch_size * seq_len * hidden_size
    input_emb_mb = input_emb_size * 4 / (1024**2)
    print(f"  Input embeddings: {input_emb_mb:.1f} MB")
    
    # Attention matrices (per layer)
    attn_scores_size = micro_batch_size * num_heads * seq_len * seq_len
    attn_scores_mb = attn_scores_size * 4 / (1024**2) * num_layers
    print(f"  Attention scores ({num_layers} layers): {attn_scores_mb:.1f} MB")
    
    # FFN intermediate activations (per layer)
    ffn_intermediate_size = micro_batch_size * seq_len * intermediate_size
    ffn_intermediate_mb = ffn_intermediate_size * 4 / (1024**2) * num_layers
    print(f"  FFN intermediate ({num_layers} layers): {ffn_intermediate_mb:.1f} MB")
    
    # Output logits
    output_logits_size = micro_batch_size * seq_len * vocab_size
    output_logits_mb = output_logits_size * 4 / (1024**2)
    print(f"  Output logits: {output_logits_mb:.1f} MB")
    
    # Total activation memory
    total_activation_mb = input_emb_mb + attn_scores_mb + ffn_intermediate_mb + output_logits_mb
    print(f"  Total activations: {total_activation_mb:.1f} MB")
    
    # Total memory usage
    total_memory_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + total_activation_mb
    
    print(f"\nTOTAL MEMORY USAGE:")
    print(f"  Parameters: {param_memory_mb:.1f} MB")
    print(f"  Gradients: {grad_memory_mb:.1f} MB")
    print(f"  Optimizer: {optimizer_memory_mb:.1f} MB")
    print(f"  Activations: {total_activation_mb:.1f} MB")
    print(f"  TOTAL: {total_memory_mb:.1f} MB")
    print(f"  Target: <2048 MB (2GB)")
    print(f"  Utilization: {100 * total_memory_mb / 2048:.1f}%")
    
    if total_memory_mb < 2048:
        print(f"  Status: [OK] Well within budget")
    else:
        print(f"  Status: [WARNING] Exceeds target")
    
    return {
        'parameters_mb': param_memory_mb,
        'gradients_mb': grad_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activations_mb': total_activation_mb,
        'total_mb': total_memory_mb
    }

def benchmark_performance() -> Dict[str, float]:
    """
    Benchmark forward and backward pass performance.
    
    Returns:
        Performance metrics
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Initialize model and components
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    
    # Test configuration
    micro_batch_size = 8  
    seq_len = 64  # Match model's max sequence length
    vocab_size = MODEL_CONFIG.vocab_size
    mask_token_id = 4
    
    print(f"Benchmark Configuration:")
    print(f"  Micro-batch size: {micro_batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Create test batch
    np.random.seed(42)
    batch = np.random.randint(5, 1000, (micro_batch_size, seq_len))
    
    # Warmup (JIT compilation, cache warming)
    print(f"\nWarming up...")
    for _ in range(5):
        input_ids, target_ids, mask_positions = mask_tokens(batch, vocab_size, mask_token_id)
        logits, cache = model.forward(input_ids)
        loss, accuracy = mlm_cross_entropy(logits, target_ids)
    
    # Benchmark forward pass
    print(f"Benchmarking forward pass...")
    forward_times = []
    
    for i in range(20):
        input_ids, target_ids, mask_positions = mask_tokens(batch, vocab_size, mask_token_id)
        
        start_time = time.perf_counter()
        logits, cache = model.forward(input_ids)
        end_time = time.perf_counter()
        
        forward_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_forward_ms = np.mean(forward_times[5:])  # Skip first 5 for stability
    std_forward_ms = np.std(forward_times[5:])
    
    print(f"  Forward pass: {avg_forward_ms:.1f} ± {std_forward_ms:.1f} ms")
    
    # Benchmark backward pass
    print(f"Benchmarking backward pass...")
    backward_times = []
    
    for i in range(20):
        input_ids, target_ids, mask_positions = mask_tokens(batch, vocab_size, mask_token_id)
        logits, cache = model.forward(input_ids)
        loss, accuracy = mlm_cross_entropy(logits, target_ids)
        
        grad_computer.zero_gradients()
        
        start_time = time.perf_counter()
        
        # Compute gradient w.r.t. logits (simplified for benchmark)
        batch_size, seq_len, vocab_size = logits.shape
        valid_mask = (target_ids != -100)
        num_valid = np.sum(valid_mask)
        
        # Softmax gradient
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        grad_logits = softmax_probs.copy()
        if num_valid > 0:
            valid_positions = np.where(valid_mask)
            valid_targets = target_ids[valid_mask]
            grad_logits[valid_positions[0], valid_positions[1], valid_targets] -= 1.0
            grad_logits /= num_valid
        
        # Backward through model
        grad_computer.backward_from_logits(grad_logits, cache)
        
        end_time = time.perf_counter()
        backward_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_backward_ms = np.mean(backward_times[5:])
    std_backward_ms = np.std(backward_times[5:])
    
    print(f"  Backward pass: {avg_backward_ms:.1f} ± {std_backward_ms:.1f} ms")
    
    # Total step time
    total_step_ms = avg_forward_ms + avg_backward_ms
    print(f"  Total step: {total_step_ms:.1f} ms")
    
    # Performance analysis
    target_step_ms = 70  # Target: ≤70ms per micro-batch
    performance_ratio = total_step_ms / target_step_ms
    
    print(f"\nPerformance Analysis:")
    print(f"  Target: <={target_step_ms} ms per micro-batch")
    print(f"  Actual: {total_step_ms:.1f} ms")
    print(f"  Ratio: {performance_ratio:.2f}x")
    
    if total_step_ms <= target_step_ms:
        print(f"  Status: [OK] Meets performance target")
    else:
        print(f"  Status: [WARNING] Exceeds target by {performance_ratio:.1f}x")
    
    # Throughput calculations
    steps_per_second = 1000 / total_step_ms
    samples_per_second = steps_per_second * micro_batch_size
    tokens_per_second = samples_per_second * seq_len
    
    print(f"\nThroughput:")
    print(f"  Steps/second: {steps_per_second:.1f}")
    print(f"  Samples/second: {samples_per_second:.1f}")
    print(f"  Tokens/second: {tokens_per_second:,.0f}")
    
    return {
        'forward_ms': avg_forward_ms,
        'backward_ms': avg_backward_ms,
        'total_ms': total_step_ms,
        'steps_per_sec': steps_per_second,
        'tokens_per_sec': tokens_per_second
    }

def estimate_training_time(performance_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Estimate total training time based on performance benchmarks.
    
    Args:
        performance_metrics: Results from benchmark_performance()
        
    Returns:
        Training time estimates
    """
    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATION")
    print("=" * 60)
    
    # Training configuration
    total_steps = 100_000
    micro_batch_size = 8
    logical_batch_size = 32
    accumulation_steps = logical_batch_size // micro_batch_size  # 4
    
    # Time per logical step (including accumulation)
    step_ms = performance_metrics['total_ms']
    logical_step_ms = step_ms * accumulation_steps
    
    # Total training time
    total_ms = logical_step_ms * total_steps
    total_hours = total_ms / (1000 * 3600)
    
    print(f"Training Configuration:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Micro-batch size: {micro_batch_size}")
    print(f"  Logical batch size: {logical_batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    
    print(f"\nTime Estimates:")
    print(f"  Time per micro-batch: {step_ms:.1f} ms")
    print(f"  Time per logical step: {logical_step_ms:.1f} ms")
    print(f"  Total training time: {total_hours:.1f} hours")
    
    # Compare to target
    target_hours = 12  # Target: ≤12 hours
    time_ratio = total_hours / target_hours
    
    print(f"\nTime Budget Analysis:")
    print(f"  Target: <={target_hours} hours")
    print(f"  Estimated: {total_hours:.1f} hours")
    print(f"  Ratio: {time_ratio:.2f}x")
    
    if total_hours <= target_hours:
        print(f"  Status: [OK] Within time budget")
    else:
        print(f"  Status: [WARNING] Exceeds budget by {time_ratio:.1f}x")
    
    return {
        'total_hours': total_hours,
        'logical_step_ms': logical_step_ms,
        'time_ratio': time_ratio
    }

def generate_scaling_analysis():
    """Generate scaling analysis for batch sizes and sequence lengths."""
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    print("Memory scaling with batch size (seq_len=128):")
    print("Batch Size | Activations | Total Memory | Status")
    print("-" * 50)
    
    base_param_mem = 54.0  # Parameters + gradients + optimizer
    
    for batch_size in [4, 8, 16, 32]:
        # Rough activation memory estimate
        activation_mem = batch_size * 128 * 192 * 4 / (1024**2)  # Input embeddings
        activation_mem += batch_size * 4 * 128 * 128 * 3 * 4 / (1024**2)  # Attention
        activation_mem += batch_size * 128 * 768 * 3 * 4 / (1024**2)  # FFN
        activation_mem += batch_size * 128 * 8192 * 4 / (1024**2)  # Output logits
        
        total_mem = base_param_mem + activation_mem
        status = "OK" if total_mem < 2048 else "WARN"
        
        print(f"{batch_size:9d} | {activation_mem:10.1f} | {total_mem:11.1f} | {status}")
    
    print("\nMemory scaling with sequence length (batch_size=8):")
    print("Seq Length | Activations | Total Memory | Status")
    print("-" * 50)
    
    for seq_len in [64, 128, 256, 512]:
        # Activation memory scales quadratically with seq_len due to attention
        activation_mem = 8 * seq_len * 192 * 4 / (1024**2)  # Input embeddings
        activation_mem += 8 * 4 * seq_len * seq_len * 3 * 4 / (1024**2)  # Attention (O(T²))
        activation_mem += 8 * seq_len * 768 * 3 * 4 / (1024**2)  # FFN
        activation_mem += 8 * seq_len * 8192 * 4 / (1024**2)  # Output logits
        
        total_mem = base_param_mem + activation_mem
        status = "OK" if total_mem < 2048 else "WARN"
        
        print(f"{seq_len:9d} | {activation_mem:10.1f} | {total_mem:11.1f} | {status}")

def main():
    """Run complete performance analysis."""
    print("Mini-BERT Performance Analysis")
    print("Target Hardware: Dell XPS with i7-13620H CPU")
    print("Memory Target: <=2GB RAM")
    print("Performance Target: <=70ms per micro-batch")
    
    # Memory analysis
    memory_stats = analyze_memory_usage()
    
    # Performance benchmark
    perf_stats = benchmark_performance()
    
    # Training time estimation
    time_stats = estimate_training_time(perf_stats)
    
    # Scaling analysis
    generate_scaling_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Memory Usage: {memory_stats['total_mb']:.1f} MB / 2048 MB target")
    print(f"Performance: {perf_stats['total_ms']:.1f} ms / 70 ms target")
    print(f"Training Time: {time_stats['total_hours']:.1f} hours / 12 hours target")
    
    # Overall status
    memory_ok = memory_stats['total_mb'] < 2048
    perf_ok = perf_stats['total_ms'] <= 70
    time_ok = time_stats['total_hours'] <= 12
    
    all_ok = memory_ok and perf_ok and time_ok
    
    print(f"\nOverall Status: {'[OK] All targets met' if all_ok else '[WARNING] Some targets exceeded'}")
    
    if not memory_ok:
        print(f"  - Memory usage exceeds target")
    if not perf_ok:
        print(f"  - Performance slower than target")
    if not time_ok:
        print(f"  - Training time exceeds target")

if __name__ == "__main__":
    main()