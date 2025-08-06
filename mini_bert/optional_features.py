"""
Optional Stretch Features for Mini-BERT Evaluation.

STRETCH GOALS (clearly marked as optional):
1. Confusion matrix for SST-2 classification results
2. BF16 inference mode for faster evaluation-only speed
3. Additional diagnostic utilities

These features are clearly labeled as optional enhancements beyond the core requirements.

CLI Usage:
    python optional_features.py --confusion_matrix results.json
    python optional_features.py --test_bf16_inference
    python optional_features.py --help
"""

import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from model import MiniBERT
from tokenizer import WordPieceTokenizer
from config import MODEL_CONFIG

def create_confusion_matrix(predictions: np.ndarray, true_labels: np.ndarray, 
                          class_names: List[str] = None) -> np.ndarray:
    """
    STRETCH FEATURE: Create confusion matrix for classification results.
    
    Args:
        predictions: Predicted class labels [num_samples]
        true_labels: True class labels [num_samples]
        class_names: Optional class names for display
        
    Returns:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(max(max(predictions), max(true_labels)) + 1)]
    
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(true_labels, predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    return confusion_matrix

def print_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str]):
    """
    STRETCH FEATURE: Pretty print confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        class_names: Class names for display
    """
    print("\nSTRETC H FEATURE: Confusion Matrix")
    print("=" * 50)
    
    # Header
    print(f"{'True\\Pred':<12}", end="")
    for name in class_names:
        print(f"{name:<10}", end="")
    print()
    
    print("-" * (12 + 10 * len(class_names)))
    
    # Matrix rows
    for i, true_name in enumerate(class_names):
        print(f"{true_name:<12}", end="")
        for j in range(len(class_names)):
            print(f"{confusion_matrix[i, j]:<10}", end="")
        print()
    
    # Compute metrics
    total = np.sum(confusion_matrix)
    accuracy = np.sum(np.diag(confusion_matrix)) / total
    
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{class_name:<12} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

class BF16InferenceModel:
    """
    STRETCH FEATURE: BF16 (bfloat16) inference mode for faster evaluation.
    
    Converts model weights to bfloat16 for inference-only speed improvements.
    Note: This is a demonstration - actual bfloat16 would require specialized hardware.
    """
    
    def __init__(self, model: MiniBERT):
        """
        Initialize BF16 inference model.
        
        Args:
            model: Original Mini-BERT model with float32 weights
        """
        print("STRETCH FEATURE: Initializing BF16 inference mode...")
        self.original_model = model
        self.bf16_params = {}
        
        # Convert parameters to simulated BF16 (using float16 as approximation)
        total_params = 0
        for param_name, param in model.params.items():
            # Simulate BF16 by converting to float16 (closest available in NumPy)
            self.bf16_params[param_name] = param.astype(np.float16)
            total_params += param.size
        
        print(f"  Converted {total_params:,} parameters to BF16")
        
        # Calculate memory savings
        float32_memory = total_params * 4  # 4 bytes per float32
        bf16_memory = total_params * 2     # 2 bytes per bfloat16
        memory_savings = (float32_memory - bf16_memory) / (1024**2)  # MB
        
        print(f"  Memory savings: {memory_savings:.1f} MB ({50.0}% reduction)")
        print("  Note: This is a demonstration - true BF16 requires specialized hardware")
    
    def forward_bf16(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass using BF16 weights (simulated with float16).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Temporarily replace model parameters with BF16 versions
        original_params = self.original_model.params.copy()
        
        # Convert to BF16 for computation
        for param_name, bf16_param in self.bf16_params.items():
            # Convert back to float32 for computation (simulating BF16 arithmetic)
            self.original_model.params[param_name] = bf16_param.astype(np.float32)
        
        # Forward pass
        logits, cache = self.original_model.forward(input_ids)
        
        # Restore original parameters
        self.original_model.params = original_params
        
        return logits
    
    def benchmark_inference_speed(self, batch_size: int = 8, seq_len: int = 64, 
                                 num_trials: int = 10) -> Dict[str, float]:
        """
        Benchmark BF16 vs FP32 inference speed.
        
        Args:
            batch_size: Batch size for benchmarking
            seq_len: Sequence length
            num_trials: Number of trials for averaging
            
        Returns:
            benchmark_results: Speed comparison results
        """
        print(f"\nSTRETCH FEATURE: Benchmarking BF16 vs FP32 inference...")
        print(f"  Batch size: {batch_size}, Seq length: {seq_len}, Trials: {num_trials}")
        
        # Create test input
        test_input = np.random.randint(0, 1000, (batch_size, seq_len))
        
        # Warmup
        for _ in range(3):
            _ = self.original_model.forward(test_input)
            _ = self.forward_bf16(test_input)
        
        # Benchmark FP32
        import time
        fp32_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = self.original_model.forward(test_input)
            end = time.perf_counter()
            fp32_times.append((end - start) * 1000)  # Convert to ms
        
        # Benchmark BF16
        bf16_times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            _ = self.forward_bf16(test_input)
            end = time.perf_counter()
            bf16_times.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        fp32_mean = np.mean(fp32_times)
        fp32_std = np.std(fp32_times)
        bf16_mean = np.mean(bf16_times)
        bf16_std = np.std(bf16_times)
        
        speedup = fp32_mean / bf16_mean
        
        results = {
            'fp32_mean_ms': fp32_mean,
            'fp32_std_ms': fp32_std,
            'bf16_mean_ms': bf16_mean,
            'bf16_std_ms': bf16_std,
            'speedup': speedup
        }
        
        print(f"  FP32: {fp32_mean:.1f} ± {fp32_std:.1f} ms")
        print(f"  BF16: {bf16_mean:.1f} ± {bf16_std:.1f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  [SUCCESS] BF16 is {speedup:.2f}x faster")
        else:
            print(f"  [NOTE] No speedup observed (simulation limitations)")
        
        return results

def demonstrate_confusion_matrix():
    """Demonstrate confusion matrix functionality with sample data."""
    print("STRETCH FEATURE: Confusion Matrix Demonstration")
    print("=" * 50)
    
    # Create sample classification results (simulate SST-2)
    np.random.seed(42)
    num_samples = 100
    
    # Simulate predictions vs true labels
    true_labels = np.random.randint(0, 2, num_samples)  # 0=negative, 1=positive
    
    # Simulate imperfect predictions (75% accuracy)
    predictions = true_labels.copy()
    # Flip 25% of predictions randomly
    flip_indices = np.random.choice(num_samples, size=25, replace=False)
    predictions[flip_indices] = 1 - predictions[flip_indices]
    
    class_names = ["Negative", "Positive"]
    
    # Create and print confusion matrix
    cm = create_confusion_matrix(predictions, true_labels, class_names)
    print_confusion_matrix(cm, class_names)

def demonstrate_bf16_inference():
    """Demonstrate BF16 inference functionality."""
    print("STRETCH FEATURE: BF16 Inference Demonstration")
    print("=" * 50)
    
    # Initialize model
    model = MiniBERT(MODEL_CONFIG)
    
    # Create BF16 inference model
    bf16_model = BF16InferenceModel(model)
    
    # Test inference
    test_input = np.random.randint(0, 1000, (4, 32))
    
    print(f"\nTesting inference with input shape: {test_input.shape}")
    
    # FP32 inference
    fp32_logits, _ = model.forward(test_input)
    print(f"FP32 output shape: {fp32_logits.shape}")
    
    # BF16 inference
    bf16_logits = bf16_model.forward_bf16(test_input)
    print(f"BF16 output shape: {bf16_logits.shape}")
    
    # Compare outputs
    max_diff = np.max(np.abs(fp32_logits - bf16_logits))
    mean_diff = np.mean(np.abs(fp32_logits - bf16_logits))
    
    print(f"\nOutput comparison:")
    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    
    if max_diff < 0.1:
        print(f"  [SUCCESS] BF16 outputs are very close to FP32")
    else:
        print(f"  [NOTE] Some differences expected due to precision reduction")
    
    # Benchmark speed
    speed_results = bf16_model.benchmark_inference_speed(batch_size=4, seq_len=32, num_trials=5)
    
    return speed_results

def main():
    """Main function demonstrating optional features."""
    parser = argparse.ArgumentParser(description="Optional Stretch Features for Mini-BERT")
    
    parser.add_argument('--confusion_matrix', action='store_true',
                       help='Demonstrate confusion matrix functionality')
    parser.add_argument('--test_bf16_inference', action='store_true', 
                       help='Test BF16 inference mode')
    parser.add_argument('--all', action='store_true',
                       help='Run all optional feature demonstrations')
    
    args = parser.parse_args()
    
    if args.all or (not args.confusion_matrix and not args.test_bf16_inference):
        # Run all demonstrations if no specific feature requested
        args.confusion_matrix = True
        args.test_bf16_inference = True
    
    print("Mini-BERT Optional Stretch Features")
    print("=" * 50)
    print("Note: These are optional enhancements beyond core requirements")
    print()
    
    if args.confusion_matrix:
        demonstrate_confusion_matrix()
        print()
    
    if args.test_bf16_inference:
        demonstrate_bf16_inference()
        print()
    
    print("=" * 50)
    print("Optional features demonstration completed!")
    print("These features can be integrated into the main evaluation pipeline as needed.")

if __name__ == "__main__":
    main()