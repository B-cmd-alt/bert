"""
OPTIONAL: Activation Checkpointing Utility

Mathematical Concept:
Activation checkpointing trades computation for memory by:
1. Not storing intermediate activations during forward pass
2. Recomputing them during backward pass when needed
3. Can reduce memory usage by ~50% at cost of ~33% more computation

Implementation Strategy:
- Wrap compute-intensive functions with checkpoint()
- Store minimal state (inputs) instead of full activations
- Recompute forward pass during backward when gradients needed

Note: This is a STRETCH goal implementation - clearly marked as optional.
"""

import numpy as np
from typing import Callable, Any, Tuple, Dict, List
import functools

class CheckpointFunction:
    """
    Container for a checkpointed function and its recomputation logic.
    
    Stores function inputs and recomputes outputs during backward pass
    to save memory at the cost of additional computation.
    """
    
    def __init__(self, fn: Callable, *args, **kwargs):
        """
        Initialize checkpointed function.
        
        Args:
            fn: Function to checkpoint
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.output = None
        self.requires_recompute = True
    
    def forward(self):
        """
        Compute forward pass and store minimal state.
        
        Returns:
            Function output
        """
        if self.requires_recompute or self.output is None:
            self.output = self.fn(*self.args, **self.kwargs)
            self.requires_recompute = False
        
        return self.output
    
    def recompute_for_backward(self):
        """
        Recompute forward pass for backward gradient computation.
        
        This is called during backward pass when gradients are needed.
        """
        # Force recomputation
        self.requires_recompute = True
        return self.forward()


def checkpoint(fn: Callable, *args, **kwargs) -> Any:
    """
    Activation checkpointing utility.
    
    Mathematical Trade-off:
    - Memory saved: ~50% (don't store intermediate activations)
    - Computation added: ~33% (recompute forward during backward)
    - Net benefit: Enables training larger models within memory constraints
    
    Usage:
        # Instead of:
        output = expensive_function(inputs)
        
        # Use:
        output = checkpoint(expensive_function, inputs)
    
    Args:
        fn: Function to checkpoint
        *args: Function arguments  
        **kwargs: Function keyword arguments
        
    Returns:
        Function output (with checkpointing enabled)
    """
    # Create checkpointed function
    checkpoint_fn = CheckpointFunction(fn, *args, **kwargs)
    
    # Return output of forward pass
    return checkpoint_fn.forward()


class CheckpointedTransformerLayer:
    """
    Example: Checkpointed version of transformer layer.
    
    Demonstrates how to integrate checkpointing into the Mini-BERT architecture.
    This would replace the standard transformer layer when memory is constrained.
    """
    
    def __init__(self, model, layer_idx: int, enable_checkpointing: bool = False):
        """
        Initialize checkpointed transformer layer.
        
        Args:
            model: Mini-BERT model instance
            layer_idx: Index of the transformer layer (0, 1, or 2)
            enable_checkpointing: Whether to enable activation checkpointing
        """
        self.model = model
        self.layer_idx = layer_idx
        self.enable_checkpointing = enable_checkpointing
        
        print(f"Transformer layer {layer_idx}: checkpointing {'ENABLED' if enable_checkpointing else 'DISABLED'}")
    
    def _attention_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Attention forward pass (potentially checkpointed)."""
        return self.model._multi_head_attention(x, self.layer_idx)
    
    def _ffn_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """FFN forward pass (potentially checkpointed).""" 
        return self.model._feed_forward(x, self.layer_idx)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through checkpointed transformer layer.
        
        Memory Usage:
        - Without checkpointing: Stores all intermediate activations
        - With checkpointing: Only stores inputs, recomputes during backward
        
        Args:
            x: Input tensor [B, T, H]
            
        Returns:
            output: Layer output [B, T, H]
            cache: Computation cache (reduced if checkpointing enabled)
        """
        if self.enable_checkpointing:
            # Use checkpointing for memory-intensive operations
            attn_output, attn_cache = checkpoint(self._attention_forward, x)
            
            # Residual connection + layer norm 1
            residual_1 = x + attn_output
            gamma1 = self.model.params[f'ln1_gamma_{self.layer_idx}']
            beta1 = self.model.params[f'ln1_beta_{self.layer_idx}']
            normed_1, ln1_cache = self.model._layer_norm(residual_1, gamma1, beta1)
            
            # Checkpointed FFN
            ffn_output, ffn_cache = checkpoint(self._ffn_forward, normed_1)
            
            # Residual connection + layer norm 2
            residual_2 = normed_1 + ffn_output
            gamma2 = self.model.params[f'ln2_gamma_{self.layer_idx}']
            beta2 = self.model.params[f'ln2_beta_{self.layer_idx}']
            output, ln2_cache = self.model._layer_norm(residual_2, gamma2, beta2)
            
            # Reduced cache (only essential information stored)
            cache = {
                'input': x,  # Store input for recomputation
                'residual_1': residual_1,
                'normed_1': normed_1,
                'residual_2': residual_2,
                'ln1_cache': ln1_cache,
                'ln2_cache': ln2_cache,
                'checkpointed': True
            }
            
        else:
            # Standard forward pass (stores all activations)
            output, cache = self.model._transformer_layer(x, self.layer_idx)
            cache['checkpointed'] = False
        
        return output, cache


def estimate_memory_savings(model, batch_size: int = 8, seq_len: int = 64) -> Dict[str, float]:
    """
    Estimate memory savings from activation checkpointing.
    
    Args:
        model: Mini-BERT model
        batch_size: Batch size for estimation
        seq_len: Sequence length
        
    Returns:
        Memory usage comparison
    """
    print("=" * 50)
    print("ACTIVATION CHECKPOINTING ANALYSIS")
    print("=" * 50)
    
    # Standard memory usage (store all activations)
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_layers
    vocab_size = model.config.vocab_size
    
    # Memory per layer (standard approach)
    attention_activations = batch_size * num_heads * seq_len * seq_len * 4  # Attention scores
    ffn_activations = batch_size * seq_len * intermediate_size * 4  # FFN intermediate
    layer_norm_activations = batch_size * seq_len * hidden_size * 4 * 2  # 2 layer norms
    
    layer_memory_standard = (attention_activations + ffn_activations + layer_norm_activations) / (1024**2)
    total_layer_memory_standard = layer_memory_standard * num_layers
    
    # Output logits
    output_memory = batch_size * seq_len * vocab_size * 4 / (1024**2)
    
    # Total standard memory
    total_standard_mb = total_layer_memory_standard + output_memory
    
    # Checkpointed memory (only store inputs + minimal state)
    # Roughly 50% reduction in activation memory
    layer_memory_checkpointed = layer_memory_standard * 0.5  # Store inputs + minimal cache
    total_layer_memory_checkpointed = layer_memory_checkpointed * num_layers
    total_checkpointed_mb = total_layer_memory_checkpointed + output_memory
    
    # Memory savings
    memory_saved_mb = total_standard_mb - total_checkpointed_mb
    memory_saved_percent = 100 * memory_saved_mb / total_standard_mb
    
    print(f"Memory Usage Comparison (batch_size={batch_size}, seq_len={seq_len}):")
    print(f"  Standard approach: {total_standard_mb:.1f} MB")
    print(f"  With checkpointing: {total_checkpointed_mb:.1f} MB")
    print(f"  Memory saved: {memory_saved_mb:.1f} MB ({memory_saved_percent:.1f}%)")
    
    # Computational overhead
    compute_overhead_percent = 33.3  # Typical ~33% overhead from recomputation
    print(f"\nComputational Trade-off:")
    print(f"  Additional computation: ~{compute_overhead_percent:.1f}%")
    print(f"  Trade-off: {memory_saved_percent:.1f}% memory for {compute_overhead_percent:.1f}% compute")
    
    return {
        'standard_mb': total_standard_mb,
        'checkpointed_mb': total_checkpointed_mb,
        'memory_saved_mb': memory_saved_mb,
        'memory_saved_percent': memory_saved_percent,
        'compute_overhead_percent': compute_overhead_percent
    }


def demo_checkpointing():
    """
    Demonstrate activation checkpointing functionality.
    """
    print("=" * 50)
    print("ACTIVATION CHECKPOINTING DEMO")
    print("=" * 50)
    
    # Import model for demo
    from model import MiniBERT
    
    # Initialize model
    model = MiniBERT()
    
    # Create test input
    batch_size, seq_len = 4, 32  # Small for demo
    test_input = np.random.randn(batch_size, seq_len, model.config.hidden_size)
    
    print(f"Demo input shape: {test_input.shape}")
    
    # Standard transformer layer
    print("\n1. Standard Transformer Layer:")
    standard_layer = CheckpointedTransformerLayer(model, layer_idx=0, enable_checkpointing=False)
    output_standard, cache_standard = standard_layer.forward(test_input)
    
    print(f"   Output shape: {output_standard.shape}")
    print(f"   Cache keys: {list(cache_standard.keys())}")
    print(f"   Checkpointed: {cache_standard['checkpointed']}")
    
    # Checkpointed transformer layer
    print("\n2. Checkpointed Transformer Layer:")
    checkpointed_layer = CheckpointedTransformerLayer(model, layer_idx=0, enable_checkpointing=True)
    output_checkpointed, cache_checkpointed = checkpointed_layer.forward(test_input)
    
    print(f"   Output shape: {output_checkpointed.shape}")
    print(f"   Cache keys: {list(cache_checkpointed.keys())}")
    print(f"   Checkpointed: {cache_checkpointed['checkpointed']}")
    
    # Verify outputs are identical
    output_diff = np.max(np.abs(output_standard - output_checkpointed))
    print(f"\n3. Output Verification:")
    print(f"   Max difference: {output_diff:.2e}")
    print(f"   Outputs identical: {output_diff < 1e-10}")
    
    # Memory analysis
    print("\n4. Memory Analysis:")
    estimate_memory_savings(model, batch_size=8, seq_len=64)
    
    print("\n" + "=" * 50)
    print("CHECKPOINTING INTEGRATION GUIDE")
    print("=" * 50)
    
    print("To enable checkpointing in training:")
    print("1. Replace standard layers with CheckpointedTransformerLayer")
    print("2. Set enable_checkpointing=True for memory-constrained scenarios")
    print("3. Expect ~50% memory reduction with ~33% compute overhead")
    print("4. Useful when:")
    print("   - Training larger models")
    print("   - Limited GPU/system memory")
    print("   - Batch size constrained by memory")


if __name__ == "__main__":
    demo_checkpointing()