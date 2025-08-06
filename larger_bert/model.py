"""
Larger BERT Model Architecture in Pure NumPy.
Scales up from Mini-BERT (3.2M params) to ~40M parameters with 50k vocabulary.

Key Differences from Mini-BERT:
- Vocabulary: 8,192 → 50,000 tokens
- Layers: 3 → 6 transformer layers  
- Hidden Size: 192 → 384 dimensions
- Attention Heads: 4 → 8 heads
- Sequence Length: 64 → 128 tokens
- Parameters: 3.2M → ~40M

Mathematical formulations remain the same, just scaled up.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import pickle
import os
import sys

# Add parent directory to path to import from mini_bert
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larger_bert.config import LargerBERTConfig, LARGER_BERT_CONFIG

# Import stable_softmax from mini_bert model
from mini_bert.model import stable_softmax

class LargerBERT:
    """
    Larger BERT encoder with L=6 layers, H=384, A=8 heads, V=50k.
    Built upon Mini-BERT principles but scaled for real applications.
    """
    
    def __init__(self, config: LargerBERTConfig = LARGER_BERT_CONFIG):
        self.config = config
        self.L = config.num_layers          # 6 layers
        self.H = config.hidden_size         # 384 hidden size  
        self.A = config.num_attention_heads # 8 attention heads
        self.I = config.intermediate_size   # 1536 FFN size
        self.T = config.max_sequence_length # 128 seq length
        self.V = config.vocab_size          # 50000 vocab size
        
        # Computed dimensions
        self.d_k = self.H // self.A  # 48 per head (same as mini)
        assert self.H % self.A == 0, f"Hidden size {self.H} must be divisible by heads {self.A}"
        
        # Initialize all parameters
        self.params = self._init_parameters()
        
        # Cache for activations (needed for backprop)
        self.cache = {}
        
        # Dropout masks (if training)
        self.training = True
        self.dropout_rate = config.hidden_dropout_prob
    
    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """
        Initialize all model parameters with careful initialization for stability.
        
        Total parameters: ~40M (vs 3.2M in Mini-BERT)
        - Token embeddings: 50k × 384 = 19.2M
        - Position embeddings: 128 × 384 = 49K  
        - 6 transformer layers: ~20M
        - MLM head: 50k × 384 = 19.2M
        """  
        params = {}
        rng = np.random.RandomState(42)  # Reproducible initialization
        
        # Embedding layers
        # Token embeddings: Xavier initialization for 50k vocab
        # Smaller initialization range for larger vocabulary
        embed_range = np.sqrt(1.0 / self.V)
        params['token_embeddings'] = rng.uniform(-embed_range, embed_range, (self.V, self.H))
        
        # Position embeddings: learned positional encoding
        pos_range = 0.02  # Fixed small range
        params['position_embeddings'] = rng.uniform(-pos_range, pos_range, (self.T, self.H))
        
        # Transformer layers
        for layer in range(self.L):
            # Multi-head attention parameters
            # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
            xavier_attn = np.sqrt(2.0 / (self.H + self.H))
            
            # Initialize with slightly smaller values for stability
            init_scale = 0.02 / np.sqrt(2 * self.L)  # Scale down with depth
            
            params[f'W_Q_{layer}'] = rng.normal(0, xavier_attn * init_scale, (self.H, self.H))
            params[f'W_K_{layer}'] = rng.normal(0, xavier_attn * init_scale, (self.H, self.H))
            params[f'W_V_{layer}'] = rng.normal(0, xavier_attn * init_scale, (self.H, self.H))
            params[f'W_O_{layer}'] = rng.normal(0, xavier_attn * init_scale, (self.H, self.H))
            
            # Layer norm parameters (attention)
            params[f'ln1_gamma_{layer}'] = np.ones(self.H)
            params[f'ln1_beta_{layer}'] = np.zeros(self.H)
            
            # Feed-forward parameters  
            # He initialization for ReLU: std = sqrt(2 / fan_in)
            he_w1 = np.sqrt(2.0 / self.H) * init_scale
            he_w2 = np.sqrt(2.0 / self.I) * init_scale
            params[f'W1_{layer}'] = rng.normal(0, he_w1, (self.H, self.I))
            params[f'b1_{layer}'] = np.zeros(self.I)
            params[f'W2_{layer}'] = rng.normal(0, he_w2, (self.I, self.H))
            params[f'b2_{layer}'] = np.zeros(self.H)
            
            # Layer norm parameters (FFN)
            params[f'ln2_gamma_{layer}'] = np.ones(self.H)
            params[f'ln2_beta_{layer}'] = np.zeros(self.H)
        
        # Final layer norm
        params['final_ln_gamma'] = np.ones(self.H)
        params['final_ln_beta'] = np.zeros(self.H)
        
        # MLM prediction head with careful initialization for 50k vocab
        mlm_scale = 0.02
        params['mlm_head_W'] = rng.normal(0, mlm_scale, (self.H, self.V))
        params['mlm_head_b'] = np.zeros(self.V)
        
        # Count parameters
        total_params = sum(p.size for p in params.values())
        print(f"Initialized Larger-BERT: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        
        # Parameter breakdown
        embedding_params = params['token_embeddings'].size + params['position_embeddings'].size
        transformer_params = sum(p.size for k, p in params.items() 
                               if any(k.startswith(prefix) for prefix in ['W_Q_', 'W_K_', 'W_V_', 'W_O_', 
                                                                          'W1_', 'W2_', 'b1_', 'b2_',
                                                                          'ln1_', 'ln2_']))
        mlm_params = params['mlm_head_W'].size + params['mlm_head_b'].size
        
        print(f"  - Embeddings: {embedding_params:,} ({embedding_params/1e6:.2f}M)")
        print(f"  - Transformer: {transformer_params:,} ({transformer_params/1e6:.2f}M)")
        print(f"  - MLM Head: {mlm_params:,} ({mlm_params/1e6:.2f}M)")
        
        return params
    
    def _dropout(self, x: np.ndarray, rate: float = None) -> np.ndarray:
        """Apply dropout during training."""
        if not self.training or rate is None:
            return x
        
        rate = rate if rate is not None else self.dropout_rate
        mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
        return x * mask
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-12) -> Tuple[np.ndarray, Dict]:
        """
        Layer normalization (same as Mini-BERT).
        """
        # Compute statistics along hidden dimension (axis=-1)
        mu = np.mean(x, axis=-1, keepdims=True)  # [B, T, 1]
        variance = np.var(x, axis=-1, keepdims=True)  # [B, T, 1]
        
        # Normalize
        x_centered = x - mu  # [B, T, H]
        std = np.sqrt(variance + eps)  # [B, T, 1]  
        x_norm = x_centered / std  # [B, T, H]
        
        # Scale and shift
        output = gamma * x_norm + beta  # [B, T, H]
        
        # Cache for backprop
        cache = {
            'x': x, 'mu': mu, 'variance': variance, 'std': std,
            'x_centered': x_centered, 'x_norm': x_norm,
            'gamma': gamma, 'beta': beta
        }
        
        return output, cache
    
    def _multi_head_attention(self, x: np.ndarray, layer: int, 
                            attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head self-attention (scaled up from Mini-BERT).
        Now with 8 heads instead of 4, processing 384-dim hidden states.
        """
        B, T, H = x.shape
        assert H == self.H, f"Expected hidden size {self.H}, got {H}"
        
        # Linear projections
        W_Q = self.params[f'W_Q_{layer}']  # [H, H]
        W_K = self.params[f'W_K_{layer}']  # [H, H]  
        W_V = self.params[f'W_V_{layer}']  # [H, H]
        W_O = self.params[f'W_O_{layer}']  # [H, H]
        
        Q = x @ W_Q  # [B, T, H] @ [H, H] -> [B, T, H]
        K = x @ W_K  # [B, T, H] @ [H, H] -> [B, T, H]
        V = x @ W_V  # [B, T, H] @ [H, H] -> [B, T, H]
        
        # Apply dropout to Q, K, V
        Q = self._dropout(Q)
        K = self._dropout(K)
        V = self._dropout(V)
        
        # Reshape to multi-head format
        # [B, T, H] -> [B, T, A, d_k] -> [B, A, T, d_k]
        Q = Q.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)  
        V = V.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # Q @ K^T: [B, A, T, d_k] @ [B, A, d_k, T] -> [B, A, T, T]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T] for broadcasting
            mask_expanded = attention_mask[:, np.newaxis, np.newaxis, :]  # [B, 1, 1, T]
            # Set masked positions to large negative value before softmax
            scores = scores + (1 - mask_expanded) * (-1e9)
        
        # Apply softmax to attention scores
        attention_weights = stable_softmax(scores, axis=-1)  # [B, A, T, T]
        
        # Apply attention dropout
        attention_weights = self._dropout(attention_weights, self.config.attention_probs_dropout_prob)
        
        # Apply attention to values
        # [B, A, T, T] @ [B, A, T, d_k] -> [B, A, T, d_k]
        context = attention_weights @ V
        
        # Concatenate heads: [B, A, T, d_k] -> [B, T, A, d_k] -> [B, T, H]
        context = context.transpose(0, 2, 1, 3).reshape(B, T, H)
        
        # Output projection
        output = context @ W_O  # [B, T, H] @ [H, H] -> [B, T, H]
        output = self._dropout(output)
        
        # Cache for backpropagation
        cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V, 'scores': scores,
            'attention_weights': attention_weights, 'context': context,
            'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O
        }
        
        return output, cache
    
    def _feed_forward(self, x: np.ndarray, layer: int) -> Tuple[np.ndarray, Dict]:
        """
        Position-wise feed-forward network.
        Scaled to 1536 intermediate size (4x hidden).
        """
        W1 = self.params[f'W1_{layer}']  # [H, I]
        b1 = self.params[f'b1_{layer}']  # [I]
        W2 = self.params[f'W2_{layer}']  # [I, H]
        b2 = self.params[f'b2_{layer}']  # [H]
        
        # First linear layer + ReLU
        hidden = x @ W1 + b1  # [B, T, H] @ [H, I] + [I] -> [B, T, I]
        hidden_relu = np.maximum(0, hidden)  # ReLU activation
        
        # Apply dropout
        hidden_relu = self._dropout(hidden_relu)
        
        # Second linear layer
        output = hidden_relu @ W2 + b2  # [B, T, I] @ [I, H] + [H] -> [B, T, H]
        output = self._dropout(output)
        
        cache = {
            'x': x, 'hidden': hidden, 'hidden_relu': hidden_relu,
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
        }
        
        return output, cache
    
    def _transformer_layer(self, x: np.ndarray, layer: int, 
                         attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Single transformer layer (same structure as Mini-BERT, scaled up).
        """
        # Multi-head attention block
        attn_output, attn_cache = self._multi_head_attention(x, layer, attention_mask)
        
        # Residual connection + layer norm 1
        residual_1 = x + attn_output  # [B, T, H]
        gamma1 = self.params[f'ln1_gamma_{layer}']
        beta1 = self.params[f'ln1_beta_{layer}']
        normed_1, ln1_cache = self._layer_norm(residual_1, gamma1, beta1)
        
        # Feed-forward block  
        ffn_output, ffn_cache = self._feed_forward(normed_1, layer)
        
        # Residual connection + layer norm 2
        residual_2 = normed_1 + ffn_output  # [B, T, H]
        gamma2 = self.params[f'ln2_gamma_{layer}']
        beta2 = self.params[f'ln2_beta_{layer}']
        output, ln2_cache = self._layer_norm(residual_2, gamma2, beta2)
        
        cache = {
            'attn_output': attn_output, 'attn_cache': attn_cache,
            'residual_1': residual_1, 'ln1_cache': ln1_cache,
            'ffn_output': ffn_output, 'ffn_cache': ffn_cache, 
            'residual_2': residual_2, 'ln2_cache': ln2_cache
        }
        
        return output, cache
    
    def forward(self, input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through Larger BERT encoder.
        
        Args:
            input_ids: Token IDs [B, T], values in range [0, 50000)
            attention_mask: Attention mask [B, T] (1=attend, 0=ignore)
            
        Returns:
            logits: MLM prediction logits [B, T, V=50k]
            cache: All intermediate activations for backprop
        """
        B, T = input_ids.shape
        assert T <= self.T, f"Sequence length {T} exceeds maximum {self.T}"
        assert np.max(input_ids) < self.V, f"Token ID exceeds vocabulary size {self.V}"
        
        # Input embeddings
        # Token embeddings: [B, T] -> [B, T, H]
        token_emb = self.params['token_embeddings'][input_ids]  # [B, T, H]
        
        # Position embeddings: [T, H] -> [B, T, H] (broadcast)
        pos_emb = self.params['position_embeddings'][:T]  # [T, H]
        pos_emb = pos_emb[np.newaxis, :, :]  # [1, T, H]
        
        # Combined embeddings with dropout
        embeddings = token_emb + pos_emb  # [B, T, H]
        embeddings = self._dropout(embeddings)
        
        # Pass through transformer layers
        x = embeddings
        layer_caches = {}
        
        for layer in range(self.L):
            x, layer_cache = self._transformer_layer(x, layer, attention_mask)
            layer_caches[f'layer_{layer}'] = layer_cache
        
        # Final layer normalization
        final_gamma = self.params['final_ln_gamma']
        final_beta = self.params['final_ln_beta']
        final_hidden, final_ln_cache = self._layer_norm(x, final_gamma, final_beta)
        
        # MLM prediction head
        mlm_W = self.params['mlm_head_W']  # [H, V]
        mlm_b = self.params['mlm_head_b']  # [V]
        logits = final_hidden @ mlm_W + mlm_b  # [B, T, H] @ [H, V] + [V] -> [B, T, V]
        
        # Comprehensive cache for backpropagation
        cache = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_emb': token_emb,
            'pos_emb': pos_emb,
            'embeddings': embeddings,
            'layer_caches': layer_caches,
            'final_hidden': final_hidden,
            'final_ln_cache': final_ln_cache,
            'logits': logits
        }
        
        return logits, cache
    
    def set_training(self, mode: bool):
        """Set training mode for dropout."""
        self.training = mode
    
    def get_parameter_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.size for p in self.params.values())
    
    def get_memory_usage(self, batch_size: int = 4) -> Dict[str, float]:
        """Estimate memory usage in MB."""
        param_count = self.get_parameter_count()
        
        # Parameter memory (weights + gradients + optimizer state)
        param_mb = param_count * 4 / (1024**2)  # Float32
        grad_mb = param_count * 4 / (1024**2)
        opt_mb = param_count * 8 / (1024**2)  # Adam: m + v
        
        # Activation memory (rough estimate)
        seq_len = self.T
        # Input embeddings + attention matrices + FFN intermediate
        activations = (batch_size * seq_len * self.H +  # embeddings
                      batch_size * self.A * seq_len * seq_len * self.L +  # attention
                      batch_size * seq_len * self.I * self.L)  # FFN
        
        activation_mb = activations * 4 / (1024**2)
        
        return {
            'parameters_mb': param_mb,
            'gradients_mb': grad_mb,
            'optimizer_mb': opt_mb,
            'activations_mb': activation_mb,
            'total_mb': param_mb + grad_mb + opt_mb + activation_mb
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model parameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'params': self.params,
            'config': self.config,
            'model_type': 'larger_bert'
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model parameters from disk."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.params = checkpoint['params']
        print(f"Model checkpoint loaded from {filepath}")
    
    def compare_with_mini_bert(self):
        """Print comparison with Mini-BERT architecture."""
        print("\nArchitecture Comparison: Mini-BERT vs Larger-BERT")
        print("=" * 60)
        print(f"{'Component':<25} {'Mini-BERT':<15} {'Larger-BERT':<15}")
        print("-" * 60)
        
        comparisons = [
            ("Vocabulary Size", 8192, self.V),
            ("Layers", 3, self.L),
            ("Hidden Size", 192, self.H),
            ("Attention Heads", 4, self.A),
            ("Head Size", 48, self.d_k),
            ("FFN Size", 768, self.I),
            ("Max Sequence Length", 64, self.T),
            ("Parameters", "3.2M", f"{self.get_parameter_count()/1e6:.1f}M")
        ]
        
        for name, mini_val, larger_val in comparisons:
            print(f"{name:<25} {str(mini_val):<15} {str(larger_val):<15}")


if __name__ == "__main__":
    # Test model initialization and forward pass
    print("Testing Larger-BERT model...")
    print("=" * 60)
    
    model = LargerBERT()
    model.compare_with_mini_bert()
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))  # Use smaller vocab for test
    attention_mask = np.ones((batch_size, seq_len))
    
    print(f"\nTest Forward Pass:")
    print(f"Input shape: {input_ids.shape}")
    
    # Set to eval mode for testing
    model.set_training(False)
    
    logits, cache = model.forward(input_ids, attention_mask)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {model.V}]")
    
    # Memory usage
    memory = model.get_memory_usage(batch_size)
    print(f"\nMemory usage (batch_size={batch_size}):")
    for key, value in memory.items():
        print(f"  {key}: {value:.1f} MB")
    
    # Test shapes throughout the network
    print(f"\nShape verification:")
    print(f"  Token embeddings: {cache['token_emb'].shape}")
    print(f"  Position embeddings: {cache['pos_emb'].shape}") 
    print(f"  Combined embeddings: {cache['embeddings'].shape}")
    print(f"  Final hidden states: {cache['final_hidden'].shape}")
    print(f"  MLM logits: {cache['logits'].shape}")
    
    print("\n✓ Larger-BERT forward pass completed successfully!")