"""
Mini-BERT Model Architecture in Pure NumPy.

Mathematical Formulations:

1. **Input Embeddings**: 
   E = TokenEmb(x) + PositionEmb(pos)
   Shape: [B, T, H] where B=batch, T=seq_len, H=hidden_size

2. **Multi-Head Attention**:
   Q = E @ W_Q, K = E @ W_K, V = E @ W_V  # [B, T, H] -> [B, T, H]
   Q_h = reshape(Q, [B, T, A, d_k])       # Split into heads: d_k = H/A
   Attention_h = softmax(Q_h @ K_h^T / √d_k) @ V_h
   MultiHead = concat(Attention_1, ..., Attention_A) @ W_O

3. **Feed-Forward Network**:
   FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
   W1: [H, I], W2: [I, H] where I = intermediate_size

4. **Layer Normalization**:
   LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
   μ = mean(x, axis=-1), σ² = var(x, axis=-1)

5. **Transformer Layer**:
   x' = LayerNorm(x + MultiHeadAttention(x))
   output = LayerNorm(x' + FFN(x'))
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from config import ModelConfig, MODEL_CONFIG

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    
    Args:
        x: Input array
        axis: Axis along which to apply softmax
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MiniBERT:
    """
    Mini-BERT encoder with L=3 layers, H=192, A=4 heads.
    Pure NumPy implementation with explicit shape tracking.
    """
    
    def __init__(self, config: ModelConfig = MODEL_CONFIG):
        self.config = config
        self.L = config.num_layers          # 3 layers
        self.H = config.hidden_size         # 192 hidden size  
        self.A = config.num_attention_heads # 4 attention heads
        self.I = config.intermediate_size   # 768 FFN size
        self.T = config.max_sequence_length # 64 seq length
        self.V = config.vocab_size          # 8192 vocab size
        
        # Computed dimensions
        self.d_k = self.H // self.A  # 48 per head
        assert self.H % self.A == 0, f"Hidden size {self.H} must be divisible by heads {self.A}"
        
        # Initialize all parameters
        self.params = self._init_parameters()
        
        # Cache for activations (needed for backprop)
        self.cache = {}
    
    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """
        Initialize all model parameters with Xavier/He initialization.
        
        Parameter shapes:
        - token_embeddings: [V, H] = [8192, 192]
        - position_embeddings: [T, H] = [64, 192]  
        - For each layer l ∈ {0,1,2}:
          - W_Q_l, W_K_l, W_V_l: [H, H] = [192, 192]
          - W_O_l: [H, H] = [192, 192]
          - ln1_gamma_l, ln1_beta_l: [H] = [192]
          - W1_l: [H, I] = [192, 768]
          - b1_l: [I] = [768]  
          - W2_l: [I, H] = [768, 192]
          - b2_l: [H] = [192]
          - ln2_gamma_l, ln2_beta_l: [H] = [192]
        - final_ln_gamma, final_ln_beta: [H] = [192]
        - mlm_head_W: [H, V] = [192, 8192]
        - mlm_head_b: [V] = [8192]
        """  
        params = {}
        rng = np.random.RandomState(42)  # Reproducible initialization
        
        # Embedding layers
        # Token embeddings: uniform initialization in [-0.1, 0.1]
        params['token_embeddings'] = rng.uniform(-0.1, 0.1, (self.V, self.H))
        
        # Position embeddings: learned positional encoding
        params['position_embeddings'] = rng.uniform(-0.1, 0.1, (self.T, self.H))
        
        # Transformer layers
        for layer in range(self.L):
            # Multi-head attention parameters
            # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
            xavier_attn = np.sqrt(2.0 / (self.H + self.H))
            params[f'W_Q_{layer}'] = rng.normal(0, xavier_attn, (self.H, self.H))
            params[f'W_K_{layer}'] = rng.normal(0, xavier_attn, (self.H, self.H))
            params[f'W_V_{layer}'] = rng.normal(0, xavier_attn, (self.H, self.H))
            params[f'W_O_{layer}'] = rng.normal(0, xavier_attn, (self.H, self.H))
            
            # Layer norm parameters (attention)
            params[f'ln1_gamma_{layer}'] = np.ones(self.H)
            params[f'ln1_beta_{layer}'] = np.zeros(self.H)
            
            # Feed-forward parameters  
            # He initialization for ReLU: std = sqrt(2 / fan_in)
            he_w1 = np.sqrt(2.0 / self.H)
            he_w2 = np.sqrt(2.0 / self.I)
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
        
        # MLM prediction head
        xavier_mlm = np.sqrt(2.0 / (self.H + self.V))
        params['mlm_head_W'] = rng.normal(0, xavier_mlm, (self.H, self.V))
        params['mlm_head_b'] = np.zeros(self.V)
        
        # Count parameters
        total_params = sum(p.size for p in params.values())
        print(f"Initialized Mini-BERT: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        
        return params
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-12) -> Tuple[np.ndarray, Dict]:
        """
        Layer normalization with explicit math.
        
        Formula: LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
        
        Args:
            x: Input tensor [B, T, H]
            gamma: Scale parameter [H]  
            beta: Shift parameter [H]
            eps: Numerical stability constant
            
        Returns:
            output: Normalized tensor [B, T, H]
            cache: Intermediate values for backprop
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
    
    def _multi_head_attention(self, x: np.ndarray, layer: int, attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head self-attention with explicit shape transformations.
        
        Mathematical derivation:
        1. Linear projections: Q = x @ W_Q, K = x @ W_K, V = x @ W_V
        2. Reshape to heads: Q_heads = reshape(Q, [B, T, A, d_k])
        3. Scaled dot-product: Attention = softmax(Q @ K^T / √d_k) @ V
        4. Concatenate heads: MultiHead = concat(head_1, ..., head_A) @ W_O
        
        Args:
            x: Input tensor [B, T, H]
            layer: Layer index for parameter lookup
            attention_mask: Optional attention mask [B, T] (1=attend, 0=ignore)
            
        Returns:
            output: Attention output [B, T, H]
            cache: Intermediate activations for backprop
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
        
        # Reshape to multi-head format
        # [B, T, H] -> [B, T, A, d_k] -> [B, A, T, d_k]
        Q = Q.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)  
        V = V.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # Q @ K^T: [B, A, T, d_k] @ [B, A, d_k, T] -> [B, A, T, T]
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # Apply attention mask if provided (CRITICAL FIX: apply to scores, not embeddings)
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T] for broadcasting
            mask_expanded = attention_mask[:, np.newaxis, np.newaxis, :]  # [B, 1, 1, T]
            # Set masked positions to large negative value before softmax
            scores = scores + (1 - mask_expanded) * (-1e9)
        
        # Apply softmax to attention scores (numerically stable)
        attention_weights = stable_softmax(scores, axis=-1)  # [B, A, T, T]
        
        # Apply attention to values
        # [B, A, T, T] @ [B, A, T, d_k] -> [B, A, T, d_k]
        context = attention_weights @ V
        
        # Concatenate heads: [B, A, T, d_k] -> [B, T, A, d_k] -> [B, T, H]
        context = context.transpose(0, 2, 1, 3).reshape(B, T, H)
        
        # Output projection
        output = context @ W_O  # [B, T, H] @ [H, H] -> [B, T, H]
        
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
        
        Formula: FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
        
        Args:
            x: Input tensor [B, T, H]
            layer: Layer index
            
        Returns:
            output: FFN output [B, T, H]
            cache: Intermediate activations
        """
        W1 = self.params[f'W1_{layer}']  # [H, I]
        b1 = self.params[f'b1_{layer}']  # [I]
        W2 = self.params[f'W2_{layer}']  # [I, H]
        b2 = self.params[f'b2_{layer}']  # [H]
        
        # First linear layer + ReLU
        hidden = x @ W1 + b1  # [B, T, H] @ [H, I] + [I] -> [B, T, I]
        hidden_relu = np.maximum(0, hidden)  # ReLU activation
        
        # Second linear layer
        output = hidden_relu @ W2 + b2  # [B, T, I] @ [I, H] + [H] -> [B, T, H]
        
        cache = {
            'x': x, 'hidden': hidden, 'hidden_relu': hidden_relu,
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
        }
        
        return output, cache
    
    def _transformer_layer(self, x: np.ndarray, layer: int, attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Single transformer layer with residual connections and layer norm.
        
        Architecture:
        1. x' = LayerNorm(x + MultiHeadAttention(x))
        2. output = LayerNorm(x' + FFN(x'))
        
        Args:
            x: Input tensor [B, T, H]
            layer: Layer index
            attention_mask: Optional attention mask [B, T] (1=attend, 0=ignore)
            
        Returns:
            output: Layer output [B, T, H]
            cache: All intermediate computations
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
        Forward pass through Mini-BERT encoder.
        
        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T] (1=attend, 0=ignore)
            
        Returns:
            logits: MLM prediction logits [B, T, V]
            cache: All intermediate activations for backprop
        """
        B, T = input_ids.shape
        assert T <= self.T, f"Sequence length {T} exceeds maximum {self.T}"
        
        # Input embeddings
        # Token embeddings: [B, T] -> [B, T, H]
        token_emb = self.params['token_embeddings'][input_ids]  # [B, T, H]
        
        # Position embeddings: [T, H] -> [B, T, H] (broadcast)
        pos_emb = self.params['position_embeddings'][:T]  # [T, H]
        pos_emb = pos_emb[np.newaxis, :, :]  # [1, T, H]
        
        # Combined embeddings
        embeddings = token_emb + pos_emb  # [B, T, H]
        
        # Pass through transformer layers (attention mask applied in attention, not embeddings)
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
    
    def get_parameter_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.size for p in self.params.values())
    
    def get_memory_usage(self, batch_size: int = 8) -> Dict[str, float]:
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

if __name__ == "__main__":
    # Test model initialization and forward pass
    print("Testing Mini-BERT model...")
    
    model = MiniBERT()
    print(f"Model initialized with {model.get_parameter_count():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = np.random.randint(0, model.V, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
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
    
    print("✓ Mini-BERT forward pass completed successfully!")