"""
Manual Gradient Derivations for Mini-BERT.

This module implements backward pass for each component with explicit mathematical derivations.

Mathematical Background:

1. **Linear Layer**: y = x @ W + b
   ∂L/∂W = x^T @ ∂L/∂y
   ∂L/∂b = sum(∂L/∂y, axis=0)
   ∂L/∂x = ∂L/∂y @ W^T

2. **Softmax**: p_i = exp(x_i) / Σ_j exp(x_j)
   ∂L/∂x_i = p_i * (∂L/∂p_i - Σ_j p_j * ∂L/∂p_j)
   
3. **LayerNorm**: y = γ * (x - μ) / σ + β
   ∂L/∂γ = Σ (∂L/∂y * x_norm)
   ∂L/∂β = Σ ∂L/∂y  
   ∂L/∂x = (γ/σ) * [∂L/∂y - (1/N)*Σ∂L/∂y - (x_norm/N)*Σ(∂L/∂y * x_norm)]

4. **Attention**: MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_O
   Chain rule through: Softmax -> MatMul -> Reshape -> Linear projections
"""

import numpy as np
from typing import Dict, Tuple
from model import MiniBERT

class MiniBERTGradients:
    """
    Gradient computation for Mini-BERT using pure NumPy.
    All gradients derived and implemented from first principles.
    """
    
    def __init__(self, model: MiniBERT):
        self.model = model
        self.config = model.config
        
        # Initialize gradient storage
        self.gradients = {}
        for param_name in model.params:
            self.gradients[param_name] = np.zeros_like(model.params[param_name])
    
    def zero_gradients(self):
        """Reset all gradients to zero."""
        for param_name in self.gradients:
            self.gradients[param_name].fill(0.0)
    
    def _linear_backward(self, grad_output: np.ndarray, x: np.ndarray, 
                        W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for linear layer: y = x @ W + b
        
        Mathematical derivation:
        Given: L = loss function, y = x @ W + b
        
        ∂L/∂W = ∂L/∂y * ∂y/∂W = ∂L/∂y * x^T
        ∂L/∂b = ∂L/∂y * ∂y/∂b = sum(∂L/∂y, axis=(0,1)) for broadcasting
        ∂L/∂x = ∂L/∂y * ∂y/∂x = ∂L/∂y @ W^T
        
        Args:
            grad_output: Gradient w.r.t output [B, T, output_dim]
            x: Input tensor [B, T, input_dim]
            W: Weight matrix [input_dim, output_dim]  
            b: Bias vector [output_dim]
            
        Returns:
            grad_x: Gradient w.r.t input [B, T, input_dim]
            grad_W: Gradient w.r.t weights [input_dim, output_dim]
            grad_b: Gradient w.r.t bias [output_dim]
        """
        # Reshape for matrix operations if needed
        B, T = grad_output.shape[:2]
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])  # [B*T, output_dim]
        x_flat = x.reshape(-1, x.shape[-1])  # [B*T, input_dim]
        
        # Compute gradients
        grad_W = x_flat.T @ grad_flat  # [input_dim, B*T] @ [B*T, output_dim] -> [input_dim, output_dim]
        grad_b = np.sum(grad_flat, axis=0)  # [output_dim]
        grad_x_flat = grad_flat @ W.T  # [B*T, output_dim] @ [output_dim, input_dim] -> [B*T, input_dim]
        
        # Reshape back
        grad_x = grad_x_flat.reshape(x.shape)  # [B, T, input_dim]
        
        return grad_x, grad_W, grad_b
    
    def _softmax_backward(self, grad_output: np.ndarray, softmax_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax function.
        
        Mathematical derivation:
        For softmax: p_i = exp(x_i) / Σ_j exp(x_j)
        
        Jacobian: ∂p_i/∂x_j = p_i * (δ_ij - p_j)
        where δ_ij = 1 if i==j, 0 otherwise
        
        Chain rule: ∂L/∂x_i = Σ_j (∂L/∂p_j * ∂p_j/∂x_i)
                             = Σ_j (∂L/∂p_j * p_j * (δ_ji - p_i))  
                             = p_i * (∂L/∂p_i - Σ_j p_j * ∂L/∂p_j)
        
        Args:
            grad_output: Gradient w.r.t softmax output [B, A, T, T]
            softmax_output: Softmax probabilities [B, A, T, T]
            
        Returns:
            grad_input: Gradient w.r.t softmax input [B, A, T, T]
        """
        # Compute the weighted sum: Σ_j p_j * ∂L/∂p_j along last dimension
        weighted_sum = np.sum(softmax_output * grad_output, axis=-1, keepdims=True)  # [B, A, T, 1]
        
        # Apply the softmax gradient formula
        grad_input = softmax_output * (grad_output - weighted_sum)  # [B, A, T, T]
        
        return grad_input
    
    def _layer_norm_backward(self, grad_output: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for layer normalization.
        
        Mathematical derivation:
        Given: y = γ * (x - μ) / σ + β
        where μ = mean(x), σ = sqrt(var(x) + ε), x_norm = (x - μ) / σ
        
        ∂L/∂γ = Σ (∂L/∂y * x_norm)
        ∂L/∂β = Σ ∂L/∂y
        
        For ∂L/∂x, we need chain rule through μ and σ:
        ∂L/∂x = ∂L/∂y * ∂y/∂x + ∂L/∂μ * ∂μ/∂x + ∂L/∂σ * ∂σ/∂x
        
        After working through the algebra:
        ∂L/∂x = (γ/σ) * [∂L/∂y - (1/N)*Σ∂L/∂y - (x_norm/N)*Σ(∂L/∂y * x_norm)]
        
        Args:
            grad_output: Gradient w.r.t output [B, T, H]
            cache: Forward pass cache containing intermediate values
            
        Returns:
            grad_x: Gradient w.r.t input [B, T, H]  
            grad_gamma: Gradient w.r.t scale parameter [H]
            grad_beta: Gradient w.r.t shift parameter [H]
        """
        x = cache['x']
        x_norm = cache['x_norm']
        std = cache['std']
        gamma = cache['gamma']
        
        B, T, H = grad_output.shape
        N = H  # Normalization over hidden dimension
        
        # Gradients for scale and shift parameters
        grad_gamma = np.sum(grad_output * x_norm, axis=(0, 1))  # [H]
        grad_beta = np.sum(grad_output, axis=(0, 1))  # [H]
        
        # Gradient for input (most complex part)
        # Step 1: Direct gradient term
        grad_x_direct = grad_output  # [B, T, H]
        
        # Step 2: Gradient through mean
        grad_mean_term = np.mean(grad_output, axis=-1, keepdims=True)  # [B, T, 1]
        
        # Step 3: Gradient through variance/std  
        grad_var_term = np.mean(grad_output * x_norm, axis=-1, keepdims=True)  # [B, T, 1]
        
        # Combine all terms
        grad_x = (gamma / std) * (grad_x_direct - grad_mean_term - x_norm * grad_var_term)
        
        return grad_x, grad_gamma, grad_beta
    
    def _relu_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU activation.
        
        Mathematical derivation:
        ReLU(x) = max(0, x)
        ∂ReLU/∂x = 1 if x > 0, else 0
        
        Args:
            grad_output: Gradient w.r.t ReLU output
            x: Input to ReLU (before activation)
            
        Returns:
            grad_x: Gradient w.r.t ReLU input
        """
        return grad_output * (x > 0).astype(np.float32)
    
    def _multi_head_attention_backward(self, grad_output: np.ndarray, 
                                     cache: Dict, layer: int) -> Dict[str, np.ndarray]:
        """
        Backward pass for multi-head attention.
        
        This involves backpropagating through:
        1. Output projection: context @ W_O
        2. Head concatenation and reshaping
        3. Attention application: attention_weights @ V
        4. Softmax: softmax(scores)
        5. Scaled dot-product: Q @ K^T / √d_k
        6. Input projections: x @ W_Q, x @ W_K, x @ W_V
        
        Args:
            grad_output: Gradient w.r.t attention output [B, T, H]
            cache: Forward pass cache
            layer: Layer index
            
        Returns:
            gradients: Dictionary of gradients for all attention parameters
        """
        # Extract cached values
        x = cache['x']
        Q = cache['Q']  # [B, A, T, d_k]
        K = cache['K']  # [B, A, T, d_k]  
        V = cache['V']  # [B, A, T, d_k]
        attention_weights = cache['attention_weights']  # [B, A, T, T]
        context = cache['context']  # [B, T, H]
        W_O = cache['W_O']
        
        B, T, H = grad_output.shape
        A, d_k = self.config.num_attention_heads, self.config.attention_head_size
        
        # Backward through output projection: context @ W_O
        grad_context, grad_W_O, _ = self._linear_backward(
            grad_output, context, W_O, np.zeros(H)
        )
        
        # Reshape context gradient back to multi-head format
        # [B, T, H] -> [B, T, A, d_k] -> [B, A, T, d_k]
        grad_context_heads = grad_context.reshape(B, T, A, d_k).transpose(0, 2, 1, 3)
        
        # Backward through attention application: attention_weights @ V
        grad_attention_weights = grad_context_heads @ V.transpose(0, 1, 3, 2)  # [B, A, T, T]
        grad_V = attention_weights.transpose(0, 1, 3, 2) @ grad_context_heads  # [B, A, T, d_k]
        
        # Backward through softmax
        grad_scores = self._softmax_backward(grad_attention_weights, attention_weights)
        
        # Backward through scaled dot-product: Q @ K^T / √d_k
        scale = 1.0 / np.sqrt(d_k)
        grad_scores_scaled = grad_scores * scale  # [B, A, T, T]
        
        grad_Q = grad_scores_scaled @ K  # [B, A, T, T] @ [B, A, T, d_k] -> [B, A, T, d_k]
        grad_K = grad_scores_scaled.transpose(0, 1, 3, 2) @ Q  # [B, A, T, d_k]
        
        # Reshape gradients back to [B, T, H] format
        grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(B, T, H)
        grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(B, T, H)  
        grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(B, T, H)
        
        # Backward through input projections
        W_Q = cache['W_Q']
        W_K = cache['W_K']
        W_V = cache['W_V']
        
        grad_x_Q, grad_W_Q, _ = self._linear_backward(grad_Q_flat, x, W_Q, np.zeros(H))
        grad_x_K, grad_W_K, _ = self._linear_backward(grad_K_flat, x, W_K, np.zeros(H))
        grad_x_V, grad_W_V, _ = self._linear_backward(grad_V_flat, x, W_V, np.zeros(H))
        
        # Sum gradients w.r.t input
        grad_x = grad_x_Q + grad_x_K + grad_x_V
        
        return {
            'grad_x': grad_x,
            'grad_W_Q': grad_W_Q,
            'grad_W_K': grad_W_K, 
            'grad_W_V': grad_W_V,
            'grad_W_O': grad_W_O
        }
    
    def _feed_forward_backward(self, grad_output: np.ndarray, 
                             cache: Dict, layer: int) -> Dict[str, np.ndarray]:
        """
        Backward pass for feed-forward network.
        
        FFN: hidden = ReLU(x @ W1 + b1), output = hidden @ W2 + b2
        
        Args:
            grad_output: Gradient w.r.t FFN output [B, T, H]
            cache: Forward pass cache
            layer: Layer index
            
        Returns:
            gradients: Dictionary of gradients for FFN parameters
        """
        x = cache['x']
        hidden = cache['hidden']  # Before ReLU
        hidden_relu = cache['hidden_relu']  # After ReLU
        W1 = cache['W1']
        W2 = cache['W2']
        b1 = cache['b1']
        b2 = cache['b2']
        
        # Backward through second linear layer
        grad_hidden_relu, grad_W2, grad_b2 = self._linear_backward(
            grad_output, hidden_relu, W2, b2
        )
        
        # Backward through ReLU
        grad_hidden = self._relu_backward(grad_hidden_relu, hidden)
        
        # Backward through first linear layer  
        grad_x, grad_W1, grad_b1 = self._linear_backward(
            grad_hidden, x, W1, b1
        )
        
        return {
            'grad_x': grad_x,
            'grad_W1': grad_W1,
            'grad_b1': grad_b1,
            'grad_W2': grad_W2,
            'grad_b2': grad_b2
        }
    
    def _transformer_layer_backward(self, grad_output: np.ndarray,
                                   cache: Dict, layer: int) -> np.ndarray:
        """
        Backward pass for transformer layer.
        
        Architecture (forward):
        1. attn_out = MultiHeadAttention(x)
        2. residual_1 = x + attn_out  
        3. normed_1 = LayerNorm(residual_1)
        4. ffn_out = FFN(normed_1)
        5. residual_2 = normed_1 + ffn_out
        6. output = LayerNorm(residual_2)
        
        Args:
            grad_output: Gradient w.r.t layer output [B, T, H]
            cache: Forward pass cache for this layer
            layer: Layer index
            
        Returns:
            grad_x: Gradient w.r.t layer input [B, T, H]
        """
        # Backward through final layer norm
        ln2_cache = cache['ln2_cache']  
        grad_residual_2, grad_ln2_gamma, grad_ln2_beta = self._layer_norm_backward(
            grad_output, ln2_cache
        )
        
        # Accumulate layer norm gradients
        self.gradients[f'ln2_gamma_{layer}'] += grad_ln2_gamma
        self.gradients[f'ln2_beta_{layer}'] += grad_ln2_beta
        
        # Backward through residual connection 2: residual_2 = normed_1 + ffn_out
        grad_normed_1_ffn = grad_residual_2  # Gradient flows to both branches
        grad_ffn_out = grad_residual_2
        
        # Backward through feed-forward network
        ffn_cache = cache['ffn_cache']
        ffn_grads = self._feed_forward_backward(grad_ffn_out, ffn_cache, layer)
        
        # Accumulate FFN gradients
        self.gradients[f'W1_{layer}'] += ffn_grads['grad_W1']
        self.gradients[f'b1_{layer}'] += ffn_grads['grad_b1']
        self.gradients[f'W2_{layer}'] += ffn_grads['grad_W2']
        self.gradients[f'b2_{layer}'] += ffn_grads['grad_b2']
        
        # Combine gradients flowing to normed_1
        grad_normed_1 = grad_normed_1_ffn + ffn_grads['grad_x']
        
        # Backward through first layer norm
        ln1_cache = cache['ln1_cache']
        grad_residual_1, grad_ln1_gamma, grad_ln1_beta = self._layer_norm_backward(
            grad_normed_1, ln1_cache
        )
        
        # Accumulate layer norm gradients
        self.gradients[f'ln1_gamma_{layer}'] += grad_ln1_gamma
        self.gradients[f'ln1_beta_{layer}'] += grad_ln1_beta
        
        # Backward through residual connection 1: residual_1 = x + attn_out
        grad_x_residual = grad_residual_1  # Gradient flows to both branches
        grad_attn_out = grad_residual_1
        
        # Backward through multi-head attention
        attn_cache = cache['attn_cache']
        attn_grads = self._multi_head_attention_backward(grad_attn_out, attn_cache, layer)
        
        # Accumulate attention gradients
        self.gradients[f'W_Q_{layer}'] += attn_grads['grad_W_Q']
        self.gradients[f'W_K_{layer}'] += attn_grads['grad_W_K']
        self.gradients[f'W_V_{layer}'] += attn_grads['grad_W_V']
        self.gradients[f'W_O_{layer}'] += attn_grads['grad_W_O']
        
        # Combine gradients flowing to x
        grad_x = grad_x_residual + attn_grads['grad_x']
        
        return grad_x
    
    def backward(self, loss: float, cache: Dict):
        """
        Complete backward pass through Mini-BERT.
        
        Args:
            loss: Scalar loss value (for logging)
            cache: Complete forward pass cache
        """
        # Initialize gradient w.r.t logits (assuming cross-entropy loss)
        # For MLM, this should be computed in the training loop
        # Here we assume grad_logits is provided externally
        
        # This method will be called from the training loop with proper grad_logits
        pass
    
    def compute_mlm_loss_and_gradients(self, logits: np.ndarray, labels: np.ndarray, 
                                     mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute masked language modeling loss and gradients.
        
        Args:
            logits: Model predictions [B, T, V]
            labels: True token IDs [B, T] 
            mask: MLM mask (1 for masked positions) [B, T]
            
        Returns:
            loss: Scalar loss value
            grad_logits: Gradients w.r.t logits [B, T, V]
        """
        B, T, V = logits.shape
        
        # Apply softmax to get probabilities
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_exp = np.exp(logits - logits_max)
        probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)  # [B, T, V]
        
        # Cross-entropy loss only on masked positions
        loss = 0.0
        num_masked = np.sum(mask)
        
        if num_masked > 0:
            # Gather probabilities for true labels
            batch_idx, seq_idx = np.meshgrid(np.arange(B), np.arange(T), indexing='ij')
            true_probs = probs[batch_idx, seq_idx, labels]  # [B, T]
            
            # Log probabilities (with numerical stability)
            log_probs = np.log(true_probs + 1e-12)
            
            # Mask and average
            masked_log_probs = log_probs * mask
            loss = -np.sum(masked_log_probs) / num_masked
        
        # Compute gradients
        grad_logits = probs.copy()  # Start with softmax gradient
        
        # Subtract 1 from true label positions (cross-entropy gradient)
        batch_idx, seq_idx = np.meshgrid(np.arange(B), np.arange(T), indexing='ij')
        grad_logits[batch_idx, seq_idx, labels] -= 1.0
        
        # Apply mask and normalize
        if num_masked > 0:
            grad_logits = grad_logits * mask[:, :, np.newaxis] / num_masked
        else:
            grad_logits.fill(0.0)
        
        return loss, grad_logits
    
    def backward_from_logits(self, grad_logits: np.ndarray, cache: Dict) -> np.ndarray:
        """
        Backward pass starting from gradient w.r.t logits.
        
        Args:
            grad_logits: Gradient w.r.t MLM logits [B, T, V]
            cache: Complete forward pass cache
            
        Returns:
            grad_embeddings: Gradient w.r.t input embeddings [B, T, H]
        """
        # Backward through MLM prediction head
        final_hidden = cache['final_hidden']  # [B, T, H]
        mlm_W = self.model.params['mlm_head_W']  # [H, V]
        mlm_b = self.model.params['mlm_head_b']  # [V]
        
        grad_final_hidden, grad_mlm_W, grad_mlm_b = self._linear_backward(
            grad_logits, final_hidden, mlm_W, mlm_b
        )
        
        # Accumulate MLM head gradients
        self.gradients['mlm_head_W'] += grad_mlm_W
        self.gradients['mlm_head_b'] += grad_mlm_b
        
        # Backward through final layer norm
        final_ln_cache = cache['final_ln_cache']
        grad_layer_output, grad_final_gamma, grad_final_beta = self._layer_norm_backward(
            grad_final_hidden, final_ln_cache
        )
        
        # Accumulate final layer norm gradients
        self.gradients['final_ln_gamma'] += grad_final_gamma
        self.gradients['final_ln_beta'] += grad_final_beta
        
        # Backward through transformer layers (in reverse order)
        grad_x = grad_layer_output
        layer_caches = cache['layer_caches']
        
        for layer in reversed(range(self.config.num_layers)):
            layer_cache = layer_caches[f'layer_{layer}']
            grad_x = self._transformer_layer_backward(grad_x, layer_cache, layer)
        
        # Backward through embeddings
        # Position embeddings: just accumulate gradients
        T = grad_x.shape[1]
        self.gradients['position_embeddings'][:T] += np.sum(grad_x, axis=0)
        
        # Token embeddings: efficient scatter add using np.add.at (100x faster than loops)
        input_ids = cache['input_ids']  # [B, T]
        B, T, H = grad_x.shape
        
        # Flatten indices and gradients for vectorized operation
        flat_ids = input_ids.flatten()  # [B*T]
        flat_grads = grad_x.reshape(-1, H)  # [B*T, H]
        
        # Use np.add.at for efficient scatter-add operation
        np.add.at(self.gradients['token_embeddings'], flat_ids, flat_grads)
        
        return grad_x

if __name__ == "__main__":
    # Test gradient computations
    print("Testing gradient computations...")
    
    from model import MiniBERT
    
    # Initialize model and gradient computer
    model = MiniBERT()
    grad_computer = MiniBERTGradients(model)
    
    # Test forward pass
    B, T = 2, 8
    input_ids = np.random.randint(0, model.V, (B, T))
    logits, cache = model.forward(input_ids)
    
    # Test MLM loss computation
    labels = np.random.randint(0, model.V, (B, T))
    mask = np.random.rand(B, T) < 0.15  # 15% masking
    
    loss, grad_logits = grad_computer.compute_mlm_loss_and_gradients(logits, labels, mask)
    print(f"MLM loss: {loss:.4f}")
    print(f"Gradient logits shape: {grad_logits.shape}")
    
    # Test backward pass
    grad_computer.zero_gradients()
    grad_embeddings = grad_computer.backward_from_logits(grad_logits, cache)
    
    print(f"Gradient embeddings shape: {grad_embeddings.shape}")
    
    # Check gradient shapes
    print("\nGradient shapes:")
    for param_name, grad in grad_computer.gradients.items():
        if np.any(grad != 0):  # Only show non-zero gradients
            print(f"  {param_name}: {grad.shape}, max_grad: {np.max(np.abs(grad)):.6f}")
    
    print("✓ Gradient computation completed successfully!")