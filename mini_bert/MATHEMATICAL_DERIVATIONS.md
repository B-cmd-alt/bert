# Mini-BERT Mathematical Derivations

## Complete Mathematical Foundation for Mini-BERT Implementation

This document provides detailed mathematical derivations for all components in our pure NumPy Mini-BERT implementation.

## 1. Model Architecture Overview

**Mini-BERT Specifications:**
- L = 3 layers (Transformer blocks)
- H = 192 hidden dimensions
- A = 4 attention heads (H % A = 0)
- I = 768 intermediate FFN size (≈ 4H)
- T = 64 maximum sequence length
- V = 8192 vocabulary size

**Parameter Count:**
```
Total Parameters = 4,486,656 ≈ 4.5M parameters

Breakdown:
- Token embeddings: V × H = 8,192 × 192 = 1,572,864
- Position embeddings: T × H = 64 × 192 = 12,288
- 3 × Transformer layers: 3 × 442,752 = 1,328,256
- MLM head: H × V = 192 × 8,192 = 1,572,864
```

## 2. Forward Pass Mathematics

### 2.1 Input Embeddings

**Token Embedding Lookup:**
```
E_token[b,t] = TokenEmbedding[input_ids[b,t]]  # [B,T] → [B,T,H]
```

**Position Embeddings:**
```
E_pos[t] = PositionEmbedding[t]  # [T] → [T,H] → broadcast to [B,T,H]
```

**Combined Embeddings:**
```
E[b,t,h] = E_token[b,t,h] + E_pos[t,h]  # [B,T,H]
```

### 2.2 Multi-Head Self-Attention

**Linear Projections:**
```
Q = E @ W_Q    # [B,T,H] @ [H,H] → [B,T,H]
K = E @ W_K    # [B,T,H] @ [H,H] → [B,T,H] 
V = E @ W_V    # [B,T,H] @ [H,H] → [B,T,H]
```

**Reshape to Multi-Head Format:**
```
d_k = H / A = 192 / 4 = 48

Q_heads = reshape(Q, [B,T,A,d_k]) → transpose → [B,A,T,d_k]
K_heads = reshape(K, [B,T,A,d_k]) → transpose → [B,A,T,d_k]
V_heads = reshape(V, [B,T,A,d_k]) → transpose → [B,A,T,d_k]
```

**Scaled Dot-Product Attention:**
```
Scores = Q_heads @ K_heads^T / √d_k  # [B,A,T,d_k] @ [B,A,d_k,T] → [B,A,T,T]

Attention_weights = softmax(Scores)  # [B,A,T,T]

Context = Attention_weights @ V_heads  # [B,A,T,T] @ [B,A,T,d_k] → [B,A,T,d_k]
```

**Concatenate Heads and Output Projection:**
```
Context_concat = reshape(Context, [B,T,H])  # [B,A,T,d_k] → [B,T,H]

Attention_output = Context_concat @ W_O  # [B,T,H] @ [H,H] → [B,T,H]
```

### 2.3 Layer Normalization

**Mathematical Formula:**
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

where:
μ = mean(x, axis=-1, keepdims=True)      # [B,T,1]
σ² = var(x, axis=-1, keepdims=True)      # [B,T,1]
x_norm = (x - μ) / √(σ² + ε)            # [B,T,H]
output = γ * x_norm + β                  # [B,T,H]
```

### 2.4 Feed-Forward Network

**Two-Layer MLP with ReLU:**
```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Hidden = x @ W1 + b1           # [B,T,H] @ [H,I] + [I] → [B,T,I]
Hidden_ReLU = max(0, Hidden)   # Element-wise ReLU
Output = Hidden_ReLU @ W2 + b2 # [B,T,I] @ [I,H] + [H] → [B,T,H]
```

### 2.5 Transformer Layer

**Residual Connections with Layer Norm:**
```
# Attention block
x' = LayerNorm(x + MultiHeadAttention(x))

# Feed-forward block  
output = LayerNorm(x' + FFN(x'))
```

### 2.6 MLM Prediction Head

**Final Prediction:**
```
hidden_final = LayerNorm(x_L)                    # After L transformer layers
logits = hidden_final @ W_mlm + b_mlm           # [B,T,H] @ [H,V] + [V] → [B,T,V]
```

## 3. Backward Pass Derivations

### 3.1 Linear Layer Gradients

**Forward:** `y = x @ W + b`

**Gradients:**
```
∂L/∂W = x^T @ ∂L/∂y
∂L/∂b = sum(∂L/∂y, axis=(0,1))  # Sum over batch and sequence dimensions
∂L/∂x = ∂L/∂y @ W^T
```

**Shape Analysis:**
```
x: [B,T,H_in], W: [H_in,H_out], y: [B,T,H_out]

∂L/∂y: [B,T,H_out]
∂L/∂W: [H_in,H_out] = [B*T,H_in]^T @ [B*T,H_out]
∂L/∂b: [H_out] = sum([B,T,H_out])
∂L/∂x: [B,T,H_in] = [B,T,H_out] @ [H_out,H_in]
```

### 3.2 Softmax Gradient Derivation

**Forward:** `p_i = exp(x_i) / Σ_j exp(x_j)`

**Jacobian Matrix:**
```
∂p_i/∂x_j = p_i * (δ_ij - p_j)

where δ_ij = 1 if i=j, 0 otherwise
```

**Chain Rule Application:**
```
∂L/∂x_i = Σ_j (∂L/∂p_j * ∂p_j/∂x_i)
        = Σ_j (∂L/∂p_j * p_j * (δ_ji - p_i))
        = p_i * (∂L/∂p_i - Σ_j p_j * ∂L/∂p_j)
```

**Implementation:**
```python
weighted_sum = sum(softmax_output * grad_output, axis=-1, keepdims=True)
grad_input = softmax_output * (grad_output - weighted_sum)
```

### 3.3 Layer Normalization Gradients

**Forward:**
```
y = γ * (x - μ) / σ + β
μ = mean(x), σ = sqrt(var(x) + ε)
```

**Intermediate Variables:**
```
x_centered = x - μ
x_norm = x_centered / σ
```

**Direct Gradients:**
```
∂L/∂γ = Σ (∂L/∂y * x_norm)
∂L/∂β = Σ ∂L/∂y
```

**Input Gradient (Chain Rule):**
```
∂L/∂x = ∂L/∂y * ∂y/∂x + ∂L/∂μ * ∂μ/∂x + ∂L/∂σ * ∂σ/∂x

Working through the algebra:
∂μ/∂x = 1/N
∂σ/∂x = (x - μ) / (N * σ)

Final result:
∂L/∂x = (γ/σ) * [∂L/∂y - (1/N)*Σ∂L/∂y - (x_norm/N)*Σ(∂L/∂y * x_norm)]
```

### 3.4 Attention Gradient Flow

**Backward Through Output Projection:**
```
grad_context = grad_output @ W_O^T
grad_W_O = context^T @ grad_output
```

**Backward Through Head Concatenation:**
```
grad_context_heads = reshape(grad_context, [B,A,T,d_k])
```

**Backward Through Attention Application:**
```
grad_attention_weights = grad_context_heads @ V_heads^T
grad_V_heads = attention_weights^T @ grad_context_heads
```

**Backward Through Softmax:**
```
grad_scores = softmax_backward(grad_attention_weights, attention_weights)
```

**Backward Through Scaled Dot-Product:**
```
scale = 1 / √d_k
grad_Q_heads = (grad_scores * scale) @ K_heads
grad_K_heads = (grad_scores * scale)^T @ Q_heads
```

**Backward Through Input Projections:**
```
grad_x_Q = grad_Q_flat @ W_Q^T, grad_W_Q = x^T @ grad_Q_flat
grad_x_K = grad_K_flat @ W_K^T, grad_W_K = x^T @ grad_K_flat  
grad_x_V = grad_V_flat @ W_V^T, grad_W_V = x^T @ grad_V_flat

grad_x_total = grad_x_Q + grad_x_K + grad_x_V
```

### 3.5 ReLU Gradient

**Forward:** `ReLU(x) = max(0, x)`

**Gradient:**
```
∂ReLU/∂x = 1 if x > 0, else 0

Implementation:
grad_x = grad_output * (x > 0)
```

## 4. Loss Function: Masked Language Modeling

### 4.1 Cross-Entropy Loss

**Softmax Probability:**
```
p[b,t,v] = exp(logits[b,t,v]) / Σ_k exp(logits[b,t,k])
```

**MLM Loss (only on masked positions):**
```
L = -Σ_{masked} log(p[b,t,labels[b,t]]) / N_masked

where N_masked = sum(mlm_mask)
```

### 4.2 MLM Loss Gradient

**Gradient w.r.t. Logits:**
```
∂L/∂logits[b,t,v] = (p[b,t,v] - δ[v,labels[b,t]]) * mlm_mask[b,t] / N_masked

where δ[v,k] = 1 if v=k, else 0
```

## 5. Optimization: Adam Algorithm

### 5.1 Adam Update Rules

**Exponential Moving Averages:**
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t    # First moment
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²   # Second moment
```

**Bias Correction:**
```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

**Parameter Update:**
```
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**Default Hyperparameters:**
- α = 1e-4 (learning rate)
- β₁ = 0.9 
- β₂ = 0.999
- ε = 1e-8

## 6. Memory and Computational Complexity

### 6.1 Parameter Memory

```
Model parameters: 4.5M × 4 bytes = 18 MB
Gradients: 4.5M × 4 bytes = 18 MB  
Adam state: 4.5M × 8 bytes = 36 MB
Total parameter memory: 72 MB
```

### 6.2 Activation Memory (per batch)

```
Batch size = 8, Seq length = 64

Input embeddings: 8 × 64 × 192 = 98,304 values = 393 KB
Attention scores: 8 × 4 × 64 × 64 × 3 = 393,216 values = 1.6 MB
FFN intermediate: 8 × 64 × 768 × 3 = 1,179,648 values = 4.8 MB
Total activations: ~7 MB per batch
```

### 6.3 Computational Complexity

**Forward Pass Operations (per layer):**
- Attention: O(T² × H + T × H²) 
- FFN: O(T × H × I)
- Layer Norm: O(T × H)

**For T=64, H=192, I=768:**
- Attention: ~2.4M FLOPs
- FFN: ~9.8M FLOPs  
- Total per layer: ~12.2M FLOPs
- Full model: ~37M FLOPs per forward pass

## 7. Numerical Stability Considerations

### 7.1 Softmax Numerical Stability

**Standard softmax can overflow:**
```
p_i = exp(x_i) / Σ_j exp(x_j)
```

**Numerically stable version:**
```
x_max = max(x)
p_i = exp(x_i - x_max) / Σ_j exp(x_j - x_max)
```

### 7.2 Layer Norm Stability

**Add epsilon to variance:**
```
σ = sqrt(var(x) + ε)  where ε = 1e-12
```

### 7.3 Gradient Clipping

**Global norm clipping:**
```
global_norm = sqrt(Σ ||grad_param||²)
if global_norm > max_norm:
    scale = max_norm / global_norm
    for param: grad_param *= scale
```

## 8. Training Dynamics

### 8.1 Learning Rate Schedule

**Linear Warmup + Decay:**
```
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = peak_lr * (1 - progress)
```

### 8.2 Gradient Accumulation

**Effective batch size = micro_batch_size × accumulation_steps**
```
For each micro_batch:
    compute gradients
    accumulate_gradients += gradients
    
After accumulation_steps:
    final_gradients = accumulate_gradients / accumulation_steps
    optimizer.step(final_gradients)
    accumulate_gradients = 0
```

## 9. Implementation Verification

### 9.1 Gradient Checking

**Finite Difference Approximation:**
```
f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)

Compare with analytical gradient:
relative_error = |analytical - numerical| / (|analytical| + |numerical| + ε)
```

**Typical Results:**
- ε = 1e-5
- tolerance = 1e-4  
- Expected relative error < 1e-4 for correct implementation

This completes the mathematical foundation for our Mini-BERT implementation. Every formula has been implemented exactly as derived in the corresponding Python modules.