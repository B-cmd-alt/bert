# Comprehensive Guide to Transformer Layers and Components

## Introduction: Building Blocks of Modern AI

**Core Insight**: Transformers are built from simple, composable layers that can be mixed and matched to create different architectures.

**Linear Algebra Perspective**: Each layer is a differentiable function F(x) that transforms input vectors while preserving important information through residual connections.

## 1. Core Transformer Components

### 1.1 Multi-Head Attention Layer

**Mathematical Foundation**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Multi-Head Attention:
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)  
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
```

**Your Mini-BERT Implementation**:
```python
def _multi_head_attention(self, x, layer):
    # Linear projections: [B, T, H] -> [B, T, H]
    Q = x @ self.params[f'W_Q_{layer}']  # Query
    K = x @ self.params[f'W_K_{layer}']  # Key  
    V = x @ self.params[f'W_V_{layer}']  # Value
    
    # Reshape to heads: [B, T, H] -> [B, A, T, d_k]
    Q = Q.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, self.A, self.d_k).transpose(0, 2, 1, 3)
    
    # Scaled dot-product: [B, A, T, T]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
    attention_weights = softmax(scores)
    
    # Apply to values: [B, A, T, d_k]
    context = attention_weights @ V
    
    # Concatenate heads: [B, T, H]
    context = context.transpose(0, 2, 1, 3).reshape(B, T, H)
    
    # Output projection
    output = context @ self.params[f'W_O_{layer}']
    return output
```

**Variations and Extensions**:

#### 1.1.1 Grouped Query Attention (GQA)
```python
# Instead of num_heads key/value pairs, use fewer groups
num_query_heads = 32
num_kv_heads = 8  # Fewer K,V heads than Q heads
group_size = num_query_heads // num_kv_heads  # 4 queries per K,V pair
```

#### 1.1.2 Multi-Query Attention (MQA)
```python  
# Extreme case: single K,V head shared across all Q heads
num_query_heads = 32
num_kv_heads = 1  # Single shared K,V
```

#### 1.1.3 Sliding Window Attention
```python
def sliding_window_attention(Q, K, V, window_size=512):
    """Only attend to nearby tokens within window_size"""
    seq_len = Q.shape[2]
    mask = np.zeros((seq_len, seq_len))
    
    for i in range(seq_len):
        start = max(0, i - window_size//2)
        end = min(seq_len, i + window_size//2 + 1)
        mask[i, start:end] = 1
    
    # Apply mask to attention scores
    scores = Q @ K.T / sqrt(d_k)
    scores = scores + (1 - mask) * (-1e9)
    return softmax(scores) @ V
```

#### 1.1.4 Sparse Attention Patterns
```python
# Longformer: Local + Global attention
def longformer_attention_pattern(seq_len, window_size=512, global_tokens=None):
    mask = np.zeros((seq_len, seq_len))
    
    # Local attention (sliding window)
    for i in range(seq_len):
        start = max(0, i - window_size//2)
        end = min(seq_len, i + window_size//2 + 1)
        mask[i, start:end] = 1
    
    # Global attention for special tokens
    if global_tokens:
        for token_pos in global_tokens:
            mask[token_pos, :] = 1  # Global token attends to all
            mask[:, token_pos] = 1  # All tokens attend to global token
    
    return mask

# BigBird: Random + Local + Global
def bigbird_attention_pattern(seq_len, block_size=64, num_random_blocks=3):
    # Combine local, global, and random attention patterns
    pass
```

### 1.2 Feed-Forward Network (FFN) Layer

**Standard FFN** (Your Mini-BERT):
```python
def _feed_forward(self, x, layer):
    # First linear layer: [B, T, H] -> [B, T, I]
    hidden = x @ self.params[f'W1_{layer}'] + self.params[f'b1_{layer}']
    
    # ReLU activation
    hidden_relu = np.maximum(0, hidden)
    
    # Second linear layer: [B, T, I] -> [B, T, H]  
    output = hidden_relu @ self.params[f'W2_{layer}'] + self.params[f'b2_{layer}']
    return output
```

**FFN Variants**:

#### 1.2.1 Gated Linear Units (GLU)
```python
def glu_ffn(x, W_gate, W_up, W_down):
    """Used in LLaMA, PaLM"""
    # Split into gate and value
    gate = x @ W_gate      # [B, T, I]
    value = x @ W_up       # [B, T, I]
    
    # Gated activation
    hidden = sigmoid(gate) * value  # Element-wise gating
    
    # Output projection
    output = hidden @ W_down  # [B, T, H]
    return output

def swiglu_ffn(x, W_gate, W_up, W_down):
    """SwiGLU variant (used in LLaMA)"""
    gate = x @ W_gate
    value = x @ W_up
    
    # SiLU activation on gate
    hidden = (gate / (1 + np.exp(-gate))) * value  # SiLU(gate) * value
    
    output = hidden @ W_down
    return output
```

#### 1.2.2 Mixture of Experts (MoE)
```python
def mixture_of_experts_ffn(x, expert_networks, gating_network, top_k=2):
    """
    Route tokens to different expert networks
    Used in: Switch Transformer, GLaM, PaLM-2
    """
    batch_size, seq_len, hidden_size = x.shape
    num_experts = len(expert_networks)
    
    # Compute gating scores
    gate_logits = x @ gating_network  # [B, T, num_experts]
    gate_probs = softmax(gate_logits, axis=-1)
    
    # Select top-k experts
    top_k_indices = np.argsort(gate_probs, axis=-1)[..., -top_k:]  # [B, T, k]
    top_k_probs = np.take_along_axis(gate_probs, top_k_indices, axis=-1)
    
    # Normalize probabilities
    top_k_probs = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)
    
    # Route to experts
    output = np.zeros_like(x)
    for i in range(top_k):
        expert_idx = top_k_indices[..., i]  # [B, T]
        expert_prob = top_k_probs[..., i]   # [B, T]
        
        for expert_id in range(num_experts):
            mask = (expert_idx == expert_id)  # [B, T]
            if np.any(mask):
                expert_input = x[mask]  # Tokens for this expert
                expert_output = expert_networks[expert_id](expert_input)
                output[mask] += expert_prob[mask][..., np.newaxis] * expert_output
    
    return output
```

### 1.3 Layer Normalization

**Standard Layer Norm** (Your Mini-BERT):
```python
def _layer_norm(self, x, gamma, beta, eps=1e-12):
    # Compute statistics along hidden dimension
    mu = np.mean(x, axis=-1, keepdims=True)      # [B, T, 1]
    variance = np.var(x, axis=-1, keepdims=True) # [B, T, 1]
    
    # Normalize
    x_norm = (x - mu) / np.sqrt(variance + eps)  # [B, T, H]
    
    # Scale and shift
    output = gamma * x_norm + beta               # [B, T, H]
    return output
```

**Normalization Variants**:

#### 1.3.1 Root Mean Square Layer Normalization (RMSNorm)
```python
def rms_norm(x, gamma, eps=1e-6):
    """
    Used in: T5, LLaMA, PaLM
    Simpler than LayerNorm - no mean subtraction or bias
    """
    # Root mean square
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    
    # Normalize and scale
    output = gamma * x / rms
    return output
```

#### 1.3.2 Pre-Norm vs Post-Norm
```python
# Post-Norm (Original Transformer, BERT)
def post_norm_layer(x, attention_fn, ffn_fn, ln1, ln2):
    # Attention block
    x = ln1(x + attention_fn(x))
    
    # FFN block  
    x = ln2(x + ffn_fn(x))
    return x

# Pre-Norm (GPT-2, T5, many modern models)
def pre_norm_layer(x, attention_fn, ffn_fn, ln1, ln2):
    # Attention block
    x = x + attention_fn(ln1(x))
    
    # FFN block
    x = x + ffn_fn(ln2(x))
    return x
```

## 2. Advanced Layer Types

### 2.1 Cross-Attention Layers

**Use Case**: Encoder-decoder models, multimodal transformers

```python
def cross_attention(query, key_value_source, W_Q, W_K, W_V, W_O):
    """
    Query comes from decoder, Keys/Values from encoder
    Used in: T5, BART, multimodal models
    """
    Q = query @ W_Q              # From decoder
    K = key_value_source @ W_K   # From encoder
    V = key_value_source @ W_V   # From encoder
    
    scores = Q @ K.T / sqrt(d_k)
    attention = softmax(scores) @ V
    output = attention @ W_O
    return output
```

### 2.2 Retrieval-Augmented Layers

#### 2.2.1 RETRO (DeepMind)
```python
def retro_layer(x, retrieved_chunks, cross_attn_layer):
    """
    Augment with retrieved text chunks
    """
    # Standard self-attention
    x_self = self_attention(x)
    
    # Cross-attention with retrieved chunks
    x_retrieved = cross_attn_layer(
        query=x_self,
        key_value_source=retrieved_chunks
    )
    
    return x_self + x_retrieved
```

#### 2.2.2 FiD (Fusion-in-Decoder)
```python
def fusion_in_decoder(decoder_hidden, encoder_outputs_list):
    """
    Fuse multiple encoder outputs in decoder
    """
    fused_output = []
    for encoder_output in encoder_outputs_list:
        cross_attn_output = cross_attention(
            query=decoder_hidden,
            key_value_source=encoder_output
        )
        fused_output.append(cross_attn_output)
    
    # Combine all sources
    return sum(fused_output) / len(fused_output)
```

### 2.3 Memory-Augmented Layers

#### 2.3.1 Transformer-XL Recurrence
```python
def transformer_xl_layer(x, memory, relative_pos_emb):
    """
    Extend context with cached previous segments
    """
    # Concatenate current input with memory
    extended_context = np.concatenate([memory, x], axis=1)  # [B, mem_len+seq_len, H]
    
    # Self-attention with relative positioning
    attention_output = relative_attention(
        query=x,  # Only current segment
        key_value=extended_context,  # Current + memory
        relative_pos_emb=relative_pos_emb
    )
    
    # Update memory for next segment
    new_memory = extended_context[:, -memory_length:]
    
    return attention_output, new_memory
```

#### 2.3.2 Compressive Transformer
```python
def compressive_transformer_layer(x, memory, compressed_memory):
    """
    Two-level memory: recent + compressed
    """
    # Attention over recent memory
    recent_attn = attention(query=x, key_value=memory)
    
    # Attention over compressed memory  
    compressed_attn = attention(query=x, key_value=compressed_memory)
    
    # Combine
    output = x + recent_attn + compressed_attn
    
    # Update memories
    new_memory = update_memory(memory, x)
    new_compressed = compress_old_memory(memory)
    
    return output, new_memory, new_compressed
```

### 2.4 Adaptive Computation Layers

#### 2.4.1 Universal Transformer (Recurrent Layers)
```python
def universal_transformer_step(x, step, shared_layer_params):
    """
    Apply same layer multiple times with adaptive computation
    """
    # Halting probability (learned)
    halt_prob = sigmoid(x @ W_halt + b_halt)  # [B, T, 1]
    
    # Continue computation mask
    continue_mask = (halt_prob < threshold)  # [B, T, 1]
    
    # Apply transformer layer
    layer_output = transformer_layer(x, shared_layer_params)
    
    # Adaptive mixing
    output = continue_mask * layer_output + (1 - continue_mask) * x
    
    return output, halt_prob
```

#### 2.4.2 PonderNet (Adaptive Computation Time)
```python
def ponder_layer(x, max_steps=10):
    """
    Dynamically determine how many steps to compute
    """
    batch_size, seq_len, hidden_size = x.shape
    accumulated_output = np.zeros_like(x)
    accumulated_prob = np.zeros((batch_size, seq_len, 1))
    
    for step in range(max_steps):
        # Compute halting probability
        halt_logits = x @ W_halt + b_halt
        halt_prob = sigmoid(halt_logits)
        
        # Update accumulated probability
        accumulated_prob += halt_prob
        
        # Transformer computation
        layer_output = transformer_layer(x)
        
        # Accumulate weighted output
        accumulated_output += halt_prob * layer_output
        
        # Check if we should stop
        if np.all(accumulated_prob > 0.99):
            break
            
        # Update state
        x = layer_output
    
    return accumulated_output
```

## 3. Architectural Patterns

### 3.1 Encoder-Only (BERT-style)
```python
class EncoderOnlyTransformer:
    def __init__(self, num_layers, hidden_size, num_heads):
        self.layers = []
        for i in range(num_layers):
            layer = {
                'self_attention': MultiHeadAttention(hidden_size, num_heads),
                'ffn': FeedForward(hidden_size),
                'ln1': LayerNorm(hidden_size),
                'ln2': LayerNorm(hidden_size)
            }
            self.layers.append(layer)
    
    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            # Self-attention + residual + norm
            attn_out = layer['self_attention'](x, x, x, attention_mask)
            x = layer['ln1'](x + attn_out)
            
            # FFN + residual + norm
            ffn_out = layer['ffn'](x)
            x = layer['ln2'](x + ffn_out)
        
        return x
```

### 3.2 Decoder-Only (GPT-style)  
```python
class DecoderOnlyTransformer:
    def forward(self, x, past_key_values=None):
        for i, layer in enumerate(self.layers):
            # Causal self-attention
            attn_out, new_kv = layer['self_attention'](
                x, x, x, 
                causal_mask=True,
                past_key_values=past_key_values[i] if past_key_values else None
            )
            x = layer['ln1'](x + attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['ln2'](x + ffn_out)
            
            # Cache key-values for next iteration
            if past_key_values is not None:
                past_key_values[i] = new_kv
        
        return x, past_key_values
```

### 3.3 Encoder-Decoder (T5-style)
```python
class EncoderDecoderTransformer:
    def forward(self, encoder_input, decoder_input, encoder_attention_mask=None):
        # Encoder: bidirectional attention
        encoder_output = self.encoder(encoder_input, encoder_attention_mask)
        
        # Decoder: causal self-attention + cross-attention to encoder
        decoder_output = self.decoder(
            decoder_input,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            causal_mask=True
        )
        
        return decoder_output

class DecoderWithCrossAttention:
    def forward(self, x, encoder_hidden_states, causal_mask=True):
        for layer in self.layers:
            # Causal self-attention  
            self_attn_out = layer['self_attention'](x, x, x, causal_mask)
            x = layer['ln1'](x + self_attn_out)
            
            # Cross-attention to encoder
            cross_attn_out = layer['cross_attention'](
                query=x,
                key=encoder_hidden_states,
                value=encoder_hidden_states
            )
            x = layer['ln2'](x + cross_attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['ln3'](x + ffn_out)
        
        return x
```

## 4. Efficiency and Optimization Layers

### 4.1 Linear Attention
```python
def linear_attention(Q, K, V):
    """
    O(N) complexity instead of O(N²)
    Used in: Performer, Linear Transformer
    """
    # Apply feature map to queries and keys
    Q_prime = feature_map(Q)  # [B, H, T, D']
    K_prime = feature_map(K)  # [B, H, T, D']
    
    # Compute attention in linear time
    # Standard: O(T²D) = softmax(QK^T)V  
    # Linear: O(TD'D) = Q'(K'^TV)
    KV = K_prime.transpose(-1, -2) @ V  # [B, H, D', D]
    output = Q_prime @ KV                # [B, H, T, D]
    
    return output

def feature_map(x, random_features=None):
    """Random Fourier Features for approximating softmax"""
    if random_features is None:
        # Use positive random features
        return np.exp(x - np.max(x, axis=-1, keepdims=True))
    else:
        # Random Fourier approximation
        return np.exp(x @ random_features.T)
```

### 4.2 Flash Attention
```python
def flash_attention_concept(Q, K, V, block_size=64):
    """
    Memory-efficient attention computation
    Key insight: compute attention in blocks, never materialize full attention matrix
    """
    seq_len, head_dim = Q.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    
    output = np.zeros_like(Q)
    
    # Process in blocks to save memory
    for i in range(num_blocks):
        q_start, q_end = i * block_size, min((i + 1) * block_size, seq_len)
        Q_block = Q[q_start:q_end]  # [block_size, head_dim]
        
        # Compute attention for this block
        block_output = np.zeros((q_end - q_start, head_dim))
        max_score = -np.inf
        sum_exp = 0
        
        for j in range(num_blocks):
            k_start, k_end = j * block_size, min((j + 1) * block_size, seq_len)
            K_block = K[k_start:k_end]
            V_block = V[k_start:k_end]
            
            # Compute scores for this block pair
            scores = Q_block @ K_block.T  # [q_block, k_block]
            
            # Update statistics incrementally (online softmax)
            block_max = np.max(scores)
            new_max = max(max_score, block_max)
            
            # Reweight previous computation
            if max_score != -np.inf:
                block_output *= np.exp(max_score - new_max)
                sum_exp *= np.exp(max_score - new_max)
            
            # Add current block contribution
            exp_scores = np.exp(scores - new_max)
            block_output += exp_scores @ V_block
            sum_exp += np.sum(exp_scores)
            max_score = new_max
        
        # Normalize
        output[q_start:q_end] = block_output / sum_exp
    
    return output
```

## 5. Layer Combinations and Patterns

### 5.1 Parallel vs Sequential Processing
```python
# Sequential (traditional)
def sequential_processing(x, attention_layer, ffn_layer):
    x = x + attention_layer(x)
    x = x + ffn_layer(x)
    return x

# Parallel (PaLM, GLM)
def parallel_processing(x, attention_layer, ffn_layer):
    attn_out = attention_layer(x)
    ffn_out = ffn_layer(x)
    return x + attn_out + ffn_out
```

### 5.2 Sandwich Layers (BigBird)
```python
def sandwich_transformer_block(x):
    """
    Attention → FFN → Attention sandwich
    """
    # First attention
    x = x + self_attention_1(layer_norm(x))
    
    # FFN in the middle
    x = x + ffn(layer_norm(x))
    
    # Second attention  
    x = x + self_attention_2(layer_norm(x))
    
    return x
```

### 5.3 Interleaved Attention Patterns
```python
def interleaved_attention_block(x, local_attn, global_attn, block_id):
    """
    Alternate between local and global attention
    Used in: Longformer variants
    """
    if block_id % 2 == 0:
        # Even layers: local attention
        attn_out = local_attn(x)
    else:
        # Odd layers: global attention
        attn_out = global_attn(x)
    
    x = layer_norm(x + attn_out)
    x = layer_norm(x + ffn(x))
    return x
```

## 6. Practical Implementation Tips

### 6.1 Layer Initialization
```python
def init_transformer_layer(hidden_size, intermediate_size):
    """Proper initialization is crucial for training stability"""
    
    # Xavier/Glorot for attention weights
    attn_std = np.sqrt(2.0 / (hidden_size + hidden_size))
    W_Q = np.random.normal(0, attn_std, (hidden_size, hidden_size))
    W_K = np.random.normal(0, attn_std, (hidden_size, hidden_size))
    W_V = np.random.normal(0, attn_std, (hidden_size, hidden_size))
    W_O = np.random.normal(0, attn_std, (hidden_size, hidden_size))
    
    # He initialization for FFN (ReLU activation)
    ffn_std = np.sqrt(2.0 / hidden_size)
    W1 = np.random.normal(0, ffn_std, (hidden_size, intermediate_size))
    W2 = np.random.normal(0, np.sqrt(2.0 / intermediate_size), (intermediate_size, hidden_size))
    
    # Layer norm parameters
    gamma = np.ones(hidden_size)
    beta = np.zeros(hidden_size)
    
    return {
        'attention': {'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O},
        'ffn': {'W1': W1, 'W2': W2},
        'layer_norm': {'gamma': gamma, 'beta': beta}
    }
```

### 6.2 Gradient Flow Considerations
```python
def transformer_layer_with_gradient_flow(x, layer_params, dropout_rate=0.1):
    """
    Best practices for gradient flow
    """
    # Pre-norm (better gradient flow than post-norm)
    normed_x = layer_norm(x, layer_params['ln1'])
    
    # Attention with dropout
    attn_out = multi_head_attention(normed_x, layer_params['attn'])
    attn_out = dropout(attn_out, dropout_rate)
    
    # First residual connection
    x = x + attn_out
    
    # FFN block
    normed_x = layer_norm(x, layer_params['ln2'])
    ffn_out = feed_forward(normed_x, layer_params['ffn'])
    ffn_out = dropout(ffn_out, dropout_rate)
    
    # Second residual connection
    x = x + ffn_out
    
    return x
```

### 6.3 Memory Optimization
```python
def memory_efficient_attention(Q, K, V, chunk_size=512):
    """
    Process attention in chunks to reduce memory usage
    """
    seq_len = Q.shape[1]
    output = []
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        Q_chunk = Q[:, i:end_i]  # [B, chunk_size, D]
        
        # Compute attention for this chunk against all keys/values
        scores = Q_chunk @ K.transpose(-1, -2) / np.sqrt(Q.shape[-1])
        attn_weights = softmax(scores, axis=-1)
        chunk_output = attn_weights @ V
        
        output.append(chunk_output)
    
    return np.concatenate(output, axis=1)
```

## Summary: Building Your Understanding

**For Linear Algebra Students**, think of transformers as:

1. **Attention**: Weighted averages based on similarity (dot products)
2. **FFN**: Non-linear transformations (matrix multiplications + activation)
3. **Layer Norm**: Standardization (mean=0, std=1) per example
4. **Residuals**: Adding input to output (x + f(x)) for gradient flow

**Key Insight**: Each layer is a function composition:
```
Layer(x) = x + FFN(LayerNorm(x + Attention(LayerNorm(x))))
```

The beauty of transformers is in their modularity - you can swap in different attention mechanisms, normalization methods, or FFN variants while keeping the overall structure intact. This is why the same basic architecture scales from your 3.2M parameter Mini-BERT to 175B parameter GPT-3!