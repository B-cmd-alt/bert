# Visual Guide to Mini-BERT

## 🎯 Quick Visual Overview

This guide provides visual representations of key concepts to help you understand Mini-BERT better.

## 1. The Big Picture: BERT Architecture

```
Input Text: "The cat sat"
     ↓
[Tokenization]
     ↓
Token IDs: [5, 6, 7]
     ↓
┌─────────────────────────┐
│   EMBEDDINGS LAYER      │
│ ┌─────────────────────┐ │
│ │ Token Embeddings    │ │ [5,6,7] → [B,T,192]
│ │        +            │ │
│ │ Position Embeddings │ │ [0,1,2] → [B,T,192]
│ └─────────────────────┘ │
└───────────┬─────────────┘
            ↓ [B,T,192]
┌─────────────────────────┐
│  TRANSFORMER BLOCK 1    │
│ ┌─────────────────────┐ │
│ │ Multi-Head Attention│ │
│ │ + Residual + Norm   │ │
│ └─────────────────────┘ │
│ ┌─────────────────────┐ │
│ │ Feed-Forward Network│ │
│ │ + Residual + Norm   │ │
│ └─────────────────────┘ │
└───────────┬─────────────┘
            ↓ [B,T,192]
┌─────────────────────────┐
│  TRANSFORMER BLOCK 2    │
│         (same)          │
└───────────┬─────────────┘
            ↓ [B,T,192]
┌─────────────────────────┐
│  TRANSFORMER BLOCK 3    │
│         (same)          │
└───────────┬─────────────┘
            ↓ [B,T,192]
┌─────────────────────────┐
│    FINAL LAYER NORM     │
└───────────┬─────────────┘
            ↓ [B,T,192]
┌─────────────────────────┐
│    MLM PROJECTION       │
│    [B,T,192] → [B,T,V]  │
└───────────┬─────────────┘
            ↓
Output: [B,T,8192] (logits for each position)
```

## 2. Understanding Dimensions

### Batch Processing
```
Single sentence:    [T, H]           e.g., [64, 192]
Batch of sentences: [B, T, H]        e.g., [8, 64, 192]

Where:
B = Batch size (how many sentences at once)
T = Sequence length (max tokens per sentence)  
H = Hidden size (dimension of each token's vector)
V = Vocabulary size (total unique tokens)
```

### Shape Transformations
```
Input IDs:        [B, T]        e.g., [8, 64]
After embedding:  [B, T, H]     e.g., [8, 64, 192]
After attention:  [B, T, H]     e.g., [8, 64, 192] (same!)
Final output:     [B, T, V]     e.g., [8, 64, 8192]
```

## 3. Inside Multi-Head Attention

### Step 1: Create Q, K, V
```
Input X: [B, T, H=192]
    ↓
Linear projections:
Q = X @ W_Q  →  [B, T, 192]
K = X @ W_K  →  [B, T, 192]
V = X @ W_V  →  [B, T, 192]
```

### Step 2: Split into Heads
```
[B, T, 192] → [B, T, 4 heads, 48 dims/head] → [B, 4, T, 48]

Head 0: [..., 0:48]
Head 1: [..., 48:96]
Head 2: [..., 96:144]
Head 3: [..., 144:192]
```

### Step 3: Attention for Each Head
```
For each head h:
    Scores = Q_h @ K_h^T / √48     [B, T, T]
    Weights = Softmax(Scores)      [B, T, T]
    Output_h = Weights @ V_h       [B, T, 48]
```

### Step 4: Concatenate Heads
```
[Output_0 | Output_1 | Output_2 | Output_3] → [B, T, 192]
                    ↓
              Linear projection
                    ↓
              Final output [B, T, 192]
```

## 4. Attention Visualization

### What Attention Looks Like
```
Query: "it"
        The  cat  sat  on  the  mat  because  it  was  tired
The     0.1  0.0  0.0  0.0  0.1  0.0   0.0    0.0  0.0  0.0
cat     0.0  0.2  0.1  0.0  0.0  0.0   0.0    0.0  0.0  0.0
sat     0.0  0.1  0.3  0.1  0.0  0.0   0.0    0.0  0.0  0.0
...
it  →   0.1  0.7  0.0  0.0  0.0  0.0   0.0    0.1  0.0  0.0
              ↑
        High attention to "cat" (it refers to cat!)
```

## 5. The Feed-Forward Network

```
Input: [B, T, 192]
    ↓
Linear 1: [192 → 768]  (expand)
    ↓
ReLU: max(0, x)  (non-linearity)
    ↓
Linear 2: [768 → 192]  (compress)
    ↓
Output: [B, T, 192]
```

### Why expand then compress?
```
192 dims → 768 dims → 192 dims

This creates a "bottleneck" that forces the model to:
1. Extract important features (expand)
2. Combine them efficiently (compress)
```

## 6. Training Process: MLM

### Masking Example
```
Original:  "The cat sat on the mat"
           ↓
Masked:    "The [MASK] sat on the mat"
           ↓
Model predicts: P(cat)=0.8, P(dog)=0.1, P(bird)=0.05, ...
           ↓
Loss = -log(P(cat)) = -log(0.8) = 0.22
```

### Gradient Flow
```
Loss
 ↓ gradient
MLM Head
 ↓ gradient  
Layer Norm
 ↓ gradient
Transformer Block 3
 ↓ gradient
Transformer Block 2
 ↓ gradient
Transformer Block 1
 ↓ gradient
Embeddings
```

## 7. Key Mathematical Operations

### Matrix Multiplication
```
[m×n] @ [n×p] = [m×p]

Example:
[8,64,192] @ [192,768] = [8,64,768]
```

### Softmax
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)

Turns scores into probabilities that sum to 1
```

### Layer Normalization
```
x_norm = (x - mean(x)) / sqrt(var(x) + ε)
output = γ * x_norm + β

Stabilizes training by normalizing activations
```

## 8. Memory Layout

### Parameter Storage
```
Token Embeddings:    8192 × 192 = 1.57M params
Position Embeddings:   64 × 192 = 12K params
Per Transformer:      ~443K params × 3 = 1.33M params
MLM Head:            192 × 8192 = 1.57M params
                     _______________
Total:               ~4.5M parameters

At 4 bytes/param:    ~18MB model size
```

### Computation Flow
```
Forward Pass:  ~20ms per batch
Backward Pass: ~40ms per batch
Total:         ~60ms per training step

Memory per batch: ~7MB activations + gradients
```

## 9. Quick Reference Card

### Shapes Cheat Sheet
```
Operation               Input Shape      Output Shape
---------               -----------      ------------
Token Embedding         [B,T]           [B,T,H]
Position Embedding      [T]             [T,H]
Multi-Head Attention    [B,T,H]         [B,T,H]
Feed-Forward           [B,T,H]         [B,T,H]
Layer Norm             [B,T,H]         [B,T,H]
MLM Projection         [B,T,H]         [B,T,V]
```

### Key Hyperparameters
```
hidden_size (H):        192    # Vector dimension
num_layers (L):         3      # Transformer blocks
num_heads (A):          4      # Attention heads
intermediate_size (I):  768    # FFN hidden size
max_seq_length (T):     64     # Max tokens
vocab_size (V):         8192   # Unique tokens
```

## 10. Common Patterns to Remember

### Residual Connection Pattern
```
output = sublayer(x) + x
```

### Transformer Block Pattern
```
1. x → Multi-Head Attention → Add x → LayerNorm → x'
2. x' → Feed-Forward → Add x' → LayerNorm → output
```

### Training Loop Pattern
```
for batch in data:
    1. Forward: input → model → logits
    2. Loss: compare logits with targets
    3. Backward: compute gradients
    4. Update: adjust parameters
```

---

This visual guide gives you the essential patterns and shapes to keep in mind while learning Mini-BERT. Reference it whenever you need a quick reminder!