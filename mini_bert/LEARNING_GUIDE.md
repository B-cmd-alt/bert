# Mini-BERT Learning Guide: From Linear Algebra to Transformers

## 🎯 Welcome! This Guide is For You If...

- You understand basic linear algebra (matrices, vectors, dot products)
- You want to understand how BERT works from the ground up
- You prefer learning by doing with actual code
- You want to see the math operations step-by-step

## 📚 Prerequisites Check

Before starting, you should be comfortable with:
- **Matrix multiplication**: A[m×n] @ B[n×p] = C[m×p]
- **Vector operations**: Addition, scaling, dot products
- **Basic Python**: Loops, functions, NumPy arrays
- **Derivatives** (helpful but we'll explain as we go)

## 🗺️ Your Learning Journey: 6 Phases

### Phase 1: Understanding the Building Blocks (Days 1-3)
### Phase 2: The Attention Mechanism (Days 4-6)
### Phase 3: Building the Full Model (Days 7-9)
### Phase 4: Training Mechanics (Days 10-12)
### Phase 5: Evaluation & Understanding (Days 13-15)
### Phase 6: Experimentation & Mastery (Days 16+)

---

## 📖 Phase 1: Understanding the Building Blocks (Days 1-3)

### Day 1: Matrix Operations in BERT

**Goal**: See how BERT is just clever matrix operations

#### Step 1: Understanding Embeddings
```python
# Run this to understand embeddings
python -c "
import numpy as np

# Think of embeddings as a lookup table
vocab_size = 10  # We have 10 words
hidden_size = 4  # Each word becomes a 4D vector

# This is our embedding matrix - each row is a word's vector
embedding_matrix = np.random.randn(vocab_size, hidden_size)

# Let's look up word #3
word_id = 3
word_vector = embedding_matrix[word_id]  # Just indexing!

print(f'Embedding matrix shape: {embedding_matrix.shape}')
print(f'Word {word_id} vector: {word_vector}')
print(f'This is just row {word_id} of our matrix!')
"
```

**Key Insight**: Embeddings convert discrete words → continuous vectors using a simple lookup

#### Step 2: Linear Transformations
```python
# Understand how linear layers work
python -c "
import numpy as np

# A linear layer is just matrix multiplication + bias
input_size = 4
output_size = 3

# Weight matrix and bias vector
W = np.random.randn(input_size, output_size) * 0.1
b = np.zeros(output_size)

# Input vector
x = np.array([1, 2, 3, 4])

# Linear transformation: y = xW + b
y = x @ W + b

print(f'Input x: {x} (shape: {x.shape})')
print(f'Weight W shape: {W.shape}')
print(f'Output y: {y} (shape: {y.shape})')
print(f'We transformed a {input_size}D vector to {output_size}D!')
"
```

**Key Insight**: Linear layers change vector dimensions via matrix multiplication

#### Step 3: Run the Simple Test
```bash
cd C:\Users\bqian\bert\mini_bert
python simple_test.py
```

This shows all components working together!

### Day 2: Activation Functions & Normalization

#### Step 1: ReLU Activation
```python
# Understand ReLU - keeps positive, zeros negative
python -c "
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
relu_output = np.maximum(0, x)  # That's it!

# ReLU is element-wise, not matrix operation
vector = np.array([-1, 0.5, -0.3, 2])
relu_vector = np.maximum(0, vector)
print(f'Input:  {vector}')
print(f'ReLU:   {relu_vector}')
print('ReLU keeps positive values, zeros negative ones')
"
```

#### Step 2: Layer Normalization
```python
# Layer norm - makes values have mean=0, std=1
python -c "
import numpy as np

# Sample hidden states
x = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8]])  # 2 examples, 4 features

# Compute mean and std for EACH example
mean = x.mean(axis=1, keepdims=True)  # [2, 1]
std = x.std(axis=1, keepdims=True)    # [2, 1]

# Normalize
x_norm = (x - mean) / (std + 1e-6)

print('Original:')
print(x)
print('\\nNormalized (each row has mean≈0, std≈1):')
print(x_norm)
print(f'\\nRow 0 mean: {x_norm[0].mean():.3f}, std: {x_norm[0].std():.3f}')
"
```

**Key Insight**: Layer norm stabilizes training by normalizing each example

### Day 3: Putting It Together - Forward Pass

#### Step 1: Trace Through One Layer
```python
# See how data flows through one transformer layer
cd C:\Users\bqian\bert\mini_bert
python -c "
from model import MiniBERT
import numpy as np

# Create model and input
model = MiniBERT()
input_ids = np.array([[5, 10, 15, 20]])  # 1 sequence, 4 tokens

# Get embeddings
embeddings = model.token_embeddings[input_ids]  # [1, 4, 192]
embeddings += model.position_embeddings[:4]     # Add position info

print(f'Input shape: {input_ids.shape}')
print(f'After embeddings: {embeddings.shape}')
print(f'Hidden size: {model.config.hidden_size}')
"
```

#### Step 2: Understand Model Architecture
```python
# Look at model structure
python config.py
```

Study the configuration - every number has a purpose!

---

## 📖 Phase 2: The Attention Mechanism (Days 4-6)

### Day 4: Understanding Self-Attention

**The Big Idea**: Attention lets each word look at all other words to understand context

#### Step 1: Simplified Attention
```python
# Understand attention with a tiny example
python -c "
import numpy as np

# Imagine 3 words, each is a 4D vector
hidden_size = 4
seq_len = 3

# Our 'sentence' as vectors
X = np.array([
    [1, 0, 1, 0],  # 'The'
    [0, 2, 0, 1],  # 'cat'  
    [1, 1, 0, 0]   # 'sat'
])

# Attention asks: how much should each word look at each other word?
# We compute this with dot products

# Step 1: Create Query, Key, Value matrices (simplified - no weights)
Q = X  # Queries: what each word is looking for
K = X  # Keys: what each word offers
V = X  # Values: what each word actually provides

# Step 2: Compute attention scores (how much word i looks at word j)
scores = Q @ K.T  # [3,4] @ [4,3] = [3,3]
print('Attention scores (before softmax):')
print(scores)
print('Row i, column j = how much word i looks at word j')

# Step 3: Apply softmax to get probabilities
def softmax(x):
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)
print('\\nAttention weights (after softmax):')
print(attention_weights)
print('Each row sums to 1!')

# Step 4: Apply attention - weighted sum of values
output = attention_weights @ V
print('\\nOutput (weighted combinations):')
print(output)
"
```

**Key Insight**: Attention computes weighted averages where weights come from dot products

#### Step 2: Scaled Dot-Product Attention
```python
# Why we scale by sqrt(d_k)
python -c "
import numpy as np

d_k = 64  # Dimension of each head

# Without scaling, dot products can be huge
a = np.random.randn(d_k)
b = np.random.randn(d_k)
dot_product = np.dot(a, b)
scaled_dot = np.dot(a, b) / np.sqrt(d_k)

print(f'Dimension d_k: {d_k}')
print(f'Unscaled dot product: {dot_product:.2f}')
print(f'Scaled dot product: {scaled_dot:.2f}')
print(f'Scaling keeps values in reasonable range for softmax!')
"
```

### Day 5: Multi-Head Attention

#### Step 1: Why Multiple Heads?
```python
# Each head can learn different relationships
python -c "
# Think of heads like different 'lenses' to look at the sentence
# Head 1 might learn: subject-verb relationships
# Head 2 might learn: adjective-noun relationships
# Head 3 might learn: position/order information
# Head 4 might learn: semantic similarity

# In code, we split our vectors across heads
hidden_size = 192
num_heads = 4
head_dim = hidden_size // num_heads  # 48

print(f'Hidden size: {hidden_size}')
print(f'Number of heads: {num_heads}')
print(f'Dimension per head: {head_dim}')
print(f'Total dimensions: {num_heads} × {head_dim} = {num_heads * head_dim}')
"
```

#### Step 2: See Multi-Head Attention in Action
```python
# Trace through actual multi-head attention
cd C:\Users\bqian\bert\mini_bert
python -c "
from model import MiniBERT
import numpy as np

model = MiniBERT()

# Create simple input
batch_size = 1
seq_len = 4
hidden = 192
X = np.random.randn(batch_size, seq_len, hidden)

# Run attention
output, cache = model._multi_head_attention(X, layer=0)

print(f'Input shape: {X.shape}')
print(f'Output shape: {output.shape}')
print(f'\\nCache contents:')
for key, value in cache.items():
    if isinstance(value, np.ndarray):
        print(f'  {key}: shape {value.shape}')
"
```

### Day 6: Attention Patterns Visualization

#### Step 1: Create Attention Visualization
```python
# Visualize what attention learns
cd C:\Users\bqian\bert\mini_bert
python -c "
import numpy as np
import matplotlib.pyplot as plt
from model import MiniBERT
from tokenizer import WordPieceTokenizer

# Load model and tokenizer
model = MiniBERT()
tokenizer = WordPieceTokenizer.load('tokenizer_8k.pkl')

# Example sentence
text = 'The cat sat on mat'
input_ids = tokenizer.encode(text)
input_ids = np.array([input_ids])  # Add batch dimension

# Get attention weights
logits, cache = model.forward(input_ids)

# Extract attention from first layer, first head
attn = cache['attention_weights_0'][0, 0]  # [seq_len, seq_len]

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(attn, cmap='Blues')
plt.colorbar()
tokens = tokenizer.decode(input_ids[0]).split()
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel('Attending to')
plt.ylabel('Query token')
plt.title('Attention Pattern: Which words look at which?')
plt.tight_layout()
plt.savefig('attention_viz.png')
print('Saved attention visualization to attention_viz.png')
"
```

---

## 📖 Phase 3: Building the Full Model (Days 7-9)

### Day 7: The Transformer Block

#### Step 1: Understand Residual Connections
```python
# Why residual connections matter
python -c "
import numpy as np

# Residual connections solve the vanishing gradient problem
x = np.array([1, 2, 3, 4])

# Without residual
def layer_without_residual(x):
    return x * 0.1  # Imagine this shrinks the signal

# With residual  
def layer_with_residual(x):
    return x + (x * 0.1)  # Add the original back!

# After many layers
y_without = x
y_with = x
for _ in range(10):
    y_without = layer_without_residual(y_without)
    y_with = layer_with_residual(y_with)

print(f'Original: {x}')
print(f'After 10 layers WITHOUT residual: {y_without}')
print(f'After 10 layers WITH residual: {y_with}')
print('\\nResidual connections preserve information!')
"
```

#### Step 2: Complete Transformer Block
```python
# See how all parts connect
cd C:\Users\bqian\bert\mini_bert
python -c "
from model import MiniTransformerLayer
import numpy as np

# One complete transformer layer includes:
# 1. Multi-head attention
# 2. Residual + LayerNorm
# 3. Feed-forward network  
# 4. Residual + LayerNorm

# Let's trace through it
print('Transformer block components:')
print('1. Input → Multi-Head Attention → Add & Norm')
print('2. → Feed-Forward Network → Add & Norm → Output')
print('\\nEach arrow includes a residual connection!')
"
```

### Day 8: The Complete BERT Model

#### Step 1: Stack Multiple Layers
```python
# BERT is just transformer blocks stacked
python -c "
# Our Mini-BERT has:
# 1. Embedding layer (token + position)
# 2. 3 transformer blocks
# 3. Final layer norm
# 4. Output projection for MLM

layers = [
    'Token + Position Embeddings',
    'Transformer Block 1',
    'Transformer Block 2', 
    'Transformer Block 3',
    'Final LayerNorm',
    'MLM Projection (predict masked words)'
]

print('Mini-BERT Architecture:')
for i, layer in enumerate(layers):
    print(f'{i}: {layer}')
"
```

#### Step 2: Parameter Count
```python
# Understand where parameters come from
cd C:\Users\bqian\bert\mini_bert
python -c "
from model import MiniBERT

model = MiniBERT()

# Count parameters
total_params = 0
param_breakdown = {}

# Embeddings
token_emb_params = model.token_embeddings.size
pos_emb_params = model.position_embeddings.size
param_breakdown['embeddings'] = token_emb_params + pos_emb_params

# Each transformer layer
per_layer = 0
# Attention: Q,K,V,O projections
per_layer += 4 * (192 * 192)
# FFN: two linear layers
per_layer += (192 * 768) + (768 * 192)
# LayerNorms: 2 per layer
per_layer += 4 * 192  # gamma and beta for each

param_breakdown['transformer_layers'] = 3 * per_layer

# MLM head
param_breakdown['mlm_head'] = 192 * 8192

# Total
total_params = sum(param_breakdown.values())

print('Parameter breakdown:')
for component, count in param_breakdown.items():
    print(f'  {component}: {count:,} ({count/1e6:.1f}M)')
print(f'\\nTotal: {total_params:,} ({total_params/1e6:.1f}M parameters)')
"
```

### Day 9: Forward Pass Deep Dive

#### Step 1: Step Through Forward Pass
```python
# Detailed forward pass tracing
cd C:\Users\bqian\bert\mini_bert
python -c "
from model import MiniBERT
import numpy as np

model = MiniBERT()

# Simple input
input_ids = np.array([[5, 10, 15]])  # 3 tokens

# Manual forward pass
print('=== Forward Pass Trace ===\\n')

# 1. Embeddings
token_emb = model.token_embeddings[input_ids]  # [1, 3, 192]
pos_emb = model.position_embeddings[:3]        # [3, 192]
x = token_emb + pos_emb
print(f'After embeddings: {x.shape}')

# 2. Through each transformer layer
for i in range(3):
    print(f'\\nLayer {i}:')
    # ... attention, residual, layernorm, ffn, residual, layernorm ...
    print(f'  Input shape: {x.shape}')
    print(f'  (Attention → Add&Norm → FFN → Add&Norm)')
    print(f'  Output shape: {x.shape} (same!)')

# 3. Final projection
print(f'\\nFinal projection: {x.shape} → [1, 3, 8192]')
print('(Project to vocabulary size for word prediction)')
"
```

---

## 📖 Phase 4: Training Mechanics (Days 10-12)

### Day 10: Understanding MLM (Masked Language Modeling)

#### Step 1: The MLM Task
```python
# See how MLM works
cd C:\Users\bqian\bert\mini_bert
python -c "
from mlm import mask_tokens
from tokenizer import WordPieceTokenizer
import numpy as np

tokenizer = WordPieceTokenizer.load('tokenizer_8k.pkl')

# Original sentence
text = 'The cat sat on the mat'
input_ids = tokenizer.encode(text)
print(f'Original: {tokenizer.decode(input_ids)}')

# Apply masking (15% of tokens)
masked_ids, target_ids, mask_positions = mask_tokens(
    np.array([input_ids]), 
    mask_token_id=4,  # [MASK] token
    vocab_size=8192,
    mask_prob=0.15
)

print(f'Masked: {tokenizer.decode(masked_ids[0])}')
print(f'Target IDs at masked positions: {target_ids}')
print(f'Mask positions: {mask_positions}')
print('\\nBERT learns by predicting the masked words!')
"
```

#### Step 2: Computing Loss
```python
# Understand cross-entropy loss
python -c "
import numpy as np

# Simplified example: predict 1 of 4 words
vocab_size = 4
true_word_id = 2  # The correct word is word #2

# Model's predictions (logits)
logits = np.array([1.0, 0.5, 3.0, 0.2])  # Model thinks word #2 is likely!

# Convert to probabilities with softmax
exp_logits = np.exp(logits - logits.max())
probs = exp_logits / exp_logits.sum()

print(f'Logits: {logits}')
print(f'Probabilities: {probs}')
print(f'Probability of correct word: {probs[true_word_id]:.3f}')

# Cross-entropy loss
loss = -np.log(probs[true_word_id])
print(f'\\nLoss: {loss:.3f}')
print('Lower loss = better prediction!')
"
```

### Day 11: Gradients and Backpropagation

#### Step 1: Understanding Gradients
```python
# What are gradients?
cd C:\Users\bqian\bert\mini_bert
python -c "
# A gradient tells us: 
# 'If I change this parameter slightly, how much does the loss change?'

import numpy as np

# Simple function: f(x) = x^2
# Derivative: f'(x) = 2x

x = 3.0
gradient = 2 * x  # Derivative at x=3

print(f'Function: f(x) = x²')
print(f'At x = {x}:')
print(f'  Value: f({x}) = {x**2}')
print(f'  Gradient: f\'({x}) = {gradient}')
print(f'\\nMeaning: Increasing x by 0.1 increases f(x) by ≈ {gradient * 0.1}')
"
```

#### Step 2: Gradient Checking
```python
# Verify our gradients are correct
cd C:\Users\bqian\bert\mini_bert
python -c "
from gradients import gradient_check
import numpy as np

# gradient_check compares:
# 1. Our computed gradients (fast, using calculus)
# 2. Numerical gradients (slow, using tiny differences)

print('Gradient checking ensures our backward pass is correct!')
print('If analytical ≈ numerical gradients, our math is right!')
"
```

### Day 12: Optimization with Adam

#### Step 1: Why Adam?
```python
# Understand Adam optimizer
python -c "
import numpy as np

# Adam combines:
# 1. Momentum (running average of gradients)
# 2. RMSprop (running average of squared gradients)
# 3. Bias correction

# Simplified Adam step
gradient = 0.1
m = 0  # First moment (momentum)
v = 0  # Second moment (variance)
beta1, beta2 = 0.9, 0.999

# Update moments
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient**2

# Bias correction (important in early steps)
m_hat = m / (1 - beta1)
v_hat = v / (1 - beta2)

# Parameter update
lr = 0.001
update = -lr * m_hat / (np.sqrt(v_hat) + 1e-8)

print(f'Gradient: {gradient}')
print(f'Momentum (m): {m:.4f}')
print(f'Variance (v): {v:.4f}')
print(f'Parameter update: {update:.6f}')
print('\\nAdam adapts learning rate per parameter!')
"
```

#### Step 2: Training Loop
```python
# See the complete training loop
cd C:\Users\bqian\bert\mini_bert
python -c "
print('''Training Loop Structure:

for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        logits = model(batch['input_ids'])
        
        # 2. Compute loss (only at masked positions)
        loss = cross_entropy(logits[mask_positions], targets)
        
        # 3. Backward pass (compute gradients)
        gradients = backward(loss)
        
        # 4. Update parameters
        optimizer.step(gradients)
        
        # 5. Zero gradients for next step
        optimizer.zero_grad()
''')
"
```

---

## 📖 Phase 5: Evaluation & Understanding (Days 13-15)

### Day 13: Model Evaluation Metrics

#### Step 1: Basic Metrics
```python
# Run evaluation metrics
cd C:\Users\bqian\bert\mini_bert
python metrics.py --test
```

#### Step 2: Understanding Perplexity
```python
# What is perplexity?
python -c "
import numpy as np

# Perplexity = exp(average cross-entropy loss)
# Lower = better

# Example losses
good_model_loss = 2.3  # Predicts well
bad_model_loss = 5.0   # Predicts poorly

good_perplexity = np.exp(good_model_loss)
bad_perplexity = np.exp(bad_model_loss)

print(f'Good model: loss={good_model_loss:.1f}, perplexity={good_perplexity:.1f}')
print(f'Bad model: loss={bad_model_loss:.1f}, perplexity={bad_perplexity:.1f}')
print('\\nPerplexity ≈ How many words the model is choosing between')
print(f'Good model is like choosing from ~{good_perplexity:.0f} words')
print(f'Bad model is like choosing from ~{bad_perplexity:.0f} words')
"
```

### Day 14: Probing Tasks

#### Step 1: What Are Probing Tasks?
```python
# Probing tests what linguistic knowledge BERT learned
cd C:\Users\bqian\bert\mini_bert
python -c "
print('''Probing Tasks Test Linguistic Knowledge:

1. POS (Part-of-Speech) Tagging:
   - Can BERT identify nouns, verbs, adjectives?
   - Tests: Syntactic understanding

2. Dependency Parsing:
   - Can BERT identify subject-verb relationships?
   - Tests: Grammatical structure understanding

3. Named Entity Recognition:
   - Can BERT identify people, places, organizations?
   - Tests: Semantic understanding

We train a simple classifier on BERT's representations.
If it works well, BERT has learned that linguistic feature!
''')
"
```

#### Step 2: Run POS Probing
```python
# Test syntactic knowledge
cd C:\Users\bqian\bert\mini_bert
python probe_pos.py --max_sentences 100
```

### Day 15: Fine-tuning for Downstream Tasks

#### Step 1: Understanding Fine-tuning
```python
# What is fine-tuning?
cd C:\Users\bqian\bert\mini_bert
python -c "
print('''Fine-tuning Process:

1. Pre-training (what we did):
   - Task: Predict masked words (MLM)
   - Data: Large text corpus
   - Result: General language understanding

2. Fine-tuning (adaptation):
   - Task: Specific task (e.g., sentiment analysis)
   - Data: Task-specific labeled data
   - Result: Task-specific model

Key: We start from pre-trained weights, not random!
''')
"
```

#### Step 2: Fine-tune on Sentiment
```python
# Fine-tune for sentiment analysis
cd C:\Users\bqian\bert\mini_bert
python finetune_sst2.py --epochs 1 --max_samples 500
```

---

## 📖 Phase 6: Experimentation & Mastery (Days 16+)

### Advanced Experiments to Try

#### 1. Attention Analysis
```python
# Create attention analysis tool
cd C:\Users\bqian\bert\mini_bert
# Create a new file: attention_analysis.py
```

#### 2. Model Variants
- Try different hidden sizes
- Experiment with layer numbers
- Test different attention head counts

#### 3. Training Improvements
- Implement learning rate scheduling
- Try different masking strategies
- Experiment with batch sizes

---

## 🎯 Quick Reference: Key Concepts

### Matrix Shapes in Mini-BERT
```
Input: [B, T] (batch_size, sequence_length)
After Embedding: [B, T, H] (add hidden dimension)
After Attention: [B, T, H] (same shape preserved)
After FFN: [B, T, H] (same shape preserved)
Final Output: [B, T, V] (project to vocabulary)

Where:
- B = Batch size (e.g., 8)
- T = Sequence length (e.g., 64)  
- H = Hidden size (192)
- V = Vocabulary size (8192)
```

### Key Operations
1. **Embedding Lookup**: Index → Vector
2. **Linear Layer**: Matrix multiplication + bias
3. **Attention**: Weighted average based on similarity
4. **Layer Norm**: Normalize to mean=0, std=1
5. **Softmax**: Convert scores to probabilities

---

## 🚀 Next Steps

### After Completing This Guide:

1. **Read the Mathematical Derivations**
   - `MATHEMATICAL_DERIVATIONS.md` for complete proofs

2. **Explore Advanced Topics**
   - `IMPROVEMENT_ROADMAP.md` for research ideas

3. **Build Your Own Variants**
   - Modify the architecture
   - Try new training objectives
   - Implement recent papers

4. **Join the Community**
   - Share your experiments
   - Ask questions
   - Contribute improvements

---

## 💡 Learning Tips

1. **Run Code First, Theory Second**
   - See it work, then understand why

2. **Modify and Break Things**
   - Change parameters and see effects
   - Understanding comes from experimentation

3. **Visualize Everything**
   - Plot attention patterns
   - Graph training curves
   - Visualize embeddings

4. **Start Small**
   - Use tiny examples first
   - Scale up gradually
   - Build intuition incrementally

Remember: BERT is just clever matrix operations applied repeatedly. You've got this! 🎉