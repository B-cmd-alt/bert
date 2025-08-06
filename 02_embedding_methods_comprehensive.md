# Comprehensive Guide to Embedding Methods

## What Are Embeddings?

**Simple Definition**: Embeddings convert discrete tokens (words, characters, subwords) into dense numerical vectors that capture semantic meaning.

**Linear Algebra Perspective**: 
- Input: One-hot vector [0,0,1,0,...,0] (sparse, size = vocabulary)
- Output: Dense vector [0.2, -0.5, 1.1, ...] (dense, size = embedding_dim)
- Operation: Matrix lookup or multiplication

## 1. Token/Word Embeddings

### 1.1 One-Hot Encoding (Baseline)
```
Word: "cat" → [0,0,1,0,0,...,0]
Problems: 
- Sparse (99.99% zeros)
- No semantic similarity
- Vocabulary size = dimensionality
```

### 1.2 Word2Vec (Google, 2013)
**Algorithm**: Skip-gram and CBOW
```
Skip-gram: predict context from center word
CBOW: predict center word from context

Math: maximize log P(context|word)
Vector size: typically 100-300 dimensions
```

**Example**:
```
"cat" → [0.2, -0.1, 0.8, 0.3, ...]
"dog" → [0.3, -0.2, 0.7, 0.4, ...]  # Similar to cat
"car" → [-0.5, 0.9, -0.3, 0.1, ...] # Different from cat
```

### 1.3 GloVe (Stanford, 2014)
**Algorithm**: Global matrix factorization
```
Minimize: Σ f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))^2
Where X_ij = co-occurrence count of words i,j
```

### 1.4 FastText (Facebook, 2017)
**Algorithm**: Word2Vec + subword information
```
Word embedding = Σ (subword embeddings)
"cat" = embedding("c") + embedding("ca") + embedding("cat") + ...
Handles out-of-vocabulary words!
```

### 1.5 Contextual Embeddings (Modern)
**Key Insight**: Same word, different contexts → different embeddings
```
"bank" in "river bank" ≠ "bank" in "money bank"
```

**Examples**:
- **ELMo** (2018): BiLSTM-based
- **BERT** (2018): Transformer-based
- **GPT** (2018): Autoregressive transformer

## 2. Subword Embeddings

### 2.1 Byte Pair Encoding (BPE)
**Algorithm**: Iteratively merge most frequent character pairs
```
Initial: ["c", "a", "t", "s"]
Merge: "ca" appears often → ["ca", "t", "s"]  
Continue until desired vocabulary size
```

**Advantages**:
- Handles rare words
- Consistent vocabulary size
- Language-agnostic

### 2.2 WordPiece (Google, used in BERT)
**Algorithm**: Similar to BPE but uses likelihood maximization
```
Split: "unbelievable" → ["un", "##believable"]
       "believable" → ["believ", "##able"]
```

### 2.3 SentencePiece (Google, 2018)
**Algorithm**: Treats text as sequence of Unicode characters
```
No pre-tokenization required
Handles whitespace as special tokens
Language-independent
```

## 3. Position Embeddings

### 3.1 Learned Positional Embeddings (BERT style)
```python
# In your Mini-BERT
position_embeddings = np.random.uniform(-0.1, 0.1, (max_seq_len, hidden_size))
# Position 0: [0.05, -0.02, 0.08, ...]
# Position 1: [0.01, 0.07, -0.03, ...]
```

### 3.2 Sinusoidal Positional Encoding (Original Transformer)
```python
def sinusoidal_encoding(position, d_model):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    encoding = np.zeros(d_model)
    for i in range(d_model // 2):
        encoding[2*i] = np.sin(position / (10000 ** (2*i / d_model)))
        encoding[2*i+1] = np.cos(position / (10000 ** (2*i / d_model)))
    return encoding
```

### 3.3 Relative Positional Encoding (Transformer-XL, DeBERTa)
**Idea**: Encode relative distances instead of absolute positions
```
Instead of: "word at position 5"
Use: "word is 2 positions after current word"
```

### 3.4 Rotary Position Embedding (RoPE) - GPT-NeoX, LLaMA
**Algorithm**: Rotates query/key vectors by position
```python
def apply_rotary_embedding(x, position):
    # Rotate x by angle proportional to position
    # Preserves inner products for relative positions
    return rotate_vector(x, position * θ)
```

## 4. Specialized Embedding Types

### 4.1 Character Embeddings
```python
# Each character gets an embedding
"cat" → [embed('c'), embed('a'), embed('t')]
# Good for morphologically rich languages
```

### 4.2 N-gram Embeddings
```python
# Trigram example
"the cat sat" → [embed("the cat"), embed("cat sat")]
```

### 4.3 Paragraph/Document Embeddings
- **Doc2Vec**: Extension of Word2Vec to documents
- **Sentence-BERT**: BERT fine-tuned for sentence similarities
- **Universal Sentence Encoder**: Google's sentence embedding model

### 4.4 Multilingual Embeddings
- **Multilingual BERT**: Shared vocabulary across 104 languages
- **XLM-R**: Cross-lingual model with better coverage
- **LASER**: Facebook's multilingual sentence embeddings

## 5. Mathematical Deep Dive

### 5.1 Embedding Lookup (What Your Mini-BERT Does)
```python
# Token IDs: [2, 5, 8, 1]  (batch_size=1, seq_len=4)
# Embedding matrix: [vocab_size, hidden_size] = [8192, 192]

token_embeddings = embedding_matrix[input_ids]  # [4, 192]

# Mathematically equivalent to:
one_hot = np.eye(vocab_size)[input_ids]  # [4, 8192]
token_embeddings = one_hot @ embedding_matrix  # [4, 8192] @ [8192, 192] = [4, 192]
```

### 5.2 Combining Embeddings
```python
# In BERT: Token + Position + Segment
final_embedding = token_emb + position_emb + segment_emb

# Shape verification:
# token_emb: [batch, seq_len, hidden] = [8, 64, 192]
# position_emb: [seq_len, hidden] = [64, 192] → broadcast to [8, 64, 192]
# Result: [8, 64, 192]
```

### 5.3 Embedding Similarity
```python
def cosine_similarity(a, b):
    """Measure semantic similarity between embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example:
cat_emb = [0.8, 0.2, 0.1]
dog_emb = [0.7, 0.3, 0.2] 
similarity = cosine_similarity(cat_emb, dog_emb)  # ~0.95 (very similar)
```

## 6. Modern Trends and Future Directions

### 6.1 Contextualized Embeddings Evolution
```
Static (Word2Vec) → Contextual (BERT) → Dynamic (GPT-3) → Adaptive (GPT-4)
```

### 6.2 Larger Vocabularies
- **GPT-3**: 50,257 tokens
- **PaLM**: 256,000 tokens  
- **Trend**: Larger vocabularies, better rare word handling

### 6.3 Multimodal Embeddings
- **CLIP**: Text + Image embeddings in same space
- **DALL-E**: Text → Image generation
- **Flamingo**: Few-shot multimodal learning

### 6.4 Efficient Embeddings
- **Product Key Memory**: Reduce embedding matrix size
- **Adaptive Input**: Different embedding sizes for frequency tiers
- **Hash Embeddings**: Use hashing to reduce parameters

## 7. Practical Implementation Tips

### 7.1 Initialization Strategies
```python
# Xavier/Glorot (your Mini-BERT uses this)
std = np.sqrt(2.0 / (fan_in + fan_out))
embeddings = np.random.normal(0, std, (vocab_size, hidden_size))

# He initialization (for ReLU)
std = np.sqrt(2.0 / fan_in)

# Uniform (BERT original)
embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, hidden_size))
```

### 7.2 Normalization
```python
# L2 normalize embeddings (common in retrieval)
embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
```

### 7.3 Vocabulary Considerations
```python
# Special tokens (your Mini-BERT)
vocab = {
    '[PAD]': 0,    # Padding
    '[UNK]': 1,    # Unknown
    '[CLS]': 2,    # Classification
    '[SEP]': 3,    # Separator  
    '[MASK]': 4,   # Masked token
    # ... regular tokens
}
```

## 8. Embedding Quality Evaluation

### 8.1 Intrinsic Evaluation
- **Word Similarity**: Correlation with human judgments
- **Word Analogy**: "king - man + woman = queen"
- **Clustering**: Do similar words cluster together?

### 8.2 Extrinsic Evaluation  
- **Downstream Tasks**: Use embeddings in classification, NER, etc.
- **Performance**: Better embeddings → better task performance

### 8.3 Visualization
```python
# t-SNE or UMAP for 2D visualization
from sklearn.manifold import TSNE
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
# Plot to see semantic clusters
```

## Summary for Linear Algebra Students

**Core Concept**: Embeddings are lookup tables (matrices) that map discrete symbols to continuous vectors.

**Key Operations**:
1. **Lookup**: `embedding_matrix[token_id]` 
2. **Similarity**: `cosine(vec1, vec2)`
3. **Combination**: `vec1 + vec2` (element-wise addition)
4. **Transformation**: `vec @ weight_matrix`

**Why It Works**: 
- Similar concepts have similar vectors
- Vector arithmetic captures semantic relationships
- Neural networks learn meaningful representations through gradient descent

The magic happens during training - the model learns to adjust these vectors so that semantically similar tokens have similar embeddings!