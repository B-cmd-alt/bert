# Larger BERT: Scaling Up from Mini-BERT

This implementation scales up Mini-BERT (3.2M parameters) to a more practical size (~40M parameters) while maintaining educational clarity and pure NumPy implementation.

## Architecture Comparison

| Component | Mini-BERT | Larger-BERT | Scale Factor |
|-----------|-----------|-------------|--------------|
| **Vocabulary Size** | 8,192 | 50,000 | 6.1x |
| **Layers (L)** | 3 | 6 | 2x |
| **Hidden Size (H)** | 192 | 384 | 2x |
| **Attention Heads (A)** | 4 | 8 | 2x |
| **Head Size (d_k)** | 48 | 48 | 1x |
| **FFN Size (I)** | 768 | 1,536 | 2x |
| **Max Sequence Length** | 64 | 128 | 2x |
| **Total Parameters** | 3.2M | ~40M | 12.5x |

## Key Design Decisions

### 1. **Vocabulary Scaling**
- Uses existing 50k vocabulary from `models/50k/bert_50k_vocab.txt`
- Better coverage of real-world text
- Subword tokenization for handling out-of-vocabulary words

### 2. **Architecture Scaling**
- **Doubled depth**: 6 layers provide better feature extraction
- **Doubled width**: 384 hidden dimensions capture richer representations  
- **Maintained head size**: 48 dimensions per head (same as Mini-BERT) for stability
- **Quadratic FFN scaling**: FFN size = 4×hidden_size (standard practice)

### 3. **Memory Optimization**
- Gradient accumulation for larger effective batch sizes
- Dropout for regularization (not in Mini-BERT)
- Careful initialization for training stability with larger model

## Mathematical Formulation

The core mathematics remains identical to Mini-BERT:

### Multi-Head Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
MultiHead = Concat(head_1, ..., head_8)W^O

where each head computes:
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Feed-Forward Network
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
W_1: [384, 1536], W_2: [1536, 384]
```

### Layer Structure
```
x' = LayerNorm(x + MultiHeadAttention(x))
output = LayerNorm(x' + FFN(x'))
```

## Memory Requirements

### Training Memory Estimation (per batch size)

| Batch Size | Parameters | Gradients | Optimizer | Activations | Total |
|------------|------------|-----------|-----------|-------------|-------|
| 1 | 152 MB | 152 MB | 304 MB | ~50 MB | ~658 MB |
| 4 | 152 MB | 152 MB | 304 MB | ~200 MB | ~808 MB |
| 8 | 152 MB | 152 MB | 304 MB | ~400 MB | ~1 GB |

### Parameter Breakdown

- **Embeddings**: 19.2M parameters
  - Token embeddings: 50,000 × 384 = 19.2M
  - Position embeddings: 128 × 384 = 49K
  
- **Transformer Layers**: 20.3M parameters (6 layers × 3.4M each)
  - Attention (Q,K,V,O): 4 × 384² = 590K
  - FFN: 384×1536 + 1536×384 = 1.18M
  - Layer norms: negligible
  
- **MLM Head**: 19.2M parameters
  - Projection: 384 × 50,000 = 19.2M

## Usage

### 1. Model Initialization
```python
from larger_bert import LargerBERT, LARGER_BERT_CONFIG

# Initialize model
model = LargerBERT(LARGER_BERT_CONFIG)
print(f"Model parameters: {model.get_parameter_count():,}")
```

### 2. Forward Pass
```python
import numpy as np

# Example input
batch_size, seq_len = 4, 32
input_ids = np.random.randint(0, 50000, (batch_size, seq_len))
attention_mask = np.ones((batch_size, seq_len))

# Forward pass
logits, cache = model.forward(input_ids, attention_mask)
print(f"Output shape: {logits.shape}")  # [4, 32, 50000]
```

### 3. Training
```python
from larger_bert.train import LargerBERTTrainer
from mini_bert.data import MLMDataGenerator

# Initialize trainer
trainer = LargerBERTTrainer(model)

# Load tokenizer and data
tokenizer = trainer.load_tokenizer()
data_generator = MLMDataGenerator(
    data_path="data/bert_50k_sample.txt",
    tokenizer=tokenizer,
    max_seq_length=128,
    batch_size=4
)

# Train model
trainer.train(data_generator, num_steps=1000)
```

### 4. Loading Pretrained Model
```python
# Load checkpoint
model.load_checkpoint("larger_bert/checkpoints/larger_bert_50k_best.pkl")

# Use for inference
model.set_training(False)
logits, _ = model.forward(input_ids, attention_mask)
```

## Comparison with Standard BERT

| Model | Parameters | Hidden | Layers | Heads | Vocab |
|-------|------------|--------|--------|-------|-------|
| **Mini-BERT** | 3.2M | 192 | 3 | 4 | 8K |
| **Larger-BERT** | 40M | 384 | 6 | 8 | 50K |
| **BERT-Base** | 110M | 768 | 12 | 12 | 30K |
| **BERT-Large** | 340M | 1024 | 24 | 16 | 30K |

## Educational Benefits

1. **Gradual Scaling**: Shows how to scale from toy models to practical sizes
2. **Same Core Principles**: Uses identical mathematics as Mini-BERT
3. **Memory Awareness**: Demonstrates real memory constraints
4. **Production Techniques**: Includes dropout, gradient accumulation, checkpointing

## Implementation Highlights

### 1. Careful Initialization
```python
# Scaled initialization for deeper network
init_scale = 0.02 / np.sqrt(2 * num_layers)
W = np.random.normal(0, xavier_std * init_scale, shape)
```

### 2. Gradient Accumulation
```python
# Accumulate gradients over multiple micro-batches
for micro_batch in micro_batches:
    loss, grads = compute_gradients(micro_batch)
    accumulated_grads += grads

# Update after accumulation
optimizer.step(accumulated_grads / num_accumulation_steps)
```

### 3. Memory-Efficient Training
- Dropout during training only
- Gradient checkpointing (optional)
- Mixed precision training support (framework)

## Next Steps

1. **Fine-tuning**: Adapt for downstream tasks (classification, QA)
2. **Optimization**: Implement Flash Attention, gradient checkpointing
3. **Scaling Further**: Path to 100M+ parameter models
4. **Deployment**: Model quantization and serving

## Learning Path

1. Start with **Mini-BERT** to understand core concepts
2. Move to **Larger-BERT** to see practical scaling
3. Compare with **production BERT** implementations
4. Experiment with different architectures and hyperparameters

The beauty of this implementation is that it bridges the gap between educational toy models and production-scale systems while maintaining complete transparency in pure NumPy!