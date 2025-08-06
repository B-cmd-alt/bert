# Mini-BERT from Scratch: Complete Learning Journey

## 🎯 **Mathematical Understanding Through Pure NumPy Implementation**

This repository contains a complete Mini-BERT implementation designed for **deep mathematical understanding** of transformer architectures. Every component is implemented from scratch in pure NumPy with explicit mathematical derivations.

**✅ RECENTLY UPDATED**: Critical bug fixes applied (Dec 2024) - attention mechanism now works correctly, ~100x faster gradient computation, improved numerical stability. See `CRITICAL_FIXES_APPLIED.md` for details.

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install -r requirements.txt

# Test all components
python simple_test.py

# Verify critical fixes (recommended)
python test_critical_fixes.py

# Run comprehensive evaluation
python evaluate.py --quick

# Start learning journey
# See IMPROVEMENT_ROADMAP.md for detailed learning path
```

## Memory Budget Analysis (Dell XPS: 32GB RAM, i7-13620H)

### Model Parameters:
- **L=3 layers, H=192, A=4 heads, I=768 FFN, T=64 seq_len, V=8192 vocab**

### Parameter Count:
```
Embeddings:
- Token embeddings: V × H = 8,192 × 192 = 1,572,864
- Position embeddings: T × H = 64 × 192 = 12,288
- Total embeddings: 1,585,152 params

Per Transformer Layer:
- Multi-head attention:
  - Q,K,V projections: 3 × (H × H) = 3 × 192² = 110,592
  - Output projection: H × H = 192² = 36,864
  - Layer norm: 2 × H = 384
- Feed-forward:
  - W1: H × I = 192 × 768 = 147,456
  - W2: I × H = 768 × 192 = 147,456
  - Layer norm: 2 × H = 384
- Total per layer: 442,752 params

Total Model Parameters:
- Embeddings: 1,585,152
- 3 × Transformer layers: 3 × 442,752 = 1,328,256
- Final layer norm: 384
- MLM head: H × V = 192 × 8,192 = 1,572,864
- **Grand Total: 4,486,656 params ≈ 4.5M parameters**
```

### Memory Usage (Float32):
```
Model weights: 4.5M × 4 bytes = 18 MB
Gradients: 4.5M × 4 bytes = 18 MB
Optimizer state (Adam): 4.5M × 8 bytes = 36 MB
Total parameter memory: 72 MB

Per-batch activations (batch_size=8, seq_len=64):
- Input embeddings: 8 × 64 × 192 = 98,304 values = 393 KB
- Attention matrices: 8 × 4 × 64 × 64 × 3 layers = 393,216 values = 1.6 MB
- FFN activations: 8 × 64 × 768 × 3 layers = 1,179,648 values = 4.8 MB
- Total activations per batch: ~7 MB

Peak memory estimate: 72 MB + 7 MB = 79 MB per batch
With gradient accumulation (4 microbatches): ~100 MB total
```

### Timing Estimates (i7-13620H, 8 cores):
```
Forward pass: ~20ms per batch
Backward pass: ~40ms per batch
Total step time: ~60ms
Steps per second: ~16
Target: 100K steps in ~1.7 hours (well under 12h limit)
```

## 📚 **Learning Path Overview**

### **🎓 Beginner (Weeks 1-2): Foundation**
1. **Day 1-3**: Run basic tests, understand config, trace forward pass
2. **Day 4-7**: Study mathematical derivations, implement key functions
3. **Day 8-14**: Understand training dynamics, run overfit tests

### **🔬 Intermediate (Weeks 3-4): Deep Dive**
1. **Week 3**: Training dynamics analysis, gradient flow understanding
2. **Week 4**: Evaluation metrics, probing tasks, fine-tuning mechanics

### **🚀 Advanced (Weeks 5-8): Research Level**
1. **Weeks 5-6**: Architectural improvements, optimization variants
2. **Weeks 7-8**: Novel research experiments, scaling studies

**📖 See `IMPROVEMENT_ROADMAP.md` for detailed learning path with specific exercises and code examples.**

## 📁 **Project Structure**
```
mini_bert/
├── Core Implementation
│   ├── config.py              # Hyperparameters & model configuration
│   ├── tokenizer.py           # WordPiece tokenizer (8K vocab)
│   ├── model.py               # Mini-BERT architecture & forward pass
│   ├── gradients.py           # Manual backward pass implementations
│   └── utils.py               # Gradient checking & diagnostics
│
├── Training Pipeline
│   ├── mlm.py                 # MLM masking & loss functions
│   ├── optimizer.py           # AdamW optimizer with bias correction
│   ├── train_updated.py       # Complete training loop
│   └── data.py                # Data preprocessing pipeline
│
├── Evaluation Suite
│   ├── metrics.py             # Intrinsic metrics & sanity checks
│   ├── probe_pos.py           # POS tagging probe
│   ├── finetune_sst2.py       # SST-2 sentiment fine-tuning
│   └── evaluate.py            # Master evaluation script
│
├── Learning Tools
│   ├── simple_test.py         # Basic functionality tests
│   ├── test_integration.py    # End-to-end integration tests
│   ├── performance_analysis.py # Memory & timing analysis
│   └── optional_features.py   # Advanced features demo
│
└── Documentation
    ├── MATHEMATICAL_DERIVATIONS.md     # Complete mathematical foundation
    ├── IMPLEMENTATION_SUMMARY.md       # Technical implementation details
    ├── STEP6_EVALUATION_SUMMARY.md     # Evaluation framework summary
    └── IMPROVEMENT_ROADMAP.md          # Future improvements & learning path
```

## File Descriptions:

### tokenizer.py (~200 LOC)
- `WordPieceTokenizer` class with 8K vocabulary
- `encode()` / `decode()` with ## boundary markers
- Save/load functionality
- Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]

### model.py (~220 LOC)
- `MiniTransformerLayer` class
- `MiniBERT` class with L=3 layers
- Forward pass with explicit shape assertions
- Attention, FFN, LayerNorm components

### gradients.py (~250 LOC)
- Manual backward pass implementations
- `linear_backward()`, `softmax_backward()`, `layernorm_backward()`
- Gradient accumulation utilities

### train.py (~240 LOC)
- MLM training loop with 15% masking
- Gradient checking vs finite differences
- Batch accumulation (logical=32, physical=8)
- Loss tracking and diagnostics

### utils.py (~180 LOC)
- Memory profiling functions
- Gradient magnitude histograms  
- Shape checking utilities
- Timing and logging helpers

### data.py (~200 LOC)
- Text preprocessing pipeline
- Sequence preparation for BERT
- Dynamic masking for MLM
- Batch generation with padding

### config.py (~80 LOC)
- Hyperparameters in dataclasses
- Model configuration
- Training settings

## 🎯 **Best Learning Path for Mathematical Understanding**

### **Level 1: Mathematical Foundation (Days 1-7)**

#### **Day 1: Basic Understanding**
```bash
# Start here - see the system working
python simple_test.py
python config.py               # Understand hyperparameters
```

**Learn**: Model architecture, parameter counting, memory usage

#### **Day 2-3: Forward Pass Mathematics**
```bash
# Trace through forward pass step by step
python model.py

# Study mathematical derivations
# Read: MATHEMATICAL_DERIVATIONS.md sections 1-2
```

**Learn**: Embeddings, attention mechanism, feed-forward networks, layer normalization

#### **Day 4-5: Gradient Mathematics**
```bash
# Understand backward pass
python gradients.py

# Read: MATHEMATICAL_DERIVATIONS.md section 3
```

**Learn**: Chain rule application, gradient computation for each component

#### **Day 6-7: Training Mechanics**
```bash
# Understand MLM and optimization
python mlm.py
python optimizer.py

# Run overfit test to see learning
python evaluate.py --overfit_one_batch --quick
```

**Learn**: MLM masking strategy, AdamW optimization, loss computation

### **Level 2: Training Dynamics (Days 8-14)**

#### **Day 8-10: Training Analysis**
```bash
# Study training loop
python train_updated.py --help

# Analyze gradient behavior
python metrics.py --test
```

**Learn**: Gradient accumulation, learning rate scheduling, training stability

#### **Day 11-14: Evaluation Understanding**
```bash
# Run each evaluation component
python probe_pos.py --max_sentences 100
python finetune_sst2.py --epochs 1 --max_samples 500
python evaluate.py --quick
```

**Learn**: Probing tasks, transfer learning, evaluation metrics

### **Level 3: Advanced Analysis (Days 15-30)**

#### **Week 3: Deep Mathematical Insights**
- Implement attention visualization
- Analyze gradient flow patterns
- Study weight evolution during training
- Explore different architectural variants

#### **Week 4: Research-Level Understanding**
- Design custom probing tasks
- Implement novel optimization techniques
- Study representation geometry
- Conduct scaling experiments

## 🧮 **Key Mathematical Concepts to Master**

### **Essential Mathematics**
1. **Matrix Calculus**: Chain rule, Jacobians, gradients
2. **Attention Mechanism**: Scaled dot-product, multi-head architecture
3. **Optimization Theory**: AdamW, learning rate scheduling, gradient clipping
4. **Information Theory**: Cross-entropy, perplexity, KL divergence
5. **Linear Algebra**: Eigenvalues, matrix norms, SVD

### **Practical Skills**
1. **Gradient Checking**: Finite difference validation
2. **Numerical Stability**: Log-sum-exp tricks, overflow prevention
3. **Memory Management**: Activation checkpointing, efficient computation
4. **Evaluation Methodology**: Intrinsic vs extrinsic evaluation
5. **Debugging Techniques**: Overfit tests, gradient analysis

## 🔍 **Using Metrics for Mathematical Understanding**

### **Training Metrics**
```python
# Monitor these during training
- MLM loss and accuracy         # Language modeling quality
- Gradient norms by layer       # Learning dynamics
- Attention entropy             # Attention pattern diversity
- Parameter update magnitudes   # Which components are learning
```

### **Evaluation Metrics**
```python
# Use these to understand model quality
- Masked token accuracy         # Direct MLM performance
- Perplexity                   # Language modeling quality
- Probe task accuracy          # Linguistic knowledge
- Transfer learning performance # Representation quality
```

### **Analysis Metrics**
```python
# Advanced analysis tools
- Attention pattern analysis   # What patterns are learned
- Representation geometry      # How embeddings are structured
- Gradient flow analysis       # Training stability insights
- Weight evolution tracking    # Learning progression
```

## 📊 **Recommended Metrics Workflow**

### **Phase 1: Basic Training Validation**
1. **Overfit Test**: Verify model can learn (loss → 0 on single batch)
2. **Gradient Health**: Check for NaN, explosion, vanishing
3. **Memory Usage**: Ensure within budget throughout training
4. **Loss Curve**: Smooth decrease without oscillations

### **Phase 2: Learning Quality Assessment**
1. **MLM Accuracy**: Should reach 60-70% on held-out data
2. **Perplexity**: Should decrease from ~10000 to ~15-30
3. **POS Probe**: Should achieve >90% with trained model
4. **Attention Patterns**: Should show linguistic structure

### **Phase 3: Advanced Analysis**
1. **Layer-wise Analysis**: Different layers learn different features
2. **Head Specialization**: Different attention heads focus on different patterns
3. **Transfer Learning**: Representations should transfer to downstream tasks
4. **Scaling Behavior**: Understand how performance scales with model size

## 🚀 **Next Steps for Improvement**

### **Immediate Improvements (Week 1)**
1. **Enhanced Metrics**: Add attention analysis, weight tracking
2. **Better Logging**: More detailed training diagnostics
3. **Visualization Tools**: Plot attention patterns, gradient flows
4. **Interactive Analysis**: Jupyter notebooks for exploration

### **Medium-term Improvements (Weeks 2-4)**
1. **Architectural Variants**: Try different layer configurations
2. **Advanced Optimizers**: Implement Lion, Sophia, or other modern optimizers
3. **Curriculum Learning**: Progressive difficulty during training
4. **Multi-task Learning**: Additional pre-training objectives

### **Long-term Research (Weeks 5-8)**
1. **Novel Architectures**: Implement latest research findings
2. **Scaling Studies**: Understand parameter vs performance relationships
3. **Efficiency Improvements**: Better memory/compute trade-offs
4. **Domain Adaptation**: Adapt to specific domains or languages

**📖 See `IMPROVEMENT_ROADMAP.md` for detailed implementation guides and code examples.**

This completes your comprehensive Mini-BERT learning journey from mathematical foundations to advanced research topics.