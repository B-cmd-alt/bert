"""
Configuration for Larger BERT implementation with 50k vocabulary.
Scales up from Mini-BERT while maintaining learning clarity.
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class LargerBERTConfig:
    """Larger BERT model architecture configuration."""
    # Architecture scaling from Mini-BERT
    # Mini-BERT: L=3, H=192, A=4, I=768
    # Larger-BERT: L=6, H=384, A=8, I=1536
    
    num_layers: int = 6           # L = 6 transformer layers (2x mini)
    hidden_size: int = 384        # H = 384 hidden dimensions (2x mini)
    num_attention_heads: int = 8  # A = 8 attention heads (2x mini)
    intermediate_size: int = 1536 # I = 1536 FFN inner size (4H)
    max_sequence_length: int = 128 # T = 128 max sequence length (2x mini)
    vocab_size: int = 50000       # V = 50K vocabulary size
    
    # Computed properties
    @property
    def attention_head_size(self) -> int:
        """Size of each attention head: H/A = 384/8 = 48."""
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads
    
    # Activation and regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Special tokens (consistent with 50k vocab)
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    
    # Model type identifier
    model_type: str = "larger_bert"

@dataclass 
class LargerBERTTrainingConfig:
    """Training hyperparameters for Larger BERT."""
    # Batch sizes (designed for 32GB RAM with larger model)
    train_batch_size: int = 16      # Logical batch size (reduced from 32)
    train_micro_batch_size: int = 4  # Physical batch size (reduced from 8)
    gradient_accumulation_steps: int = 4  # 16/4 = 4 accumulation steps
    
    # Learning rate and optimization
    learning_rate: float = 5e-5    # Slightly lower for larger model
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1      # 10% warmup
    
    # MLM masking strategy (same as mini-BERT)
    mlm_probability: float = 0.15  # 15% token masking
    replace_prob: float = 0.8      # 80% [MASK], 10% random, 10% unchanged
    random_prob: float = 0.1
    
    # Training schedule
    num_train_steps: int = 200_000   # More steps for larger model
    save_steps: int = 10_000
    logging_steps: int = 100
    eval_steps: int = 1000
    
    # Data settings
    max_seq_length: int = 128
    short_seq_prob: float = 0.1      # Probability of shorter sequences
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    mixed_precision: bool = False        # Can enable if supported

@dataclass
class LargerBERTDataConfig:
    """Data configuration for Larger BERT."""
    # Data paths
    train_data_path: str = "data/bert_50k_sample.txt"
    vocab_path: str = "models/50k/bert_50k_vocab.txt"
    tokenizer_path: str = "models/50k/bert_50k_tokenizer.pkl"
    
    # Model save paths
    model_save_dir: str = "larger_bert/checkpoints/"
    model_name: str = "larger_bert_50k"
    
    # Data loading
    buffer_size: int = 10_000
    num_workers: int = 4

# Global configuration instances
LARGER_BERT_CONFIG = LargerBERTConfig()
LARGER_BERT_TRAINING_CONFIG = LargerBERTTrainingConfig()
LARGER_BERT_DATA_CONFIG = LargerBERTDataConfig()

def get_parameter_count(config: LargerBERTConfig) -> int:
    """Calculate total number of parameters for the larger model."""
    # Embeddings
    token_embeddings = config.vocab_size * config.hidden_size
    position_embeddings = config.max_sequence_length * config.hidden_size
    
    # Per transformer layer
    # Attention: Q,K,V,O projections + 2 layer norms
    attention_params = (4 * config.hidden_size * config.hidden_size + 
                       2 * config.hidden_size)
    
    # Feed-forward: W1, b1, W2, b2 + 2 layer norms  
    ffn_params = (config.hidden_size * config.intermediate_size +
                  config.intermediate_size +
                  config.intermediate_size * config.hidden_size +
                  config.hidden_size +
                  2 * config.hidden_size)
    
    layer_params = attention_params + ffn_params
    
    # Final layer norm + MLM head
    final_ln = 2 * config.hidden_size
    mlm_head = config.hidden_size * config.vocab_size + config.vocab_size
    
    total = (token_embeddings + position_embeddings + 
             config.num_layers * layer_params + 
             final_ln + mlm_head)
    
    return total

def estimate_memory_usage(config: LargerBERTConfig, batch_size: int) -> dict:
    """Estimate memory usage for model and training."""
    param_count = get_parameter_count(config)
    
    # Parameters (weights + gradients + optimizer state)
    param_memory_mb = param_count * 4 / (1024**2)      # Float32 weights
    grad_memory_mb = param_count * 4 / (1024**2)       # Float32 gradients  
    optimizer_memory_mb = param_count * 8 / (1024**2)  # Adam state (m,v)
    
    # Activations per batch (rough estimate)
    seq_len = config.max_sequence_length
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    heads = config.num_attention_heads
    layers = config.num_layers
    
    # Rough activation memory estimate
    input_emb = batch_size * seq_len * hidden
    attention_scores = batch_size * heads * seq_len * seq_len * layers
    ffn_intermediate = batch_size * seq_len * intermediate * layers
    
    activation_memory_mb = (input_emb + attention_scores + ffn_intermediate) * 4 / (1024**2)
    
    return {
        "parameters_mb": param_memory_mb,
        "gradients_mb": grad_memory_mb, 
        "optimizer_mb": optimizer_memory_mb,
        "activations_mb": activation_memory_mb,
        "total_mb": param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb,
        "parameter_count": param_count
    }

def compare_with_mini_bert():
    """Compare Larger BERT with Mini-BERT configurations."""
    from mini_bert.config import MODEL_CONFIG as MINI_CONFIG
    
    print("Model Comparison: Mini-BERT vs Larger-BERT")
    print("=" * 50)
    
    # Architecture comparison
    print("\nArchitecture:")
    print(f"{'Parameter':<20} {'Mini-BERT':<15} {'Larger-BERT':<15} {'Scale Factor':<10}")
    print("-" * 60)
    
    params = [
        ("Layers (L)", MINI_CONFIG.num_layers, LARGER_BERT_CONFIG.num_layers),
        ("Hidden Size (H)", MINI_CONFIG.hidden_size, LARGER_BERT_CONFIG.hidden_size),
        ("Attention Heads (A)", MINI_CONFIG.num_attention_heads, LARGER_BERT_CONFIG.num_attention_heads),
        ("FFN Size (I)", MINI_CONFIG.intermediate_size, LARGER_BERT_CONFIG.intermediate_size),
        ("Max Seq Length", MINI_CONFIG.max_sequence_length, LARGER_BERT_CONFIG.max_sequence_length),
        ("Vocabulary Size", MINI_CONFIG.vocab_size, LARGER_BERT_CONFIG.vocab_size),
    ]
    
    for name, mini_val, larger_val in params:
        scale = larger_val / mini_val
        print(f"{name:<20} {mini_val:<15} {larger_val:<15} {scale:<10.1f}x")
    
    # Parameter count comparison
    mini_params = get_parameter_count(MINI_CONFIG)
    larger_params = get_parameter_count(LARGER_BERT_CONFIG)
    
    print(f"\nTotal Parameters:")
    print(f"Mini-BERT:   {mini_params:,} ({mini_params/1e6:.1f}M)")
    print(f"Larger-BERT: {larger_params:,} ({larger_params/1e6:.1f}M)")
    print(f"Scale Factor: {larger_params/mini_params:.1f}x")
    
    # Memory comparison
    batch_size = 4
    mini_memory = estimate_memory_usage(MINI_CONFIG, batch_size)
    larger_memory = estimate_memory_usage(LARGER_BERT_CONFIG, batch_size)
    
    print(f"\nMemory Usage (batch_size={batch_size}):")
    print(f"{'Component':<20} {'Mini-BERT (MB)':<15} {'Larger-BERT (MB)':<15}")
    print("-" * 50)
    
    for key in ["parameters_mb", "gradients_mb", "optimizer_mb", "activations_mb", "total_mb"]:
        print(f"{key.replace('_mb', ''):<20} {mini_memory[key]:<15.1f} {larger_memory[key]:<15.1f}")

if __name__ == "__main__":
    # Print configuration summary
    print("Larger BERT Configuration")
    print("=" * 40)
    print(f"Model: L={LARGER_BERT_CONFIG.num_layers}, H={LARGER_BERT_CONFIG.hidden_size}, "
          f"A={LARGER_BERT_CONFIG.num_attention_heads}, I={LARGER_BERT_CONFIG.intermediate_size}")
    print(f"Sequence length: {LARGER_BERT_CONFIG.max_sequence_length}")
    print(f"Vocabulary size: {LARGER_BERT_CONFIG.vocab_size}")
    
    param_count = get_parameter_count(LARGER_BERT_CONFIG)
    print(f"\nTotal parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    memory_est = estimate_memory_usage(LARGER_BERT_CONFIG, LARGER_BERT_TRAINING_CONFIG.train_micro_batch_size)
    print(f"\nMemory estimate (batch_size={LARGER_BERT_TRAINING_CONFIG.train_micro_batch_size}):")
    for key, value in memory_est.items():
        if key.endswith("_mb"):
            print(f"  {key}: {value:.1f} MB")
        else:
            print(f"  {key}: {value:,}")
    
    print("\n" + "=" * 40)
    compare_with_mini_bert()