"""
Configuration for Mini-BERT implementation.
All hyperparameters and settings centralized here.
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ModelConfig:
    """Mini-BERT model architecture configuration."""
    # Core architecture (fixed for this implementation)
    num_layers: int = 3           # L = 3 transformer layers
    hidden_size: int = 192        # H = 192 hidden dimensions
    num_attention_heads: int = 4  # A = 4 attention heads (H % A == 0)
    intermediate_size: int = 768  # I = 768 FFN inner size (≈4H)
    max_sequence_length: int = 64 # T = 64 max sequence length
    vocab_size: int = 8192        # V = 8K vocabulary size
    
    # Computed properties
    @property
    def attention_head_size(self) -> int:
        """Size of each attention head: H/A = 192/4 = 48."""
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads
    
    # Activation and dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Special tokens
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4

@dataclass 
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Batch sizes (designed for 32GB RAM)
    train_batch_size: int = 32     # Logical batch size
    train_micro_batch_size: int = 8 # Physical batch size (for memory)
    gradient_accumulation_steps: int = 4  # 32/8 = 4 accumulation steps
    
    # Learning rate and optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # MLM masking strategy
    mlm_probability: float = 0.15  # 15% token masking
    replace_prob: float = 0.8      # 80% [MASK], 10% random, 10% unchanged
    random_prob: float = 0.1
    
    # Training schedule
    num_train_steps: int = 100_000
    warmup_steps: int = 10_000
    save_steps: int = 10_000
    logging_steps: int = 100
    
    # Data settings
    max_seq_length: int = 64
    short_seq_prob: float = 0.1    # Probability of shorter sequences
    
    # Gradient checking
    gradient_check_freq: int = 1000  # Check gradients every N steps
    finite_diff_epsilon: float = 1e-5

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    # Data paths (using existing repository data)
    train_data_path: str = "data/bert_50k_sample.txt"  # 53MB sample for testing
    vocab_save_path: str = "mini_bert/tokenizer_vocab.pkl"
    model_save_path: str = "mini_bert/model_checkpoint.pkl"
    
    # Text preprocessing
    max_lines_for_vocab: Optional[int] = 100_000  # Limit for vocab building
    min_word_freq: int = 2         # Minimum frequency for vocabulary
    
    # Data loading
    buffer_size: int = 10_000      # Number of examples to buffer
    num_workers: int = 4           # Parallel data loading

@dataclass  
class SystemConfig:
    """System and resource configuration."""
    # Hardware constraints
    max_memory_gb: float = 10.0    # Target max memory usage
    num_cpu_cores: int = 8         # Available CPU cores
    
    # Monitoring
    memory_check_freq: int = 100   # Check memory every N steps
    profiling_enabled: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    # Paths
    project_root: str = "mini_bert/"
    log_dir: str = "mini_bert/logs/"
    checkpoint_dir: str = "mini_bert/checkpoints/"

# Global configuration instances
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
SYSTEM_CONFIG = SystemConfig()

def get_parameter_count(config: ModelConfig) -> int:
    """Calculate total number of parameters for the model."""
    # Embeddings
    token_embeddings = config.vocab_size * config.hidden_size
    position_embeddings = config.max_sequence_length * config.hidden_size
    
    # Per transformer layer
    # Attention: Q,K,V projections + output projection + 2 layer norms
    attention_params = (3 * config.hidden_size * config.hidden_size + 
                       config.hidden_size * config.hidden_size + 
                       2 * config.hidden_size)
    
    # Feed-forward: W1 + W2 + 2 layer norms  
    ffn_params = (config.hidden_size * config.intermediate_size +
                  config.intermediate_size * config.hidden_size +
                  2 * config.hidden_size)
    
    layer_params = attention_params + ffn_params
    
    # Final layer norm + MLM head
    final_ln = 2 * config.hidden_size
    mlm_head = config.hidden_size * config.vocab_size
    
    total = (token_embeddings + position_embeddings + 
             config.num_layers * layer_params + 
             final_ln + mlm_head)
    
    return total

def estimate_memory_usage(config: ModelConfig, batch_size: int) -> dict:
    """Estimate memory usage for model and activations."""
    param_count = get_parameter_count(config)
    
    # Parameters (weights + gradients + optimizer state)
    param_memory_mb = param_count * 4 / (1024**2)      # Float32 weights
    grad_memory_mb = param_count * 4 / (1024**2)       # Float32 gradients  
    optimizer_memory_mb = param_count * 8 / (1024**2)  # Adam state (m,v)
    
    # Activations per batch
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

if __name__ == "__main__":
    # Print configuration summary
    print("Mini-BERT Configuration Summary")
    print("=" * 40)
    print(f"Model: L={MODEL_CONFIG.num_layers}, H={MODEL_CONFIG.hidden_size}, "
          f"A={MODEL_CONFIG.num_attention_heads}, I={MODEL_CONFIG.intermediate_size}")
    print(f"Sequence length: {MODEL_CONFIG.max_sequence_length}")
    print(f"Vocabulary size: {MODEL_CONFIG.vocab_size}")
    
    param_count = get_parameter_count(MODEL_CONFIG)
    print(f"\nTotal parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    memory_est = estimate_memory_usage(MODEL_CONFIG, TRAINING_CONFIG.train_micro_batch_size)
    print(f"\nMemory estimate (batch_size={TRAINING_CONFIG.train_micro_batch_size}):")
    for key, value in memory_est.items():
        if key.endswith("_mb"):
            print(f"  {key}: {value:.1f} MB")
        else:
            print(f"  {key}: {value:,}")
    
    print(f"\nTraining: {TRAINING_CONFIG.num_train_steps:,} steps, "
          f"batch_size={TRAINING_CONFIG.train_batch_size} "
          f"(accumulate {TRAINING_CONFIG.gradient_accumulation_steps})")