"""
Modern BERT configuration using Hugging Face ecosystem.
Demonstrates production-ready approach vs educational NumPy implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import BertConfig
import torch


@dataclass
class ModernBertConfig:
    """Configuration for modern BERT implementation."""
    
    # Model selection (can use any BERT variant from HuggingFace)
    model_name: str = "bert-base-uncased"  # Default to standard BERT
    custom_vocab_path: Optional[str] = "models/50k/bert_50k_vocab.txt"
    
    # Architecture overrides (if creating custom model)
    num_hidden_layers: int = 6  # Matching our larger_bert design
    hidden_size: int = 384
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    max_position_embeddings: int = 128
    vocab_size: int = 50000  # Using our 50k vocabulary
    
    # Training configuration
    learning_rate: float = 5e-5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Advanced training options
    fp16: bool = True  # Mixed precision training
    gradient_checkpointing: bool = True  # Memory efficiency
    use_8bit: bool = False  # 8-bit Adam optimizer
    use_lora: bool = False  # Parameter-efficient fine-tuning
    
    # LoRA configuration (if enabled)
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Data configuration
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 4
    
    # Paths
    output_dir: str = "modern_bert/outputs"
    logging_dir: str = "modern_bert/logs"
    cache_dir: str = "modern_bert/cache"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "modern-bert-50k"
    experiment_name: str = "bert-50k-custom"
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    
    # Hardware
    no_cuda: bool = False
    seed: int = 42
    
    def to_hf_config(self) -> BertConfig:
        """Convert to HuggingFace BertConfig."""
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Dataset selection
    dataset_name: str = "wikitext"  # Can use any HF dataset
    dataset_config: str = "wikitext-103-raw-v1"
    custom_data_path: Optional[str] = "data/bert_50k_sample.txt"
    
    # Preprocessing
    tokenizer_name: str = "bert-base-uncased"  # Or custom tokenizer
    use_custom_tokenizer: bool = True
    custom_tokenizer_path: str = "models/50k/bert_50k_tokenizer.pkl"
    
    # Data splits
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    max_train_samples: Optional[int] = None  # Limit samples for debugging
    max_eval_samples: Optional[int] = 1000
    
    # Processing options
    line_by_line: bool = True  # Process text line by line
    pad_to_max_length: bool = True
    return_special_tokens_mask: bool = True
    

@dataclass 
class ModelComparison:
    """Configuration to compare different approaches."""
    
    implementations: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "mini_bert_numpy": {
            "type": "numpy",
            "params": 3_200_000,
            "vocab": 8_192,
            "layers": 3,
            "hidden": 192,
            "library": "pure numpy"
        },
        "larger_bert_numpy": {
            "type": "numpy", 
            "params": 40_000_000,
            "vocab": 50_000,
            "layers": 6,
            "hidden": 384,
            "library": "pure numpy"
        },
        "modern_bert_small": {
            "type": "huggingface",
            "params": 40_000_000,
            "vocab": 50_000,
            "layers": 6,
            "hidden": 384,
            "library": "transformers + pytorch"
        },
        "bert_base_uncased": {
            "type": "huggingface",
            "params": 110_000_000,
            "vocab": 30_522,
            "layers": 12,
            "hidden": 768,
            "library": "transformers + pytorch"
        }
    })


# Global instances
MODERN_BERT_CONFIG = ModernBertConfig()
DATA_CONFIG = DataConfig()
MODEL_COMPARISON = ModelComparison()


# Preset configurations for common scenarios
class PresetConfigs:
    """Preset configurations for different use cases."""
    
    @staticmethod
    def debug_config() -> ModernBertConfig:
        """Small config for debugging."""
        config = ModernBertConfig()
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.train_batch_size = 4
        config.num_train_epochs = 1
        config.fp16 = False
        return config
    
    @staticmethod
    def efficient_config() -> ModernBertConfig:
        """Memory-efficient configuration."""
        config = ModernBertConfig()
        config.gradient_checkpointing = True
        config.use_8bit = True
        config.use_lora = True
        config.train_batch_size = 16
        return config
    
    @staticmethod
    def production_config() -> ModernBertConfig:
        """Production-ready configuration."""
        config = ModernBertConfig()
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.intermediate_size = 3072
        config.train_batch_size = 256  # With gradient accumulation
        config.fp16 = True
        config.use_wandb = True
        return config