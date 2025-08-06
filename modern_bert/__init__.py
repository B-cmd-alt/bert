"""
Modern BERT: Production-ready implementation using HuggingFace ecosystem.
Complements the educational Mini-BERT with industry best practices.
"""

from .model import ModernBert, ModernBertForClassification
from .config import (
    ModernBertConfig,
    DataConfig,
    PresetConfigs,
    MODERN_BERT_CONFIG,
    DATA_CONFIG
)
from .data_pipeline import ModernDataPipeline
from .train import ModernBertTrainer
from .fine_tune import FineTuner, FineTuningConfig, TaskConfigs
from .inference import ModernBertInference, ModernBertEvaluator, ModelOptimizer

__version__ = "1.0.0"

__all__ = [
    # Core model classes
    'ModernBert',
    'ModernBertForClassification',
    
    # Configuration
    'ModernBertConfig',
    'DataConfig', 
    'FineTuningConfig',
    'PresetConfigs',
    'TaskConfigs',
    'MODERN_BERT_CONFIG',
    'DATA_CONFIG',
    
    # Training and fine-tuning
    'ModernBertTrainer',
    'FineTuner',
    'ModernDataPipeline',
    
    # Inference and evaluation
    'ModernBertInference',
    'ModernBertEvaluator', 
    'ModelOptimizer',
]