"""
Adaptive BERT Training System

An intelligent BERT training system that automatically adjusts resource usage
based on other Python processes running on the system.
"""

from .adaptive_resource_manager import adaptive_manager, AdaptiveResourceManager
from .adaptive_bert_trainer import AdaptiveBERTTrainer  
from .run_adaptive_bert import InteractiveController

__version__ = "1.0.0"
__all__ = [
    "adaptive_manager",
    "AdaptiveResourceManager", 
    "AdaptiveBERTTrainer",
    "InteractiveController"
]