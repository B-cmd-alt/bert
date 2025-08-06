"""
Modern BERT implementation using Hugging Face Transformers.
Shows the production-ready approach vs educational NumPy implementation.
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, 
    BertForMaskedLM,
    BertConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, Optional, Union, Tuple
import os


class ModernBert:
    """
    Modern BERT implementation using HuggingFace ecosystem.
    Supports custom architectures, pretrained models, and efficient training.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        
        # Initialize model
        if config.model_name == "custom":
            self.model = self._create_custom_model()
        else:
            self.model = self._load_pretrained_model()
        
        # Move model to device
        self.model.to(self.device)
        
        # Apply optimizations
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if config.use_lora:
            self.model = self._apply_lora()
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
        
        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {self.count_parameters():,}")
        if config.use_lora:
            print(f"Trainable parameters: {self.count_trainable_parameters():,}")
    
    def _create_custom_model(self) -> BertForMaskedLM:
        """Create custom BERT model with specified architecture."""
        bert_config = self.config.to_hf_config()
        model = BertForMaskedLM(bert_config)
        
        # Initialize weights with better initialization
        model.apply(self._init_weights)
        
        return model
    
    def _load_pretrained_model(self) -> BertForMaskedLM:
        """Load pretrained model from HuggingFace."""
        model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        # Resize token embeddings if using custom vocabulary
        if self.config.custom_vocab_path and self.config.vocab_size != model.config.vocab_size:
            model.resize_token_embeddings(self.config.vocab_size)
            
        return model
    
    def _apply_lora(self) -> nn.Module:
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.MASKED_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _load_tokenizer(self):
        """Load tokenizer (HuggingFace or custom)."""
        if self.config.use_custom_tokenizer and os.path.exists(self.config.custom_tokenizer_path):
            # For custom tokenizer, we'd need to implement a wrapper
            # For now, use HuggingFace tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                cache_dir=self.config.cache_dir
            )
            print("Note: Using HuggingFace tokenizer. Custom tokenizer integration needed.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name or self.config.model_name,
                cache_dir=self.config.cache_dir
            )
        
        return tokenizer
    
    @staticmethod
    def _init_weights(module):
        """Initialize weights with better defaults."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Labels for MLM loss [batch_size, seq_length]
            
        Returns:
            Dict containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }
    
    def generate_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate embeddings for input sequences."""
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Return [CLS] token embeddings
        return outputs.last_hidden_state[:, 0, :]
    
    def predict_masked_tokens(
        self,
        texts: Union[str, List[str]],
        return_top_k: int = 5
    ) -> List[Dict]:
        """
        Predict masked tokens in text.
        
        Args:
            texts: Text or list of texts with [MASK] tokens
            return_top_k: Number of top predictions to return
            
        Returns:
            List of predictions for each masked position
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Find masked positions and get top predictions
        results = []
        for i, text in enumerate(texts):
            masked_indices = (inputs.input_ids[i] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            
            for idx in masked_indices:
                logits = predictions[i, idx]
                top_k_tokens = torch.topk(logits, return_top_k, dim=-1)
                
                predictions_list = []
                for j in range(return_top_k):
                    token_id = top_k_tokens.indices[j].item()
                    score = torch.softmax(logits, dim=-1)[token_id].item()
                    token = self.tokenizer.decode([token_id])
                    predictions_list.append({
                        "token": token,
                        "score": score,
                        "token_id": token_id
                    })
                
                results.append({
                    "position": idx.item(),
                    "predictions": predictions_list
                })
        
        return results
    
    def count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_model(self, save_path: str):
        """Save model to disk."""
        os.makedirs(save_path, exist_ok=True)
        
        if self.config.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        # Save config
        torch.save(self.config, os.path.join(save_path, "training_config.pt"))
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model from disk."""
        if self.config.use_lora:
            # Load LoRA weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, load_path)
        else:
            # Load full model
            self.model = AutoModelForMaskedLM.from_pretrained(load_path)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")
    
    def compare_with_numpy_implementation(self):
        """Compare with NumPy implementation."""
        print("\n" + "="*60)
        print("Comparison: NumPy vs Modern Implementation")
        print("="*60)
        
        print("\nNumPy Implementation (Educational):")
        print("- Pure NumPy arrays")
        print("- Explicit mathematical operations")
        print("- Manual backpropagation")
        print("- No GPU support")
        print("- ~1000 lines of code")
        
        print("\nModern Implementation (Production):")
        print("- PyTorch tensors with autograd")
        print("- Optimized CUDA kernels")
        print("- Automatic differentiation")
        print("- Multi-GPU support")
        print("- ~100 lines of code")
        
        print("\nPerformance Comparison (estimated):")
        print("- Training speed: 100-1000x faster")
        print("- Memory efficiency: 2-5x better")
        print("- Development time: 10x faster")
        
        print("\nFeature Comparison:")
        features = [
            ("Mixed precision training", "❌", "✅"),
            ("Gradient checkpointing", "❌", "✅"),
            ("Multi-GPU training", "❌", "✅"),
            ("Pretrained models", "❌", "✅"),
            ("Educational clarity", "✅", "❌"),
            ("Implementation transparency", "✅", "❌"),
        ]
        
        print(f"\n{'Feature':<30} {'NumPy':<10} {'Modern':<10}")
        print("-"*50)
        for feature, numpy_support, modern_support in features:
            print(f"{feature:<30} {numpy_support:<10} {modern_support:<10}")


class ModernBertForClassification(ModernBert):
    """Extended version for sequence classification tasks."""
    
    def __init__(self, config, num_labels: int = 2):
        # Modify config for classification
        config.model_name = config.model_name.replace("ForMaskedLM", "ForSequenceClassification")
        super().__init__(config)
        
        # Replace MLM head with classification head
        self.model = self._create_classification_model(num_labels)
        self.model.to(self.device)
    
    def _create_classification_model(self, num_labels: int):
        """Create classification model."""
        from transformers import BertForSequenceClassification
        
        if self.config.model_name == "custom":
            bert_config = self.config.to_hf_config()
            bert_config.num_labels = num_labels
            model = BertForSequenceClassification(bert_config)
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_labels,
                cache_dir=self.config.cache_dir
            )
        
        return model
    
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Predict class labels for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()


if __name__ == "__main__":
    # Test modern BERT implementation
    from modern_bert.config import ModernBertConfig
    
    # Create config
    config = ModernBertConfig()
    config.model_name = "bert-base-uncased"
    config.use_lora = True
    
    # Initialize model
    model = ModernBert(config)
    
    # Test prediction
    text = "The capital of France is [MASK]."
    results = model.predict_masked_tokens(text)
    
    print(f"\nPredictions for: '{text}'")
    for result in results:
        print(f"\nPosition {result['position']}:")
        for pred in result['predictions']:
            print(f"  {pred['token']}: {pred['score']:.3f}")
    
    # Compare implementations
    model.compare_with_numpy_implementation()