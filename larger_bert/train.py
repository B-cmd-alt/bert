"""
Training script for Larger BERT with 50k vocabulary.
Adapted from Mini-BERT training with optimizations for larger model.
"""

import numpy as np
import pickle
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from larger_bert.model import LargerBERT
from larger_bert.config import (
    LARGER_BERT_CONFIG, 
    LARGER_BERT_TRAINING_CONFIG,
    LARGER_BERT_DATA_CONFIG
)

# Import utilities from mini_bert
from mini_bert.optimizer import AdamOptimizer
from mini_bert.gradients import compute_mlm_gradients
from mini_bert.data import create_mlm_batch, MLMDataGenerator


class LargerBERTTrainer:
    """Trainer class for Larger BERT model."""
    
    def __init__(self, model: LargerBERT, config=LARGER_BERT_TRAINING_CONFIG):
        self.model = model
        self.config = config
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer(
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        self.optimizer.initialize(model.params)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Logging
        self.training_history = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'time_per_step': []
        }
        
        # Create checkpoint directory
        os.makedirs(LARGER_BERT_DATA_CONFIG.model_save_dir, exist_ok=True)
    
    def load_tokenizer(self):
        """Load the 50k tokenizer."""
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            LARGER_BERT_DATA_CONFIG.tokenizer_path
        )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        print(f"Loaded tokenizer with {len(tokenizer.word_to_id)} tokens")
        return tokenizer
    
    def compute_loss_and_gradients(self, batch: Dict) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute MLM loss and gradients."""
        # Forward pass
        logits, cache = self.model.forward(
            batch['input_ids'], 
            batch['attention_mask']
        )
        
        # Compute loss only for masked positions
        masked_positions = batch['masked_positions']
        batch_size, seq_len = batch['input_ids'].shape
        
        # Get logits for masked positions
        masked_logits = []
        masked_labels = []
        
        for b in range(batch_size):
            for pos in masked_positions[b]:
                if pos >= 0:  # Valid position
                    masked_logits.append(logits[b, pos])
                    masked_labels.append(batch['labels'][b, pos])
        
        if not masked_logits:
            return 0.0, {}
        
        masked_logits = np.array(masked_logits)  # [num_masked, vocab_size]
        masked_labels = np.array(masked_labels)  # [num_masked]
        
        # Compute cross-entropy loss
        # Numerical stability: subtract max before exp
        shifted_logits = masked_logits - np.max(masked_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Loss for each masked token
        num_masked = len(masked_labels)
        loss = -np.sum(np.log(probs[np.arange(num_masked), masked_labels] + 1e-10)) / num_masked
        
        # Compute gradients using mini_bert's gradient computation
        gradients = compute_mlm_gradients(
            self.model, logits, batch, cache
        )
        
        return loss, gradients
    
    def get_learning_rate(self, step: int) -> float:
        """Linear warmup and linear decay learning rate schedule."""
        warmup_steps = int(self.config.num_train_steps * self.config.warmup_ratio)
        
        if step < warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / warmup_steps)
        else:
            # Linear decay
            remaining_steps = self.config.num_train_steps - step
            total_decay_steps = self.config.num_train_steps - warmup_steps
            return self.config.learning_rate * (remaining_steps / total_decay_steps)
    
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> float:
        """Clip gradients by global norm."""
        # Compute global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.config.max_grad_norm:
            clip_ratio = self.config.max_grad_norm / total_norm
            for key in gradients:
                gradients[key] *= clip_ratio
        
        return total_norm
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        start_time = time.time()
        
        # Set model to training mode
        self.model.set_training(True)
        
        # Compute loss and gradients
        loss, gradients = self.compute_loss_and_gradients(batch)
        
        # Clip gradients
        grad_norm = self.clip_gradients(gradients)
        
        # Get current learning rate
        current_lr = self.get_learning_rate(self.global_step)
        self.optimizer.learning_rate = current_lr
        
        # Update parameters
        self.optimizer.step(gradients, self.model.params)
        
        # Update global step
        self.global_step += 1
        
        # Compute step time
        step_time = time.time() - start_time
        
        # Return metrics
        return {
            'loss': loss,
            'learning_rate': current_lr,
            'gradient_norm': grad_norm,
            'time_per_step': step_time
        }
    
    def evaluate(self, data_generator: MLMDataGenerator, num_steps: int = 100) -> float:
        """Evaluate model on validation data."""
        self.model.set_training(False)
        total_loss = 0.0
        
        for step in range(num_steps):
            batch = data_generator.generate_batch()
            loss, _ = self.compute_loss_and_gradients(batch)
            total_loss += loss
        
        avg_loss = total_loss / num_steps
        self.model.set_training(True)
        return avg_loss
    
    def save_checkpoint(self, filepath: str = None):
        """Save training checkpoint."""
        if filepath is None:
            filepath = os.path.join(
                LARGER_BERT_DATA_CONFIG.model_save_dir,
                f"{LARGER_BERT_DATA_CONFIG.model_name}_step_{self.global_step}.pkl"
            )
        
        checkpoint = {
            'model_params': self.model.params,
            'optimizer_state': {
                'm': self.optimizer.m,
                'v': self.optimizer.v,
                't': self.optimizer.t
            },
            'config': self.model.config,
            'training_config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model.params = checkpoint['model_params']
        self.optimizer.m = checkpoint['optimizer_state']['m']
        self.optimizer.v = checkpoint['optimizer_state']['v']
        self.optimizer.t = checkpoint['optimizer_state']['t']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from step {self.global_step}")
    
    def train(self, data_generator: MLMDataGenerator, num_steps: int = None):
        """Main training loop."""
        if num_steps is None:
            num_steps = self.config.num_train_steps
        
        print(f"Starting training for {num_steps} steps...")
        print(f"Batch size: {self.config.train_micro_batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.train_batch_size}")
        
        # Training loop
        accumulated_loss = 0.0
        accumulated_gradients = None
        
        for step in range(self.global_step, num_steps):
            # Generate batch
            batch = data_generator.generate_batch()
            
            # Forward and backward pass
            loss, gradients = self.compute_loss_and_gradients(batch)
            
            # Accumulate gradients
            if accumulated_gradients is None:
                accumulated_gradients = {k: v.copy() for k, v in gradients.items()}
            else:
                for k in gradients:
                    accumulated_gradients[k] += gradients[k]
            
            accumulated_loss += loss
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Average gradients
                for k in accumulated_gradients:
                    accumulated_gradients[k] /= self.config.gradient_accumulation_steps
                
                # Training step with accumulated gradients
                metrics = {
                    'loss': accumulated_loss / self.config.gradient_accumulation_steps,
                    'learning_rate': self.get_learning_rate(self.global_step),
                    'gradient_norm': self.clip_gradients(accumulated_gradients),
                    'time_per_step': 0  # Will be updated in train_step
                }
                
                # Update parameters
                self.optimizer.learning_rate = metrics['learning_rate']
                self.optimizer.step(accumulated_gradients, self.model.params)
                
                # Update history
                for key, value in metrics.items():
                    self.training_history[key].append(value)
                
                # Reset accumulation
                accumulated_loss = 0.0
                accumulated_gradients = None
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = np.mean(self.training_history['loss'][-100:])
                    print(f"Step {self.global_step}/{num_steps} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"LR: {metrics['learning_rate']:.2e} | "
                          f"Grad Norm: {metrics['gradient_norm']:.2f}")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate(data_generator, num_steps=50)
                    print(f"Evaluation loss: {eval_loss:.4f}")
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        best_path = os.path.join(
                            LARGER_BERT_DATA_CONFIG.model_save_dir,
                            f"{LARGER_BERT_DATA_CONFIG.model_name}_best.pkl"
                        )
                        self.save_checkpoint(best_path)
                        print(f"New best model saved with loss: {eval_loss:.4f}")
        
        print("Training completed!")
        self.save_checkpoint()
        
        # Save training history
        history_path = os.path.join(
            LARGER_BERT_DATA_CONFIG.model_save_dir,
            "training_history.json"
        )
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Main training function."""
    print("Initializing Larger BERT training...")
    print("=" * 60)
    
    # Initialize model
    model = LargerBERT(LARGER_BERT_CONFIG)
    
    # Initialize trainer
    trainer = LargerBERTTrainer(model, LARGER_BERT_TRAINING_CONFIG)
    
    # Load tokenizer
    tokenizer = trainer.load_tokenizer()
    
    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        LARGER_BERT_DATA_CONFIG.train_data_path
    )
    
    print(f"Loading training data from {data_path}")
    
    # Initialize data generator
    data_generator = MLMDataGenerator(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=LARGER_BERT_CONFIG.max_sequence_length,
        batch_size=LARGER_BERT_TRAINING_CONFIG.train_micro_batch_size,
        mlm_probability=LARGER_BERT_TRAINING_CONFIG.mlm_probability
    )
    
    # Start training
    try:
        trainer.train(data_generator, num_steps=1000)  # Train for 1000 steps as demo
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Training can be resumed later.")
    
    print("\nTraining complete!")
    print(f"Final model saved to {LARGER_BERT_DATA_CONFIG.model_save_dir}")


if __name__ == "__main__":
    main()