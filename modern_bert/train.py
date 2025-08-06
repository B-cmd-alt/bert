"""
Modern BERT training using HuggingFace Trainer and Accelerate.
Production-ready training with automatic optimization and monitoring.
"""

import os
import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from torch.optim import AdamW
import numpy as np
from typing import Dict, Optional
import json
from dataclasses import asdict

from modern_bert.model import ModernBert
from modern_bert.data_pipeline import ModernDataPipeline, create_mlm_pipeline
from modern_bert.config import ModernBertConfig, DataConfig


class ModernBertTrainer:
    """
    Modern BERT trainer using HuggingFace ecosystem.
    Supports distributed training, mixed precision, and experiment tracking.
    """
    
    def __init__(self, config: ModernBertConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision='fp16' if config.fp16 else 'no',
            gradient_accumulation_steps=1,  # Will be handled by Trainer
            log_with='wandb' if config.use_wandb else None
        )
        
        # Initialize wandb if enabled
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=asdict(config)
            )
        
        # Initialize model
        self.model = ModernBert(config)
        
        # Initialize data pipeline
        self.data_pipeline = ModernDataPipeline(data_config, self.model.tokenizer)
        
        # Training state
        self.training_history = []
        
    def prepare_training_args(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            logging_dir=self.config.logging_dir,
            
            # Training schedule
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=1,  # Adjust based on memory
            
            # Optimization
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            
            # Mixed precision and optimization
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_pin_memory=True,
            
            # Logging and evaluation
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            
            # Model saving
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            
            # Monitoring
            report_to="wandb" if self.config.use_wandb else None,
            run_name=self.config.experiment_name,
            
            # Hardware
            no_cuda=self.config.no_cuda,
            seed=self.config.seed,
            
            # Advanced features
            push_to_hub=False,  # Set to True if you want to push to HF Hub
            hub_model_id=None,
        )
        
        return training_args
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and learning rate scheduler."""
        # Create optimizer
        if self.config.use_8bit:
            # 8-bit AdamW
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                self.model.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            # Standard AdamW
            optimizer = AdamW(
                self.model.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return optimizer, scheduler
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        
        # For MLM, we compute perplexity
        # Predictions are logits, labels are token IDs
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # Filter out -100 labels (padding/non-masked tokens)
        mask = (labels != -100)
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(labels) == 0:
            return {"perplexity": float('inf')}
        
        # Compute cross-entropy loss
        log_probs = torch.nn.functional.log_softmax(torch.tensor(predictions), dim=-1)
        loss = torch.nn.functional.nll_loss(log_probs, torch.tensor(labels), reduction='mean')
        
        # Compute perplexity
        perplexity = torch.exp(loss).item()
        
        # Compute accuracy
        pred_ids = torch.argmax(torch.tensor(predictions), dim=-1)
        accuracy = (pred_ids == torch.tensor(labels)).float().mean().item()
        
        return {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "eval_loss": loss.item()
        }
    
    def train_with_hf_trainer(self):
        """Train using HuggingFace Trainer (recommended approach)."""
        print("Starting training with HuggingFace Trainer...")
        
        # Prepare data
        dataset = self.data_pipeline.load_dataset()
        dataset = self.data_pipeline.preprocess_dataset(dataset)
        
        # Create training arguments
        training_args = self.prepare_training_args()
        
        # Create trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.model.tokenizer,
            data_collator=self.data_pipeline.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        
        # Evaluate on test set if available
        if 'test' in dataset:
            test_results = trainer.evaluate(dataset['test'])
            print(f"Test results: {test_results}")
        
        return trainer
    
    def train_with_accelerate(self):
        """Train using Accelerate for more control (advanced approach)."""
        print("Starting training with Accelerate...")
        
        # Prepare data
        dataset = self.data_pipeline.load_dataset()
        dataset = self.data_pipeline.preprocess_dataset(dataset)
        dataloaders = self.data_pipeline.create_dataloaders(dataset)
        
        # Calculate training steps
        num_training_steps = len(dataloaders['train']) * self.config.num_train_epochs
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(num_training_steps)
        
        # Prepare everything with accelerator
        model, optimizer, train_dataloader, eval_dataloader, scheduler = self.accelerator.prepare(
            self.model.model, optimizer, dataloaders['train'], dataloaders['validation'], scheduler
        )
        
        # Training loop
        model.train()
        global_step = 0
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Training
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    
                    if self.accelerator.is_main_process:
                        print(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "train_loss": avg_loss,
                                "learning_rate": lr,
                                "step": global_step
                            })
                    
                    total_loss = 0
                
                # Evaluation
                if global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate_with_accelerate(model, eval_dataloader)
                    
                    if self.accelerator.is_main_process:
                        print(f"Eval loss: {eval_loss:.4f}")
                        
                        if self.config.use_wandb:
                            wandb.log({"eval_loss": eval_loss, "step": global_step})
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    if self.accelerator.is_main_process:
                        self.save_checkpoint(model, optimizer, scheduler, global_step)
        
        # Final save
        if self.accelerator.is_main_process:
            self.save_checkpoint(model, optimizer, scheduler, global_step, is_final=True)
    
    def evaluate_with_accelerate(self, model, eval_dataloader):
        """Evaluate model with accelerate."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, model, optimizer, scheduler, step, is_final=False):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir, 
            f"checkpoint-{step}" if not is_final else "final"
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional info
        checkpoint_info = {
            "step": step,
            "config": asdict(self.config),
            "training_history": self.training_history
        }
        
        with open(os.path.join(checkpoint_dir, "training_info.json"), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    @staticmethod
    def compare_with_numpy_training():
        """Compare with NumPy training approach."""
        print("\n" + "="*60)
        print("Training Comparison: NumPy vs Modern")
        print("="*60)
        
        print("\nNumPy Implementation:")
        print("- Manual gradient computation")
        print("- Manual optimizer updates")
        print("- Single GPU only")
        print("- Manual mixed precision")
        print("- Custom learning rate scheduling")
        print("- Manual checkpointing")
        
        print("\nModern Implementation:")
        print("- Automatic differentiation")
        print("- Optimized optimizers (8-bit, etc.)")
        print("- Multi-GPU/TPU support")
        print("- Automatic mixed precision")
        print("- Built-in schedulers")
        print("- Automatic checkpointing")
        
        print("\nProduction Features:")
        features = [
            ("Distributed training", "❌", "✅"),
            ("Mixed precision (FP16)", "❌", "✅"),
            ("Gradient accumulation", "✅", "✅"),
            ("Early stopping", "❌", "✅"),
            ("Experiment tracking", "❌", "✅"),
            ("Model hub integration", "❌", "✅"),
            ("ONNX export", "❌", "✅"),
            ("Parameter-efficient tuning", "❌", "✅"),
        ]
        
        print(f"\n{'Feature':<30} {'NumPy':<10} {'Modern':<10}")
        print("-"*50)
        for feature, numpy_support, modern_support in features:
            print(f"{feature:<30} {numpy_support:<10} {modern_support:<10}")


def main():
    """Main training function."""
    # Configuration
    config = ModernBertConfig()
    data_config = DataConfig()
    
    # Override for quick testing
    if os.getenv("DEBUG", "False").lower() == "true":
        from modern_bert.config import PresetConfigs
        config = PresetConfigs.debug_config()
        data_config.max_train_samples = 1000
        data_config.max_eval_samples = 100
    
    # Create trainer
    trainer = ModernBertTrainer(config, data_config)
    
    # Choose training method
    training_method = os.getenv("TRAINING_METHOD", "hf_trainer")
    
    if training_method == "hf_trainer":
        # Recommended: Use HuggingFace Trainer
        hf_trainer = trainer.train_with_hf_trainer()
        print("Training completed with HuggingFace Trainer!")
        
    elif training_method == "accelerate":
        # Advanced: Use Accelerate for more control
        trainer.train_with_accelerate()
        print("Training completed with Accelerate!")
    
    else:
        raise ValueError(f"Unknown training method: {training_method}")
    
    # Compare approaches
    ModernBertTrainer.compare_with_numpy_training()


if __name__ == "__main__":
    # Set environment variables for configuration
    # os.environ["DEBUG"] = "True"  # Uncomment for debug mode
    # os.environ["TRAINING_METHOD"] = "accelerate"  # or "hf_trainer"
    
    main()