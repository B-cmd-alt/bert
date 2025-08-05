import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import gc
from .adaptive_resource_manager import adaptive_manager
from transformers import AutoTokenizer, AutoModel, AdamW
import logging

class AdaptiveBERTTrainer:
    def __init__(self, model_name="bert-base-uncased", initial_batch_size=8):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Training parameters that will adapt
        self.current_batch_size = initial_batch_size
        self.current_learning_rate = 1e-5
        self.optimizer = None
        self.scheduler = None
        
        # Data loader will be recreated when batch size changes
        self.train_loader = None
        self.current_epoch = 0
        self.training_active = False
        
        # Setup adaptive resource manager
        self.setup_adaptive_manager()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_adaptive_manager(self):
        """Setup adaptive resource manager with callbacks"""
        def on_resource_change(limits):
            if self.training_active:
                self.adapt_training_parameters(limits)
        
        adaptive_manager.add_callback(on_resource_change)
        adaptive_manager.start_monitoring()
        
    def adapt_training_parameters(self, limits):
        """Adapt training parameters based on available resources"""
        new_batch_size = limits['batch_size']
        new_learning_rate = limits['learning_rate']
        
        # Batch size adaptation
        if new_batch_size != self.current_batch_size:
            self.logger.info(f"🔄 Adapting batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            self._recreate_data_loader()
        
        # Learning rate adaptation
        if abs(new_learning_rate - self.current_learning_rate) > 1e-6:
            self.logger.info(f"🔄 Adapting learning rate: {self.current_learning_rate:.2e} → {new_learning_rate:.2e}")
            self.current_learning_rate = new_learning_rate
            self._update_optimizer()
        
        # Memory management based on available memory
        memory_limit_gb = limits['memory_gb']
        if memory_limit_gb < 3:
            self._aggressive_memory_cleanup()
        elif memory_limit_gb < 2:
            self._emergency_memory_cleanup()
    
    def _recreate_data_loader(self):
        """Recreate data loader with new batch size"""
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.current_batch_size, 
                shuffle=True,
                num_workers=min(2, adaptive_manager.get_current_limits()['cpu_cores'])
            )
            self.logger.info(f"✅ Data loader recreated with batch size {self.current_batch_size}")
    
    def _update_optimizer(self):
        """Update optimizer with new learning rate"""
        if self.optimizer is not None:
            # Update learning rate in existing optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_learning_rate
            self.logger.info(f"✅ Optimizer learning rate updated to {self.current_learning_rate:.2e}")
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.logger.info("🧹 Performed aggressive memory cleanup")
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup - pause training temporarily"""
        self._aggressive_memory_cleanup()
        self.logger.warning("⚠️ Emergency memory cleanup - pausing briefly")
        time.sleep(2)  # Brief pause to let other processes use memory
    
    def prepare_data(self, texts, max_length=512):
        """Prepare training data"""
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        self.train_dataset = TextDataset(encodings)
        self._recreate_data_loader()
        self.logger.info(f"📊 Data prepared: {len(texts)} samples")
    
    def train_epoch(self):
        """Train one epoch with adaptive resource management"""
        if not self.train_loader:
            raise ValueError("No training data prepared. Call prepare_data() first.")
        
        self.model.train()
        self.training_active = True
        
        # Create/update optimizer if needed
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.current_learning_rate)
        
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Check if we should pause for other processes
            current_limits = adaptive_manager.get_current_limits()
            if current_limits.get('resource_ratio', 1) < 0.2:
                self.logger.info("⏸️ Low resources detected - brief pause")
                time.sleep(1)
                continue
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            
            # Simple MLM loss (for demonstration)
            loss = outputs.last_hidden_state.mean()  # Simplified loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Periodic status update
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, "
                               f"Resources: {current_limits.get('resource_ratio', 0):.2f}")
            
            # Adaptive memory cleanup
            if batch_idx % 5 == 0:
                self._check_memory_usage()
        
        self.training_active = False
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        self.current_epoch += 1
        
        self.logger.info(f"✅ Epoch {self.current_epoch-1} completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def _check_memory_usage(self):
        """Check and manage memory usage during training"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_limit = adaptive_manager.get_current_limits().get('memory_gb', 4)
            
            if memory_used > memory_limit * 0.8:  # 80% of limit
                self._aggressive_memory_cleanup()
    
    def train(self, num_epochs=10, save_every=5):
        """Train the model with adaptive resource management"""
        self.logger.info(f"🚀 Starting adaptive BERT training for {num_epochs} epochs")
        self.logger.info(adaptive_manager.get_status())
        
        try:
            for epoch in range(num_epochs):
                # Check system resources before each epoch
                current_limits = adaptive_manager.get_current_limits()
                self.logger.info(f"\n📊 Epoch {epoch+1}/{num_epochs}")
                self.logger.info(f"Current batch size: {self.current_batch_size}")
                self.logger.info(f"Current learning rate: {self.current_learning_rate:.2e}")
                self.logger.info(f"Resource ratio: {current_limits.get('resource_ratio', 0):.2f}")
                
                # Train epoch
                avg_loss = self.train_epoch()
                
                # Save checkpoint periodically
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
                
                # Brief pause between epochs for system stability
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Training interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
        finally:
            self.training_active = False
            self.logger.info("🏁 Training completed")
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'batch_size': self.current_batch_size,
            'learning_rate': self.current_learning_rate
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_batch_size = checkpoint['batch_size']
        self.current_learning_rate = checkpoint['learning_rate']
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"📂 Checkpoint loaded: {filepath}")
    
    def __del__(self):
        """Cleanup when trainer is destroyed"""
        adaptive_manager.stop_monitoring()

# Example usage
if __name__ == "__main__":
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require significant computational resources.",
        "Adaptive resource management optimizes system performance."
    ] * 100  # Repeat for more training data
    
    # Create trainer
    trainer = AdaptiveBERTTrainer()
    
    # Prepare data
    trainer.prepare_data(sample_texts)
    
    # Start training
    trainer.train(num_epochs=5)
    
    print("\n" + "="*50)
    print("🎯 Final Resource Status:")
    print(adaptive_manager.get_status())