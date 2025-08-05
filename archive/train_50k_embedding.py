#!/usr/bin/env python3
"""
50K Vocabulary BERT Embedding Training
Optimized for fast training with high-quality 50k vocabulary embeddings.
"""

import os
import sys
import time 
import logging
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_50k_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_50k_bert_embedding():
    """Train a high-quality 50k vocabulary BERT embedding."""
    
    logger.info("=== 50K BERT Embedding Training Started ===")
    start_time = time.time()
    
    # Use existing combined data
    data_file = "large_bert_models/combined_full_training_data.txt"
    output_dir = Path("bert_50k_models")
    output_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(data_file):
        logger.error(f"Training data not found: {data_file}")
        logger.info("Please run full_scale_bert_trainer.py first to prepare data")
        return
    
    # Check data size
    data_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    logger.info(f"Training data: {data_file} ({data_size_mb:.1f} MB)")
    
    # Initialize tokenizer with 50k vocabulary
    logger.info("Initializing 50K vocabulary tokenizer...")
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    
    # Train the tokenizer
    logger.info("Training WordPiece tokenizer...")
    training_start = time.time()
    
    try:
        tokenizer.train(data_file)
        training_time = time.time() - training_start
        
        # Save the model
        model_file = output_dir / "bert_50k_tokenizer.pkl"
        vocab_file = output_dir / "bert_50k_vocab.txt"
        
        logger.info(f"Saving model to {model_file}...")
        tokenizer.save_model(str(model_file))
        
        logger.info(f"Saving vocabulary to {vocab_file}...")
        tokenizer.save_vocab(str(vocab_file))
        
        # Training results
        final_vocab_size = len(tokenizer.vocab)
        total_time = time.time() - start_time
        
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final vocabulary size: {final_vocab_size:,} tokens")
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Model saved: {model_file}")
        logger.info(f"Vocabulary saved: {vocab_file}")
        
        # Validate results
        if final_vocab_size >= 40000:
            logger.info("✅ EXCELLENT: High-quality 50K vocabulary achieved!")
        elif final_vocab_size >= 30000:
            logger.info("✅ GOOD: Substantial vocabulary size for BERT embeddings")
        else:
            logger.info("⚠️  LIMITED: Vocabulary smaller than expected")
            
        return str(model_file), {
            "final_vocab_size": final_vocab_size,
            "training_time_minutes": training_time/60,
            "total_time_minutes": total_time/60,
            "data_size_mb": data_size_mb
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, None

if __name__ == "__main__":
    model_file, results = train_50k_bert_embedding()
    
    if model_file:
        logger.info(f"🎉 SUCCESS: 50K BERT embedding trained successfully!")
        logger.info(f"Model ready at: {model_file}")
    else:
        logger.error("❌ Training failed")