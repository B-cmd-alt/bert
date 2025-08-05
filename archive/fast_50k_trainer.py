#!/usr/bin/env python3
"""
Fast 50K BERT Embedding Training
Uses existing high-quality training data but with optimized processing for speed.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_50k_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fast_train_50k():
    """Fast training of 50K vocabulary using existing high-quality data."""
    
    logger.info("=== Fast 50K BERT Embedding Training ===")
    start_time = time.time()
    
    # Use the existing enhanced training data (smaller but high quality)
    data_file = "enhanced_training_data_4.0gb.txt"
    
    if not os.path.exists(data_file):
        # Fallback to other available data
        potential_files = [
            "high_quality_training_data.txt",
            "large_bert_models/combined_full_training_data.txt",
            "large_bert_models/full_data/wikipedia_full.txt"
        ]
        
        for file in potential_files:
            if os.path.exists(file):
                data_file = file
                break
        else:
            logger.error("No training data found!")
            return None, None
    
    data_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    logger.info(f"Using training data: {data_file} ({data_size_mb:.1f} MB)")
    
    # Create output directory
    output_dir = Path("bert_50k_models")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize 50K tokenizer
    logger.info("Initializing 50K vocabulary tokenizer...")
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    
    # Train with progress tracking
    logger.info("Starting training (this may take 20-40 minutes)...")
    training_start = time.time()
    
    try:
        # Train the tokenizer
        tokenizer.train(data_file)
        
        training_time = time.time() - training_start
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        
        # Save the model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_file = output_dir / f"bert_50k_tokenizer_{timestamp}.pkl"
        vocab_file = output_dir / f"bert_50k_vocab_{timestamp}.txt"
        
        logger.info(f"Saving model to {model_file}...")
        tokenizer.save_model(str(model_file))
        
        logger.info(f"Saving vocabulary to {vocab_file}...")
        tokenizer.save_vocab(str(vocab_file))
        
        # Results
        final_vocab_size = len(tokenizer.vocab)
        total_time = time.time() - start_time
        
        logger.info("=== TRAINING COMPLETED ===")
        logger.info(f"Final vocabulary size: {final_vocab_size:,} tokens")
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        # Test the tokenizer
        test_sentences = [
            "This is a test sentence.",
            "BERT embeddings are powerful for NLP tasks.",
            "Machine learning requires high-quality tokenization."
        ]
        
        logger.info("=== Testing Tokenizer ===")
        for sentence in test_sentences:
            tokens = tokenizer.tokenize(sentence)
            logger.info(f"'{sentence}' -> {tokens}")
        
        # Quality assessment
        if final_vocab_size >= 40000:
            logger.info("✅ EXCELLENT: High-quality 50K vocabulary achieved!")
        elif final_vocab_size >= 25000:
            logger.info("✅ GOOD: Substantial vocabulary for BERT embeddings")
        elif final_vocab_size >= 10000:
            logger.info("⚠️  ADEQUATE: Basic vocabulary size achieved")
        else:
            logger.info("⚠️  LIMITED: Vocabulary smaller than expected")
        
        return str(model_file), {
            "final_vocab_size": final_vocab_size,
            "training_time_minutes": training_time/60,
            "total_time_minutes": total_time/60,
            "data_size_mb": data_size_mb,
            "model_file": str(model_file),
            "vocab_file": str(vocab_file)
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    logger.info("Starting fast 50K BERT embedding training...")
    
    model_file, results = fast_train_50k()
    
    if model_file and results:
        logger.info("🎉 SUCCESS: 50K BERT embedding trained successfully!")
        logger.info(f"Model: {model_file}")
        logger.info(f"Vocabulary size: {results['final_vocab_size']:,}")
        logger.info(f"Training time: {results['training_time_minutes']:.1f} minutes")
    else:
        logger.error("❌ Training failed")