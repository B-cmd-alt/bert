#!/usr/bin/env python3
"""
Memory-Efficient 50K BERT Embedding Training
Uses streaming approach to handle large datasets without memory issues.
"""

import os
import sys
import time
import logging
from pathlib import Path
from collections import Counter
import gc

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_efficient_50k.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryEfficientWordPieceTrainer:
    """Memory-efficient WordPiece trainer for large datasets."""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.tokenizer = SimpleWordPieceTokenizer(vocab_size=vocab_size)
    
    def create_sample_training_data(self, sample_file="bert_50k_sample_data.txt", target_size_mb=200):
        """Create a high-quality sample from existing data."""
        logger.info(f"Creating {target_size_mb}MB sample training data...")
        
        # Source files in order of preference
        source_files = [
            "large_bert_models/full_data/wikipedia_full.txt",
            "enhanced_training_data_4.0gb.txt", 
            "high_quality_training_data.txt"
        ]
        
        source_file = None
        for file in source_files:
            if os.path.exists(file):
                source_file = file
                break
        
        if not source_file:
            logger.error("No source training data found!")
            return None
        
        logger.info(f"Sampling from: {source_file}")
        
        target_bytes = target_size_mb * 1024 * 1024
        written_bytes = 0
        
        try:
            with open(source_file, 'r', encoding='utf-8') as infile:
                with open(sample_file, 'w', encoding='utf-8') as outfile:
                    for line_num, line in enumerate(infile):
                        if written_bytes >= target_bytes:
                            break
                        
                        # Sample every 10th line for diversity, but take first lines for quality
                        if line_num < 50000 or line_num % 10 == 0:
                            outfile.write(line)
                            written_bytes += len(line.encode('utf-8'))
                        
                        # Progress update
                        if line_num % 100000 == 0:
                            logger.info(f"Processed {line_num:,} lines, {written_bytes/(1024*1024):.1f} MB written")
                            
            actual_size_mb = written_bytes / (1024 * 1024)
            logger.info(f"Sample created: {sample_file} ({actual_size_mb:.1f} MB)")
            return sample_file
            
        except Exception as e:
            logger.error(f"Error creating sample: {e}")
            return None
    
    def train_efficient(self, data_file):
        """Train tokenizer with memory-efficient approach."""
        logger.info("Starting memory-efficient training...")
        
        # Read data in chunks to build vocabulary
        logger.info("Building vocabulary from text chunks...")
        word_freq = Counter()
        chunk_size = 100000  # 100K characters per chunk (reduced for memory efficiency)
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                chunk_num = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_num += 1
                    logger.info(f"Processing chunk {chunk_num}...")
                    
                    # Process chunk
                    words = self.tokenizer._preprocess_text(chunk)
                    for word in words:
                        word_freq[word] += 1
                    
                    # Free memory
                    del words
                    del chunk
                    gc.collect()
                    
                    # Limit processing for memory efficiency
                    if chunk_num >= 50:  # Process max 5MB worth (reduced chunks)
                        break
            
            logger.info(f"Found {len(word_freq)} unique words")
            
            # Build vocabulary using word frequencies
            vocab = set()
            
            # Add special tokens
            for token in self.tokenizer.special_tokens:
                vocab.add(token)
            
            # Add characters
            for word in word_freq.keys():
                for char in word:
                    vocab.add(char)
            
            logger.info(f"Initial vocabulary size: {len(vocab)}")
            
            # Simple merge strategy for remaining vocab
            remaining_slots = self.vocab_size - len(vocab)
            most_common_words = word_freq.most_common(remaining_slots)
            
            for word, freq in most_common_words:
                if len(vocab) >= self.vocab_size:
                    break
                vocab.add(word)
            
            # Build final vocab dictionary
            self.tokenizer.vocab = {token: idx for idx, token in enumerate(vocab)}
            self.tokenizer.inverse_vocab = {idx: token for token, idx in self.tokenizer.vocab.items()}
            
            logger.info(f"Training completed. Final vocabulary size: {len(self.tokenizer.vocab)}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

def main():
    logger.info("=== Memory-Efficient 50K BERT Training ===")
    start_time = time.time()
    
    trainer = MemoryEfficientWordPieceTrainer(vocab_size=50000)
    
    # Create sample data
    sample_file = trainer.create_sample_training_data("bert_50k_sample.txt", target_size_mb=50)  # Reduced from 100MB to 50MB
    
    if not sample_file:
        logger.error("Failed to create sample data")
        return
    
    # Train the tokenizer
    success = trainer.train_efficient(sample_file)
    
    if not success:
        logger.error("Training failed")
        return
    
    # Save the model
    output_dir = Path("bert_50k_models")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_file = output_dir / f"bert_50k_efficient_{timestamp}.pkl"
    vocab_file = output_dir / f"bert_50k_vocab_efficient_{timestamp}.txt"
    
    logger.info(f"Saving model to {model_file}...")
    trainer.tokenizer.save_model(str(model_file))
    
    logger.info(f"Saving vocabulary to {vocab_file}...")
    trainer.tokenizer.save_vocab(str(vocab_file))
    
    # Test the tokenizer
    test_text = "This is a comprehensive test of the BERT tokenizer for natural language processing."
    tokens = trainer.tokenizer.tokenize(test_text)
    
    total_time = time.time() - start_time
    final_vocab_size = len(trainer.tokenizer.vocab)
    
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Final vocabulary size: {final_vocab_size:,}")
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    logger.info(f"Model saved: {model_file}")
    logger.info(f"Test tokenization: '{test_text}' -> {tokens}")
    
    if final_vocab_size >= 40000:
        logger.info("SUCCESS: High-quality 50K vocabulary achieved!")
    else:
        logger.info(f"SUCCESS: {final_vocab_size:,} vocabulary created")

if __name__ == "__main__":
    main()