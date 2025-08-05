#!/usr/bin/env python3
"""
Enhanced 8-Hour Continuous Training Pipeline with High-Quality Diverse Data

This version uses much more diverse, high-quality data that should provide
enough vocabulary diversity to actually utilize the full 100,000 token target
and take the full 8 hours as intended.
"""

import os
import sys
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add the learning-examples directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'bert-wordpiece-tokenizer'))

from wordpiece_simple import SimpleWordPieceTokenizer
from high_quality_data_generator import HighQualityDataGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrainingMonitor:
    """Enhanced monitor for training progress and system resources."""
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        self.last_checkpoint = time.time()
        
    def log_system_stats(self):
        """Log current system resource usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / 1024 / 1024 / 1024
        available_memory_gb = system_memory.available / 1024 / 1024 / 1024
        used_memory_percent = system_memory.percent
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        logger.info(f"=== Enhanced System Stats ===")
        logger.info(f"Elapsed Time: {elapsed_str}")
        logger.info(f"Process Memory: {memory_mb:.1f} MB")
        logger.info(f"System Memory: {available_memory_gb:.1f}GB available / {system_memory_gb:.1f}GB total ({used_memory_percent:.1f}% used)")
        logger.info(f"CPU Usage: {cpu_percent:.1f}%")
        
        # Check if we're approaching memory limits
        if used_memory_percent > 85:
            logger.warning(f"High memory usage detected: {used_memory_percent:.1f}%")
        
        if cpu_percent > 90:
            logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
            
        logger.info(f"============================")
    
    def should_checkpoint(self, interval_seconds=1800):  # 30 minutes
        """Check if it's time for a checkpoint."""
        return time.time() - self.last_checkpoint > interval_seconds
    
    def create_checkpoint(self, tokenizer, checkpoint_dir, iteration=None):
        """Create a checkpoint of the current training state."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}"
        if iteration:
            checkpoint_name += f"_iter{iteration}"
        
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pkl"
        tokenizer.save_model(str(checkpoint_file))
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        self.last_checkpoint = time.time()
        return checkpoint_file

def main():
    parser = argparse.ArgumentParser(description="Enhanced 8-Hour Continuous WordPiece Training")
    parser.add_argument("--vocab-size", type=int, default=100000, 
                       help="Target vocabulary size (default: 100,000)")
    parser.add_argument("--data-size-gb", type=float, default=1.0,
                       help="Training data size in GB (default: 1.0 - reduced for memory efficiency)")
    parser.add_argument("--output-dir", type=str, default="enhanced_models",
                       help="Output directory for models")
    parser.add_argument("--checkpoint-interval", type=int, default=1800,
                       help="Checkpoint interval in seconds (default: 30 minutes)")
    parser.add_argument("--max-training-hours", type=float, default=8.0,
                       help="Maximum training time in hours (default: 8.0)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    monitor = EnhancedTrainingMonitor()
    
    logger.info("=== Enhanced 8-Hour Continuous WordPiece Training Started ===")
    logger.info(f"Target vocabulary size: {args.vocab_size:,}")
    logger.info(f"Training data size: {args.data_size_gb:.1f} GB")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Maximum training time: {args.max_training_hours:.1f} hours")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval} seconds")
    
    monitor.log_system_stats()
    
    try:
        # Step 1: Generate high-quality diverse training dataset
        logger.info("Step 1: Generating high-quality diverse training dataset...")
        data_generator = HighQualityDataGenerator()
        
        training_file = data_generator.generate_diverse_dataset(
            filename=f"enhanced_training_data_{args.data_size_gb}gb.txt",
            target_size_gb=args.data_size_gb
        )
        
        monitor.log_system_stats()
        
        # Step 2: Analyze data diversity
        logger.info("Step 2: Analyzing data diversity...")
        with open(training_file, 'r', encoding='utf-8') as f:
            sample_lines = [f.readline().strip() for _ in range(10)]
            logger.info("Sample sentences from generated data:")
            for i, line in enumerate(sample_lines, 1):
                logger.info(f"  {i}: {line[:100]}...")
        
        # Quick vocabulary analysis (memory optimized)
        unique_words = set()
        total_words = 0
        with open(training_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 5000:  # Sample first 5k lines for analysis (reduced from 10k)
                    break
                words = line.strip().lower().split()
                unique_words.update(words)
                total_words += len(words)
                
                # Clear variables periodically to free memory
                if i % 1000 == 0:
                    import gc
                    gc.collect()
        
        logger.info(f"Data diversity analysis (first 10k lines):")
        logger.info(f"  Unique words: {len(unique_words):,}")
        logger.info(f"  Total words: {total_words:,}")
        logger.info(f"  Vocabulary richness: {len(unique_words)/total_words:.3f}")
        
        # Step 3: Initialize tokenizer with enhanced settings
        logger.info(f"Step 3: Initializing enhanced tokenizer...")
        tokenizer = SimpleWordPieceTokenizer(vocab_size=args.vocab_size)
        
        # Step 4: Start training with monitoring
        logger.info("Step 4: Starting enhanced tokenizer training...")
        training_start = time.time()
        max_training_seconds = args.max_training_hours * 3600
        
        # This is where we'd need to modify the tokenizer to support checkpointing
        # For now, we'll train normally but with enhanced monitoring
        logger.info("Beginning BPE training process...")
        
        # Create initial checkpoint
        initial_checkpoint = monitor.create_checkpoint(tokenizer, checkpoint_dir, "initial")
        
        # Start the actual training
        tokenizer.train(training_file)
        
        training_end = time.time()
        training_duration = training_end - training_start
        
        logger.info(f"Training completed in {timedelta(seconds=int(training_duration))}")
        
        # Step 5: Save final models
        logger.info("Step 5: Saving enhanced trained models...")
        
        final_vocab_size = len(tokenizer.vocab)
        model_file = output_dir / f"enhanced_tokenizer_vocab{final_vocab_size//1000}k.pkl"
        vocab_file = output_dir / f"enhanced_vocab_{final_vocab_size//1000}k.txt"
        
        tokenizer.save_model(str(model_file))
        tokenizer.save_vocab(str(vocab_file))
        
        # Step 6: Comprehensive quality testing
        logger.info("Step 6: Running comprehensive quality tests...")
        
        test_sentences = [
            "Advanced machine learning algorithms utilize sophisticated mathematical optimization techniques for enhanced performance",
            "Biomedical research encompasses genomics proteomics bioinformatics and computational biology methodologies",
            "Quantum computational systems demonstrate unprecedented processing capabilities for cryptographic applications",
            "Interdisciplinary collaborative research projects integrate diverse scientific methodologies and analytical frameworks",
            "Revolutionary breakthrough discoveries transform theoretical understanding into practical technological innovations",
            "Enterprise-grade distributed architectures enable scalable microservices deployment across multi-cloud environments",
            "Artificial intelligence systems leverage deep learning neural networks for complex pattern recognition tasks",
            "Cybersecurity professionals implement multi-layered defense strategies against sophisticated threat vectors",
            "Sustainable development initiatives require comprehensive environmental impact assessment methodologies",
            "International collaboration facilitates cross-cultural knowledge exchange through digital transformation platforms"
        ]
        
        logger.info("Enhanced Quality Test Results:")
        logger.info("-" * 80)
        
        total_accuracy = 0
        total_compression = 0
        
        for i, sentence in enumerate(test_sentences, 1):
            tokens = tokenizer.tokenize(sentence)
            encoded = tokenizer.encode(sentence)
            decoded = tokenizer.decode(encoded)
            
            # Calculate accuracy
            original_words = set(sentence.lower().split())
            decoded_words = set(decoded.lower().split())
            overlap = len(original_words.intersection(decoded_words))
            accuracy = overlap / len(original_words) if original_words else 0
            total_accuracy += accuracy
            
            # Calculate compression ratio
            original_chars = len(sentence)
            token_count = len(tokens)
            compression = token_count / len(sentence.split())
            total_compression += compression
            
            logger.info(f"Test {i}: '{sentence[:60]}...'")
            logger.info(f"  Original words: {len(sentence.split())}, Tokens: {len(tokens)}")
            logger.info(f"  Compression ratio: {compression:.2f} tokens/word")
            logger.info(f"  Sample tokens: {tokens[:8]}...")
            logger.info(f"  Accuracy: {accuracy:.1%}")
            logger.info("")
        
        avg_accuracy = total_accuracy / len(test_sentences)
        avg_compression = total_compression / len(test_sentences)
        
        logger.info(f"=== Enhanced Training Results ===")
        logger.info(f"Average Round-trip Accuracy: {avg_accuracy:.1%}")
        logger.info(f"Average Compression Ratio: {avg_compression:.2f} tokens/word")
        logger.info(f"Final vocabulary size: {final_vocab_size:,}")
        logger.info(f"Training duration: {timedelta(seconds=int(training_duration))}")
        logger.info(f"Model saved to: {model_file}")
        logger.info(f"Vocabulary saved to: {vocab_file}")
        logger.info(f"Data diversity: {len(unique_words):,} unique words analyzed")
        
        monitor.log_system_stats()
        
        # Cleanup training data
        try:
            os.remove(training_file)
            logger.info(f"Cleaned up training file: {training_file}")
        except:
            pass
        
        logger.info("Enhanced training pipeline completed successfully!")
        
        # Final assessment
        if final_vocab_size >= args.vocab_size * 0.8:  # At least 80% of target
            logger.info("✅ SUCCESS: Achieved substantial vocabulary size!")
        elif final_vocab_size >= args.vocab_size * 0.5:  # At least 50% of target
            logger.info("⚠️  PARTIAL: Achieved reasonable vocabulary size, but could be improved")
        else:
            logger.info("❌ LIMITED: Vocabulary size lower than expected, need more diverse data")
        
        if training_duration > 3600:  # More than 1 hour
            logger.info("✅ SUCCESS: Training took substantial time as expected!")
        else:
            logger.info("⚠️  QUICK: Training completed faster than expected")
        
    except Exception as e:
        logger.error(f"Enhanced training failed with error: {e}")
        monitor.log_system_stats()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())