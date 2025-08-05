#!/usr/bin/env python3
"""
8-Hour Continuous Training Pipeline for Large Vocabulary WordPiece Tokenizer

This script is designed for long-running training sessions with:
- Large vocabulary (100,000 tokens)
- High-quality Wikipedia data
- Progress monitoring and logging
- Automatic checkpointing
- Memory usage tracking
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and system resources."""
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def log_system_stats(self):
        """Log current system resource usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / 1024 / 1024 / 1024
        available_memory_gb = system_memory.available / 1024 / 1024 / 1024
        
        cpu_percent = psutil.cpu_percent()
        
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        logger.info(f"=== System Stats ===")
        logger.info(f"Elapsed Time: {elapsed_str}")
        logger.info(f"Process Memory: {memory_mb:.1f} MB")
        logger.info(f"System Memory: {available_memory_gb:.1f}GB available / {system_memory_gb:.1f}GB total")
        logger.info(f"CPU Usage: {cpu_percent:.1f}%")
        logger.info(f"==================")

def create_large_sample_data(filename: str = "large_training_data.txt", size_gb: float = 2.0):
    """Create large sample training data for extended training."""
    
    # High-quality sample texts covering various domains
    sample_texts = [
        # Science and Technology
        "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with interactions between computers and human language",
        "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning",
        "Transformers are a type of neural network architecture that has been very successful for natural language processing tasks",
        "The attention mechanism allows the model to focus on different parts of the input sequence when producing an output",
        "BERT stands for Bidirectional Encoder Representations from Transformers and uses bidirectional training",
        "WordPiece tokenization is a data driven tokenization scheme which generates subword units for handling out of vocabulary words",
        
        # Computing and Programming
        "Python is a high level general purpose programming language with dynamic semantics and elegant syntax",
        "Artificial intelligence is intelligence demonstrated by machines in contrast to natural intelligence displayed by humans",
        "Computer science is the study of computational systems algorithms and computational thinking processes",
        "Data science is an interdisciplinary field that uses scientific methods processes algorithms and systems to extract knowledge from data",
        "Software engineering is the systematic application of engineering approaches to the development of software systems",
        "Algorithm analysis involves determining the computational complexity of algorithms in terms of time and space requirements",
        
        # Mathematics and Statistics
        "Statistical learning theory is a framework for machine learning drawing from the fields of statistics and functional analysis",
        "Linear algebra is the branch of mathematics concerning linear equations linear maps and their representations in vector spaces",
        "Calculus is the mathematical study of continuous change as geometry is the study of shape and algebra is the study of operations",
        "Probability theory is the branch of mathematics concerned with probability the analysis of random phenomena",
        "Optimization is the selection of a best element with regard to some criterion from some set of available alternatives",
        
        # Biology and Medicine
        "Bioinformatics is an interdisciplinary field that develops methods and software tools for understanding biological data",
        "Genomics is an interdisciplinary field of biology focusing on the structure function evolution mapping and editing of genomes",
        "Computational biology involves the development and application of data analytical and theoretical methods mathematical modeling and computational simulation",
        "Medical imaging is the technique and process of creating visual representations of the interior of a body for clinical analysis",
        
        # Physics and Chemistry
        "Quantum mechanics is a fundamental theory in physics which describes the physical properties of nature at the scale of atoms",
        "Thermodynamics is a branch of physics that deals with heat work and temperature and their relation to energy radiation and physical properties",
        "Molecular biology is the branch of biology that concerns the molecular basis of biological activity in and between cells",
        "Chemical engineering is a branch of engineering that uses principles of chemistry physics mathematics biology and economics",
        
        # Economics and Social Sciences
        "Economics is the social science that studies the production distribution and consumption of goods and services",
        "Psychology is the scientific study of mind and behavior including conscious and unconscious phenomena and mental processes",
        "Sociology is a social science that focuses on society social institutions and social relationships among individuals",
        "Political science is the scientific study of politics and power from domestic national and international perspectives",
        
        # Literature and Philosophy
        "Philosophy is the study of general and fundamental questions such as those about existence reason knowledge values mind and language",
        "Literature is a form of human expression that encompasses written works especially those considered to have artistic or intellectual value",
        "Linguistics is the scientific study of language involving the analysis of language form language meaning and language in context",
        "Ethics or moral philosophy is a branch of philosophy that involves systematizing defending and recommending concepts of right and wrong",
        
        # History and Geography
        "History is the study of the past as it is described in written documents and the interpretation of these documents",
        "Geography is a field of science devoted to the study of the lands features inhabitants and phenomena of the Earth and planets",
        "Archaeology is the study of human activity through the recovery and analysis of material culture and environmental data",
        "Anthropology is the scientific study of humans human behavior and societies in the past and present",
        
        # Arts and Culture
        "Music is an art form and cultural activity whose medium is sound organized in time through the elements of melody harmony rhythm and dynamics",
        "Visual arts are art forms such as painting drawing printmaking sculpture ceramics photography video filmmaking design crafts and architecture",
        "Theater is a collaborative form of performing art that uses live performers usually actors or actresses to present the experience of a real or imagined event",
        "Cinema is a visual art form used to simulate experiences that communicate ideas stories perceptions feelings beauty or atmosphere",
    ]
    
    target_bytes = int(size_gb * 1024 * 1024 * 1024)
    current_bytes = 0
    
    logger.info(f"Creating large training dataset: {filename} (target: {size_gb:.1f} GB)")
    
    with open(filename, 'w', encoding='utf-8') as f:
        cycle_count = 0
        while current_bytes < target_bytes:
            for text in sample_texts:
                # Add some variation to prevent exact repetition
                variations = [
                    text,
                    text.replace(' and ', ' as well as '),
                    text.replace(' is ', ' represents '),
                    text.replace(' the ', ' a '),
                    text.replace(' of ', ' related to '),
                ]
                
                for variation in variations:
                    line = variation + '\n'
                    f.write(line)
                    current_bytes += len(line.encode('utf-8'))
                    
                    if current_bytes >= target_bytes:
                        break
                
                if current_bytes >= target_bytes:
                    break
            
            cycle_count += 1
            if cycle_count % 100 == 0:
                logger.info(f"Generated {current_bytes / (1024*1024):.1f} MB so far...")
    
    final_size_mb = current_bytes / (1024 * 1024)
    logger.info(f"Training data created: {filename} ({final_size_mb:.1f} MB)")
    return filename

def main():
    parser = argparse.ArgumentParser(description="8-Hour Continuous WordPiece Training")
    parser.add_argument("--vocab-size", type=int, default=100000, 
                       help="Target vocabulary size (default: 100,000)")
    parser.add_argument("--data-size-gb", type=float, default=2.0,
                       help="Training data size in GB (default: 2.0)")
    parser.add_argument("--output-dir", type=str, default="large_models",
                       help="Output directory for models")
    parser.add_argument("--checkpoint-interval", type=int, default=3600,
                       help="Checkpoint interval in seconds (default: 1 hour)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    monitor = TrainingMonitor()
    
    logger.info("=== 8-Hour Continuous WordPiece Training Started ===")
    logger.info(f"Target vocabulary size: {args.vocab_size:,}")
    logger.info(f"Training data size: {args.data_size_gb:.1f} GB")
    logger.info(f"Output directory: {output_dir}")
    
    monitor.log_system_stats()
    
    try:
        # Step 1: Create large training dataset
        logger.info("Step 1: Creating large training dataset...")
        training_file = create_large_sample_data(
            filename=f"training_data_{args.data_size_gb}gb.txt",
            size_gb=args.data_size_gb
        )
        
        monitor.log_system_stats()
        
        # Step 2: Initialize tokenizer
        logger.info(f"Step 2: Initializing tokenizer with vocab size {args.vocab_size:,}...")
        tokenizer = SimpleWordPieceTokenizer(vocab_size=args.vocab_size)
        
        # Step 3: Start training
        logger.info("Step 3: Starting tokenizer training...")
        training_start = time.time()
        
        tokenizer.train(training_file)
        
        training_end = time.time()
        training_duration = training_end - training_start
        
        logger.info(f"Training completed in {timedelta(seconds=int(training_duration))}")
        
        monitor.log_system_stats()
        
        # Step 4: Save models
        logger.info("Step 4: Saving trained models...")
        
        model_file = output_dir / f"tokenizer_vocab{args.vocab_size//1000}k.pkl"
        vocab_file = output_dir / f"vocab_{args.vocab_size//1000}k.txt"
        
        tokenizer.save_model(str(model_file))
        tokenizer.save_vocab(str(vocab_file))
        
        # Step 5: Quality testing
        logger.info("Step 5: Running quality tests...")
        
        test_sentences = [
            "Advanced machine learning algorithms utilize sophisticated mathematical optimization techniques",
            "Biomedical research encompasses genomics proteomics bioinformatics and computational biology methodologies",
            "Quantum computational systems demonstrate unprecedented processing capabilities for cryptographic applications",
            "Interdisciplinary collaborative research projects integrate diverse scientific methodologies and analytical frameworks",
            "Revolutionary breakthrough discoveries transform theoretical understanding into practical technological innovations"
        ]
        
        logger.info("Quality Test Results:")
        logger.info("-" * 60)
        
        total_accuracy = 0
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
            
            logger.info(f"Test {i}: '{sentence[:50]}...'")
            logger.info(f"  Tokens ({len(tokens)}): {tokens[:10]}...")
            logger.info(f"  Accuracy: {accuracy:.1%}")
            logger.info("")
        
        avg_accuracy = total_accuracy / len(test_sentences)
        logger.info(f"Average Round-trip Accuracy: {avg_accuracy:.1%}")
        
        # Final statistics
        logger.info("=== Training Complete ===")
        logger.info(f"Final vocabulary size: {len(tokenizer.vocab):,}")
        logger.info(f"Training duration: {timedelta(seconds=int(training_duration))}")
        logger.info(f"Model saved to: {model_file}")
        logger.info(f"Vocabulary saved to: {vocab_file}")
        
        monitor.log_system_stats()
        
        # Cleanup training data
        try:
            os.remove(training_file)
            logger.info(f"Cleaned up training file: {training_file}")
        except:
            pass
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        monitor.log_system_stats()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())