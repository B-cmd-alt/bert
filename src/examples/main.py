#!/usr/bin/env python3
"""
Complete WordPiece Tokenizer Training Pipeline

This script implements a complete pipeline for training a WordPiece tokenizer
from scratch using Wikipedia data, similar to BERT's tokenizer.
"""

import os
import sys
import argparse
from pathlib import Path

from download_wiki import download_wikipedia_dump, stream_and_clean_wikipedia
from wordpiece_simple import SimpleWordPieceTokenizer

def create_sample_data(filename: str = "sample_wiki.txt", size_mb: int = 10):
    """Create sample training data for testing."""
    sample_text = [
        "Natural language processing is a subfield of linguistics computer science and artificial intelligence",
        "Machine learning is a method of data analysis that automates analytical model building",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks",
        "Transformers are a type of neural network architecture that has been very successful for natural language processing",
        "The attention mechanism allows the model to focus on different parts of the input sequence",
        "BERT stands for Bidirectional Encoder Representations from Transformers",
        "WordPiece tokenization is a data driven tokenization scheme which generates subword units",
        "Python is a high level general purpose programming language",
        "Artificial intelligence is intelligence demonstrated by machines",
        "Computer science is the study of computational systems and computational thinking",
        "Data science is an inter disciplinary field that uses scientific methods to extract knowledge from data",
        "Neural networks are computing systems vaguely inspired by the biological neural networks",
        "Supervised learning is the machine learning task of learning a function that maps an input to an output",
        "Unsupervised learning is a type of algorithm that learns patterns from untagged data",
        "Reinforcement learning is an area of machine learning concerned with how agents take actions",
        "Natural language understanding is a subtopic of natural language processing in artificial intelligence",
        "Speech recognition is an interdisciplinary subfield of computer science and computational linguistics",
        "Computer vision is an interdisciplinary scientific field that deals with how computers gain understanding from images",
        "Information retrieval is the activity of obtaining information system resources that are relevant to an information need",
        "Knowledge representation and reasoning is the field of artificial intelligence dedicated to representing information"
    ]
    
    # Repeat and expand the sample data to reach desired size
    target_bytes = size_mb * 1024 * 1024
    current_bytes = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        while current_bytes < target_bytes:
            for text in sample_text:
                line = text + '\n'
                f.write(line)
                current_bytes += len(line.encode('utf-8'))
                if current_bytes >= target_bytes:
                    break
    
    print(f"Created sample data: {filename} ({current_bytes / (1024*1024):.1f} MB)")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Train WordPiece tokenizer from scratch")
    parser.add_argument("--vocab-size", type=int, default=50000, 
                       help="Target vocabulary size")
    parser.add_argument("--sample-only", action="store_true",
                       help="Use sample data instead of downloading Wikipedia")
    parser.add_argument("--data-size-mb", type=int, default=100,
                       help="Size of training data in MB")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for models and vocab")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("WordPiece Tokenizer Training Pipeline")
    print("=" * 50)
    
    # Step 1: Get training data
    if args.sample_only:
        print("Creating sample training data...")
        training_file = create_sample_data("sample_wiki.txt", args.data_size_mb)
    else:
        print("Downloading Wikipedia dump...")
        dump_file = download_wikipedia_dump()
        
        if dump_file and os.path.exists(dump_file):
            print("Processing and cleaning Wikipedia data...")
            training_file = stream_and_clean_wikipedia(dump_file, "wiki_clean.txt", args.data_size_mb)
        else:
            print("Download failed, falling back to sample data...")
            training_file = create_sample_data("sample_wiki.txt", args.data_size_mb)
    
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        return 1
    
    file_size_mb = os.path.getsize(training_file) / (1024 * 1024)
    print(f"Training data ready: {training_file} ({file_size_mb:.1f} MB)")
    
    # Step 2: Train tokenizer
    print(f"\nTraining WordPiece tokenizer (target vocab size: {args.vocab_size})...")
    tokenizer = SimpleWordPieceTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(training_file)
    
    # Step 3: Save model and vocabulary
    vocab_file = output_dir / "vocab.txt"
    model_file = output_dir / "tokenizer.pkl"
    
    print(f"\nSaving model and vocabulary...")
    tokenizer.save_vocab(str(vocab_file))
    tokenizer.save_model(str(model_file))
    
    # Step 4: Test the tokenizer
    print(f"\nTesting tokenizer...")
    test_sentences = [
        "Hello world, this is a test.",
        "Machine learning and artificial intelligence",
        "Natural language processing with transformers",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    print("\nTest Results:")
    print("-" * 40)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: {sentence}")
        
        tokens = tokenizer.tokenize(sentence)
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: {decoded}")
        
        # Check round-trip accuracy
        original_words = set(sentence.lower().replace('.', '').replace(',', '').split())
        decoded_words = set(decoded.lower().split())
        overlap = len(original_words.intersection(decoded_words))
        accuracy = overlap / len(original_words) if original_words else 0
        print(f"  Round-trip accuracy: {accuracy:.1%}")
    
    # Step 5: Generate statistics
    print(f"\nFinal Statistics:")
    print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"  Training data size: {file_size_mb:.1f} MB")
    print(f"  Model saved to: {model_file}")
    print(f"  Vocabulary saved to: {vocab_file}")
    
    print(f"\nTraining completed successfully!")
    print(f"\nTo use the tokenizer:")
    print(f"  from wordpiece_simple import SimpleWordPieceTokenizer")
    print(f"  tokenizer = SimpleWordPieceTokenizer()")
    print(f"  tokenizer.load_model('{model_file}')")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())