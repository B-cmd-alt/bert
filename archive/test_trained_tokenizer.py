#!/usr/bin/env python3
"""
Test the trained BERT tokenizer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))

from wordpiece_simple import SimpleWordPieceTokenizer

def test_tokenizer():
    print("=== Testing Trained BERT Tokenizer ===\n")
    
    # Load the trained tokenizer
    model_path = "bert_models/bert_tokenizer_vocab15k.pkl"
    tokenizer = SimpleWordPieceTokenizer()
    tokenizer.load_model(model_path)
    
    print(f"Loaded tokenizer with {len(tokenizer.vocab):,} vocabulary tokens\n")
    
    # Test sentences (mix of Wikipedia, books, and news style)
    test_sentences = [
        # Wikipedia style
        "Machine learning is a method of data analysis that automates analytical model building.",
        
        # Literature style  
        "It was the best of times, it was the worst of times, in the age of artificial intelligence.",
        
        # News style
        "Breaking news: Technology companies continue to innovate with advanced AI solutions.",
        
        # Technical terms
        "Deep learning neural networks enable natural language processing capabilities.",
        
        # Mixed domains
        "Scientific research demonstrates the effectiveness of computational algorithms in healthcare applications."
    ]
    
    print("=== Tokenization Test Results ===\n")
    
    total_tokens = 0
    total_words = 0
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Test {i}: '{sentence}'")
        
        # Tokenize
        tokens = tokenizer.tokenize(sentence)
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        
        # Calculate stats
        word_count = len(sentence.split())
        token_count = len(tokens)
        compression_ratio = token_count / word_count
        
        total_tokens += token_count
        total_words += word_count
        
        print(f"  Words: {word_count}, Tokens: {token_count}")
        print(f"  Compression: {compression_ratio:.2f} tokens/word")
        print(f"  Sample tokens: {tokens[:10]}...")
        print(f"  Decoded: '{decoded[:100]}...'")
        
        # Check round-trip accuracy
        original_words = set(sentence.lower().replace(',', '').replace('.', '').replace(':', '').split())
        decoded_words = set(decoded.lower().replace(',', '').replace('.', '').replace(':', '').split())
        overlap = len(original_words.intersection(decoded_words))
        accuracy = overlap / len(original_words) if original_words else 0
        print(f"  Round-trip accuracy: {accuracy:.1%}")
        print()
    
    # Overall statistics
    avg_compression = total_tokens / total_words
    print("=== Overall Statistics ===")
    print(f"Total vocabulary size: {len(tokenizer.vocab):,}")
    print(f"Average compression ratio: {avg_compression:.2f} tokens/word")
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Check if ready for BERT
    print("\n=== BERT Compatibility Check ===")
    if len(tokenizer.vocab) >= 10000:
        print("✅ Vocabulary size sufficient for BERT (≥10k tokens)")
    else:
        print("⚠️  Vocabulary size may be too small for optimal BERT performance")
        
    if avg_compression < 2.0:
        print("✅ Good compression ratio for efficient BERT training")
    else:
        print("⚠️  High compression ratio - may need more diverse training data")
    
    print(f"\n🎉 BERT tokenizer training completed successfully!")
    print(f"Model ready for transformer training: {model_path}")

if __name__ == "__main__":
    test_tokenizer()