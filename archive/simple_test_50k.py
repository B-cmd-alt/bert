#!/usr/bin/env python3
"""
Simple test of the 50K BERT model without unicode characters
"""

import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

def test_50k_model():
    model_file = "bert_50k_models/bert_50k_efficient_20250804_132709.pkl"
    
    print("=== Testing 50K BERT Embedding Model ===")
    print(f"Loading model: {model_file}")
    
    # Load the model
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    tokenizer.load_model(model_file)
    
    print("SUCCESS: Model loaded!")
    print(f"Vocabulary size: {len(tokenizer.vocab):,} tokens")
    
    # Test sentences
    test_sentences = [
        "Hello world, this is a test.",
        "Machine learning and artificial intelligence are transforming technology.",
        "BERT embeddings provide powerful representations for natural language processing.",
        "Python programming language is excellent for data science."
    ]
    
    print("\n=== Tokenization Examples ===")
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        print(f"Input: '{sentence}'")
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)} tokens")
        print()
    
    # Check special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    print("Special tokens:")
    for token in special_tokens:
        if token in tokenizer.vocab:
            print(f"  Found: {token} -> {tokenizer.vocab[token]}")
    
    print(f"\nSUCCESS: 50K BERT embedding model is ready!")
    print(f"Model file: {model_file}")
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")

if __name__ == "__main__":
    test_50k_model()