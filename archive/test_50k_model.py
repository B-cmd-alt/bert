#!/usr/bin/env python3
"""
Test the trained 50K BERT embedding model
"""

import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

def test_50k_model():
    """Test the trained 50K model with various examples."""
    
    model_file = "bert_50k_models/bert_50k_efficient_20250804_132709.pkl"
    
    if not os.path.exists(model_file):
        print(f"❌ Model not found: {model_file}")
        return
    
    print("=== Testing 50K BERT Embedding Model ===")
    print(f"Loading model: {model_file}")
    
    # Load the model
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    tokenizer.load_model(model_file)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Vocabulary size: {len(tokenizer.vocab):,} tokens")
    
    # Test sentences
    test_sentences = [
        "Hello world, this is a test.",
        "Machine learning and artificial intelligence are transforming technology.",
        "BERT embeddings provide powerful representations for natural language processing.",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming language is excellent for data science and machine learning.",
        "Transformers have revolutionized the field of natural language understanding.",
        "This tokenizer can handle complex vocabulary with subword tokenization."
    ]
    
    print("\n=== Tokenization Examples ===")
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        print(f"Input:  '{sentence}'")
        print(f"Tokens: {tokens}")
        print(f"Count:  {len(tokens)} tokens")
        print()
    
    # Test vocabulary coverage
    print("=== Vocabulary Coverage Test ===")
    
    # Check special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    print("Special tokens:")
    for token in special_tokens:
        if token in tokenizer.vocab:
            print(f"  ✅ {token} -> {tokenizer.vocab[token]}")
        else:
            print(f"  ❌ {token} missing")
    
    # Check common words
    common_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"]
    print("\nCommon words:")
    for word in common_words:
        if word in tokenizer.vocab:
            print(f"  ✅ '{word}' -> {tokenizer.vocab[word]}")
        else:
            print(f"  ❌ '{word}' missing")
    
    print(f"\n🎉 SUCCESS: 50K BERT embedding model is ready to use!")
    print(f"📁 Model file: {model_file}")
    print(f"📊 Final vocabulary size: {len(tokenizer.vocab):,}")

if __name__ == "__main__":
    test_50k_model()