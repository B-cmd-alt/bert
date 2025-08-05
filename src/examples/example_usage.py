#!/usr/bin/env python3
"""
Example usage of the WordPiece tokenizer implementation.

This demonstrates how to:
1. Train a tokenizer from scratch
2. Save and load the model
3. Use the tokenizer for encoding/decoding
4. Test round-trip accuracy
"""

from wordpiece_simple import SimpleWordPieceTokenizer
from pathlib import Path

def demo_tokenizer():
    """Demonstrate complete tokenizer usage."""
    print("WordPiece Tokenizer Demo")
    print("=" * 30)
    
    # Create sample training data
    training_data = [
        "machine learning is a powerful tool for data analysis",
        "natural language processing helps computers understand human language", 
        "deep learning uses neural networks with multiple layers",
        "transformers have revolutionized natural language processing",
        "attention mechanisms allow models to focus on relevant information",
        "bert bidirectional encoder representations from transformers",
        "wordpiece tokenization breaks words into subword units",
        "python programming language for artificial intelligence",
        "supervised learning trains models on labeled data",
        "unsupervised learning finds patterns in unlabeled data",
        "reinforcement learning trains agents through rewards",
        "computer vision analyzes and understands visual information",
        "neural networks are inspired by biological brain structure",
        "backpropagation algorithm trains neural networks effectively",
        "gradient descent optimization minimizes loss functions"
    ]
    
    # Save training data to file
    train_file = "demo_train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for text in training_data:
            f.write(text + '\n')
    
    print(f"Created training data: {train_file}")
    
    # Initialize and train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    tokenizer.train(train_file)
    
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Save the model
    model_path = "demo_tokenizer.pkl"
    vocab_path = "demo_vocab.txt"
    
    tokenizer.save_model(model_path)
    tokenizer.save_vocab(vocab_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Test the tokenizer
    test_texts = [
        "machine learning and artificial intelligence",
        "natural language processing with transformers", 
        "python programming for data science",
        "new unseen words should become unknown tokens",
        "testing tokenization quality and accuracy"
    ]
    
    print("\nTokenizer Test Results:")
    print("-" * 40)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"  Tokens: {tokens}")
        
        # Encode
        encoded = tokenizer.encode(text)
        print(f"  Encoded IDs: {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"  Decoded: '{decoded}'")
        
        # Check accuracy
        original_words = set(text.lower().split())
        decoded_words = set(decoded.lower().split())
        
        if original_words:
            accuracy = len(original_words.intersection(decoded_words)) / len(original_words)
            print(f"  Round-trip accuracy: {accuracy:.1%}")
        
        # Check reversibility
        is_reversible = (tokenizer.decode(tokenizer.encode(text)).strip() == decoded.strip())
        print(f"  Reversible: {is_reversible}")
    
    # Load model and test
    print(f"\nTesting model loading...")
    new_tokenizer = SimpleWordPieceTokenizer()
    new_tokenizer.load_model(model_path)
    
    test_text = "machine learning is amazing"
    orig_encoded = tokenizer.encode(test_text)
    loaded_encoded = new_tokenizer.encode(test_text)
    
    print(f"Original model encoding: {orig_encoded}")
    print(f"Loaded model encoding: {loaded_encoded}")
    print(f"Models identical: {orig_encoded == loaded_encoded}")
    
    # Vocabulary inspection
    print(f"\nVocabulary Sample (first 20 tokens):")
    sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab[:20]:
        print(f"  {idx:3d}: '{token}'")
    
    print(f"\nSpecial tokens:")
    for token in tokenizer.special_tokens:
        if token in tokenizer.vocab:
            print(f"  {tokenizer.vocab[token]:3d}: '{token}'")
    
    # Clean up
    Path(train_file).unlink(missing_ok=True)
    print(f"\nDemo completed! Files created:")
    print(f"  - {model_path}")
    print(f"  - {vocab_path}")

def compare_with_simple_tokenization():
    """Compare WordPiece with simple whitespace tokenization."""
    print("\n" + "=" * 50)
    print("Comparison with Simple Tokenization")
    print("=" * 50)
    
    # Load the demo tokenizer
    tokenizer = SimpleWordPieceTokenizer()
    try:
        tokenizer.load_model("demo_tokenizer.pkl")
    except FileNotFoundError:
        print("Demo tokenizer not found. Run demo_tokenizer() first.")
        return
    
    test_sentences = [
        "preprocessing",
        "tokenization", 
        "subwordtokenization",
        "artificialintelligence",
        "machinelearning"
    ]
    
    print("\nSubword vs. Word-level Tokenization:")
    print("-" * 45)
    
    for sentence in test_sentences:
        print(f"\nInput: '{sentence}'")
        
        # WordPiece tokenization
        wp_tokens = tokenizer.tokenize(sentence)
        print(f"  WordPiece ({len(wp_tokens)} tokens): {wp_tokens}")
        
        # Simple tokenization (just split by spaces)
        simple_tokens = sentence.split()
        print(f"  Simple ({len(simple_tokens)} tokens): {simple_tokens}")
        
        # Show the advantage of subword tokenization
        if len(wp_tokens) > len(simple_tokens):
            print(f"  -> WordPiece breaks long words into meaningful parts")
        elif "[UNK]" in wp_tokens:
            print(f"  -> Contains unknown tokens")
        else:
            print(f"  -> Word fully recognized by vocabulary")

if __name__ == "__main__":
    demo_tokenizer()
    compare_with_simple_tokenization()