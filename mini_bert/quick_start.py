"""
Quick Start Script for Mini-BERT Learning Journey

This script helps beginners verify their setup and see BERT in action immediately.
Run this first to make sure everything is working!
"""

import numpy as np
import sys
import os

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def check_setup():
    """Check if all required files and dependencies are available."""
    print_section("CHECKING YOUR SETUP")
    
    required_files = [
        'model.py',
        'tokenizer.py',
        'config.py',
        'tokenizer_8k.pkl',
        'LEARNING_GUIDE.md'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} NOT FOUND")
            all_good = False
    
    if all_good:
        print("\n✅ All required files present!")
    else:
        print("\n❌ Some files are missing. Please check your installation.")
        return False
    
    return True

def test_basic_imports():
    """Test that all modules can be imported."""
    print_section("TESTING IMPORTS")
    
    try:
        from model import MiniBERT
        print("✓ MiniBERT model imported")
        
        from tokenizer import WordPieceTokenizer
        print("✓ Tokenizer imported")
        
        from config import Config
        print("✓ Config imported")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        return False

def demonstrate_embeddings():
    """Show how embeddings work with a simple example."""
    print_section("UNDERSTANDING EMBEDDINGS")
    
    # Simple embedding example
    vocab_size = 5
    hidden_size = 3
    
    # Create a toy embedding matrix
    embedding_matrix = np.array([
        [0.1, 0.2, 0.3],    # Word 0
        [0.4, 0.5, 0.6],    # Word 1
        [0.7, 0.8, 0.9],    # Word 2
        [1.0, 1.1, 1.2],    # Word 3
        [1.3, 1.4, 1.5]     # Word 4
    ])
    
    print("Embedding Matrix (5 words, 3 dimensions each):")
    print(embedding_matrix)
    
    # Lookup example
    word_ids = [1, 3, 2]  # "cat", "sat", "on"
    embeddings = embedding_matrix[word_ids]
    
    print(f"\nLooking up words {word_ids}:")
    print(embeddings)
    print("\nNotice: Each word ID gives us its corresponding vector!")

def demonstrate_attention_concept():
    """Explain attention with a simple example."""
    print_section("UNDERSTANDING ATTENTION")
    
    print("Attention answers: 'Which words should I pay attention to?'\n")
    
    # Simple attention scores
    words = ['The', 'cat', 'sat']
    attention_from_cat = [0.1, 0.8, 0.1]  # 'cat' mostly looks at itself
    
    print(f"When processing 'cat':")
    for word, score in zip(words, attention_from_cat):
        bar = '█' * int(score * 20)
        print(f"  {word:6s}: {bar} {score:.1f}")
    
    print("\nAttention weights always sum to 1.0 (they're probabilities)")

def run_mini_bert_demo():
    """Run a complete Mini-BERT example."""
    print_section("MINI-BERT IN ACTION")
    
    try:
        from model import MiniBERT
        from tokenizer import WordPieceTokenizer
        
        # Load model and tokenizer
        print("Loading Mini-BERT model...")
        model = MiniBERT()
        tokenizer = WordPieceTokenizer.load('tokenizer_8k.pkl')
        
        # Process a sentence
        text = "The cat sat on the mat"
        print(f"\nInput text: '{text}'")
        
        # Tokenize
        input_ids = tokenizer.encode(text)
        print(f"Token IDs: {input_ids}")
        print(f"Tokens: {tokenizer.decode(input_ids).split()}")
        
        # Forward pass
        input_batch = np.array([input_ids])
        logits, cache = model.forward(input_batch)
        
        print(f"\nModel output shape: {logits.shape}")
        print(f"  Batch size: {logits.shape[0]}")
        print(f"  Sequence length: {logits.shape[1]}")
        print(f"  Vocabulary size: {logits.shape[2]}")
        
        # Show what's in the cache
        print("\nWhat the model computed internally:")
        for key in sorted(cache.keys()):
            if isinstance(cache[key], np.ndarray):
                print(f"  {key}: shape {cache[key].shape}")
        
        print("\n✅ Mini-BERT is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error running Mini-BERT: {e}")
        return False
    
    return True

def print_next_steps():
    """Guide the user on what to do next."""
    print_section("YOUR LEARNING PATH")
    
    print("""
📚 NEXT STEPS:

1. Read the Learning Guide:
   → Open LEARNING_GUIDE.md for your complete learning path

2. Run Interactive Notebooks:
   → jupyter notebook notebooks/01_understanding_embeddings.ipynb
   → jupyter notebook notebooks/02_attention_mechanism.ipynb

3. Explore the Code:
   → Start with model.py to see the architecture
   → Look at simple_test.py for basic usage

4. Understand the Math:
   → Read MATHEMATICAL_DERIVATIONS.md alongside the code
   → Use docs/learning_materials/visual_guide.md for quick reference

5. Experiment:
   → Modify parameters in config.py
   → Try different sentences in the demos
   → Break things and fix them!

💡 REMEMBER:
- BERT is just matrix operations applied cleverly
- Start with small examples, then scale up
- Use the visual guide when confused about shapes
- Learning happens through experimentation!

Happy learning! 🚀
""")

def main():
    """Run all demonstrations."""
    print("🎯 MINI-BERT QUICK START")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        return
    
    # Test imports
    if not test_basic_imports():
        return
    
    # Run demonstrations
    demonstrate_embeddings()
    demonstrate_attention_concept()
    
    # Run full Mini-BERT
    if run_mini_bert_demo():
        print_next_steps()

if __name__ == "__main__":
    main()