#!/usr/bin/env python3
"""
Create 50K BERT Embedding from Existing 100K Model
Fast approach: adapt existing 100k vocabulary to 50k by selecting most frequent tokens.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from collections import Counter

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
from wordpiece_simple import SimpleWordPieceTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_50k_from_100k():
    """Create optimized 50K vocabulary from existing 100K model."""
    
    logger.info("=== Creating 50K BERT Embedding from 100K Model ===")
    
    # Load existing 100k model
    model_100k_file = "large_models/tokenizer_vocab100k.pkl"
    vocab_100k_file = "large_models/vocab_100k.txt"
    
    if not os.path.exists(model_100k_file):
        logger.error(f"100K model not found: {model_100k_file}")
        return None
    
    logger.info(f"Loading 100K model from {model_100k_file}...")
    
    try:
        # Load the 100k tokenizer
        tokenizer_100k = SimpleWordPieceTokenizer(vocab_size=100000)
        tokenizer_100k.load_model(model_100k_file)
        
        logger.info(f"Loaded 100K vocabulary with {len(tokenizer_100k.vocab)} tokens")
        
        # Create new 50k tokenizer
        tokenizer_50k = SimpleWordPieceTokenizer(vocab_size=50000)
        
        # Get the most important tokens (first 50k from the original vocabulary)
        # The original vocab is already ordered by importance
        vocab_items = list(tokenizer_100k.vocab.items())
        
        # Keep special tokens + most frequent 50k tokens
        selected_tokens = {}
        inverse_vocab = {}
        
        # First, add special tokens
        for token in tokenizer_100k.special_tokens:
            if token in tokenizer_100k.vocab:
                token_id = tokenizer_100k.vocab[token]
                selected_tokens[token] = len(selected_tokens)
                inverse_vocab[len(selected_tokens)-1] = token
        
        # Then add most frequent tokens up to 50k
        added_count = len(selected_tokens)
        for token, original_id in vocab_items:
            if added_count >= 50000:
                break
            if token not in selected_tokens:  # Skip if already added as special token
                selected_tokens[token] = added_count
                inverse_vocab[added_count] = token
                added_count += 1
        
        # Update the 50k tokenizer
        tokenizer_50k.vocab = selected_tokens
        tokenizer_50k.inverse_vocab = inverse_vocab
        tokenizer_50k.special_tokens = tokenizer_100k.special_tokens
        tokenizer_50k.unk_token = tokenizer_100k.unk_token
        
        # Create output directory
        output_dir = Path("bert_50k_models")
        output_dir.mkdir(exist_ok=True)
        
        # Save the new 50k model
        model_50k_file = output_dir / "bert_50k_tokenizer.pkl"
        vocab_50k_file = output_dir / "bert_50k_vocab.txt"
        
        logger.info(f"Saving 50K model to {model_50k_file}...")
        tokenizer_50k.save_model(str(model_50k_file))
        
        logger.info(f"Saving 50K vocabulary to {vocab_50k_file}...")
        tokenizer_50k.save_vocab(str(vocab_50k_file))
        
        # Verify the model works
        test_text = "This is a test sentence for the BERT tokenizer."
        tokens = tokenizer_50k.tokenize(test_text)
        
        logger.info("=== 50K BERT EMBEDDING CREATED SUCCESSFULLY ===")
        logger.info(f"Final vocabulary size: {len(tokenizer_50k.vocab):,} tokens")
        logger.info(f"Model saved: {model_50k_file}")
        logger.info(f"Vocabulary saved: {vocab_50k_file}")
        logger.info(f"Test tokenization: '{test_text}' -> {tokens}")
        
        return str(model_50k_file), {
            "final_vocab_size": len(tokenizer_50k.vocab),
            "source_vocab_size": len(tokenizer_100k.vocab),
            "model_file": str(model_50k_file),
            "vocab_file": str(vocab_50k_file)
        }
        
    except Exception as e:
        logger.error(f"Error creating 50K model: {e}")
        return None, None

if __name__ == "__main__":
    model_file, results = create_50k_from_100k()
    
    if model_file:
        logger.info("🎉 SUCCESS: 50K BERT embedding created!")
        logger.info("Ready for use in your applications!")
    else:
        logger.error("❌ Failed to create 50K model")