#!/usr/bin/env python3
"""
Tests for the SimpleWordPieceTokenizer implementation.
"""

import unittest
import tempfile
import os
from wordpiece_simple import SimpleWordPieceTokenizer

class TestSimpleWordPieceTokenizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
        
        # Create comprehensive training data
        training_text = [
            "machine learning is a powerful method for data analysis",
            "natural language processing helps computers understand text",
            "deep learning uses neural networks with multiple layers", 
            "transformers revolutionized natural language processing",
            "artificial intelligence and machine learning are related fields",
            "python programming language for data science applications",
            "supervised learning trains models on labeled datasets",
            "unsupervised learning discovers patterns in unlabeled data",
            "reinforcement learning optimizes actions through rewards",
            "computer vision analyzes images and visual information"
        ]
        
        # Create temporary training file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for text in training_text:
            self.temp_file.write(text + '\n')
        self.temp_file.close()
        
        # Train tokenizer
        self.tokenizer.train(self.temp_file.name)
    
    def tearDown(self):
        """Clean up."""
        os.unlink(self.temp_file.name)
    
    def test_special_tokens_exist(self):
        """Test that special tokens are in vocabulary."""
        for token in self.tokenizer.special_tokens:
            self.assertIn(token, self.tokenizer.vocab)
    
    def test_encode_decode_roundtrip(self):
        """Test that encode/decode is reversible for training data."""
        test_texts = [
            "machine learning",
            "natural language processing", 
            "artificial intelligence",
            "deep learning networks"
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded)
                
                self.assertIsInstance(encoded, list)
                self.assertIsInstance(decoded, str)
                
                # Check that main words are preserved
                original_words = set(text.lower().split())
                decoded_words = set(decoded.lower().split())
                overlap = len(original_words.intersection(decoded_words))
                
                # Should preserve most content words
                self.assertGreater(overlap, len(original_words) * 0.7)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
    
    def test_unknown_token_handling(self):
        """Test handling of completely unknown text."""
        unknown_text = "xyzqwerty"  # Unlikely to be in training data
        encoded = self.tokenizer.encode(unknown_text)
        decoded = self.tokenizer.decode(encoded)
        
        # Should handle gracefully
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertTrue(len(encoded) > 0)
    
    def test_deterministic_behavior(self):
        """Test that tokenization is deterministic."""
        test_text = "machine learning artificial intelligence"
        
        results = []
        for _ in range(3):
            results.append(self.tokenizer.encode(test_text))
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)
    
    def test_save_load_model(self):
        """Test saving and loading model."""
        # Create temporary model file
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        model_file.close()
        
        try:
            # Save model
            self.tokenizer.save_model(model_file.name)
            
            # Load into new tokenizer
            new_tokenizer = SimpleWordPieceTokenizer()
            new_tokenizer.load_model(model_file.name)
            
            # Test that both tokenizers work identically
            test_text = "machine learning is amazing"
            
            original_encoded = self.tokenizer.encode(test_text)
            loaded_encoded = new_tokenizer.encode(test_text)
            
            self.assertEqual(original_encoded, loaded_encoded)
            
        finally:
            os.unlink(model_file.name)
    
    def test_vocabulary_size_reasonable(self):
        """Test that vocabulary size is reasonable."""
        vocab_size = len(self.tokenizer.vocab)
        
        # Should have special tokens plus learned vocabulary
        self.assertGreaterEqual(vocab_size, len(self.tokenizer.special_tokens))
        self.assertLess(vocab_size, self.tokenizer.vocab_size * 2)  # Not too large
    
    def test_tokenize_consistency(self):
        """Test consistency between tokenize and encode/decode."""
        test_text = "natural language processing"
        
        tokens = self.tokenizer.tokenize(test_text)
        encoded = self.tokenizer.encode(test_text)
        
        # Should have same number of tokens
        self.assertEqual(len(tokens), len(encoded))
        
        # Each token should map to correct ID
        for token, token_id in zip(tokens, encoded):
            self.assertEqual(self.tokenizer.vocab[token], token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)