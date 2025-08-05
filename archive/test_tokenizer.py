import unittest
import os
import tempfile
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bert-wordpiece-tokenizer'))
from wordpiece_tokenizer import WordPieceTokenizer

class TestWordPieceTokenizer(unittest.TestCase):
    """Comprehensive tests for WordPiece tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = WordPieceTokenizer(vocab_size=50000)
        
        # Create sample training data
        self.test_data = [
            "hello world this is a test",
            "machine learning is amazing and wonderful",
            "natural language processing with transformers",
            "the quick brown fox jumps over the lazy dog",
            "python programming is fun and educational",
            "artificial intelligence will change the world",
            "deep learning models are very powerful",
            "tokenization is an important preprocessing step"
        ]
        
        # Create temporary training file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for line in self.test_data:
            self.temp_file.write(line + '\n')
        self.temp_file.close()
        
        # Train the tokenizer
        self.tokenizer.train(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_special_tokens_in_vocab(self):
        """Test that special tokens are in vocabulary."""
        for token in self.tokenizer.special_tokens:
            self.assertIn(token, self.tokenizer.vocab)
    
    def test_round_trip_simple(self):
        """Test basic round-trip encoding/decoding."""
        test_texts = [
            "hello world",
            "machine learning",
            "natural language processing",
            "the quick brown fox"
        ]
        
        for text in test_texts:
            with self.subTest(text=text):
                encoded = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(encoded)
                
                # Should be roughly equivalent (might differ in spacing/casing)
                self.assertIsInstance(encoded, list)
                self.assertIsInstance(decoded, str)
                self.assertTrue(len(encoded) > 0)
                self.assertTrue(len(decoded) > 0)
    
    def test_encode_decode_consistency(self):
        """Test that encode/decode operations are consistent."""
        text = "hello world machine learning"
        tokens = self.tokenizer.tokenize(text)
        encoded = self.tokenizer.encode(text)
        decoded_tokens = [self.tokenizer.inverse_vocab[id_] for id_ in encoded]
        
        self.assertEqual(tokens, decoded_tokens)
    
    def test_unknown_tokens(self):
        """Test handling of unknown/out-of-vocabulary tokens."""
        # Use text with characters likely not in small training set
        unknown_text = "xyz123 qwertyuiop"
        encoded = self.tokenizer.encode(unknown_text)
        decoded = self.tokenizer.decode(encoded)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_encoded = self.tokenizer.encode("")
        empty_decoded = self.tokenizer.decode([])
        
        self.assertEqual(empty_encoded, [])
        self.assertEqual(empty_decoded, "")
    
    def test_punctuation_handling(self):
        """Test handling of punctuation."""
        punct_text = "Hello, world! How are you? Fine, thanks."
        encoded = self.tokenizer.encode(punct_text)
        decoded = self.tokenizer.decode(encoded)
        
        # Should handle without crashing
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        # Test with some Unicode characters
        unicode_texts = [
            "café résumé naïve",  # Accented characters
            "Hello 世界",         # Mixed ASCII and CJK
            "émoji 😀 test"       # Emoji
        ]
        
        for text in unicode_texts:
            with self.subTest(text=text):
                try:
                    encoded = self.tokenizer.encode(text)
                    decoded = self.tokenizer.decode(encoded)
                    
                    self.assertIsInstance(encoded, list)
                    self.assertIsInstance(decoded, str)
                except Exception as e:
                    # If Unicode handling fails, should fail gracefully
                    self.fail(f"Unicode handling failed for '{text}': {e}")
    
    def test_long_input(self):
        """Test handling of very long input."""
        long_text = " ".join(["word"] * 1000)  # 1000 repeated words
        
        encoded = self.tokenizer.encode(long_text)
        decoded = self.tokenizer.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertTrue(len(encoded) > 0)
    
    def test_vocab_size_respected(self):
        """Test that vocabulary size is approximately correct."""
        # Should be close to target size (allowing for special tokens)
        vocab_size = len(self.tokenizer.vocab)
        target_size = self.tokenizer.vocab_size
        
        # Allow some flexibility as BPE might not reach exact target
        self.assertLessEqual(vocab_size, target_size * 1.1)  # Within 10% over
        self.assertGreaterEqual(vocab_size, 100)  # At least some reasonable minimum
    
    def test_tokenize_method(self):
        """Test the tokenize method returns string tokens."""
        text = "hello world machine"
        tokens = self.tokenizer.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        for token in tokens:
            self.assertIsInstance(token, str)
            self.assertIn(token, self.tokenizer.vocab)
    
    def test_save_load_vocab(self):
        """Test saving and loading vocabulary."""
        # Save vocabulary
        vocab_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        vocab_file.close()
        
        try:
            self.tokenizer.save_vocab(vocab_file.name)
            
            # Create new tokenizer and load vocab
            new_tokenizer = WordPieceTokenizer()
            new_tokenizer.load_vocab(vocab_file.name)
            
            # Should have same vocabulary
            self.assertEqual(self.tokenizer.vocab, new_tokenizer.vocab)
            self.assertEqual(self.tokenizer.inverse_vocab, new_tokenizer.inverse_vocab)
            
        finally:
            os.unlink(vocab_file.name)
    
    def test_save_load_model(self):
        """Test saving and loading entire model."""
        model_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
        model_file.close()
        
        try:
            self.tokenizer.save_model(model_file.name)
            
            # Create new tokenizer and load model
            new_tokenizer = WordPieceTokenizer()
            new_tokenizer.load_model(model_file.name)
            
            # Test that loaded model works the same
            test_text = "hello world"
            
            original_encoded = self.tokenizer.encode(test_text)
            loaded_encoded = new_tokenizer.encode(test_text)
            
            self.assertEqual(original_encoded, loaded_encoded)
            
        finally:
            os.unlink(model_file.name)
    
    def test_deterministic_behavior(self):
        """Test that tokenizer behavior is deterministic."""
        text = "hello world machine learning"
        
        # Encode the same text multiple times
        results = [self.tokenizer.encode(text) for _ in range(3)]
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)


class TestRoundTripProperty(unittest.TestCase):
    """Property-based testing for round-trip encoding/decoding."""
    
    def setUp(self):
        """Set up a tokenizer with more training data."""
        self.tokenizer = WordPieceTokenizer(vocab_size=50000)
        
        # Create more comprehensive training data
        training_sentences = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a subset of artificial intelligence",
            "natural language processing enables computers to understand human language",
            "deep learning models use neural networks with multiple layers",
            "python is a popular programming language for data science",
            "transformers have revolutionized natural language processing",
            "bert and gpt are famous transformer models",
            "tokenization is the process of converting text into tokens",
            "word piece tokenization handles out of vocabulary words",
            "subword tokenization improves model performance on rare words"
        ]
        
        # Add more diverse vocabulary
        additional_sentences = [
            "hello world goodbye universe",
            "computer science mathematics statistics",
            "algorithm data structure optimization",
            "neural network training validation testing",
            "supervised unsupervised reinforcement learning",
            "classification regression clustering dimensionality",
            "preprocessing feature engineering model selection",
            "overfitting underfitting regularization generalization"
        ]
        
        all_sentences = training_sentences + additional_sentences
        
        # Create training file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for sentence in all_sentences:
            self.temp_file.write(sentence + '\n')
        self.temp_file.close()
        
        # Train tokenizer
        self.tokenizer.train(self.temp_file.name)
    
    def tearDown(self):
        """Clean up."""
        os.unlink(self.temp_file.name)
    
    def test_round_trip_training_data(self):
        """Test round-trip on training data."""
        test_sentences = [
            "machine learning is amazing",
            "natural language processing",
            "the quick brown fox",
            "python programming language",
            "neural network training"
        ]
        
        for sentence in test_sentences:
            with self.subTest(sentence=sentence):
                encoded = self.tokenizer.encode(sentence)
                decoded = self.tokenizer.decode(encoded)
                
                # Check that we can encode and decode
                self.assertIsInstance(encoded, list)
                self.assertIsInstance(decoded, str)
                self.assertTrue(len(encoded) > 0)
                
                # The decoded text should contain the main words from original
                # (accounting for potential differences in spacing/punctuation)
                original_words = set(sentence.lower().split())
                decoded_words = set(decoded.lower().split())
                
                # Most words should be preserved (allowing some flexibility)
                overlap = len(original_words.intersection(decoded_words))
                self.assertGreater(overlap, len(original_words) * 0.5)  # At least 50% overlap


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)