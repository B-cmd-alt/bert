import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import pickle

class SimpleWordPieceTokenizer:
    """Simplified WordPiece tokenizer with better round-trip guarantees."""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.unk_token = "[UNK]"
        self.max_input_chars_per_word = 100
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into words."""
        # Simple whitespace tokenization with basic cleanup
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation, lowercase
        words = text.split()
        return [w for w in words if w]  # Remove empty strings
    
    def train(self, text_file: str):
        """Train WordPiece tokenizer with a simplified approach."""
        print("Training Simple WordPiece tokenizer...")
        
        # Read and preprocess text
        all_words = []
        word_freq = Counter()
        
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = self._preprocess_text(line.strip())
                all_words.extend(words)
                for word in words:
                    word_freq[word] += 1
        
        print(f"Found {len(word_freq)} unique words from {len(all_words)} total words")
        
        # Start with character vocabulary
        vocab = set()
        
        # Add special tokens
        for token in self.special_tokens:
            vocab.add(token)
        
        # Add individual characters
        for word in word_freq.keys():
            for char in word:
                vocab.add(char)
        
        # Create initial subword splits (each character is a token)
        word_splits = {}
        for word, freq in word_freq.items():
            if word:  # Skip empty words
                splits = list(word)  # Split into characters
                word_splits[word] = splits
        
        print(f"Initial vocabulary size: {len(vocab)}")
        
        # Iteratively merge most frequent pairs
        target_merges = self.vocab_size - len(vocab)
        print(f"Performing up to {target_merges} merges...")
        
        for iteration in range(target_merges):
            # Count all adjacent pairs across all words
            pair_counts = Counter()
            
            for word, splits in word_splits.items():
                freq = word_freq[word]
                for i in range(len(splits) - 1):
                    pair = (splits[i], splits[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                print(f"No more pairs to merge at iteration {iteration}")
                break
            
            # Get the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = ''.join(best_pair)
            
            # Update word splits by merging the best pair
            new_word_splits = {}
            for word, splits in word_splits.items():
                new_splits = []
                i = 0
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == best_pair[0] and splits[i + 1] == best_pair[1]:
                        # Merge this pair
                        new_splits.append(merged_token)
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                new_word_splits[word] = new_splits
            
            word_splits = new_word_splits
            vocab.add(merged_token)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}'")
        
        # Build the final vocabulary mapping
        vocab_list = sorted(list(vocab))
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        
        # Store the final word splits for reference
        self.word_splits = word_splits
        
        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using greedy longest-match."""
        if not word or len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        
        # If we've seen this exact word during training, use its splits
        if hasattr(self, 'word_splits') and word in self.word_splits:
            return self.word_splits[word]
        
        # Otherwise, use greedy longest-first matching
        tokens = []
        start = 0
        
        while start < len(word):
            # Try to find the longest possible match
            found_match = False
            for end in range(len(word), start, -1):  # Start from longest possible
                candidate = word[start:end]
                
                # Add ## prefix for non-first tokens if needed
                if tokens and len(candidate) < len(word):  # Only for subwords
                    candidate_with_prefix = "##" + candidate
                    if candidate_with_prefix in self.vocab:
                        tokens.append(candidate_with_prefix)
                        start = end
                        found_match = True
                        break
                
                # Try without prefix
                if candidate in self.vocab:
                    tokens.append(candidate)
                    start = end
                    found_match = True
                    break
            
            if not found_match:
                # If no match found, try single character
                char = word[start]
                if char in self.vocab:
                    tokens.append(char)
                    start += 1
                else:
                    # Give up and return UNK for the whole word
                    return [self.unk_token]
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens."""
        words = self._preprocess_text(text)
        all_tokens = []
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            all_tokens.extend(word_tokens)
        
        return all_tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""
        
        tokens = [self.inverse_vocab.get(id_, self.unk_token) for id_ in token_ids]
        
        # Reconstruct text by handling ## prefixes
        result = []
        current_word = ""
        
        for token in tokens:
            if token in self.special_tokens:
                if current_word:
                    result.append(current_word)
                    current_word = ""
                result.append(token)
            elif token.startswith("##"):
                current_word += token[2:]  # Remove ## prefix
            else:
                if current_word:
                    result.append(current_word)
                current_word = token
        
        if current_word:
            result.append(current_word)
        
        return " ".join(result)
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in sorted(self.vocab.keys(), key=lambda x: self.vocab[x]):
                f.write(token + '\n')
        print(f"Vocabulary saved to {vocab_file}")
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        vocab_list = []
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                vocab_list.append(line.strip())
        
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        print(f"Vocabulary loaded. Size: {len(self.vocab)}")
    
    def save_model(self, filepath: str):
        """Save the complete model."""
        model_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'unk_token': self.unk_token,
            'word_splits': getattr(self, 'word_splits', {})
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the complete model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.inverse_vocab = model_data['inverse_vocab']
        self.vocab_size = model_data['vocab_size']
        self.special_tokens = model_data['special_tokens']
        self.unk_token = model_data['unk_token']
        self.word_splits = model_data.get('word_splits', {})
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SimpleWordPieceTokenizer(vocab_size=50000)
    
    # Create test data
    test_data = [
        "hello world this is a test",
        "machine learning is amazing",
        "natural language processing",
        "the quick brown fox jumps over the lazy dog",
        "python programming is fun",
        "artificial intelligence and deep learning",
        "tokenization preprocessing step",
        "transformers attention mechanism"
    ]
    
    # Create training file
    with open("test_train.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")
    
    # Train the tokenizer
    tokenizer.train("test_train.txt")
    
    # Test round-trip
    test_text = "hello world machine learning"
    print(f"\nOriginal: {test_text}")
    
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Save vocabulary
    tokenizer.save_vocab("simple_vocab.txt")