import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import pickle

class WordPieceTokenizer:
    """WordPiece tokenizer implementation from scratch."""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.unk_token = "[UNK]"
        self.max_input_chars_per_word = 100
        
    def _get_word_frequency(self, text_file: str) -> Dict[str, int]:
        """Get word frequencies from text file."""
        word_freq = Counter()
        
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Basic whitespace tokenization
                words = line.strip().split()
                for word in words:
                    # Clean word: remove punctuation, lowercase
                    cleaned = re.sub(r'[^\w]', '', word.lower())
                    if cleaned:
                        word_freq[cleaned] += 1
        
        return dict(word_freq)
    
    def _get_character_stats(self, word_freq: Dict[str, int]) -> Counter:
        """Get character pair statistics for WordPiece training."""
        pairs = Counter()
        
        for word, freq in word_freq.items():
            # Convert word to character list with end-of-word marker
            word_chars = list(word) + ['</w>']
            
            # Count all adjacent pairs
            for i in range(len(word_chars) - 1):
                pair = (word_chars[i], word_chars[i + 1])
                pairs[pair] += freq
                
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freq: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary."""
        new_word_freq = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freq:
            # Split word into characters with spaces
            spaced_word = ' '.join(list(word)) + ' </w>'
            new_word = p.sub(''.join(pair), spaced_word)
            new_word_freq[new_word] = word_freq[word]
            
        return new_word_freq
    
    def train(self, text_file: str):
        """Train WordPiece tokenizer using BPE-style algorithm."""
        print("Training WordPiece tokenizer...")
        
        # Get word frequencies
        print("Computing word frequencies...")
        word_freq = self._get_word_frequency(text_file)
        print(f"Found {len(word_freq)} unique words")
        
        # Initialize vocabulary with characters and special tokens
        vocab = set()
        
        # Add special tokens first
        for token in self.special_tokens:
            vocab.add(token)
        
        # Add all characters from the corpus
        chars = set()
        for word in word_freq.keys():
            for char in word:
                chars.add(char)
        
        # Add characters to vocab
        for char in sorted(chars):
            vocab.add(char)
        
        print(f"Initial vocab size: {len(vocab)}")
        
        # Convert words to subword splits (start with characters)
        word_splits = {}
        for word, freq in word_freq.items():
            # Split into characters, first char has no ##, rest have ##
            if len(word) == 0:
                continue
            chars = [word[0]] + [f"#{char}" for char in word[1:]]
            word_splits[' '.join(chars)] = freq
        
        # BPE training loop
        target_vocab_size = min(self.vocab_size, len(vocab) + 10000)  # reasonable limit
        num_merges = target_vocab_size - len(vocab)
        print(f"Performing up to {num_merges} merges...")
        
        for i in range(num_merges):
            # Get pair statistics
            pairs = Counter()
            for word, freq in word_splits.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
            
            if not pairs:
                print(f"No more pairs to merge at iteration {i}")
                break
                
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Create the merged token
            merged_token = ''.join(best_pair)
            
            # Merge the pair in all word splits
            new_word_splits = {}
            bigram = re.escape(' '.join(best_pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            
            for word in word_splits:
                new_word = p.sub(merged_token, word)
                new_word_splits[new_word] = word_splits[word]
            
            word_splits = new_word_splits
            vocab.add(merged_token)
            
            if i % 100 == 0:
                print(f"Merge {i}: {best_pair} -> {merged_token}")
        
        # Build final vocabulary - ensure ## prefixed tokens are handled correctly
        final_vocab = set()
        for token in vocab:
            final_vocab.add(token)
            # For tokens that start with #, also add the ## version
            if token.startswith('#') and not token.startswith('##'):
                final_vocab.add('#' + token)
        
        vocab_list = sorted(list(final_vocab))
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        
        print(f"Training completed. Final vocab size: {len(self.vocab)}")
        
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file (one token per line)."""
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
        print(f"Vocabulary loaded from {vocab_file}. Size: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using the trained vocabulary."""
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            # Greedy longest-first matching
            while start < end:
                substr = word[start:end]
                # First token doesn't get ## prefix, subsequent ones do
                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                # If we can't find any valid subword, return UNK for the whole word
                return [self.unk_token]
            
            tokens.append(cur_substr)
            # Move start to after the matched substring
            # Remove ## prefix if it was added for position calculation
            actual_substr = cur_substr[2:] if cur_substr.startswith("##") else cur_substr
            start += len(actual_substr)
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subwords."""
        # Basic preprocessing
        text = text.strip().lower()
        words = text.split()
        
        all_tokens = []
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word:
                word_tokens = self._tokenize_word(clean_word)
                all_tokens.extend(word_tokens)
        
        return all_tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.inverse_vocab.get(id_, self.unk_token) for id_ in token_ids]
        
        # Join tokens and handle WordPiece prefixes
        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]  # Remove ## prefix
            else:
                if text:
                    text += " "
                text += token
        
        return text
    
    def save_model(self, filepath: str):
        """Save the entire model."""
        model_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'unk_token': self.unk_token
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the entire model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.inverse_vocab = model_data['inverse_vocab']
        self.vocab_size = model_data['vocab_size']
        self.special_tokens = model_data['special_tokens']
        self.unk_token = model_data['unk_token']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    tokenizer = WordPieceTokenizer(vocab_size=50000)
    
    # For testing with small data
    test_file = "sample.txt"
    with open(test_file, 'w') as f:
        f.write("hello world this is a test\n")
        f.write("machine learning is amazing\n")
        f.write("natural language processing\n")
    
    tokenizer.train(test_file)
    tokenizer.save_vocab("vocab.txt")
    
    # Test encoding/decoding
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")