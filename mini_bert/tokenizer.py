"""
WordPiece Tokenizer for Mini-BERT (8K vocabulary).
Greedy WordPiece training with ## boundary markers.
"""
import re
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from config import MODEL_CONFIG

class WordPieceTokenizer:
    """
    WordPiece tokenizer implementation with BPE-style training.
    
    Design choices:
    - 8K vocab (small but sufficient for learning)
    - Greedy longest-first matching for fast inference
    - ## prefix for subword continuation (BERT standard)
    - Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
    """
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        
        # Special tokens (must be first 5 IDs for config compatibility)
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]" 
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning for tokenization."""
        # Remove excessive whitespace, keep basic punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        # Split on punctuation but keep it
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        return text.lower()
    
    def _get_word_frequency(self, text_file: str, max_lines: Optional[int] = None) -> Dict[str, int]:
        """Extract word frequencies from text file."""
        word_freq = Counter()
        line_count = 0
        
        print(f"Reading text from {text_file}...")
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                    
                cleaned = self._clean_text(line)
                words = cleaned.split()
                for word in words:
                    if word.strip():  # Skip empty strings
                        word_freq[word] += 1
                
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"  Processed {line_count:,} lines, found {len(word_freq):,} unique words")
        
        print(f"Final: {line_count:,} lines, {len(word_freq):,} unique words")
        return dict(word_freq)
    
    def _get_subword_frequency(self, word_freq: Dict[str, int]) -> Counter:
        """Get subword pair frequencies for BPE training."""
        subword_freq = Counter()
        
        for word, freq in word_freq.items():
            # Convert to character list with end-of-word marker
            chars = list(word) + ['</w>']
            
            # Count all adjacent pairs
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                subword_freq[pair] += freq
        
        return subword_freq
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freq: Dict[str, int]) -> Dict[str, int]:
        """Merge most frequent pair and update word frequencies."""
        new_word_freq = {}
        bigram = ''.join(pair)
        
        for word, freq in word_freq.items():
            # Replace the pair in this word
            chars = list(word) + ['</w>']
            new_chars = []
            i = 0
            
            while i < len(chars):
                if (i < len(chars) - 1 and 
                    chars[i] == pair[0] and chars[i + 1] == pair[1]):
                    # Found the pair - merge it
                    new_chars.append(bigram)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            # Convert back to word (remove </w>)
            if new_chars and new_chars[-1] == '</w>':
                new_chars = new_chars[:-1]
            
            new_word = ''.join(new_chars)
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def train(self, text_file: str, max_lines: Optional[int] = None) -> None:
        """
        Train WordPiece tokenizer using BPE algorithm.
        
        Algorithm:
        1. Start with character-level vocabulary
        2. Iteratively merge most frequent adjacent pairs
        3. Continue until reaching target vocab size
        """
        print(f"Training WordPiece tokenizer (target vocab: {self.vocab_size})")
        
        # Get initial word frequencies
        word_freq = self._get_word_frequency(text_file, max_lines)
        
        # Initialize vocabulary with individual characters
        all_chars = set()
        for word in word_freq.keys():
            all_chars.update(word)
        
        # Add characters to vocab (after special tokens)
        current_vocab_size = len(self.special_tokens)
        for char in sorted(all_chars):
            if current_vocab_size >= self.vocab_size:
                break
            if char not in self.vocab:
                self.vocab[char] = current_vocab_size
                self.inverse_vocab[current_vocab_size] = char
                current_vocab_size += 1
        
        print(f"Initialized with {current_vocab_size} tokens ({len(all_chars)} characters)")
        
        # BPE training loop
        iteration = 0
        while current_vocab_size < self.vocab_size:
            # Get current subword frequencies
            subword_freq = self._get_subword_frequency(word_freq)
            
            if not subword_freq:
                print("No more pairs to merge")
                break
            
            # Find most frequent pair
            most_frequent_pair = subword_freq.most_common(1)[0][0]
            freq_count = subword_freq[most_frequent_pair]
            
            # Create new token (with ## prefix for continuation)
            first, second = most_frequent_pair
            if second != '</w>':
                # Standard subword merge
                new_token = first + second
                if not first.startswith('##') and iteration > 0:
                    # Add ## prefix for subword pieces (except root words)
                    new_token = '##' + new_token
            else:
                # End of word - just use first part
                new_token = first
            
            # Add to vocabulary
            self.vocab[new_token] = current_vocab_size
            self.inverse_vocab[current_vocab_size] = new_token
            current_vocab_size += 1
            
            # Merge the pair in all words
            word_freq = self._merge_vocab(most_frequent_pair, word_freq)
            
            iteration += 1
            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}: merged '{first}' + '{second}' -> '{new_token}' "
                      f"(freq: {freq_count:,}, vocab: {current_vocab_size:,})")
        
        print(f"Training complete: {len(self.vocab):,} tokens in vocabulary")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize single word using greedy longest-first matching.
        
        Algorithm:
        1. Try to match longest possible substring from start
        2. If no match, use [UNK]
        3. Continue with remaining characters
        """
        if not word:
            return []
        
        tokens = []
        start = 0
        
        while start < len(word):
            # Try to find longest substring match
            end = len(word)
            found_match = False
            
            while end > start:
                substr = word[start:end]
                
                # For non-first subwords, try with ## prefix
                if start > 0:
                    prefixed_substr = '##' + substr
                    if prefixed_substr in self.vocab:
                        tokens.append(prefixed_substr)
                        start = end
                        found_match = True
                        break
                
                # Try without prefix (for first subword or full words)
                if substr in self.vocab:
                    tokens.append(substr)
                    start = end
                    found_match = True
                    break
                
                end -= 1
            
            if not found_match:
                # No valid subword found - use UNK and advance by 1 char
                tokens.append(self.unk_token)
                start += 1
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens."""
        if not text:
            return []
        
        cleaned = self._clean_text(text)
        words = cleaned.split()
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP]
            max_length: Maximum sequence length (pad/truncate)
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
        
        # Handle max_length
        if max_length:
            if len(token_ids) > max_length:
                # Truncate (keep [CLS] if present)
                if add_special_tokens:
                    token_ids = token_ids[:max_length-1] + [self.vocab[self.sep_token]]
                else:
                    token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length:
                # Pad
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.vocab[self.pad_token]] * pad_length)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
        
        # Join tokens and clean up ## prefixes
        text = ' '.join(tokens)
        
        # Remove ## prefixes and join subwords
        text = re.sub(r' ##', '', text)
        
        return text.strip()
    
    def save_model(self, filepath: str) -> None:
        """Save tokenizer to pickle file."""
        model_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load tokenizer from pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.inverse_vocab = model_data['inverse_vocab'] 
        self.vocab_size = model_data['vocab_size']
        self.special_tokens = model_data['special_tokens']
        
        print(f"Tokenizer loaded from {filepath} ({len(self.vocab):,} tokens)")
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

if __name__ == "__main__":
    # Test tokenizer training and functionality
    import os
    
    # Initialize tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=8192)
    
    # Train on sample data
    data_path = "../data/bert_50k_sample.txt"
    if os.path.exists(data_path):
        tokenizer.train(data_path, max_lines=50000)  # Use subset for testing
        
        # Save trained tokenizer
        tokenizer.save_model("tokenizer_8k.pkl")
        
        # Test encoding/decoding
        test_texts = [
            "Hello world! This is a test.",
            "Natural language processing with transformers.",
            "WordPiece tokenization splits words into subwords."
        ]
        
        print("\nTesting tokenization:")
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text, max_length=20)
            decoded = tokenizer.decode(ids)
            
            print(f"Original: {text}")
            print(f"Tokens:   {tokens}")
            print(f"IDs:      {ids}")
            print(f"Decoded:  {decoded}")
            print()
        
        print(f"Final vocabulary size: {tokenizer.get_vocab_size():,}")
        
        # Show some vocabulary examples
        print("\nVocabulary samples:")
        sample_tokens = list(tokenizer.vocab.keys())[:20]
        for token in sample_tokens:
            print(f"  '{token}' -> {tokenizer.vocab[token]}")
    
    else:
        print(f"Data file not found: {data_path}")
        print("Please check the path and try again.")