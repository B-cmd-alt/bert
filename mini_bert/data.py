"""
Data preprocessing pipeline for Mini-BERT MLM training.

Handles:
- Text loading from Wikipedia/BookCorpus format
- Sequence preparation and padding  
- Dynamic MLM masking (15% tokens)
- Batch generation with memory efficiency
"""

import numpy as np
import os
import random
from typing import List, Dict, Tuple, Iterator, Optional
from collections import deque
from tokenizer import WordPieceTokenizer
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

class MLMDataProcessor:
    """
    Masked Language Model data processor.
    
    Implements BERT-style MLM preprocessing:
    - 15% of tokens are selected for masking
    - Of those: 80% -> [MASK], 10% -> random token, 10% -> unchanged
    - Special tokens are never masked
    """
    
    def __init__(self, tokenizer: WordPieceTokenizer, 
                 max_seq_length: int = 64,
                 mlm_probability: float = 0.15,
                 replace_prob: float = 0.8,
                 random_prob: float = 0.1):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        
        # Token IDs for special tokens
        self.pad_id = tokenizer.vocab.get("[PAD]", 0)
        self.cls_id = tokenizer.vocab.get("[CLS]", 2) 
        self.sep_id = tokenizer.vocab.get("[SEP]", 3)
        self.mask_id = tokenizer.vocab.get("[MASK]", 4)
        self.unk_id = tokenizer.vocab.get("[UNK]", 1)
        
        # Special tokens that should never be masked
        self.special_token_ids = {self.pad_id, self.cls_id, self.sep_id, self.unk_id}
        
        self.vocab_size = tokenizer.get_vocab_size()
        print(f"MLM processor initialized: {self.mlm_probability*100:.1f}% masking, "
              f"max_seq_len={self.max_seq_length}")
    
    def _create_mlm_predictions(self, input_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Create MLM predictions for a sequence.
        
        Args:
            input_ids: Original token sequence
            
        Returns:
            masked_input_ids: Input with some tokens masked
            labels: Original tokens at masked positions (-100 for non-masked)
            mask: Binary mask (1 for masked positions)
        """
        masked_input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 = ignore in loss computation
        mask = [0] * len(input_ids)
        
        # Select tokens for masking (excluding special tokens)
        maskable_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id not in self.special_token_ids:
                maskable_positions.append(i)
        
        # Calculate number of tokens to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        num_to_mask = min(num_to_mask, len(maskable_positions))
        
        if num_to_mask == 0:
            return masked_input_ids, labels, mask
        
        # Randomly select positions to mask
        masked_positions = random.sample(maskable_positions, num_to_mask)
        
        for pos in masked_positions:
            original_token = input_ids[pos]
            labels[pos] = original_token  # Store original token as label
            mask[pos] = 1  # Mark as masked
            
            rand_val = random.random()
            
            if rand_val < self.replace_prob:
                # 80% of the time: replace with [MASK]
                masked_input_ids[pos] = self.mask_id
            elif rand_val < self.replace_prob + self.random_prob:
                # 10% of the time: replace with random token
                # Avoid special tokens in random replacement
                random_token = random.randint(5, self.vocab_size - 1)
                masked_input_ids[pos] = random_token
            # 10% of the time: keep original token (no change needed)
        
        return masked_input_ids, labels, mask
    
    def _truncate_and_pad(self, token_ids: List[int], labels: List[int], 
                         mask: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Truncate or pad sequences to max_seq_length.
        
        Args:
            token_ids: Token sequence
            labels: MLM labels  
            mask: MLM mask
            
        Returns:
            Padded/truncated arrays of shape [max_seq_length]
        """
        # Truncate if too long (keep [CLS] and [SEP])
        if len(token_ids) > self.max_seq_length:
            if token_ids[0] == self.cls_id and token_ids[-1] == self.sep_id:
                # Keep [CLS] and [SEP], truncate middle
                keep_length = self.max_seq_length - 2
                token_ids = [token_ids[0]] + token_ids[1:keep_length+1] + [token_ids[-1]]
                labels = [labels[0]] + labels[1:keep_length+1] + [labels[-1]]
                mask = [mask[0]] + mask[1:keep_length+1] + [mask[-1]]
            else:
                token_ids = token_ids[:self.max_seq_length]
                labels = labels[:self.max_seq_length]  
                mask = mask[:self.max_seq_length]
        
        # Pad if too short
        pad_length = self.max_seq_length - len(token_ids)
        if pad_length > 0:
            token_ids.extend([self.pad_id] * pad_length)
            labels.extend([-100] * pad_length)  # Ignore padded positions
            mask.extend([0] * pad_length)
        
        return np.array(token_ids), np.array(labels), np.array(mask)
    
    def process_sequence(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single text sequence for MLM training.
        
        Args:
            text: Input text string
            
        Returns:
            input_ids: Processed token IDs [max_seq_length]
            labels: MLM labels [max_seq_length] 
            mlm_mask: MLM mask [max_seq_length]
        """
        # Tokenize text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, 
                                        max_length=None)  # Don't pad yet
        
        # Apply MLM masking
        masked_ids, labels, mask = self._create_mlm_predictions(token_ids)
        
        # Truncate and pad
        input_ids, labels, mlm_mask = self._truncate_and_pad(masked_ids, labels, mask)
        
        return input_ids, labels, mlm_mask

class TextDataLoader:
    """
    Efficient text data loader for large corpora.
    
    Features:
    - Streaming data loading to handle large files
    - Sentence-level processing
    - Memory-efficient buffering
    - Random sampling
    """
    
    def __init__(self, data_files: List[str], 
                 processor: MLMDataProcessor,
                 buffer_size: int = 10000,
                 shuffle_buffer: bool = True):
        self.data_files = data_files
        self.processor = processor
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        
        # Validate data files exist
        for filepath in data_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Data loader initialized with {len(data_files)} files, "
              f"buffer_size={buffer_size}")
    
    def _read_lines(self) -> Iterator[str]:
        """Generator that yields lines from all data files."""
        for filepath in self.data_files:
            print(f"Reading from {filepath}...")
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line and len(line) > 20:  # Skip very short lines
                            yield line
                        
                        if line_num % 100000 == 0 and line_num > 0:
                            print(f"  Processed {line_num:,} lines from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    def _create_sentences(self, lines: Iterator[str]) -> Iterator[str]:
        """
        Convert lines to sentences, handling paragraph breaks.
        
        Args:
            lines: Iterator over text lines
            
        Yields:
            Individual sentences suitable for BERT training
        """
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    # Join paragraph and split into sentences
                    paragraph_text = ' '.join(current_paragraph)
                    sentences = self._split_sentences(paragraph_text)
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10:  # Minimum sentence length
                            yield sentence
                    
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Handle final paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            sentences = self._split_sentences(paragraph_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    yield sentence
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting on punctuation.
        
        Args:
            text: Input paragraph text
            
        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        import re
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length
                cleaned.append(sentence)
        
        return cleaned
    
    def create_batches(self, batch_size: int) -> Iterator[Dict[str, np.ndarray]]:
        """
        Create batches of MLM training data.
        
        Args:
            batch_size: Number of sequences per batch
            
        Yields:
            Batch dictionaries with 'input_ids', 'labels', 'mlm_mask'
        """
        sentence_iter = self._create_sentences(self._read_lines())
        buffer = deque(maxlen=self.buffer_size)
        
        # Fill initial buffer
        print("Filling data buffer...")
        for i, sentence in enumerate(sentence_iter):
            buffer.append(sentence)
            
            if len(buffer) >= self.buffer_size:
                break
            
            if i % 10000 == 0 and i > 0:
                print(f"  Buffer: {len(buffer):,} sentences")
        
        print(f"Buffer filled with {len(buffer):,} sentences")
        
        # Generate batches
        batch_input_ids = []
        batch_labels = []  
        batch_mlm_masks = []
        
        sentences_processed = 0
        
        while True:
            # Refill buffer periodically
            if len(buffer) < self.buffer_size // 2:
                try:
                    for _ in range(self.buffer_size // 4):
                        buffer.append(next(sentence_iter))
                except StopIteration:
                    # End of data - finish current batch if any
                    if batch_input_ids:
                        yield {
                            'input_ids': np.array(batch_input_ids),
                            'labels': np.array(batch_labels),
                            'mlm_mask': np.array(batch_mlm_masks)
                        }
                    break
            
            # Sample sentence from buffer
            if not buffer:
                break
                
            if self.shuffle_buffer:
                sentence = buffer.popleft() if random.random() < 0.5 else buffer.pop()
            else:
                sentence = buffer.popleft()
            
            # Process sentence
            try:
                input_ids, labels, mlm_mask = self.processor.process_sequence(sentence)
                
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_mlm_masks.append(mlm_mask)
                
                sentences_processed += 1
                
                # Yield batch when full
                if len(batch_input_ids) == batch_size:
                    yield {
                        'input_ids': np.array(batch_input_ids),
                        'labels': np.array(batch_labels), 
                        'mlm_mask': np.array(batch_mlm_masks)
                    }
                    
                    batch_input_ids = []
                    batch_labels = []
                    batch_mlm_masks = []
                    
                    if sentences_processed % 1000 == 0:
                        print(f"Processed {sentences_processed:,} sentences")
            
            except Exception as e:
                print(f"Error processing sentence: {e}")
                continue

def prepare_data_loaders(data_config=DATA_CONFIG, training_config=TRAINING_CONFIG) -> Tuple[TextDataLoader, WordPieceTokenizer]:
    """
    Prepare data loaders for training.
    
    Args:
        data_config: Data configuration
        training_config: Training configuration
        
    Returns:
        data_loader: Configured data loader
        tokenizer: Trained tokenizer
    """
    # Load or train tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=MODEL_CONFIG.vocab_size)
    
    tokenizer_path = data_config.vocab_save_path
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer.load_model(tokenizer_path)
    else:
        print("Training new tokenizer...")
        if os.path.exists(data_config.train_data_path):
            tokenizer.train(data_config.train_data_path, 
                          max_lines=data_config.max_lines_for_vocab)
            tokenizer.save_model(tokenizer_path)
        else:
            raise FileNotFoundError(f"Training data not found: {data_config.train_data_path}")
    
    # Create MLM processor
    processor = MLMDataProcessor(
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        mlm_probability=training_config.mlm_probability,
        replace_prob=training_config.replace_prob,
        random_prob=training_config.random_prob
    )
    
    # Create data loader
    data_files = [data_config.train_data_path] if isinstance(data_config.train_data_path, str) else data_config.train_data_path
    data_loader = TextDataLoader(
        data_files=data_files,
        processor=processor,
        buffer_size=data_config.buffer_size,
        shuffle_buffer=True
    )
    
    return data_loader, tokenizer

if __name__ == "__main__":
    # Test data processing
    print("Testing data processing pipeline...")
    
    # Test with sample data
    sample_data_path = "../data/bert_50k_sample.txt"
    
    if os.path.exists(sample_data_path):
        # Test tokenizer
        tokenizer = WordPieceTokenizer(vocab_size=1000)  # Small vocab for testing
        
        # Train on small sample
        tokenizer.train(sample_data_path, max_lines=1000)
        print(f"Tokenizer trained: {tokenizer.get_vocab_size()} tokens")
        
        # Test MLM processor
        processor = MLMDataProcessor(tokenizer, max_seq_length=32)
        
        test_text = "This is a test sentence for BERT training."
        input_ids, labels, mlm_mask = processor.process_sequence(test_text)
        
        print(f"\nMLM processing test:")
        print(f"Original text: {test_text}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"MLM mask shape: {mlm_mask.shape}")
        print(f"Masked positions: {np.sum(mlm_mask)}")
        
        # Test data loader
        data_loader = TextDataLoader([sample_data_path], processor, buffer_size=100)
        
        print(f"\nTesting batch generation...")
        batch_count = 0
        for batch in data_loader.create_batches(batch_size=4):
            print(f"Batch {batch_count}: input_ids {batch['input_ids'].shape}, "
                  f"labels {batch['labels'].shape}, mlm_mask {batch['mlm_mask'].shape}")
            
            # Show masking statistics
            total_tokens = np.sum(batch['input_ids'] != 0)  # Non-padding tokens
            masked_tokens = np.sum(batch['mlm_mask'])
            mask_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
            print(f"    Masking ratio: {mask_ratio:.3f} ({masked_tokens}/{total_tokens})")
            
            batch_count += 1
            if batch_count >= 3:  # Just test a few batches
                break
        
        print("✓ Data processing pipeline tested successfully!")
    
    else:
        print(f"Sample data not found at {sample_data_path}")
        print("Please ensure data file exists for testing")