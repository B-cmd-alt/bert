# BERT WordPiece Tokenizer

A complete implementation of WordPiece tokenization from scratch, compatible with BERT's tokenization approach. This project provides the essential preprocessing component needed before BERT model training.

## 📁 Project Structure

```
bert-wordpiece-tokenizer/
├── wordpiece_tokenizer.py    # Main tokenizer implementation
├── download_wiki.py          # Wikipedia data preparation
├── vocab.txt                 # Trained vocabulary (442 tokens)
├── tokenizer.pkl            # Complete saved model
└── README.md               # This file
```

## 📋 File Descriptions

### `wordpiece_tokenizer.py` - Core Implementation
**The main WordPiece tokenizer class with complete functionality:**

- **Training Algorithm**: BPE-style subword learning from text corpus
- **BERT Compatibility**: Uses exact same special tokens and `##` prefix convention
- **Tokenization**: Greedy longest-first matching for text → subwords
- **Encoding/Decoding**: Text ↔ Token ID conversion
- **Persistence**: Save/load vocabulary and complete model state

**Key Features:**
- Handles out-of-vocabulary words through subword segmentation
- Memory-efficient streaming through large text files
- Configurable vocabulary size (default: 8000 tokens)
- Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`

### `download_wiki.py` - Data Preparation
**Wikipedia corpus preparation for tokenizer training:**

- **Downloads**: March 2024 English Wikipedia dump (25GB compressed)
- **Streams & Cleans**: Processes XML without loading into memory
- **Filters**: Removes XML tags, extracts clean sentences
- **Outputs**: Creates clean text file (default: 100MB) ready for training

**Usage**: Provides the large-scale text corpus needed for meaningful vocabulary learning.

### `vocab.txt` - Vocabulary File (442 lines)
**Human-readable token list in ID order:**

```
[CLS]      # Token ID 0
[MASK]     # Token ID 1  
[PAD]      # Token ID 2
[SEP]      # Token ID 3
[UNK]      # Token ID 4
a          # Token ID 5
act        # Token ID 6
...
artificial # Token ID 46
...
```

**Format**: One token per line, line number = Token ID

### `tokenizer.pkl` - Complete Model (9.2KB)
**Binary file containing full tokenizer state:**

- Forward mapping: `{'[CLS]': 0, 'machine': 1234, ...}`
- Inverse mapping: `{0: '[CLS]', 1234: 'machine', ...}`
- Model configuration and special tokens
- Ready for immediate encoding/decoding

## 🔄 How WordPiece Tokenization Works

### Training Process

1. **Word Frequency Analysis**
   ```python
   # Count word occurrences in corpus
   word_freq = {"machine": 1000, "learning": 800, ...}
   ```

2. **Initialize with Characters**
   ```python
   # Start with all unique characters + special tokens
   vocab = {'[PAD]', '[UNK]', 'a', 'b', 'c', ..., 'z'}
   ```

3. **Character-Level Splitting**
   ```python
   # "hello" → ["h", "##e", "##l", "##l", "##o"]
   # First char has no ##, continuation chars have ##
   ```

4. **Iterative Pair Merging (BPE)**
   ```python
   # Find most frequent adjacent pair: ("##l", "##l")
   # Merge to create new token: "##ll" 
   # Update all words: ["h", "##e", "##ll", "##o"]
   # Repeat until target vocabulary size reached
   ```

### Tokenization Process

**Example**: `"Machine learning is revolutionizing artificial intelligence"`

1. **Preprocessing**: Lowercase, split words, remove punctuation
   ```
   ["machine", "learning", "is", "revolutionizing", "artificial", "intelligence"]
   ```

2. **Per-Word Tokenization** (Greedy Longest-First):
   ```python
   "machine" → ["machine"]                    # Complete word in vocab
   "learning" → ["learn", "##ing"]            # Split into subwords  
   "revolutionizing" → ["revolution", "##izing"]  # Long word split
   "artificial" → ["art", "##ificial"]        # Subword segmentation
   ```

3. **Final Token Sequence**:
   ```
   ["machine", "learn", "##ing", "is", "revolution", "##izing", 
    "art", "##ificial", "intel", "##ligence"]
   ```

### Encoding & Decoding

**Encoding (Text → IDs)**:
```python
text = "machine learning"
tokens = tokenizer.tokenize(text)    # ["machine", "learn", "##ing"]
ids = tokenizer.encode(text)         # [1234, 1235, 1236]
```

**Decoding (IDs → Text)**:
```python
ids = [1234, 1235, 1236]
tokens = [inverse_vocab[id] for id in ids]  # ["machine", "learn", "##ing"]

# Smart reconstruction:
# "machine" → text = "machine" 
# "learn" → text = "machine learn" (space added)
# "##ing" → text = "machine learning" (no space, attach)
```

## 🚀 Usage Examples

### Quick Start
```python
from wordpiece_tokenizer import WordPieceTokenizer

# Load pre-trained tokenizer
tokenizer = WordPieceTokenizer()
tokenizer.load_model("tokenizer.pkl")

# Tokenize text
text = "artificial intelligence"
tokens = tokenizer.tokenize(text)     # ["art", "##ificial", "intel", "##ligence"]
ids = tokenizer.encode(text)          # [43, 44, 124, 125]
decoded = tokenizer.decode(ids)       # "artificial intelligence"
```

### Training New Tokenizer
```python
# Train on your own corpus
tokenizer = WordPieceTokenizer(vocab_size=8000)
tokenizer.train("your_text_corpus.txt")
tokenizer.save_model("my_tokenizer.pkl")
tokenizer.save_vocab("my_vocab.txt")
```

### Data Preparation
```python
from download_wiki import download_wikipedia_dump, stream_and_clean_wikipedia

# Prepare Wikipedia training data
dump_file = download_wikipedia_dump()
clean_text = stream_and_clean_wikipedia(dump_file, max_size_mb=500)
```

## 🗂️ Vocabulary Storage Formats

### Two Storage Methods

**1. `vocab.txt` - Human-Readable**
- One token per line, in ID order
- Line number (0-indexed) = Token ID
- BERT-compatible format
- Can be inspected and edited manually
- Used for integration with other tools

**2. `tokenizer.pkl` - Complete Model**
- Binary pickle format
- Contains both forward and inverse mappings
- Stores all model configuration
- Faster loading for production use
- Complete tokenizer state preservation

### Mapping Structure
```python
# Forward mapping (token → ID)
vocab = {
    '[CLS]': 0, '[MASK]': 1, '[PAD]': 2,
    'machine': 1234, 'learn': 1235, '##ing': 1236
}

# Inverse mapping (ID → token)  
inverse_vocab = {
    0: '[CLS]', 1: '[MASK]', 2: '[PAD]',
    1234: 'machine', 1235: 'learn', 1236: '##ing'
}
```

## 🎯 Key Advantages

1. **BERT Compatibility**: Drop-in replacement for BERT tokenization
2. **Subword Robustness**: Handles any text without `[UNK]` tokens
3. **Memory Efficient**: Streams through large files during training
4. **Configurable**: Adjustable vocabulary size for different needs
5. **Complete Pipeline**: Training, tokenization, persistence all included
6. **Educational**: Clear, well-documented implementation for learning

## 🔧 Requirements

- Python 3.7+
- Standard library only (no external dependencies for core functionality)
- `urllib`, `bz2` for Wikipedia download
- `pickle`, `json` for model persistence

## 📚 Technical Details

### WordPiece vs BPE
- **WordPiece**: Uses `##` prefix for continuation tokens (BERT standard)
- **BPE**: Typically uses different continuation marking
- **Algorithm**: Both use similar frequency-based pair merging

### Special Token Handling
- `[CLS]`: Classification token (sentence start)
- `[SEP]`: Separator token (sentence boundary)  
- `[MASK]`: Masking token (masked language modeling)
- `[PAD]`: Padding token (sequence length normalization)
- `[UNK]`: Unknown token (fallback for rare cases)

### Tokenization Strategy
- **Greedy Longest-First**: Matches longest possible subword first
- **Fallback Mechanism**: Uses `[UNK]` only for extremely rare cases
- **Prefix Handling**: First subword has no `##`, continuations have `##`

This tokenizer is ready for BERT pretraining and provides the foundation for transformer-based language models!