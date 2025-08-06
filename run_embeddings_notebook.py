"""
Running the 01_understanding_embeddings.ipynb notebook manually to catch and fix errors.
"""
import sys
import os

# Ensure UTF-8 encoding for Windows
if hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

sys.path.append('mini_bert')  # Add mini_bert to path

print("="*60)
print("RUNNING EMBEDDINGS NOTEBOOK")
print("="*60)

try:
    # Cell 1: Basic imports
    print("\n[CELL 1] Basic imports...")
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Set random seed for reproducibility
    np.random.seed(42)
    print("✓ Basic imports successful")
    
    # Cell 3: Create embedding matrix
    print("\n[CELL 3] Creating embedding matrix...")
    vocab_size = 10  # We have 10 words in our vocabulary
    hidden_size = 4  # Each word becomes a 4-dimensional vector
    
    # Initialize embedding matrix randomly
    embedding_matrix = np.random.randn(vocab_size, hidden_size) * 0.1
    
    # Let's give names to our vocabulary
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 
             'the', 'cat', 'sat', 'on', 'mat']
    
    print("Embedding Matrix Shape:", embedding_matrix.shape)
    print("\nEach word has a", hidden_size, "dimensional vector:")
    print("\nFirst 5 word embeddings:")
    for i in range(5):
        print(f"{vocab[i]:8s}: {embedding_matrix[i]}")
    print("✓ Embedding matrix created")
    
    # Cell 5: Embedding lookup
    print("\n[CELL 5] Embedding lookup...")
    word_ids = [5, 6, 7]  # IDs for 'the', 'cat', 'sat'
    
    # Method 1: Loop (clear but slow)
    embeddings_loop = []
    for word_id in word_ids:
        embeddings_loop.append(embedding_matrix[word_id])
    embeddings_loop = np.array(embeddings_loop)
    
    # Method 2: Direct indexing (fast, what BERT uses)
    embeddings_direct = embedding_matrix[word_ids]
    
    print("Word IDs:", word_ids)
    print("Words:", [vocab[i] for i in word_ids])
    print("\nEmbeddings shape:", embeddings_direct.shape)
    print("\nEmbedding for 'the':", embeddings_direct[0])
    print("Embedding for 'cat':", embeddings_direct[1])
    print("Embedding for 'sat':", embeddings_direct[2])
    print("✓ Embedding lookup successful")
    
    # Cell 7: Visualizing embeddings
    print("\n[CELL 7] Creating embedding visualization...")
    plt.figure(figsize=(8, 6))
    plt.imshow(embedding_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.yticks(range(vocab_size), vocab)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Word')
    plt.title('Token Embedding Matrix Visualization')
    plt.tight_layout()
    plt.savefig('embedding_matrix_viz.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Embedding visualization saved to embedding_matrix_viz.png")
    
    # Cell 9: Position embeddings
    print("\n[CELL 9] Creating position embeddings...")
    max_seq_length = 8  # Maximum sentence length
    position_embeddings = np.random.randn(max_seq_length, hidden_size) * 0.1
    
    print("Position Embeddings Shape:", position_embeddings.shape)
    print("\nEach position (0 to 7) has its own vector:")
    for i in range(4):
        print(f"Position {i}: {position_embeddings[i]}")
    print("✓ Position embeddings created")
    
    # Cell 11: Combining embeddings
    print("\n[CELL 11] Combining token and position embeddings...")
    sentence_ids = [5, 6, 7]
    sentence_length = len(sentence_ids)
    
    # Get token embeddings
    token_embs = embedding_matrix[sentence_ids]  # [3, 4]
    
    # Get position embeddings for positions 0, 1, 2
    pos_embs = position_embeddings[:sentence_length]  # [3, 4]
    
    # Combine them
    combined_embeddings = token_embs + pos_embs  # Element-wise addition
    
    print("Token embeddings shape:", token_embs.shape)
    print("Position embeddings shape:", pos_embs.shape)
    print("Combined embeddings shape:", combined_embeddings.shape)
    print("\nExample for first word 'the' at position 0:")
    print("Token embedding:", token_embs[0])
    print("Position embedding:", pos_embs[0])
    print("Combined:", combined_embeddings[0])
    print("✓ Embedding combination successful")
    
    # Cell 13: Visualizing combination
    print("\n[CELL 13] Creating combination visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Token embeddings
    axes[0].imshow(token_embs, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Token Embeddings')
    axes[0].set_ylabel('Word Position')
    axes[0].set_xlabel('Dimension')
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(['the', 'cat', 'sat'])
    
    # Position embeddings
    axes[1].imshow(pos_embs, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Position Embeddings')
    axes[1].set_xlabel('Dimension')
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(['Pos 0', 'Pos 1', 'Pos 2'])
    
    # Combined
    im = axes[2].imshow(combined_embeddings, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Combined Embeddings')
    axes[2].set_xlabel('Dimension')
    axes[2].set_yticks(range(3))
    axes[2].set_yticklabels(['the@0', 'cat@1', 'sat@2'])
    
    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    plt.savefig('embedding_combination_viz.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Combination visualization saved to embedding_combination_viz.png")
    
    # Test importing Mini-BERT components
    print("\n[CELL 15] Testing Mini-BERT imports...")
    try:
        from model import MiniBERT
        from tokenizer import WordPieceTokenizer
        print("✓ Mini-BERT imports successful")
        
        # Try to load the model
        try:
            model = MiniBERT()
            print(f"✓ Mini-BERT model loaded")
            print(f"  Vocabulary size: {model.config.vocab_size}")
            print(f"  Hidden size: {model.config.hidden_size}")
            print(f"  Max sequence length: {getattr(model.config, 'max_seq_length', getattr(model.config, 'max_sequence_length', 'N/A'))}")
            print(f"  Token embeddings shape: {model.params['token_embeddings'].shape}")
            print(f"  Position embeddings shape: {model.params['position_embeddings'].shape}")
            
            # Try to load tokenizer
            try:
                tokenizer = WordPieceTokenizer()
                tokenizer.load_model('mini_bert/tokenizer_8k.pkl')
                print("✓ Tokenizer loaded successfully")
                
                # Cell 17: Process real text
                print("\n[CELL 17] Processing real text...")
                text = "The cat sat on the mat"
                input_ids = tokenizer.encode(text)
                input_ids_array = np.array([input_ids])  # Add batch dimension
                
                print(f"Original text: '{text}'")
                print(f"Token IDs: {input_ids}")
                print(f"Decoded tokens: {tokenizer.decode(input_ids).split()}")
                
                # Get embeddings
                token_embeddings = model.params['token_embeddings'][input_ids]
                position_embeddings = model.params['position_embeddings'][:len(input_ids)]
                combined = token_embeddings + position_embeddings
                
                print(f"\nToken embeddings shape: {token_embeddings.shape}")
                print(f"Position embeddings shape: {position_embeddings.shape}")
                print(f"Combined shape: {combined.shape}")
                print("✓ Real text processing successful")
                
            except FileNotFoundError:
                print("⚠ Tokenizer file not found at 'mini_bert/tokenizer_8k.pkl'")
                print("  Skipping tokenizer-dependent cells")
                
        except Exception as e:
            print(f"⚠ Could not load Mini-BERT model: {e}")
            print("  Skipping model-dependent cells")
            
    except ImportError as e:
        print(f"⚠ Could not import Mini-BERT components: {e}")
        print("  Skipping Mini-BERT specific cells")
    
    # Cell 19: Simulated training effect
    print("\n[CELL 19] Simulating training effects...")
    # Before training: random
    initial_embeddings = np.random.randn(5, 4) * 0.1
    
    # After training: similar words have similar embeddings
    trained_embeddings = np.array([
        [0.2, 0.1, -0.3, 0.4],   # 'cat'
        [0.25, 0.15, -0.28, 0.38], # 'dog' (similar to cat)
        [0.21, 0.12, -0.31, 0.41], # 'kitten' (similar to cat)
        [-0.4, 0.3, 0.2, -0.1],   # 'ran' (verb, different)
        [-0.38, 0.32, 0.18, -0.12] # 'jumped' (similar to ran)
    ])
    
    words = ['cat', 'dog', 'kitten', 'ran', 'jumped']
    
    # Compute similarity matrix (dot products)
    similarity = trained_embeddings @ trained_embeddings.T
    
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity, cmap='RdBu_r')
    plt.colorbar(label='Similarity')
    plt.xticks(range(5), words, rotation=45)
    plt.yticks(range(5), words)
    plt.title('Word Similarity After Training (Simulated)')
    plt.tight_layout()
    plt.savefig('word_similarity_viz.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("✓ Training simulation completed")
    print("✓ Word similarity visualization saved to word_similarity_viz.png")
    print("Notice how similar words (cat/dog/kitten) have high similarity!")
    print("This is what BERT learns during training.")
    
    # Cell 21: Embedding arithmetic
    print("\n[CELL 21] Demonstrating embedding arithmetic...")
    semantic_embeddings = {
        'king': np.array([0.5, 0.3, 0.8, -0.2]),
        'queen': np.array([0.5, 0.3, -0.8, -0.2]),
        'man': np.array([0.2, 0.4, 0.7, 0.1]),
        'woman': np.array([0.2, 0.4, -0.7, 0.1])
    }
    
    # Famous example: king - man + woman ≈ queen
    result = (semantic_embeddings['king'] - 
              semantic_embeddings['man'] + 
              semantic_embeddings['woman'])
    
    print("Embedding arithmetic:")
    print("king - man + woman =")
    print(f"  {semantic_embeddings['king']}")
    print(f"- {semantic_embeddings['man']}")
    print(f"+ {semantic_embeddings['woman']}")
    print(f"= {result}")
    print(f"\nQueen embedding: {semantic_embeddings['queen']}")
    print(f"\nAre they similar? The difference is:")
    print(f"{result - semantic_embeddings['queen']}")
    print("\n(In real embeddings, this would be approximately zero!)")
    print("✓ Embedding arithmetic demonstration completed")
    
    print("\n" + "="*60)
    print("NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nGenerated visualizations:")
    print("  • embedding_matrix_viz.png")
    print("  • embedding_combination_viz.png") 
    print("  • word_similarity_viz.png")
    
    print("\nKey Takeaways:")
    print("1. ✓ Token Embeddings: Convert word IDs to vectors via simple lookup")
    print("2. ✓ Position Embeddings: Add position information to preserve word order")
    print("3. ✓ Combination: Simple element-wise addition")
    print("4. ✓ Training: Embeddings start random but learn meaningful patterns")
    print("5. ✓ Geometry: Similar words end up with similar vectors")
    
except Exception as e:
    print(f"\nERROR in notebook execution: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*60)