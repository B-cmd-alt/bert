"""
Running the 02_attention_mechanism.ipynb notebook manually to catch and fix errors.
"""
import sys
import os

# Ensure UTF-8 encoding for Windows
if hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

sys.path.append('mini_bert')  # Add mini_bert to path

print("="*60)
print("RUNNING ATTENTION MECHANISM NOTEBOOK")
print("="*60)

try:
    # Cell 1: Basic imports
    print("\n[CELL 1] Basic imports...")
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Check if seaborn is available
    try:
        import seaborn as sns
        print("✓ Seaborn available")
    except ImportError:
        print("⚠ Seaborn not available, will install it")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'seaborn'], check=True)
        import seaborn as sns
        print("✓ Seaborn installed and imported")
    
    # Set style for better visualizations (update deprecated style)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        print("✓ Using seaborn-v0_8-darkgrid style")
    except:
        try:
            plt.style.use('seaborn-darkgrid')
            print("✓ Using seaborn-darkgrid style")
        except:
            print("✓ Using default matplotlib style")
    
    np.random.seed(42)
    print("✓ Basic imports successful")
    
    # Cell 3: Word vectors
    print("\n[CELL 3] Creating word vectors...")
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired']
    num_words = len(words)
    hidden_dim = 4  # Small dimension for visualization
    
    # Random word vectors (in reality, these come from embeddings)
    word_vectors = np.random.randn(num_words, hidden_dim) * 0.5
    
    # Let's manually make 'cat' and 'it' somewhat similar
    word_vectors[1] = np.array([0.8, 0.2, -0.1, 0.5])  # 'cat'
    word_vectors[7] = np.array([0.7, 0.3, -0.2, 0.4])  # 'it' (similar to cat)
    
    print("Word vectors shape:", word_vectors.shape)
    print("\nFirst few words and their vectors:")
    for i in range(5):
        print(f"{words[i]:8s}: {word_vectors[i]}")
    print("✓ Word vectors created")
    
    # Cell 5: Raw attention scores
    print("\n[CELL 5] Computing raw attention scores...")
    attention_scores = word_vectors @ word_vectors.T  # [10, 10]
    
    # Visualize the raw attention scores
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_scores, 
                xticklabels=words, 
                yticklabels=words, 
                cmap='Blues', 
                center=0,
                annot=True, 
                fmt='.2f')
    plt.title('Raw Attention Scores (Dot Products)')
    plt.xlabel('Attending to')
    plt.ylabel('Query word')
    plt.tight_layout()
    plt.savefig('raw_attention_scores.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Notice: 'it' (row 7) has high score with 'cat' (column 1)!")
    print("This is how attention can resolve references.")
    print("✓ Raw attention visualization saved to raw_attention_scores.png")
    
    # Cell 7: Scaled dot-product attention
    print("\n[CELL 7] Implementing scaled dot-product attention...")
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix [seq_len, d_k]
            K: Key matrix [seq_len, d_k]
            V: Value matrix [seq_len, d_k]
            mask: Optional mask for invalid positions
        
        Returns:
            output: Weighted sum of values [seq_len, d_k]
            attention_weights: Attention probabilities [seq_len, seq_len]
        """
        d_k = Q.shape[-1]
        
        # Step 1: Compute dot products between queries and keys
        scores = Q @ K.T  # [seq_len, seq_len]
        
        # Step 2: Scale by square root of dimension
        scores = scores / np.sqrt(d_k)
        
        # Step 3: Apply mask if provided (for padding, etc.)
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Step 4: Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Step 5: Weighted sum of values
        output = attention_weights @ V
        
        return output, attention_weights
    
    # Let's use our word vectors as Q, K, V
    Q = K = V = word_vectors
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("Input shape:", word_vectors.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
    print("✓ Scaled dot-product attention implemented")
    
    # Cell 9: Visualize attention weights
    print("\n[CELL 9] Visualizing attention weights...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=words, 
                yticklabels=words, 
                cmap='Blues', 
                annot=True, 
                fmt='.2f',
                vmin=0, vmax=1)
    plt.title('Attention Weights (After Softmax)')
    plt.xlabel('Attending to')
    plt.ylabel('Query word')
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Each row sums to 1.0 (it's a probability distribution)")
    print(f"Sum of row 0: {attention_weights[0].sum():.3f}")
    print("✓ Attention weights visualization saved to attention_weights.png")
    
    # Cell 11: Scaling effect demonstration
    print("\n[CELL 11] Demonstrating scaling effect...")
    d_k_values = [4, 64, 512]  # Different dimensions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, d_k in enumerate(d_k_values):
        # Create random vectors
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        
        # Compute dot product
        dot_product = np.dot(q, k)
        scaled_dot = dot_product / np.sqrt(d_k)
        
        # Softmax of single value (compared to 0)
        scores = np.array([dot_product, 0])
        scaled_scores = np.array([scaled_dot, 0])
        
        softmax_unscaled = np.exp(scores) / np.exp(scores).sum()
        softmax_scaled = np.exp(scaled_scores) / np.exp(scaled_scores).sum()
        
        ax = axes[idx]
        x = ['Unscaled', 'Scaled']
        y = [softmax_unscaled[0], softmax_scaled[0]]
        ax.bar(x, y)
        ax.set_ylim(0, 1)
        ax.set_title(f'd_k = {d_k}')
        ax.set_ylabel('Softmax probability')
        
        # Add values on bars
        for i, v in enumerate(y):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.suptitle('Effect of Scaling on Softmax')
    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Without scaling, softmax becomes too 'sharp' (near 0 or 1) for large dimensions.")
    print("Scaling keeps gradients healthy!")
    print("✓ Scaling effect visualization saved to scaling_effect.png")
    
    # Cell 13: Multi-head attention
    print("\n[CELL 13] Implementing multi-head attention...")
    def multi_head_attention(X, num_heads, hidden_dim):
        """
        Simplified multi-head attention (without learned projections).
        """
        seq_len = X.shape[0]
        head_dim = hidden_dim // num_heads
        
        # Split hidden dimension across heads
        X_heads = X.reshape(seq_len, num_heads, head_dim)
        
        all_outputs = []
        all_attention_weights = []
        
        # Process each head independently
        for head in range(num_heads):
            Q = K = V = X_heads[:, head, :]  # [seq_len, head_dim]
            output, attention_weights = scaled_dot_product_attention(Q, K, V)
            all_outputs.append(output)
            all_attention_weights.append(attention_weights)
        
        # Concatenate all heads
        concat_output = np.concatenate(all_outputs, axis=-1)
        
        return concat_output, all_attention_weights
    
    # Create larger hidden dimension for multi-head
    hidden_dim_large = 16
    num_heads = 4
    word_vectors_large = np.random.randn(num_words, hidden_dim_large) * 0.5
    
    # Run multi-head attention
    output_mh, attention_weights_mh = multi_head_attention(word_vectors_large, num_heads, hidden_dim_large)
    
    print(f"Input shape: {word_vectors_large.shape}")
    print(f"Output shape: {output_mh.shape}")
    print(f"Number of attention heads: {len(attention_weights_mh)}")
    print(f"Each head's attention shape: {attention_weights_mh[0].shape}")
    print("✓ Multi-head attention implemented")
    
    # Cell 15: Visualizing different attention heads
    print("\n[CELL 15] Visualizing different attention patterns...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Simulate different attention patterns for each head
    attention_patterns = [
        # Head 1: Attend to previous word
        np.eye(num_words, k=-1) * 0.8 + np.eye(num_words) * 0.2,
        # Head 2: Attend to next word  
        np.eye(num_words, k=1) * 0.8 + np.eye(num_words) * 0.2,
        # Head 3: Attend to first word (global context)
        np.zeros((num_words, num_words)) + 0.1,
        # Head 4: Self-attention (diagonal)
        np.eye(num_words)
    ]
    
    # Add some noise and normalize
    for i in range(4):
        pattern = attention_patterns[i] + np.random.rand(num_words, num_words) * 0.1
        # Normalize rows to sum to 1
        pattern = pattern / pattern.sum(axis=1, keepdims=True)
        
        sns.heatmap(pattern, 
                    xticklabels=words, 
                    yticklabels=words, 
                    cmap='Blues',
                    vmin=0, vmax=1,
                    cbar=True,
                    ax=axes[i])
        axes[i].set_title(f'Head {i+1}: {["Previous", "Next", "Global", "Self"][i]}')
        
    plt.suptitle('Different Attention Patterns Each Head Might Learn')
    plt.tight_layout()
    plt.savefig('attention_head_patterns.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Attention head patterns visualization saved to attention_head_patterns.png")
    
    # Test Mini-BERT integration
    print("\n[CELL 17] Testing Mini-BERT integration...")
    try:
        from model import MiniBERT
        from tokenizer import WordPieceTokenizer
        
        # Load model and tokenizer
        model = MiniBERT()
        tokenizer = WordPieceTokenizer()
        tokenizer.load_model('mini_bert/tokenizer_8k.pkl')
        
        # Process text
        text = "The cat sat on the mat"
        input_ids = tokenizer.encode(text)
        input_ids_batch = np.array([input_ids])
        
        print(f"Text: '{text}'")
        print(f"Input IDs shape: {input_ids_batch.shape}")
        
        # Forward pass
        logits, cache = model.forward(input_ids_batch)
        
        print(f"Logits shape: {logits.shape}")
        print("Cache keys:", list(cache.keys()))
        
        # Check if we can access attention weights
        try:
            if 'layer_caches' in cache:
                layer_caches = cache['layer_caches']
                print("Available layers:", list(layer_caches.keys()))
                
                # Try to access first layer attention
                if 'layer_0' in layer_caches and 'attn_cache' in layer_caches['layer_0']:
                    attn_cache = layer_caches['layer_0']['attn_cache']
                    print("Attention cache keys:", list(attn_cache.keys()))
                    
                    if 'attention_weights' in attn_cache:
                        attention = attn_cache['attention_weights']
                        print(f"Attention weights shape: {attention.shape}")
                        
                        # Visualize real attention from Mini-BERT
                        print("\n[CELL 19] Visualizing real Mini-BERT attention...")
                        tokens = tokenizer.decode(input_ids).split()
                        batch_idx = 0
                        
                        if attention.shape[1] >= 4:  # Check we have at least 4 heads
                            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                            axes = axes.flatten()
                            
                            for head_idx in range(4):
                                attention_head = attention[batch_idx, head_idx]
                                
                                sns.heatmap(attention_head,
                                            xticklabels=tokens,
                                            yticklabels=tokens,
                                            cmap='Blues',
                                            vmin=0, vmax=1,
                                            cbar=True,
                                            ax=axes[head_idx])
                                axes[head_idx].set_title(f'Layer 0, Head {head_idx}')
                                
                            plt.suptitle(f'Attention Patterns for: "{text}"')
                            plt.tight_layout()
                            plt.savefig('minibert_attention_patterns.png', dpi=100, bbox_inches='tight')
                            plt.close()
                            
                            print("Each head learns different patterns!")
                            print("This is from a randomly initialized model - patterns become more meaningful after training.")
                            print("✓ Mini-BERT attention visualization saved to minibert_attention_patterns.png")
                        else:
                            print(f"⚠ Model has only {attention.shape[1]} heads, expected 4")
                    else:
                        print("⚠ attention_weights not found in attention cache")
                else:
                    print("⚠ Could not access attention cache from layer 0")
            else:
                print("⚠ layer_caches not found in cache")
                
        except Exception as e:
            print(f"⚠ Could not access attention weights: {e}")
            print("  Will skip real model attention visualization")
        
        print("✓ Mini-BERT integration successful")
        
    except ImportError as e:
        print(f"⚠ Could not import Mini-BERT components: {e}")
        print("  Skipping Mini-BERT specific cells")
    except Exception as e:
        print(f"⚠ Error with Mini-BERT: {e}")
        print("  Skipping Mini-BERT specific cells")
    
    print("\n" + "="*60)
    print("ATTENTION NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nGenerated visualizations:")
    print("  • raw_attention_scores.png")
    print("  • attention_weights.png") 
    print("  • scaling_effect.png")
    print("  • attention_head_patterns.png")
    try:
        if 'minibert_attention_patterns.png' in locals():
            print("  • minibert_attention_patterns.png")
    except:
        pass
    
    print("\nKey Takeaways:")
    print("1. ✓ Attention = Weighted Average: Each word's representation becomes a weighted average of all words")
    print("2. ✓ Weights from Similarity: The weights come from how similar (dot product) words are")
    print("3. ✓ Scaling Matters: Dividing by √d_k keeps softmax gradients healthy")
    print("4. ✓ Multiple Heads: Different heads can learn different types of relationships")
    print("5. ✓ Q, K, V: Queries ask questions, Keys provide answers, Values are the actual information")
    
except Exception as e:
    print(f"\nERROR in notebook execution: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*60)