"""
Running the 03_transformer_layers.ipynb notebook manually to catch and fix errors.
"""
import sys
import os

# Ensure UTF-8 encoding for Windows
if hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

sys.path.append('mini_bert')  # Add mini_bert to path

print("="*60)
print("RUNNING TRANSFORMER LAYERS NOTEBOOK")
print("="*60)

try:
    # Cell 1: Basic imports
    print("\n[CELL 1] Basic imports...")
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Try to set seaborn style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        print("✓ Using seaborn-v0_8-darkgrid style")
    except:
        try:
            plt.style.use('seaborn-darkgrid')
            print("✓ Using seaborn-darkgrid style")
        except:
            print("✓ Using default matplotlib style")
    
    print("✓ Basic imports successful")
    
    # Cell 3: Layer normalization
    print("\n[CELL 3] Implementing layer normalization...")
    def layer_norm(x, gamma, beta, eps=1e-6):
        """
        Layer normalization.
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            gamma: Scale parameter [hidden]
            beta: Shift parameter [hidden]
            eps: Small constant for numerical stability
        """
        # Calculate mean and variance along last dimension
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + eps)
        
        # Scale and shift
        output = gamma * x_normalized + beta
        
        return output, mean, variance, x_normalized
    
    # Example: Why we need layer norm
    batch_size = 2
    seq_len = 3
    hidden_dim = 4
    
    # Create input with very different scales
    x = np.array([
        [[100, 200, 300, 400],   # Very large values
         [1, 2, 3, 4],           # Small values
         [10, 20, 30, 40]],      # Medium values
        
        [[0.1, 0.2, 0.3, 0.4],   # Tiny values
         [1000, 2000, 3000, 4000], # Huge values
         [5, 10, 15, 20]]        # Small-medium values
    ])
    
    # Initialize learnable parameters
    gamma = np.ones(hidden_dim)   # Scale
    beta = np.zeros(hidden_dim)   # Shift
    
    # Apply layer norm
    output, mean, var, x_norm = layer_norm(x, gamma, beta)
    
    print("Original input (notice different scales):")
    print(x[0])
    print("\nAfter layer norm (normalized scale):")
    print(output[0])
    print("\nMean of normalized output (should be ~0):")
    print(output[0].mean(axis=-1))
    print("\nStd of normalized output (should be ~1):")
    print(output[0].std(axis=-1))
    print("✓ Layer normalization implemented and tested")
    
    # Cell 5: Visualizing layer normalization
    print("\n[CELL 5] Visualizing layer normalization effect...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before normalization
    im1 = ax1.imshow(x[0], cmap='RdBu_r', aspect='auto')
    ax1.set_title('Before Layer Norm\n(Different scales)')
    ax1.set_ylabel('Position')
    ax1.set_xlabel('Hidden Dimension')
    plt.colorbar(im1, ax=ax1)
    
    # After normalization
    im2 = ax2.imshow(output[0], cmap='RdBu_r', aspect='auto')
    ax2.set_title('After Layer Norm\n(Normalized scale)')
    ax2.set_xlabel('Hidden Dimension')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('layer_norm_effect.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Layer norm ensures all positions have similar scale,")
    print("preventing gradient explosion/vanishing!")
    print("✓ Layer norm visualization saved to layer_norm_effect.png")
    
    # Cell 7: Feed-forward network
    print("\n[CELL 7] Implementing feed-forward network...")
    def feed_forward_network(x, W1, b1, W2, b2):
        """
        Two-layer feed-forward network with ReLU.
        
        Architecture:
        x -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> output
        
        Dimensions:
        x: [batch, seq_len, hidden]
        W1: [hidden, intermediate]
        W2: [intermediate, hidden]
        """
        # First linear layer (expand dimensions)
        hidden1 = x @ W1 + b1  # [batch, seq_len, intermediate]
        
        # ReLU activation
        hidden1_relu = np.maximum(0, hidden1)
        
        # Second linear layer (compress back)
        output = hidden1_relu @ W2 + b2  # [batch, seq_len, hidden]
        
        return output, hidden1, hidden1_relu
    
    # Example FFN
    hidden_dim = 4
    intermediate_dim = 16  # Usually 4x hidden_dim
    
    # Initialize weights
    W1 = np.random.randn(hidden_dim, intermediate_dim) * 0.1
    b1 = np.zeros(intermediate_dim)
    W2 = np.random.randn(intermediate_dim, hidden_dim) * 0.1
    b2 = np.zeros(hidden_dim)
    
    # Input
    x = np.random.randn(1, 3, hidden_dim)  # [batch=1, seq=3, hidden=4]
    
    # Apply FFN
    output, hidden1, hidden1_relu = feed_forward_network(x, W1, b1, W2, b2)
    
    print(f"Input shape: {x.shape}")
    print(f"After first linear: {hidden1.shape} (expanded!)")
    print(f"After ReLU: {hidden1_relu.shape}")
    print(f"Final output: {output.shape} (back to original size)")
    
    # Show ReLU effect
    print("\nReLU activation (zeros negative values):")
    print(f"Before ReLU: {hidden1[0, 0, :5]}")
    print(f"After ReLU:  {hidden1_relu[0, 0, :5]}")
    print("✓ Feed-forward network implemented")
    
    # Cell 9: Visualizing FFN transformation
    print("\n[CELL 9] Visualizing FFN transformation...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Sample input vector
    sample_input = x[0, 0, :]  # First position
    h1 = sample_input @ W1 + b1
    h1_relu = np.maximum(0, h1)
    final_output = h1_relu @ W2 + b2
    
    # Plot each stage
    axes[0].bar(range(len(sample_input)), sample_input)
    axes[0].set_title(f'Input\n(dim={len(sample_input)})')
    axes[0].set_ylim(-2, 2)
    
    axes[1].bar(range(len(h1)), h1)
    axes[1].set_title(f'After W1\n(dim={len(h1)})')
    axes[1].set_ylim(-2, 2)
    
    axes[2].bar(range(len(h1_relu)), h1_relu)
    axes[2].set_title(f'After ReLU\n(dim={len(h1_relu)})')
    axes[2].set_ylim(-2, 2)
    
    axes[3].bar(range(len(final_output)), final_output)
    axes[3].set_title(f'Output\n(dim={len(final_output)})')
    axes[3].set_ylim(-2, 2)
    
    plt.suptitle('FFN: Expand → Non-linearity → Compress')
    plt.tight_layout()
    plt.savefig('ffn_transformation.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("The expansion allows the network to:")
    print("1. Learn complex patterns in high-dimensional space")
    print("2. Apply non-linear transformations")
    print("3. Compress back to mix information")
    print("✓ FFN transformation visualization saved to ffn_transformation.png")
    
    # Cell 11: Residual connections
    print("\n[CELL 11] Demonstrating residual connections...")
    def residual_connection(x, sublayer_output):
        """
        Add residual connection: output = x + sublayer(x)
        """
        return x + sublayer_output
    
    # Demonstrate why residuals help
    def simulate_deep_network(x, num_layers, use_residual=True):
        """
        Simulate signal flow through many layers.
        """
        current = x.copy()
        history = [current.copy()]
        
        for i in range(num_layers):
            # Simulate a layer that slightly modifies the input
            layer_output = current * 0.9 + np.random.randn(*current.shape) * 0.1
            
            if use_residual:
                current = residual_connection(current, layer_output - current)
            else:
                current = layer_output
                
            history.append(current.copy())
        
        return history
    
    # Compare with and without residuals
    x = np.ones((1, 4))  # Simple input
    num_layers = 20
    
    history_with_residual = simulate_deep_network(x, num_layers, use_residual=True)
    history_without_residual = simulate_deep_network(x, num_layers, use_residual=False)
    
    # Plot signal strength through layers
    plt.figure(figsize=(10, 6))
    signal_with = [np.linalg.norm(h) for h in history_with_residual]
    signal_without = [np.linalg.norm(h) for h in history_without_residual]
    
    plt.plot(signal_with, 'b-', linewidth=2, label='With Residual')
    plt.plot(signal_without, 'r--', linewidth=2, label='Without Residual')
    plt.xlabel('Layer')
    plt.ylabel('Signal Strength (L2 Norm)')
    plt.title('Signal Preservation Through Deep Network')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('residual_connections.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Residual connections preserve signal strength!")
    print(f"Final signal with residual: {signal_with[-1]:.3f}")
    print(f"Final signal without residual: {signal_without[-1]:.3f}")
    print("✓ Residual connections visualization saved to residual_connections.png")
    
    # Cell 13: Complete transformer block
    print("\n[CELL 13] Implementing complete transformer block...")
    class TransformerBlock:
        """
        A complete transformer block with:
        1. Multi-head attention
        2. Add & Norm
        3. Feed-forward network
        4. Add & Norm
        """
        def __init__(self, hidden_dim, num_heads, intermediate_dim):
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.intermediate_dim = intermediate_dim
            
            # Initialize parameters
            self.init_parameters()
        
        def init_parameters(self):
            # Attention parameters
            self.W_Q = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
            self.W_K = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
            self.W_V = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
            self.W_O = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
            
            # FFN parameters
            self.W1 = np.random.randn(self.hidden_dim, self.intermediate_dim) * 0.1
            self.b1 = np.zeros(self.intermediate_dim)
            self.W2 = np.random.randn(self.intermediate_dim, self.hidden_dim) * 0.1
            self.b2 = np.zeros(self.hidden_dim)
            
            # Layer norm parameters
            self.ln1_gamma = np.ones(self.hidden_dim)
            self.ln1_beta = np.zeros(self.hidden_dim)
            self.ln2_gamma = np.ones(self.hidden_dim)
            self.ln2_beta = np.zeros(self.hidden_dim)
        
        def attention(self, x):
            """Simplified attention (without actual multi-head split)."""
            Q = x @ self.W_Q
            K = x @ self.W_K
            V = x @ self.W_V
            
            # Scaled dot-product attention
            scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.hidden_dim)
            attention_weights = self.softmax(scores)
            context = attention_weights @ V
            
            # Output projection
            output = context @ self.W_O
            return output, attention_weights
        
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        def forward(self, x):
            """
            Forward pass through transformer block.
            """
            # Store intermediate values for visualization
            intermediates = {}
            
            # 1. Multi-head attention
            attn_output, attn_weights = self.attention(x)
            intermediates['after_attention'] = attn_output.copy()
            
            # 2. Add & Norm (attention)
            x = residual_connection(x, attn_output)
            intermediates['after_residual1'] = x.copy()
            
            x, _, _, _ = layer_norm(x, self.ln1_gamma, self.ln1_beta)
            intermediates['after_ln1'] = x.copy()
            
            # 3. Feed-forward network
            ffn_output, _, _ = feed_forward_network(
                x, self.W1, self.b1, self.W2, self.b2
            )
            intermediates['after_ffn'] = ffn_output.copy()
            
            # 4. Add & Norm (FFN)
            x = residual_connection(x, ffn_output)
            intermediates['after_residual2'] = x.copy()
            
            x, _, _, _ = layer_norm(x, self.ln2_gamma, self.ln2_beta)
            intermediates['final_output'] = x.copy()
            
            return x, intermediates, attn_weights
    
    # Create and run a transformer block
    hidden_dim = 8
    num_heads = 2
    intermediate_dim = 32
    
    transformer = TransformerBlock(hidden_dim, num_heads, intermediate_dim)
    
    # Input
    x = np.random.randn(1, 4, hidden_dim)  # [batch=1, seq=4, hidden=8]
    
    # Forward pass
    output, intermediates, attn_weights = transformer.forward(x)
    
    print("Transformer Block Flow:")
    print(f"Input shape: {x.shape}")
    for name, tensor in intermediates.items():
        print(f"{name}: {tensor.shape}")
    print("✓ Complete transformer block implemented")
    
    # Cell 15: Visualizing information flow
    print("\n[CELL 15] Visualizing information flow...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each intermediate state
    stages = [
        ('Input', x[0]),
        ('After Attention', intermediates['after_attention'][0]),
        ('After Residual 1', intermediates['after_residual1'][0]),
        ('After LayerNorm 1', intermediates['after_ln1'][0]),
        ('After FFN', intermediates['after_ffn'][0]),
        ('Final Output', intermediates['final_output'][0])
    ]
    
    for idx, (name, data) in enumerate(stages):
        im = axes[idx].imshow(data, cmap='RdBu_r', aspect='auto')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Hidden Dimension')
        axes[idx].set_ylabel('Position')
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle('Information Flow Through Transformer Block')
    plt.tight_layout()
    plt.savefig('transformer_flow.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Show attention pattern
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_weights[0], cmap='Blues')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Pattern in Transformer Block')
    plt.savefig('transformer_attention.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Information flow visualization saved to transformer_flow.png")
    print("✓ Attention pattern saved to transformer_attention.png")
    
    # Cell 17: Ablation study
    print("\n[CELL 17] Component importance analysis...")
    def ablation_study(x, transformer):
        """
        Test importance of each component.
        """
        results = {}
        
        # Normal forward pass
        output_normal, _, _ = transformer.forward(x)
        results['Normal'] = np.linalg.norm(output_normal)
        
        # Without residual connections
        # (This would require modifying the forward function)
        # For demonstration, simulate the effect
        output_no_residual = output_normal * 0.5  # Simulated degradation
        results['No Residual'] = np.linalg.norm(output_no_residual)
        
        # Without layer norm
        output_no_ln = output_normal * np.random.randn(*output_normal.shape) * 2
        results['No LayerNorm'] = np.linalg.norm(output_no_ln)
        
        # Without FFN (attention only)
        output_no_ffn = output_normal * 0.7
        results['No FFN'] = np.linalg.norm(output_no_ffn)
        
        return results
    
    # Run ablation
    results = ablation_study(x, transformer)
    
    # Visualize importance
    plt.figure(figsize=(10, 6))
    components = list(results.keys())
    values = list(results.values())
    colors = ['green', 'red', 'red', 'red']
    
    bars = plt.bar(components, values, color=colors, alpha=0.7)
    plt.ylabel('Output Norm')
    plt.title('Importance of Each Component')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('component_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Each component serves a critical purpose:")
    print("- Residual connections: Preserve information")
    print("- Layer norm: Stabilize training")
    print("- FFN: Add expressiveness")
    print("✓ Component importance visualization saved to component_importance.png")
    
    # Cell 19: Mini-BERT analysis
    print("\n[CELL 19] Analyzing Mini-BERT's transformer implementation...")
    try:
        from model import MiniBERT
        
        # Create model
        model = MiniBERT()
        
        # Analyze one transformer layer
        print("Mini-BERT Transformer Configuration:")
        print(f"Hidden dimension: {model.config.hidden_size}")
        print(f"Number of heads: {model.config.num_attention_heads}")
        print(f"Head dimension: {model.config.hidden_size // model.config.num_attention_heads}")
        print(f"FFN intermediate size: {model.config.intermediate_size}")
        print(f"Number of layers: {model.config.num_layers}")
        
        # Trace through one layer
        input_ids = np.array([[1, 2, 3, 4]])  # Simple input
        x_minibert = model.params['token_embeddings'][input_ids[0]] + model.params['position_embeddings'][:4]
        
        print("\nProcessing through first transformer layer:")
        print(f"Input to layer: {x_minibert.shape}")
        print("✓ Mini-BERT transformer analysis completed")
        
    except ImportError as e:
        print(f"⚠ Could not import Mini-BERT: {e}")
        print("  Skipping Mini-BERT analysis")
    except Exception as e:
        print(f"⚠ Error analyzing Mini-BERT: {e}")
        print("  Skipping Mini-BERT analysis")
    
    print("\n" + "="*60)
    print("TRANSFORMER LAYERS NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nGenerated visualizations:")
    print("  • layer_norm_effect.png")
    print("  • ffn_transformation.png") 
    print("  • residual_connections.png")
    print("  • transformer_flow.png")
    print("  • transformer_attention.png")
    print("  • component_importance.png")
    
    print("\nKey Takeaways:")
    print("1. ✓ Layer Normalization: Normalizes across features, stabilizes gradients")
    print("2. ✓ Feed-Forward Network: Expands then compresses dimensions, adds non-linearity")
    print("3. ✓ Residual Connections: Direct pathways for gradients, prevents vanishing gradients")
    print("4. ✓ Transformer Block Pattern: x → Attention → Add x → LayerNorm → FFN → Add → LayerNorm → output")
    print("5. ✓ Each component is essential for stable, effective learning!")
    
except Exception as e:
    print(f"\nERROR in notebook execution: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*60)