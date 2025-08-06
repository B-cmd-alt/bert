"""
Complete Training Loop for Mini-BERT with MLM.

Features:
- Stream processing of plain-text corpus
- WordPiece tokenization with 8K vocabulary
- MLM masking (15% tokens: 80% [MASK], 10% random, 10% unchanged)
- Gradient accumulation (logical_batch=32, micro_batch=8)
- AdamW optimizer with linear warmup and decay
- Memory efficient (target ≤2GB peak RAM)
- Performance target: ≤70ms per micro-batch on i7 CPU

Memory Budget Analysis:
- Model parameters: ~18MB (4.5M × 4 bytes)
- Optimizer state: ~72MB (params + gradients + Adam moments)
- Activations per micro-batch: ~8MB (batch=8, seq=128, hidden=192)
- Total estimated: ~100MB (well under 2GB target)
"""

import numpy as np
import os
import time
import argparse
from typing import Iterator, Dict, List, Tuple, Optional
import tqdm

# Import our modules
from model import MiniBERT
from gradients import MiniBERTGradients
from tokenizer import WordPieceTokenizer
from mlm import mask_tokens, mlm_cross_entropy
from optimizer import AdamW, LRScheduler
from config import MODEL_CONFIG

class TextDataLoader:
    """
    Memory-efficient streaming dataloader for large text corpora.
    
    Processes text line-by-line, tokenizes with WordPiece, and assembles batches.
    """
    
    def __init__(self, 
                 data_paths: List[str],
                 tokenizer: WordPieceTokenizer,
                 micro_batch_size: int = 8,
                 seq_len: int = 128,
                 pad_token_id: int = 0):
        """
        Initialize dataloader.
        
        Args:
            data_paths: List of paths to text files
            tokenizer: WordPiece tokenizer
            micro_batch_size: Number of sequences per micro-batch
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
        """
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        
        # Validate data files exist
        for path in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        
        print(f"DataLoader initialized:")
        print(f"  Files: {len(data_paths)} files")
        print(f"  Micro-batch size: {micro_batch_size}")
        print(f"  Sequence length: {seq_len}")
    
    def _read_lines(self) -> Iterator[str]:
        """Generator that yields lines from all data files."""
        for data_path in self.data_paths:
            print(f"Reading from {os.path.basename(data_path)}...")
            
            try:
                with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if len(line) > 20:  # Skip very short lines
                            yield line
                        
                        if line_num % 50000 == 0 and line_num > 0:
                            print(f"  Processed {line_num:,} lines...")
                            
            except Exception as e:
                print(f"Error reading {data_path}: {e}")
                continue
    
    def _tokenize_and_batch(self, lines: Iterator[str]) -> Iterator[np.ndarray]:
        """
        Tokenize lines and assemble into batches.
        
        Yields:
            Batched token IDs [micro_batch_size, seq_len]
        """
        batch_sequences = []
        
        for line in lines:
            # Tokenize line
            token_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.seq_len)
            
            # Pad to seq_len
            if len(token_ids) < self.seq_len:
                token_ids.extend([self.pad_token_id] * (self.seq_len - len(token_ids)))
            
            batch_sequences.append(token_ids)
            
            # Yield batch when full
            if len(batch_sequences) == self.micro_batch_size:
                yield np.array(batch_sequences, dtype=np.int32)
                batch_sequences = []
        
        # Yield final partial batch if any
        if batch_sequences:
            # Pad to full batch size
            while len(batch_sequences) < self.micro_batch_size:
                padding_seq = [self.pad_token_id] * self.seq_len
                batch_sequences.append(padding_seq)
            
            yield np.array(batch_sequences, dtype=np.int32)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over batches of tokenized sequences.""" 
        lines = self._read_lines()
        yield from self._tokenize_and_batch(lines)


def train_step(model: MiniBERT, 
               grad_computer: MiniBERTGradients,
               batch: np.ndarray,
               vocab_size: int,
               mask_token_id: int) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Perform one training step (forward + backward).
    
    Args:
        model: Mini-BERT model
        grad_computer: Gradients computer
        batch: Input token IDs [micro_batch_size, seq_len]
        vocab_size: Vocabulary size
        mask_token_id: [MASK] token ID
        
    Returns:
        loss: MLM loss
        accuracy: MLM accuracy
        gradients: Parameter gradients
    """
    # Apply MLM masking
    input_ids, target_ids, mask_positions = mask_tokens(
        batch, vocab_size, mask_token_id, p_mask=0.15
    )
    
    # Forward pass
    start_time = time.time()
    logits, cache = model.forward(input_ids)
    forward_time = time.time() - start_time
    
    # Compute MLM loss
    loss, accuracy = mlm_cross_entropy(logits, target_ids, ignore_index=-100)
    
    # Backward pass
    start_time = time.time()
    grad_computer.zero_gradients()
    
    # Compute gradients w.r.t. logits
    # For cross-entropy: ∂L/∂logits = (softmax_probs - target_onehot) / num_valid
    batch_size, seq_len, vocab_size = logits.shape
    
    # Compute softmax probabilities
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Create target one-hot encoding for valid positions only
    valid_mask = (target_ids != -100)
    num_valid = np.sum(valid_mask)
    
    grad_logits = softmax_probs.copy()  # Start with softmax gradient
    
    if num_valid > 0:
        # Subtract 1 at target positions (cross-entropy gradient)
        valid_positions = np.where(valid_mask)
        valid_targets = target_ids[valid_mask]
        grad_logits[valid_positions[0], valid_positions[1], valid_targets] -= 1.0
        
        # Scale by number of valid positions
        grad_logits /= num_valid
    else:
        grad_logits.fill(0.0)
    
    # Backward pass through model
    grad_computer.backward_from_logits(grad_logits, cache)
    backward_time = time.time() - start_time
    
    # Print timing info occasionally
    total_time = forward_time + backward_time
    if np.random.random() < 0.01:  # 1% of steps
        print(f"    Timing: forward={forward_time*1000:.1f}ms, backward={backward_time*1000:.1f}ms, total={total_time*1000:.1f}ms")
    
    return loss, accuracy, grad_computer.gradients


def train_mini_bert(data_paths: List[str],
                   vocab_path: str,
                   total_steps: int = 100000,
                   micro_batch_size: int = 8,
                   logical_batch_size: int = 32,
                   seq_len: int = 128,
                   learning_rate: float = 5e-4,
                   warmup_steps: int = 10000,
                   save_every: int = 10000,
                   log_every: int = 100,
                   overfit_one_batch: bool = False) -> MiniBERT:
    """
    Complete Mini-BERT training pipeline.
    
    Args:
        data_paths: Paths to training data files
        vocab_path: Path to save/load tokenizer vocabulary
        total_steps: Total training steps (100K default)
        micro_batch_size: Physical batch size (8 default, fits in memory)
        logical_batch_size: Effective batch size via gradient accumulation (32 default)
        seq_len: Maximum sequence length (128 default)
        learning_rate: Peak learning rate (5e-4 default)
        warmup_steps: Linear warmup steps (10K default = 10% of total)
        save_every: Checkpoint saving frequency
        log_every: Logging frequency
        overfit_one_batch: Debug flag to overfit single batch
        
    Returns:
        Trained model
    """
    print("=" * 60)
    print("Mini-BERT Training Pipeline")
    print("=" * 60)
    
    # Calculate gradient accumulation steps
    accumulation_steps = logical_batch_size // micro_batch_size
    assert logical_batch_size % micro_batch_size == 0, f"logical_batch_size {logical_batch_size} must be divisible by micro_batch_size {micro_batch_size}"
    
    print(f"Training Configuration:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Logical batch size: {logical_batch_size}")
    print(f"  Micro batch size: {micro_batch_size}")
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps:,} ({100*warmup_steps/total_steps:.1f}%)")
    print(f"  Overfit one batch: {overfit_one_batch}")
    
    # Initialize model
    model = MiniBERT(MODEL_CONFIG)
    grad_computer = MiniBERTGradients(model)
    
    # Load or train tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=MODEL_CONFIG.vocab_size)
    
    if os.path.exists(vocab_path):
        print(f"Loading tokenizer from {vocab_path}")
        tokenizer.load_model(vocab_path)
    else:
        print("Training new tokenizer...")
        # Train on first data file with limited lines
        tokenizer.train(data_paths[0], max_lines=100000)
        tokenizer.save_model(vocab_path)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999, 
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=learning_rate
    )
    
    # Initialize dataloader
    dataloader = TextDataLoader(
        data_paths=data_paths,
        tokenizer=tokenizer,
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        pad_token_id=tokenizer.vocab.get("[PAD]", 0)
    )
    
    # Training setup
    mask_token_id = tokenizer.vocab.get("[MASK]", 4)
    vocab_size = tokenizer.get_vocab_size()
    
    # Training state
    step = 0
    accumulated_gradients = {}
    accumulation_count = 0
    running_loss = 0.0
    running_accuracy = 0.0
    
    # For overfitting debug
    debug_batch = None
    
    print(f"\nStarting training...")
    training_start_time = time.time()
    
    # Training loop
    try:
        # Progress bar
        pbar = tqdm.tqdm(total=total_steps, desc="Training", unit="step")
        
        # Infinite data iterator (restart when exhausted)
        data_iter = iter(dataloader)
        
        while step < total_steps:
            try:
                # Get next batch
                if overfit_one_batch:
                    if debug_batch is None:
                        debug_batch = next(data_iter)
                        print(f"Debug: Using fixed batch with shape {debug_batch.shape}")
                    batch = debug_batch
                else:
                    batch = next(data_iter)
                    
            except StopIteration:
                # Restart data iterator
                print("Restarting data iterator...")
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Training step
            step_start_time = time.time()
            
            loss, accuracy, gradients = train_step(
                model, grad_computer, batch, vocab_size, mask_token_id
            )
            
            # Gradient accumulation
            if not accumulated_gradients:
                # Initialize accumulated gradients
                accumulated_gradients = {name: grad.copy() for name, grad in gradients.items()}
            else:
                # Add to accumulated gradients
                for name, grad in gradients.items():
                    accumulated_gradients[name] += grad
            
            accumulation_count += 1
            running_loss += loss
            running_accuracy += accuracy
            
            # Update parameters when accumulation is complete
            if accumulation_count >= accumulation_steps:
                # Average accumulated gradients
                for name in accumulated_gradients:
                    accumulated_gradients[name] /= accumulation_steps
                
                # Update learning rate
                scheduler.step(step)
                current_lr = scheduler.get_lr(step)
                
                # Optimizer step
                opt_stats = optimizer.step(model.params, accumulated_gradients)
                
                # Compute averages
                avg_loss = running_loss / accumulation_steps
                avg_accuracy = running_accuracy / accumulation_steps
                
                step_time = time.time() - step_start_time
                
                # Logging
                if step % log_every == 0:
                    elapsed = time.time() - training_start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    
                    print(f"Step {step:6d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Acc: {avg_accuracy:.3f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Grad: {opt_stats['grad_norm']:.2e} | "
                          f"Time: {step_time*1000:.0f}ms | "
                          f"Speed: {steps_per_sec:.1f} steps/s")
                    
                    # Check if overfitting is working
                    if overfit_one_batch and step > 100:
                        if avg_loss > 0.1:
                            print(f"    Warning: Loss not decreasing fast enough for overfitting (current: {avg_loss:.4f})")
                        elif avg_loss < 0.01:
                            print(f"    Success: Overfitting working (loss: {avg_loss:.4f})")
                
                # Save checkpoint
                if step % save_every == 0 and step > 0:
                    checkpoint_path = f"checkpoint_step_{step}.pkl"
                    save_checkpoint(model, tokenizer, optimizer, scheduler, step, avg_loss, checkpoint_path)
                
                # Reset accumulation
                accumulated_gradients = {}
                accumulation_count = 0
                running_loss = 0.0
                running_accuracy = 0.0
                step += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_accuracy:.3f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        pbar.close()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint
        final_checkpoint_path = "final_checkpoint.pkl"
        save_checkpoint(model, tokenizer, optimizer, scheduler, step, 
                       running_loss / max(1, accumulation_count), final_checkpoint_path)
        
        # Training summary
        total_time = time.time() - training_start_time
        print(f"\nTraining Summary:")
        print(f"  Steps completed: {step:,} / {total_steps:,}")
        print(f"  Total time: {total_time/3600:.1f} hours")
        print(f"  Average speed: {step/total_time:.1f} steps/second")
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024**2)
            print(f"  Peak memory: {memory_mb:.0f} MB")
        except ImportError:
            print("  Memory info unavailable (install psutil)")
    
    return model


def save_checkpoint(model: MiniBERT, tokenizer: WordPieceTokenizer, 
                   optimizer: AdamW, scheduler: LRScheduler,
                   step: int, loss: float, filepath: str):
    """Save training checkpoint."""
    import pickle
    
    checkpoint = {
        'model_params': model.params,
        'tokenizer_vocab': tokenizer.vocab,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': {
            'warmup_steps': scheduler.warmup_steps,
            'total_steps': scheduler.total_steps,
            'base_lr': scheduler.base_lr
        },
        'step': step,
        'loss': loss,
        'model_config': MODEL_CONFIG.__dict__
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {filepath} (step {step}, loss {loss:.4f})")


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description="Train Mini-BERT with MLM")
    
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Paths to training data files')
    parser.add_argument('--vocab', type=str, default='tokenizer_vocab.pkl',
                       help='Path to tokenizer vocabulary')
    parser.add_argument('--steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--micro_batch', type=int, default=8,
                       help='Micro batch size')
    parser.add_argument('--logical_batch', type=int, default=32,
                       help='Logical batch size (via gradient accumulation)')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Peak learning rate')
    parser.add_argument('--warmup', type=int, default=10000,
                       help='Warmup steps')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--overfit_one_batch', action='store_true',
                       help='Debug flag: overfit one batch (loss should reach ~0.0)')
    
    args = parser.parse_args()
    
    # Validate data files
    for data_path in args.data:
        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {data_path}")
            return
    
    # Memory budget check
    print("Memory Budget Analysis:")
    print("  Model parameters: ~18MB")
    print("  Optimizer state: ~72MB") 
    print("  Activations (micro-batch): ~8MB")
    print("  Estimated total: ~100MB (target: <2GB)")
    print()
    
    # Start training
    model = train_mini_bert(
        data_paths=args.data,
        vocab_path=args.vocab,
        total_steps=args.steps,
        micro_batch_size=args.micro_batch,
        logical_batch_size=args.logical_batch,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        save_every=args.save_every,
        log_every=args.log_every,
        overfit_one_batch=args.overfit_one_batch
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()