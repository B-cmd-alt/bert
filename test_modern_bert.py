"""
Simple test script for Modern BERT - Windows compatible.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

print("Starting Modern BERT Test...")
print("=" * 50)

# Test imports
try:
    import torch
    print("+ PyTorch imported successfully")
except ImportError:
    print("- PyTorch not found")
    sys.exit(1)

try:
    from transformers import pipeline, BertTokenizer
    print("+ Transformers imported successfully")
except ImportError:
    print("- Transformers not found")
    sys.exit(1)

try:
    import numpy as np
    print("+ NumPy imported successfully")
except ImportError:
    print("- NumPy not found")
    sys.exit(1)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"+ Using device: {device}")

# Test masked language modeling
print("\nTesting Masked Language Modeling...")
print("-" * 30)

try:
    # Create pipeline
    mlm_pipeline = pipeline(
        "fill-mask", 
        model="distilbert-base-uncased",
        device=0 if device == "cuda" else -1
    )
    print("+ Pipeline created successfully")
    
    # Test predictions
    test_sentences = [
        "The capital of France is [MASK].",
        "Python is a [MASK] programming language.",
        "I love to [MASK] books in my free time."
    ]
    
    print("\nPrediction Results:")
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        predictions = mlm_pipeline(sentence, top_k=3)
        
        for i, pred in enumerate(predictions, 1):
            token = pred['token_str'].strip()
            score = pred['score']
            print(f"  {i}. {token}: {score:.3f}")
    
    print("\n+ Masked language modeling test passed!")
    
except Exception as e:
    print(f"- Masked language modeling failed: {e}")

# Test sentiment analysis
print("\nTesting Sentiment Analysis...")
print("-" * 30)

try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    test_texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    
    print("Sentiment Results:")
    for text in test_texts:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        print(f"  '{text}' -> {label} ({score:.3f})")
    
    print("\n+ Sentiment analysis test passed!")
    
except Exception as e:
    print(f"- Sentiment analysis failed: {e}")

# Performance benchmark
print("\nRunning Performance Benchmark...")
print("-" * 30)

try:
    import time
    
    test_text = "The capital of [MASK] is a beautiful city."
    num_runs = 10
    
    print(f"Running {num_runs} inferences...")
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = mlm_pipeline(test_text)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.3f}s")
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    throughput = 1 / avg_time
    
    print(f"\nPerformance Results:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Min time: {min_time:.3f}s")
    print(f"  Max time: {max_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    print("\n+ Performance benchmark completed!")
    
except Exception as e:
    print(f"- Performance benchmark failed: {e}")

# Mini training demo
print("\nMini Training Demo...")
print("-" * 30)

try:
    from transformers import (
        BertForMaskedLM, Trainer, TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    
    # Create small dataset
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns.",
        "Transformers revolutionized natural language understanding."
    ] * 5  # 25 examples
    
    print(f"Created {len(training_texts)} training examples")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=64,  # Short sequences for demo
            return_special_tokens_mask=True
        )
    
    # Create dataset
    dataset = Dataset.from_dict({'text': training_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting mini training (this may take 1-2 minutes)...")
    
    # Train for a few steps
    trainer.train()
    
    # Get final loss
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
        print(f"Final training loss: {final_loss}")
    
    print("+ Mini training completed successfully!")
    
except Exception as e:
    print(f"- Mini training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Modern BERT Test Summary")
print("=" * 50)

print("All tests completed!")
print("\nNext steps:")
print("1. Check outputs in ./test_results/")
print("2. Run full training: python modern_bert/train.py")
print("3. Try fine-tuning: python modern_bert/fine_tune.py")

print("\nModel comparison:")
print("- Mini-BERT (NumPy): Educational transparency")
print("- Modern BERT (PyTorch): Production efficiency")
print("- Same math, different implementations!")