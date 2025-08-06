"""
Complete training demo for Modern BERT with evaluation metrics.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    BertTokenizer, BertForMaskedLM, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
import time

print("Modern BERT Training & Evaluation Demo")
print("=" * 50)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create training data
print("\n1. Creating training dataset...")
training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world of technology.",
    "Natural language processing enables computers to understand human text.",
    "Deep learning models can learn complex patterns from data.",
    "Transformers revolutionized natural language understanding tasks.",
    "BERT is a powerful language model for many NLP applications.",
    "Attention mechanisms help models focus on relevant information.",
    "Fine-tuning allows models to adapt to specific downstream tasks.",
    "Large language models demonstrate emergent capabilities at scale.",
    "Artificial intelligence research continues to push boundaries.",
    "Neural networks are inspired by biological brain structures.",
    "Gradient descent optimizes model parameters during training.",
    "Backpropagation computes gradients for neural network learning.",
    "Cross-entropy loss measures prediction accuracy for classification.",
    "Regularization techniques prevent overfitting in machine learning."
] * 4  # 60 total examples

print(f"Created dataset with {len(training_texts)} examples")

# Initialize model and tokenizer  
print("\n2. Initializing model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=64,
        return_special_tokens_mask=True
    )

dataset = Dataset.from_dict({'text': training_texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("Dataset tokenized successfully")

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments (fixed)
training_args = TrainingArguments(
    output_dir="./training_results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=None,
    dataloader_pin_memory=False,  # Avoid potential issues
)

# Create trainer
print("\n3. Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
print("\n4. Starting training...")
print("This will take 2-5 minutes depending on your hardware...")

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.1f} seconds!")

# Get training metrics
training_history = trainer.state.log_history
if training_history:
    print("\nTraining Metrics:")
    for i, entry in enumerate(training_history):
        if 'train_loss' in entry:
            step = entry.get('step', i)
            loss = entry['train_loss']
            print(f"  Step {step}: Loss = {loss:.4f}")

# Save the model
print("\n5. Saving trained model...")
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Evaluation on new examples
print("\n6. Evaluating trained model...")

# Load the trained model for inference
from transformers import pipeline

# Create MLM pipeline with trained model
mlm_pipeline = pipeline(
    "fill-mask",
    model="./trained_model",
    tokenizer="./trained_model",
    device=0 if device == "cuda" else -1
)

# Test sentences for evaluation
test_sentences = [
    "Machine learning is a [MASK] of artificial intelligence.",
    "Neural networks learn [MASK] from training data.",
    "Deep learning models can [MASK] complex patterns.",
    "Transformers use [MASK] mechanisms for language understanding.",
    "The model was trained using [MASK] descent optimization."
]

print("\nEvaluation Results:")
print("-" * 40)

evaluation_results = []
for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    predictions = mlm_pipeline(sentence, top_k=3)
    
    print("Top predictions:")
    for i, pred in enumerate(predictions, 1):
        token = pred['token_str'].strip()
        score = pred['score']
        print(f"  {i}. {token}: {score:.3f}")
    
    evaluation_results.append({
        'sentence': sentence,
        'top_prediction': predictions[0]['token_str'],
        'confidence': predictions[0]['score']
    })

# Performance comparison: Before vs After training
print("\n7. Performance comparison...")

# Test with original model
original_pipeline = pipeline(
    "fill-mask",
    model="bert-base-uncased", 
    device=0 if device == "cuda" else -1
)

comparison_sentence = "Deep learning models can [MASK] complex patterns."

print(f"\nComparison for: {comparison_sentence}")
print("\nOriginal BERT:")
orig_preds = original_pipeline(comparison_sentence, top_k=3)
for i, pred in enumerate(orig_preds, 1):
    print(f"  {i}. {pred['token_str'].strip()}: {pred['score']:.3f}")

print("\nTrained BERT:")
trained_preds = mlm_pipeline(comparison_sentence, top_k=3)
for i, pred in enumerate(trained_preds, 1):
    print(f"  {i}. {pred['token_str'].strip()}: {pred['score']:.3f}")

# Final metrics summary
print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)

final_loss = training_history[-1]['train_loss'] if training_history else "N/A"
print(f"Final training loss: {final_loss}")
print(f"Training time: {training_time:.1f} seconds") 
print(f"Training samples: {len(training_texts)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\nBest predictions:")
for result in evaluation_results:
    sentence = result['sentence']
    prediction = result['top_prediction']
    confidence = result['confidence']
    print(f"  '{sentence}' → {prediction} ({confidence:.3f})")

print(f"\nAverage confidence: {np.mean([r['confidence'] for r in evaluation_results]):.3f}")

print("\n" + "=" * 50)
print("MODERN BERT vs MINI BERT COMPARISON")
print("=" * 50)

print("Mini-BERT (Educational):")
print("  - Pure NumPy implementation")
print("  - 3.2M parameters") 
print("  - Full mathematical transparency")
print("  - CPU only, slower training")
print("  - Perfect for learning fundamentals")

print("\nModern BERT (Production):")
print("  - PyTorch + HuggingFace implementation")
print("  - 110M parameters")
print("  - Optimized for performance")
print("  - GPU acceleration, faster training")
print("  - Perfect for real applications")

print("\nSame core mathematics, different implementations!")
print("Both use: Attention, Layer Norm, Feed-Forward, Residuals")

print("\n✓ Demo completed successfully!")
print("Check './trained_model/' for saved model files")