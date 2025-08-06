"""
Quick start script for Modern BERT - minimal setup for immediate results.
This script runs a lightweight training and evaluation demo.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Minimal imports
try:
    import torch
    from transformers import (
        BertTokenizer, BertForMaskedLM, 
        pipeline, Trainer, TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import numpy as np
    from tqdm import tqdm
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("Please run: pip install torch transformers datasets numpy tqdm")
    sys.exit(1)


class QuickBERTDemo:
    """Minimal BERT demo for immediate results."""
    
    def __init__(self):
        print("Initializing Quick BERT Demo...")
        
        # Use a smaller, faster model for demo
        self.model_name = "distilbert-base-uncased"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Using model: {self.model_name}")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mlm_pipeline = pipeline(
            "fill-mask", 
            model=self.model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        print("✓ Model and tokenizer initialized")
    
    def test_masked_prediction(self):
        """Test masked language modeling."""
        print("\n" + "="*50)
        print("Testing Masked Language Modeling")
        print("="*50)
        
        test_sentences = [
            "The capital of France is [MASK].",
            "Python is a [MASK] programming language.",
            "I love to [MASK] books.",
            "The [MASK] is shining today.",
            "Machine learning requires [MASK] data."
        ]
        
        results = []
        for sentence in test_sentences:
            print(f"\nInput: {sentence}")
            predictions = self.mlm_pipeline(sentence, top_k=3)
            
            print("Top predictions:")
            for i, pred in enumerate(predictions, 1):
                token = pred['token_str']
                score = pred['score']
                print(f"  {i}. {token}: {score:.3f}")
                
            results.append({
                'sentence': sentence,
                'top_prediction': predictions[0]['token_str'],
                'confidence': predictions[0]['score']
            })
        
        return results
    
    def run_mini_training(self):
        """Run a mini training session for demonstration."""
        print("\n" + "="*50)
        print("Running Mini Training Demo")
        print("="*50)
        
        try:
            # Create dummy training data
            training_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world.",
                "Natural language processing enables computers to understand text.",
                "Deep learning models can learn complex patterns.",
                "Transformers revolutionized natural language understanding.",
                "BERT is a powerful language model for many tasks.",
                "Attention mechanisms help models focus on relevant information.",
                "Fine-tuning allows models to adapt to specific tasks.",
                "Large language models demonstrate emergent capabilities.",
                "AI research continues to push the boundaries of what's possible."
            ] * 10  # Repeat for more data
            
            print(f"Created {len(training_texts)} training examples")
            
            # Tokenize data
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_special_tokens_mask=True
                )
            
            # Create dataset
            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            print("✓ Dataset created and tokenized")
            
            # Initialize model for training
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
            model.resize_token_embeddings(len(self.tokenizer))
            
            # Data collator for MLM
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
            
            # Training arguments (very small scale)
            training_args = TrainingArguments(
                output_dir="./demo_results",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                logging_steps=10,
                save_steps=50,
                evaluation_strategy="no",  # Skip evaluation for speed
                save_total_limit=1,
                prediction_loss_only=True,
                remove_unused_columns=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            print("✓ Trainer initialized")
            print("Starting training (this may take a few minutes)...")
            
            # Train
            trainer.train()
            
            print("✓ Training completed!")
            
            # Save model
            trainer.save_model("./demo_model")
            print("✓ Model saved to ./demo_model")
            
            # Get training metrics
            if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                final_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
                print(f"Final training loss: {final_loss}")
            
            return True
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return False
    
    def benchmark_performance(self):
        """Benchmark inference performance."""
        print("\n" + "="*50)
        print("Performance Benchmark")
        print("="*50)
        
        test_text = "The capital of [MASK] is a beautiful city."
        num_runs = 20
        
        print(f"Running {num_runs} inferences...")
        
        import time
        times = []
        
        for i in tqdm(range(num_runs)):
            start_time = time.time()
            _ = self.mlm_pipeline(test_text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\nPerformance Results:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Std deviation: {std_time:.3f}s")
        print(f"  Throughput: {1/avg_time:.1f} samples/sec")
        
        return {
            'avg_time': avg_time,
            'throughput': 1/avg_time,
            'times': times
        }
    
    def test_different_tasks(self):
        """Test different NLP tasks."""
        print("\n" + "="*50)
        print("Testing Different Tasks")
        print("="*50)
        
        # Sentiment Analysis
        print("\n1. Sentiment Analysis:")
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
            texts = [
                "I love this product!",
                "This is terrible.",
                "It's okay, nothing special."
            ]
            
            for text in texts:
                result = sentiment_pipeline(text)[0]
                print(f"  '{text}' → {result['label']} ({result['score']:.3f})")
        except Exception as e:
            print(f"  Sentiment analysis failed: {e}")
        
        # Text Classification
        print("\n2. Zero-shot Classification:")
        try:
            classifier = pipeline("zero-shot-classification")
            text = "This movie is fantastic and entertaining."
            labels = ["positive", "negative", "neutral"]
            
            result = classifier(text, labels)
            print(f"  Text: {text}")
            print(f"  Classification: {result['labels'][0]} ({result['scores'][0]:.3f})")
        except Exception as e:
            print(f"  Zero-shot classification failed: {e}")
        
        # Text Generation (if available)
        print("\n3. Text Generation:")
        try:
            generator = pipeline("text-generation", model="gpt2")
            prompt = "The future of AI is"
            result = generator(prompt, max_length=50, num_return_sequences=1)
            print(f"  Prompt: {prompt}")
            print(f"  Generated: {result[0]['generated_text']}")
        except Exception as e:
            print(f"  Text generation failed: {e}")
    
    def run_complete_demo(self):
        """Run the complete demo."""
        print("Starting Complete Modern BERT Demo")
        print("=" * 60)
        
        # Test 1: Masked predictions
        mlm_results = self.test_masked_prediction()
        
        # Test 2: Performance benchmark
        perf_results = self.benchmark_performance()
        
        # Test 3: Different tasks
        self.test_different_tasks()
        
        # Test 4: Mini training (optional - takes time)
        print("\n" + "="*50)
        print("Mini Training Demo")
        print("="*50)
        print("Would you like to run a mini training demo?")
        print("(This will take 2-5 minutes)")
        
        choice = input("Run training demo? (y/n): ").strip().lower()
        if choice == 'y':
            training_success = self.run_mini_training()
        else:
            print("Skipping training demo")
            training_success = None
        
        # Summary
        print("\n" + "="*60)
        print("DEMO SUMMARY")
        print("="*60)
        
        print(f"✓ Masked Language Modeling: {len(mlm_results)} predictions")
        print(f"✓ Performance: {perf_results['throughput']:.1f} samples/sec")
        print(f"✓ Average inference time: {perf_results['avg_time']:.3f}s")
        
        if training_success is True:
            print("✓ Mini training: Completed successfully")
        elif training_success is False:
            print("✗ Mini training: Failed")
        else:
            print("- Mini training: Skipped")
        
        print("\nBest Predictions:")
        for result in mlm_results[:3]:
            sentence = result['sentence']
            prediction = result['top_prediction']
            confidence = result['confidence']
            print(f"  '{sentence}' → {prediction} ({confidence:.3f})")
        
        print("\n" + "="*60) 
        print("Demo completed successfully!")
        print("="*60)
        
        return {
            'mlm_results': mlm_results,
            'performance': perf_results,
            'training_success': training_success
        }


def main():
    """Main function."""
    try:
        demo = QuickBERTDemo()
        results = demo.run_complete_demo()
        return results
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()