"""
Script to run Modern BERT training and evaluation.
Sets up everything needed to train and evaluate the model.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

# Add modern_bert to path
sys.path.append(str(Path(__file__).parent / "modern_bert"))

from modern_bert.config import ModernBertConfig, DataConfig, PresetConfigs
from modern_bert.model import ModernBert
from modern_bert.data_pipeline import ModernDataPipeline
from modern_bert.train import ModernBertTrainer
from modern_bert.inference import ModernBertInference, ModernBertEvaluator


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate', 
        'numpy', 'tqdm', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {missing_packages}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✓ All packages installed")
    
    return True


def setup_environment():
    """Set up the training environment."""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = [
        "modern_bert/outputs",
        "modern_bert/logs", 
        "modern_bert/cache",
        "modern_bert/checkpoints"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No GPU available, using CPU (training will be slower)")
    
    return True


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n" + "="*60)
    print("Running Quick Test")
    print("="*60)
    
    try:
        # Test model initialization
        print("1. Testing model initialization...")
        config = PresetConfigs.debug_config()  # Small config for testing
        model = ModernBert(config)
        print(f"✓ Model initialized with {model.count_parameters():,} parameters")
        
        # Test prediction
        print("\n2. Testing masked language modeling...")
        test_text = "The capital of France is [MASK]."
        predictions = model.predict_masked_tokens(test_text, return_top_k=3)
        
        print(f"Input: {test_text}")
        print("Top predictions:")
        for pred in predictions[:3]:
            print(f"  - {pred['token']}: {pred['score']:.3f}")
        
        print("✓ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False


def run_training_demo():
    """Run a demo training session."""
    print("\n" + "="*60)
    print("Running Training Demo")
    print("="*60)
    
    # Configure for demo (small scale)
    config = PresetConfigs.debug_config()
    config.num_train_epochs = 1
    config.train_batch_size = 4
    config.eval_batch_size = 8
    config.logging_steps = 10
    config.eval_steps = 50
    config.save_steps = 100
    config.use_wandb = False  # Disable wandb for demo
    
    data_config = DataConfig()
    data_config.max_train_samples = 500   # Limit for demo
    data_config.max_eval_samples = 100
    data_config.dataset_name = "wikitext"
    data_config.dataset_config = "wikitext-2-raw-v1"  # Smaller dataset
    
    print(f"Training config: {config.num_train_epochs} epochs, batch size {config.train_batch_size}")
    print(f"Data config: max {data_config.max_train_samples} train samples")
    
    try:
        # Initialize trainer
        print("\n1. Initializing trainer...")
        trainer = ModernBertTrainer(config, data_config)
        
        # Run training
        print("\n2. Starting training...")
        hf_trainer = trainer.train_with_hf_trainer()
        
        print("✓ Training completed successfully!")
        
        # Get training metrics
        train_history = hf_trainer.state.log_history
        if train_history:
            final_metrics = train_history[-1]
            print(f"\nFinal Training Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        return hf_trainer
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_evaluation(model_path=None):
    """Run comprehensive evaluation."""
    print("\n" + "="*60)
    print("Running Evaluation")
    print("="*60)
    
    try:
        # Use trained model or pretrained
        if model_path and os.path.exists(model_path):
            print(f"Using trained model: {model_path}")
            inference_model = model_path
        else:
            print("Using pretrained BERT-base-uncased")
            inference_model = "bert-base-uncased"
        
        # Initialize inference
        inference = ModernBertInference(
            model_path=inference_model,
            task="fill-mask",
            optimize=False  # Disable optimization for demo
        )
        
        # Test sentences for MLM evaluation
        test_sentences = [
            "The capital of France is [MASK].",
            "Python is a [MASK] programming language.",
            "The [MASK] is shining brightly today.",
            "I love to [MASK] books in my free time.",
            "The [MASK] flew over the mountain."
        ]
        
        print("\nMasked Language Modeling Results:")
        print("-" * 50)
        
        all_results = []
        for sentence in test_sentences:
            predictions = inference.predict(sentence, top_k=3)
            print(f"\nInput: {sentence}")
            print("Predictions:")
            
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. {pred['token_str']}: {pred['score']:.3f}")
            
            all_results.append({
                'sentence': sentence,
                'predictions': predictions
            })
        
        # Performance stats
        perf_stats = inference.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"  Average inference time: {perf_stats.get('avg_time', 0):.3f}s")
        print(f"  Total inferences: {perf_stats.get('total_inferences', 0)}")
        
        # Run benchmark
        print("\n" + "-" * 50)
        print("Performance Benchmark:")
        
        benchmark_results = inference.benchmark(
            test_inputs=test_sentences,
            num_runs=3,
            batch_sizes=[1, 2, 4]
        )
        
        for batch_size, stats in benchmark_results["batch_results"].items():
            print(f"  Batch size {batch_size}: {stats['avg_throughput']:.1f} samples/sec")
        
        return all_results
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_fine_tuning_demo():
    """Run a fine-tuning demo on sentiment analysis."""
    print("\n" + "="*60)
    print("Running Fine-tuning Demo (Sentiment Analysis)")
    print("="*60)
    
    try:
        from modern_bert.fine_tune import FineTuner, TaskConfigs
        
        # Configure for demo
        config = TaskConfigs.sentiment_analysis()
        config.num_train_epochs = 1
        config.train_batch_size = 8
        config.eval_batch_size = 16
        config.output_dir = "modern_bert/fine_tuned_demo"
        
        print(f"Fine-tuning config: {config.task_name} task")
        print(f"Training: {config.num_train_epochs} epochs, batch size {config.train_batch_size}")
        
        # Initialize fine-tuner
        print("\n1. Initializing fine-tuner...")
        fine_tuner = FineTuner(config)
        
        # Note: This would download the IMDB dataset (large)
        # For demo purposes, we'll just show the setup
        print("✓ Fine-tuner initialized")
        print("Note: Actual fine-tuning skipped in demo (dataset is large)")
        print("To run full fine-tuning: TASK=sentiment python modern_bert/fine_tune.py")
        
        # Test prediction with pretrained sentiment model
        print("\n2. Testing sentiment prediction with pretrained model...")
        from modern_bert.inference import ModernBertInference
        
        sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        cls_inference = ModernBertInference(sentiment_model, task="text-classification")
        
        test_texts = [
            "I love this product! It's amazing!",
            "This is the worst thing I've ever bought.",
            "It's okay, nothing special but works fine.",
            "Absolutely fantastic, highly recommend!",
            "Terrible quality, waste of money."
        ]
        
        results = cls_inference.predict(test_texts)
        
        print("\nSentiment Analysis Results:")
        print("-" * 40)
        for text, result in zip(test_texts, results):
            sentiment = result['label']
            confidence = result['score']
            print(f"Text: {text[:50]}...")
            print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Fine-tuning demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the complete demo."""
    print("Modern BERT Training and Evaluation Demo")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return
    
    # Step 2: Setup environment
    print("\n2. Setting up environment...")
    if not setup_environment():
        return
    
    # Step 3: Quick test
    print("\n3. Running quick test...")
    if not run_quick_test():
        print("Quick test failed. Please check your setup.")
        return
    
    # Step 4: Ask user what to run
    print("\n" + "="*60)
    print("What would you like to run?")
    print("1. Quick training demo (recommended)")
    print("2. Evaluation only")
    print("3. Fine-tuning demo")
    print("4. All of the above")
    print("="*60)
    
    choice = input("Enter your choice (1-4): ").strip()
    
    try:
        if choice in ['1', '4']:
            # Run training demo
            trainer = run_training_demo()
            if trainer:
                # Run evaluation on trained model
                model_path = trainer.args.output_dir
                run_evaluation(model_path)
        
        if choice in ['2', '4']:
            # Run evaluation only
            run_evaluation()
        
        if choice in ['3', '4']:
            # Run fine-tuning demo
            run_fine_tuning_demo()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
        print("\nNext steps:")
        print("- Check outputs in modern_bert/outputs/")
        print("- View logs in modern_bert/logs/") 
        print("- Run full training: python modern_bert/train.py")
        print("- Run fine-tuning: TASK=sentiment python modern_bert/fine_tune.py")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()