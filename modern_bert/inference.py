"""
Modern BERT inference and evaluation scripts.
Production-ready inference with optimizations and serving capabilities.
"""

import torch
import numpy as np
from transformers import pipeline, Pipeline
from typing import Dict, List, Union, Optional, Any
import json
import time
import os
from pathlib import Path
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForMaskedLM

from modern_bert.model import ModernBert, ModernBertForClassification
from modern_bert.config import ModernBertConfig


class ModernBertInference:
    """
    Production-ready inference class for BERT models.
    Supports various optimizations and deployment formats.
    """
    
    def __init__(
        self,
        model_path: str,
        task: str = "masked-lm",
        device: str = "auto",
        optimize: bool = True,
        use_onnx: bool = False
    ):
        self.model_path = model_path
        self.task = task
        self.device = device
        self.optimize = optimize
        self.use_onnx = use_onnx
        
        # Initialize pipeline
        self.pipeline = self._create_pipeline()
        
        # Performance tracking
        self.inference_times = []
        
    def _create_pipeline(self) -> Pipeline:
        """Create HuggingFace pipeline for inference."""
        pipeline_kwargs = {
            "model": self.model_path,
            "device": 0 if torch.cuda.is_available() and self.device != "cpu" else -1
        }
        
        if self.use_onnx:
            # Use ONNX Runtime for faster inference
            if self.task == "text-classification":
                model = ORTModelForSequenceClassification.from_pretrained(self.model_path)
            elif self.task == "fill-mask":
                model = ORTModelForMaskedLM.from_pretrained(self.model_path)
            else:
                raise ValueError(f"ONNX not supported for task: {self.task}")
            
            pipeline_kwargs["model"] = model
        
        # Create pipeline
        if self.task == "masked-lm":
            task_name = "fill-mask"
        elif self.task == "classification":
            task_name = "text-classification"
        else:
            task_name = self.task
        
        pipe = pipeline(task_name, **pipeline_kwargs)
        
        # Apply optimizations
        if self.optimize and not self.use_onnx:
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                pipe.model = torch.compile(pipe.model)
        
        return pipe
    
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 1,
        return_all_scores: bool = False,
        top_k: int = 5
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions on input text(s).
        
        Args:
            inputs: Single text or list of texts
            batch_size: Batch size for processing
            return_all_scores: Return all scores for classification
            top_k: Number of top predictions to return
            
        Returns:
            Predictions with scores and metadata
        """
        start_time = time.time()
        
        # Handle single vs batch inputs
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        # Process in batches
        all_results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Run inference
            if self.task in ["fill-mask", "masked-lm"]:  
                batch_results = self.pipeline(batch, top_k=top_k)
            elif self.task in ["text-classification", "classification"]:
                batch_results = self.pipeline(batch, return_all_scores=return_all_scores)
            else:
                batch_results = self.pipeline(batch)
            
            # Ensure batch_results is a list
            if not isinstance(batch_results[0], list):
                batch_results = [batch_results]
            
            all_results.extend(batch_results)
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Add metadata
        for i, result in enumerate(all_results):
            if isinstance(result, list):
                # Multiple predictions per input (e.g., fill-mask)
                for pred in result:
                    pred['input_text'] = inputs[i]
                    pred['inference_time'] = inference_time / len(all_results)
            else:
                # Single prediction per input
                result['input_text'] = inputs[i]
                result['inference_time'] = inference_time / len(all_results)
        
        return all_results[0] if single_input else all_results
    
    def predict_batch(
        self,
        inputs: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict]:
        """Optimized batch prediction."""
        from tqdm import tqdm
        
        all_results = []
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        
        iterator = tqdm(batches) if show_progress else batches
        
        for batch in iterator:
            batch_results = self.predict(batch, batch_size=len(batch))
            all_results.extend(batch_results)
        
        return all_results
    
    def benchmark(
        self,
        test_inputs: List[str],
        num_runs: int = 10,
        batch_sizes: List[int] = [1, 4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Benchmark inference performance."""
        print("Running benchmark...")
        
        results = {
            "model_path": self.model_path,
            "task": self.task,
            "device": self.device,
            "use_onnx": self.use_onnx,
            "optimize": self.optimize,
            "num_runs": num_runs,
            "batch_results": {}
        }
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            times = []
            throughputs = []
            
            for _ in range(num_runs):
                # Select test batch
                test_batch = test_inputs[:batch_size] * (batch_size // len(test_inputs) + 1)
                test_batch = test_batch[:batch_size]
                
                # Time inference
                start_time = time.time()
                _ = self.predict(test_batch, batch_size=batch_size)
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                throughputs.append(batch_size / elapsed)
            
            results["batch_results"][batch_size] = {
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "avg_throughput": np.mean(throughputs),
                "std_throughput": np.std(throughputs),
                "times": times
            }
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {"message": "No inference times recorded"}
        
        times = np.array(self.inference_times)
        return {
            "total_inferences": len(times),
            "avg_time": np.mean(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "std_time": np.std(times),
            "median_time": np.median(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99)
        }


class ModernBertEvaluator:
    """Comprehensive evaluation for BERT models."""
    
    def __init__(self, model_path: str, task: str):
        self.model_path = model_path
        self.task = task
        self.inference_engine = ModernBertInference(model_path, task)
    
    def evaluate_masked_lm(
        self,
        test_sentences: List[str],
        mask_token: str = "[MASK]"
    ) -> Dict[str, float]:
        """Evaluate masked language modeling."""
        results = {
            "total_sentences": len(test_sentences),
            "correct_predictions": 0,
            "top_1_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "avg_confidence": 0.0
        }
        
        correct_top1 = 0
        correct_top5 = 0
        total_confidence = 0.0
        
        for sentence in test_sentences:
            if mask_token not in sentence:
                continue
            
            # Get predictions
            predictions = self.inference_engine.predict(sentence, top_k=5)
            
            # For evaluation, we'd need the ground truth
            # This is a simplified example
            if predictions:
                total_confidence += predictions[0]['score']
                # Add logic to check if prediction matches ground truth
        
        results["avg_confidence"] = total_confidence / len(test_sentences)
        return results
    
    def evaluate_classification(
        self,
        test_texts: List[str],
        true_labels: List[int],
        label_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate classification performance."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Get predictions
        predictions = self.inference_engine.predict_batch(test_texts)
        pred_labels = [pred['label'] for pred in predictions]
        
        # Convert string labels to integers if needed
        if isinstance(pred_labels[0], str):
            if label_names:
                label_to_id = {name: i for i, name in enumerate(label_names)}
                pred_labels = [label_to_id[label] for label in pred_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_samples": len(test_texts)
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        results["confusion_matrix"] = cm.tolist()
        
        return results
    
    def run_comprehensive_evaluation(
        self,
        test_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation suite."""
        evaluation_results = {
            "model_path": self.model_path,
            "task": self.task,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_stats": self.inference_engine.get_performance_stats()
        }
        
        if self.task in ["masked-lm", "fill-mask"]:
            if "masked_sentences" in test_data:
                mlm_results = self.evaluate_masked_lm(test_data["masked_sentences"])
                evaluation_results["mlm_evaluation"] = mlm_results
        
        elif self.task in ["classification", "text-classification"]:
            if "texts" in test_data and "labels" in test_data:
                cls_results = self.evaluate_classification(
                    test_data["texts"],
                    test_data["labels"],
                    test_data.get("label_names")
                )
                evaluation_results["classification_evaluation"] = cls_results
        
        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Evaluation results saved to {output_path}")
        
        return evaluation_results


class ModelOptimizer:
    """Optimize models for production deployment."""
    
    @staticmethod
    def convert_to_onnx(
        model_path: str,
        output_path: str,
        task: str = "text-classification",
        opset_version: int = 14
    ):
        """Convert model to ONNX format."""
        from optimum.onnxruntime import ORTModelForSequenceClassification
        
        print(f"Converting {model_path} to ONNX...")
        
        if task == "text-classification":
            model = ORTModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError(f"ONNX conversion not implemented for task: {task}")
        
        model.save_pretrained(output_path)
        print(f"ONNX model saved to {output_path}")
    
    @staticmethod
    def quantize_model(
        model_path: str,
        output_path: str,
        quantization_type: str = "dynamic"
    ):
        """Quantize model for faster inference."""
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        
        print(f"Quantizing model: {quantization_type}")
        
        # Define quantization configuration
        if quantization_type == "dynamic":
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        elif quantization_type == "static":
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Quantize
        quantizer = ORTQuantizer.from_pretrained(model_path)
        quantizer.quantize(save_dir=output_path, quantization_config=qconfig)
        
        print(f"Quantized model saved to {output_path}")


def main():
    """Example usage of inference and evaluation."""
    
    # Example 1: Basic inference
    print("=== Basic Inference Example ===")
    
    # Use a pretrained model for demonstration
    model_path = "bert-base-uncased"
    
    # Masked LM inference
    mlm_inference = ModernBertInference(model_path, task="fill-mask")
    
    test_sentence = "The capital of France is [MASK]."
    predictions = mlm_inference.predict(test_sentence, top_k=3)
    
    print(f"Input: {test_sentence}")
    print("Predictions:")
    for pred in predictions:
        print(f"  {pred['token_str']}: {pred['score']:.3f}")
    
    # Example 2: Classification inference
    print("\n=== Classification Example ===")
    
    # For this example, we'll use a sentiment model
    sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    cls_inference = ModernBertInference(sentiment_model, task="text-classification")
    
    texts = [
        "I love this movie!",
        "This is terrible.",
        "It's okay, not great but not bad either."
    ]
    
    results = cls_inference.predict(texts)
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} ({result['score']:.3f})")
    
    # Example 3: Performance benchmark
    print("\n=== Performance Benchmark ===")
    
    benchmark_results = mlm_inference.benchmark(
        test_inputs=[test_sentence] * 5,
        num_runs=3,
        batch_sizes=[1, 2, 4]
    )
    
    print("Benchmark Results:")
    for batch_size, stats in benchmark_results["batch_results"].items():
        print(f"Batch size {batch_size}: {stats['avg_throughput']:.2f} samples/sec")
    
    # Example 4: Model optimization
    print("\n=== Model Optimization Example ===")
    
    # Note: This would require the actual model files
    # ModelOptimizer.convert_to_onnx(
    #     model_path="path/to/model",
    #     output_path="path/to/onnx/model",
    #     task="text-classification"
    # )
    
    print("Model optimization examples (commented out - requires model files)")


if __name__ == "__main__":
    main()