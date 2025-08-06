"""
Fine-tuning script for downstream tasks using modern BERT.
Supports classification, NER, QA, and other tasks with minimal code changes.
"""

import os
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from modern_bert.config import ModernBertConfig
from modern_bert.data_pipeline import ModernDataPipeline


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning tasks."""
    
    # Task configuration
    task_type: str = "classification"  # classification, ner, qa, regression
    task_name: str = "sentiment"  # sentiment, nli, ner, squad, etc.
    num_labels: int = 2
    
    # Model configuration
    base_model: str = "bert-base-uncased"
    use_pretrained_bert: bool = True  # Use our trained BERT or HF pretrained
    model_checkpoint: Optional[str] = None  # Path to our trained model
    
    # Training configuration
    learning_rate: float = 2e-5  # Lower than pre-training
    num_train_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Task-specific configuration
    max_seq_length: int = 128
    label_names: Optional[List[str]] = None
    
    # Advanced options
    use_lora: bool = False  # Parameter-efficient fine-tuning
    freeze_backbone: bool = False  # Freeze BERT, train only head
    gradient_checkpointing: bool = True
    fp16: bool = True
    
    # Paths
    output_dir: str = "modern_bert/fine_tuned"
    data_dir: str = "data"
    cache_dir: str = "modern_bert/cache"


class FineTuner:
    """Universal fine-tuner for BERT downstream tasks."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            cache_dir=config.cache_dir
        )
        
        # Initialize model based on task
        self.model = self._create_model()
        
        # Initialize metrics
        self.metrics = self._get_metrics()
        
    def _create_model(self):
        """Create model based on task type."""
        if self.config.task_type == "classification":
            model_class = AutoModelForSequenceClassification
        elif self.config.task_type == "ner":
            model_class = AutoModelForTokenClassification
        elif self.config.task_type == "qa":
            model_class = AutoModelForQuestionAnswering
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        # Load model
        if self.config.use_pretrained_bert:
            model = model_class.from_pretrained(
                self.config.base_model,
                num_labels=self.config.num_labels,
                cache_dir=self.config.cache_dir
            )
        else:
            # Load our custom trained BERT
            if self.config.model_checkpoint:
                model = model_class.from_pretrained(
                    self.config.model_checkpoint,
                    num_labels=self.config.num_labels
                )
            else:
                raise ValueError("model_checkpoint required when not using pretrained BERT")
        
        # Apply optimizations
        if self.config.freeze_backbone:
            # Freeze BERT parameters, only train classification head
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'qa_outputs' not in name:
                    param.requires_grad = False
        
        if self.config.use_lora:
            model = self._apply_lora(model)
        
        return model
    
    def _apply_lora(self, model):
        """Apply LoRA for parameter-efficient fine-tuning."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        if self.config.task_type == "classification":
            task_type = TaskType.SEQ_CLS
        elif self.config.task_type == "ner":
            task_type = TaskType.TOKEN_CLS
        elif self.config.task_type == "qa":
            task_type = TaskType.QUESTION_ANS
        else:
            task_type = TaskType.FEATURE_EXTRACTION
        
        lora_config = LoraConfig(
            task_type=task_type,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _get_metrics(self):
        """Get metrics for evaluation."""
        if self.config.task_type == "classification":
            return evaluate.load("accuracy")
        elif self.config.task_type == "ner":
            return evaluate.load("seqeval")
        elif self.config.task_type == "qa":
            return evaluate.load("squad")
        else:
            return None
    
    def load_dataset(self):
        """Load dataset for the specific task."""
        task_datasets = {
            "sentiment": ("imdb", None),
            "nli": ("multi_nli", None),
            "ner": ("conll2003", None),
            "squad": ("squad", None),
            "cola": ("glue", "cola"),
            "sst2": ("glue", "sst2"),
            "mrpc": ("glue", "mrpc"),
            "qqp": ("glue", "qqp"),
        }
        
        if self.config.task_name in task_datasets:
            dataset_name, subset = task_datasets[self.config.task_name]
            dataset = load_dataset(dataset_name, subset, cache_dir=self.config.cache_dir)
        else:
            # Try to load as a custom dataset
            dataset = load_dataset(self.config.task_name, cache_dir=self.config.cache_dir)
        
        return dataset
    
    def preprocess_classification_data(self, dataset):
        """Preprocess data for classification tasks."""
        def preprocess_function(examples):
            # Handle different text column names
            text_cols = ['text', 'sentence', 'premise', 'question']
            text_col = next((col for col in text_cols if col in examples), None)
            
            if text_col is None:
                raise ValueError(f"No text column found. Available: {list(examples.keys())}")
            
            # Handle sentence pairs (for NLI, etc.)
            if 'hypothesis' in examples:
                result = self.tokenizer(
                    examples[text_col],
                    examples['hypothesis'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_seq_length
                )
            else:
                result = self.tokenizer(
                    examples[text_col],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_seq_length
                )
            
            # Handle different label column names
            label_cols = ['label', 'labels']
            label_col = next((col for col in label_cols if col in examples), None)
            
            if label_col:
                result['labels'] = examples[label_col]
            
            return result
        
        return dataset.map(preprocess_function, batched=True)
    
    def preprocess_ner_data(self, dataset):
        """Preprocess data for NER tasks."""
        def preprocess_function(examples):
            tokenized_inputs = self.tokenizer(
                examples['tokens'],
                truncation=True,
                is_split_into_words=True,
                padding='max_length',
                max_length=self.config.max_seq_length
            )
            
            labels = []
            for i, label in enumerate(examples['ner_tags']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)  # Set to -100 for subword tokens
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        return dataset.map(preprocess_function, batched=True)
    
    def preprocess_qa_data(self, dataset):
        """Preprocess data for QA tasks."""
        def preprocess_function(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = self.tokenizer(
                questions,
                examples["context"],
                max_length=self.config.max_seq_length,
                truncation="only_second",
                return_offsets_mapping=True,
                padding='max_length'
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []
            
            for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
                start_char = answer["answer_start"][0] if answer["answer_start"] else 0
                end_char = start_char + len(answer["text"][0]) if answer["text"] else 0
                
                # Find token positions
                start_token = 0
                end_token = 0
                
                for idx, (start, end) in enumerate(offset):
                    if start <= start_char < end:
                        start_token = idx
                    if start < end_char <= end:
                        end_token = idx
                        break
                
                start_positions.append(start_token)
                end_positions.append(end_token)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs
        
        return dataset.map(preprocess_function, batched=True)
    
    def compute_classification_metrics(self, eval_pred):
        """Compute metrics for classification."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        results = {}
        
        # Accuracy
        accuracy = accuracy_score(labels, predictions)
        results["accuracy"] = accuracy
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        results.update({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        return results
    
    def compute_ner_metrics(self, eval_pred):
        """Compute metrics for NER."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.config.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.config.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = self.metrics.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"], 
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
    
    def fine_tune(self):
        """Main fine-tuning function."""
        print(f"Fine-tuning BERT for {self.config.task_name} ({self.config.task_type})")
        
        # Load and preprocess dataset
        dataset = self.load_dataset()
        
        if self.config.task_type == "classification":
            dataset = self.preprocess_classification_data(dataset)
            data_collator = DataCollatorWithPadding(self.tokenizer)
            compute_metrics = self.compute_classification_metrics
        elif self.config.task_type == "ner":
            dataset = self.preprocess_ner_data(dataset)
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
            compute_metrics = self.compute_ner_metrics
        elif self.config.task_type == "qa":
            dataset = self.preprocess_qa_data(dataset)
            data_collator = DataCollatorWithPadding(self.tokenizer)
            compute_metrics = None  # QA metrics are more complex
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            
            # Optimization
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # Evaluation and saving
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model="f1" if self.config.task_type != "qa" else "eval_loss",
            save_total_limit=2,
            
            # Misc
            seed=42,
            report_to=None,  # Disable wandb for fine-tuning
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Fine-tune
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # Save model
        trainer.save_model()
        
        return trainer, eval_results
    
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Make predictions on new texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.config.task_type == "classification":
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
            
            results = []
            for i, text in enumerate(texts):
                results.append({
                    "text": text,
                    "predicted_class": predicted_classes[i].item(),
                    "probabilities": predictions[i].cpu().numpy().tolist(),
                    "confidence": predictions[i].max().item()
                })
            
            return results[0] if len(results) == 1 else results
        
        # Add other task types as needed
        return outputs


# Preset configurations for common tasks
class TaskConfigs:
    """Preset configurations for common fine-tuning tasks."""
    
    @staticmethod
    def sentiment_analysis() -> FineTuningConfig:
        return FineTuningConfig(
            task_type="classification",
            task_name="imdb",
            num_labels=2,
            learning_rate=2e-5,
            num_train_epochs=3
        )
    
    @staticmethod
    def named_entity_recognition() -> FineTuningConfig:
        return FineTuningConfig(
            task_type="ner",
            task_name="conll2003", 
            num_labels=9,  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
            learning_rate=3e-5,
            num_train_epochs=3,
            label_names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        )
    
    @staticmethod
    def question_answering() -> FineTuningConfig:
        return FineTuningConfig(
            task_type="qa",
            task_name="squad",
            learning_rate=3e-5,
            num_train_epochs=2,
            max_seq_length=384
        )


def main():
    """Example usage of fine-tuning."""
    # Choose task
    task = os.getenv("TASK", "sentiment")  # sentiment, ner, qa
    
    if task == "sentiment":
        config = TaskConfigs.sentiment_analysis()
    elif task == "ner":
        config = TaskConfigs.named_entity_recognition()
    elif task == "qa":
        config = TaskConfigs.question_answering()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create fine-tuner
    fine_tuner = FineTuner(config)
    
    # Fine-tune
    trainer, results = fine_tuner.fine_tune()
    
    # Test prediction
    if task == "sentiment":
        test_texts = ["This movie is great!", "I hate this film."]
        predictions = fine_tuner.predict(test_texts)
        print(f"Predictions: {predictions}")
    
    print("Fine-tuning completed!")


if __name__ == "__main__":
    # Set task via environment variable
    # os.environ["TASK"] = "sentiment"  # or "ner", "qa"
    main()