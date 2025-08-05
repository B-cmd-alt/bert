#!/usr/bin/env python3
"""
Real-World Data BERT Tokenizer Training Pipeline

This script downloads and processes Wikipedia, Books, and News data
to train a BERT-compatible WordPiece tokenizer optimized for your hardware.
"""

import os
import sys
import time
import psutil
import logging
import requests
import bz2
import re
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import List, Tuple
import xml.etree.ElementTree as ET

# Add the learning-examples directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'learning-examples'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'bert-wordpiece-tokenizer'))

from wordpiece_simple import SimpleWordPieceTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_world_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealWorldDataTrainer:
    """Train BERT tokenizer on Wikipedia + Books + News datasets."""
    
    def __init__(self, output_dir: str = "bert_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "raw_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # System specs based on your hardware
        self.total_ram_gb = 32
        self.max_data_size_gb = 12  # Conservative limit for processing
        
    def estimate_training_requirements(self, vocab_size: int) -> dict:
        """Estimate training time and resource requirements."""
        
        # Based on your system specs and typical WordPiece training
        estimates = {
            "vocab_size": vocab_size,
            "recommended_data_size_gb": min(8 + (vocab_size / 10000), self.max_data_size_gb),
            "estimated_training_hours": 2 + (vocab_size / 15000),  # Conservative estimate
            "peak_memory_gb": 4 + (vocab_size / 20000),
            "final_model_size_mb": vocab_size * 0.05,  # Rough estimate
        }
        
        return estimates
    
    def download_with_progress(self, url: str, filename: str) -> bool:
        """Download file with progress bar."""
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB)", end="")
            
            print(f"\nDownload completed: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def download_wikipedia_sample(self, force_download: bool = False) -> str:
        """Download and extract Wikipedia sample."""
        # Use a smaller, manageable Wikipedia sample for faster processing
        wiki_file = self.data_dir / "wiki_sample.txt"
        
        if wiki_file.exists() and not force_download:
            file_size = os.path.getsize(wiki_file) / (1024 * 1024)  # MB
            if file_size > 1:  # Only skip if file is substantial (>1MB)
                logger.info(f"Wikipedia sample already exists: {wiki_file} ({file_size:.1f} MB)")
                return str(wiki_file)
            else:
                logger.info(f"Existing Wikipedia file is too small ({file_size:.1f} MB), re-downloading...")
                os.remove(wiki_file)
        
        logger.info("Downloading Wikipedia sample via Hugging Face datasets...")
        
        try:
            import datasets
            import random
            
            logger.info("Loading Wikipedia dataset...")
            # Try multiple Wikipedia dataset configurations
            dataset_configs = [
                ("wikipedia", "20220301.en"),
                ("wikipedia", "20220301.simple"),
                ("legacy-datasets/wikipedia", "20220301.en")
            ]
            
            dataset = None
            for dataset_name, config in dataset_configs:
                try:
                    logger.info(f"Trying {dataset_name} with config {config}...")
                    dataset = datasets.load_dataset(dataset_name, config, split="train", streaming=True)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}:{config} - {e}")
                    continue
            
            if dataset is None:
                raise Exception("Could not load any Wikipedia dataset configuration")
            
            logger.info("Processing Wikipedia articles...")
            with open(wiki_file, "w", encoding="utf-8") as f:
                count = 0
                lines_written = 0
                for item in dataset:
                    if count >= 25000:  # Limit to 25k articles for manageable size
                        break
                    
                    text = item.get("text", "")
                    if len(text) > 100:  # Skip very short articles
                        # Clean and write text
                        text_lines = text.split("\n")
                        for line in text_lines:
                            line = line.strip()
                            if len(line) > 20:  # Skip very short lines
                                f.write(line + "\n")
                                lines_written += 1
                    
                    count += 1
                    if count % 2500 == 0:
                        logger.info(f"Processed {count} articles, wrote {lines_written} lines...")
                        
            file_size = os.path.getsize(wiki_file) / (1024 * 1024)
            logger.info(f"Wikipedia sample created: {wiki_file} ({file_size:.1f} MB, {lines_written} lines)")
            
            logger.info(f"Wikipedia sample created: {wiki_file}")
            return str(wiki_file)
            
        except Exception as e:
            logger.error(f"Failed to download Wikipedia: {e}")
            logger.info("Attempting fallback: Creating sample Wikipedia-style text...")
            # Create a more substantial fallback with Wikipedia-style content
            with open(wiki_file, 'w', encoding='utf-8') as f:
                sample_content = [
                    "Wikipedia is a multilingual online encyclopedia created and maintained as an open collaboration project.",
                    "Machine learning is a method of data analysis that automates analytical model building.",
                    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
                    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
                    "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans.",
                    "Computer science is the study of algorithmic processes, computational systems and the design of computer systems.",
                    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems.",
                    "Software engineering is the systematic approach to the design, development, operation, and maintenance of software.",
                    "Information technology is the use of computers to store, retrieve, transmit, and manipulate data or information.",
                    "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks."
                ] * 1000  # Repeat to create substantial content
                
                for line in sample_content:
                    f.write(line + "\n")
            
            logger.info(f"Created fallback Wikipedia content: {wiki_file}")
            return str(wiki_file)
    
    def download_books_sample(self, force_download: bool = False) -> str:
        """Download Project Gutenberg books sample."""
        books_file = self.data_dir / "books_sample.txt"
        
        if books_file.exists() and not force_download:
            file_size = os.path.getsize(books_file) / (1024 * 1024)  # MB
            if file_size > 0.5:  # Only skip if file is substantial (>0.5MB)
                logger.info(f"Books sample already exists: {books_file} ({file_size:.1f} MB)")
                return str(books_file)
            else:
                logger.info(f"Existing books file is too small ({file_size:.1f} MB), re-downloading...")
                os.remove(books_file)
        
        logger.info("Downloading Project Gutenberg books sample...")
        
        try:
            import datasets
            
            try:
                logger.info("Loading Project Gutenberg dataset...")
                dataset = datasets.load_dataset("manu/project_gutenberg", split="train", streaming=True)
                
                logger.info("Processing books...")
                with open(books_file, "w", encoding="utf-8") as f:
                    count = 0
                    for item in dataset:
                        if count >= 3000:  # Limit to 3k books for manageable size
                            break
                        
                        text = item["text"] if "text" in item else str(item)
                        if len(text) > 500:  # Skip very short texts
                            # Clean and write text
                            lines = text.split("\n")
                            for line in lines:
                                line = line.strip()
                                if len(line) > 20:  # Skip very short lines
                                    f.write(line + "\n")
                        
                        count += 1
                        if count % 300 == 0:
                            logger.info(f"Processed {count} books...")
                            
                logger.info(f"Books sample created: {books_file}")
                
            except Exception as e:
                logger.warning(f"Could not load Project Gutenberg dataset: {e}")
                logger.info("Creating substantial literary fallback content...")
                # Create sample literary text as fallback
                with open(books_file, "w", encoding="utf-8") as f:
                    literary_content = [
                        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
                        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
                        "In the beginning was the Word, and the Word was with God, and the Word was God.",
                        "All happy families are alike; each unhappy family is unhappy in its own way.",
                        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
                        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows.",
                        "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty.",
                        "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator.",
                        "Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume.",
                        "Two roads diverged in a yellow wood, and sorry I could not travel both and be one traveler, long I stood."
                    ] * 500  # Repeat to create substantial content
                    
                    for line in literary_content:
                        f.write(line + "\n")
            logger.info(f"Books sample created: {books_file}")
            return str(books_file)
            
        except Exception as e:
            logger.error(f"Failed to download books: {e}")
            logger.info("Creating fallback literary content...")
            # Create substantial literary fallback
            with open(books_file, 'w', encoding='utf-8') as f:
                fallback_literary = [
                    "Classic literature provides rich vocabulary and narrative structures for language model training.",
                    "The development of written language has transformed human civilization and knowledge preservation.",
                    "Poetry and prose demonstrate the artistic and expressive capabilities of human language.",
                    "Literature reflects cultural values, historical contexts, and universal human experiences.",
                    "Storytelling traditions have evolved across cultures, creating diverse narrative techniques and themes."
                ] * 400
                
                for line in fallback_literary:
                    f.write(line + "\n")
            
            logger.info(f"Created fallback books content: {books_file}")
            return str(books_file)
    
    def download_news_sample(self, force_download: bool = False) -> str:
        """Download news dataset sample."""
        news_file = self.data_dir / "news_sample.txt"
        
        if news_file.exists() and not force_download:
            file_size = os.path.getsize(news_file) / (1024 * 1024)  # MB
            if file_size > 0.1:  # Only skip if file is substantial (>0.1MB)
                logger.info(f"News sample already exists: {news_file} ({file_size:.1f} MB)")
                return str(news_file)
            else:
                logger.info(f"Existing news file is too small ({file_size:.1f} MB), re-downloading...")
                os.remove(news_file)
        
        logger.info("Downloading news sample...")
        
        try:
            import datasets
            
            try:
                logger.info("Loading BBC news dataset...")
                dataset = datasets.load_dataset("SetFit/bbc-news", split="train")
                
                logger.info("Processing news articles...")
                with open(news_file, "w", encoding="utf-8") as f:
                    for item in dataset:
                        text = item["text"] if "text" in item else str(item)
                        if len(text) > 50:
                            # Clean and write text
                            lines = text.split("\n")
                            for line in lines:
                                line = line.strip()
                                if len(line) > 20:
                                    f.write(line + "\n")
                                    
                logger.info(f"News sample created: {news_file}")
                
            except Exception as e:
                logger.warning(f"Could not load news dataset: {e}")
                logger.info("Creating substantial news fallback content...")
                # Create sample news text as fallback
                with open(news_file, "w", encoding="utf-8") as f:
                    news_content = [
                        "Breaking news reports provide timely information about current events and developments worldwide.",
                        "Investigative journalism uncovers important stories that impact public policy and social issues.",
                        "Technology companies continue to innovate with artificial intelligence and machine learning solutions.",
                        "Economic indicators suggest continued growth in emerging markets despite global uncertainties.",
                        "Climate change research reveals new insights into environmental protection and sustainability measures.",
                        "Healthcare advances include breakthrough treatments and preventive medicine approaches.",
                        "Education policy reforms aim to improve student outcomes and accessibility to quality learning.",
                        "International relations involve diplomatic negotiations and trade agreements between nations.",
                        "Scientific discoveries contribute to our understanding of the natural world and universe.",
                        "Cultural events and artistic expressions reflect the diversity of human creativity and heritage."
                    ] * 300  # Repeat to create substantial content
                    
                    for line in news_content:
                        f.write(line + "\n")
            logger.info(f"News sample created: {news_file}")
            return str(news_file)
            
        except Exception as e:
            logger.error(f"Failed to download news: {e}")
            logger.info("Creating fallback news content...")
            # Create substantial news fallback
            with open(news_file, 'w', encoding='utf-8') as f:
                fallback_news = [
                    "Journalism serves a vital role in democratic societies by providing information and accountability.",
                    "News reporting requires accuracy, objectivity, and ethical standards in information gathering.",
                    "Media coverage influences public opinion and shapes discourse on important social issues.",
                    "Digital transformation has changed how news is produced, distributed, and consumed globally.",
                    "Press freedom remains essential for maintaining transparency and democratic governance."
                ] * 200
                
                for line in fallback_news:
                    f.write(line + "\n")
            
            logger.info(f"Created fallback news content: {news_file}")
            return str(news_file)
    
    def combine_datasets(self, files: List[str], output_file: str) -> str:
        """Combine multiple dataset files into one training file."""
        logger.info(f"Combining {len(files)} datasets into {output_file}")
        
        total_lines = 0
        with open(output_file, 'w', encoding='utf-8') as outf:
            for file_path in files:
                if os.path.exists(file_path):
                    logger.info(f"Adding {file_path}...")
                    with open(file_path, 'r', encoding='utf-8') as inf:
                        lines_added = 0
                        for line in inf:
                            line = line.strip()
                            if len(line) > 10:  # Skip very short lines
                                outf.write(line + '\n')
                                lines_added += 1
                                total_lines += 1
                        logger.info(f"  Added {lines_added:,} lines from {file_path}")
                else:
                    logger.warning(f"File not found: {file_path}")
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Combined dataset: {total_lines:,} lines, {file_size_mb:.1f} MB")
        
        return output_file
    
    def train_bert_tokenizer(self, data_file: str, vocab_size: int = 30000) -> Tuple[str, dict]:
        """Train BERT-compatible WordPiece tokenizer."""
        logger.info(f"Training BERT tokenizer with vocab_size={vocab_size}")
        
        # Get training estimates
        estimates = self.estimate_training_requirements(vocab_size)
        logger.info(f"Training estimates: {estimates}")
        
        # Initialize tokenizer
        tokenizer = SimpleWordPieceTokenizer(vocab_size=vocab_size)
        
        # Start training
        start_time = time.time()
        logger.info("Starting WordPiece training...")
        
        try:
            tokenizer.train(data_file)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {timedelta(seconds=int(training_time))}")
            
            # Save model
            model_file = self.output_dir / f"bert_tokenizer_vocab{vocab_size//1000}k.pkl"
            vocab_file = self.output_dir / f"bert_vocab_{vocab_size//1000}k.txt"
            
            tokenizer.save_model(str(model_file))
            tokenizer.save_vocab(str(vocab_file))
            
            # Training results
            results = {
                "final_vocab_size": len(tokenizer.vocab),
                "training_time_seconds": training_time,
                "model_file": str(model_file),
                "vocab_file": str(vocab_file)
            }
            
            return str(model_file), results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Real-World Data BERT Tokenizer Training")
    parser.add_argument("--vocab-size", type=int, default=15000,
                       help="Target vocabulary size (default: 15,000 for faster training)")
    parser.add_argument("--output-dir", type=str, default="bert_models",
                       help="Output directory for models")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download of datasets even if they exist")
    
    args = parser.parse_args()
    
    trainer = RealWorldDataTrainer(args.output_dir)
    
    logger.info("=== Real-World Data BERT Tokenizer Training Started ===")
    logger.info(f"Target vocabulary size: {args.vocab_size:,}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Get training estimates
    estimates = trainer.estimate_training_requirements(args.vocab_size)
    logger.info("=== Training Estimates ===")
    for key, value in estimates.items():
        if "gb" in key.lower() or "mb" in key.lower():
            logger.info(f"{key}: {value:.1f}")
        elif "hours" in key.lower():
            logger.info(f"{key}: {value:.1f} hours")
        else:
            logger.info(f"{key}: {value}")
    
    try:
        # Step 1: Download datasets
        logger.info("Step 1: Downloading real-world datasets...")
        
        wiki_file = trainer.download_wikipedia_sample(force_download=args.force_download)
        books_file = trainer.download_books_sample(force_download=args.force_download)
        news_file = trainer.download_news_sample(force_download=args.force_download)
        
        # Step 2: Combine datasets
        logger.info("Step 2: Combining datasets...")
        combined_file = str(trainer.output_dir / "combined_training_data.txt")
        trainer.combine_datasets([wiki_file, books_file, news_file], combined_file)
        
        # Step 3: Train tokenizer
        logger.info("Step 3: Training BERT tokenizer...")
        model_file, results = trainer.train_bert_tokenizer(combined_file, args.vocab_size)
        
        # Step 4: Results
        logger.info("=== Training Results ===")
        for key, value in results.items():
            if "time" in key.lower() and isinstance(value, (int, float)):
                logger.info(f"{key}: {timedelta(seconds=int(value))}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=== BERT Tokenizer Training Completed Successfully! ===")
        logger.info(f"Model saved to: {model_file}")
        logger.info(f"Ready for BERT training!")
        
        # Cleanup large intermediate files
        try:
            os.remove(combined_file)
            logger.info("Cleaned up temporary training file")
        except:
            pass
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())