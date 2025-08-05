#!/usr/bin/env python3
"""
Memory Optimization Utility for BERT Training
Provides tools to monitor and optimize memory usage during training.
"""

import os
import gc
import psutil
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Utility class for optimizing memory usage during BERT training."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_gb: Maximum memory usage allowed in GB
        """
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_system_memory_info(self) -> dict:
        """Get system memory information."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / 1024 / 1024 / 1024,
            'available_gb': mem.available / 1024 / 1024 / 1024,
            'used_percent': mem.percent,
            'process_mb': self.get_memory_usage_mb()
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        current_gb = self.get_memory_usage_mb() / 1024
        return current_gb > self.max_memory_gb
    
    def force_garbage_collection(self) -> int:
        """Force garbage collection and return freed objects count."""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        return collected
    
    def optimize_memory(self) -> None:
        """Perform memory optimization operations."""
        if self.check_memory_limit():
            logger.warning(f"Memory usage exceeds {self.max_memory_gb}GB limit")
            self.force_garbage_collection()
    
    def log_memory_stats(self) -> None:
        """Log current memory statistics."""
        info = self.get_system_memory_info()
        logger.info(f"Memory - Process: {info['process_mb']:.1f}MB, "
                   f"System: {info['used_percent']:.1f}% used, "
                   f"{info['available_gb']:.1f}GB available")
    
    def cleanup_large_files(self, directory: str, 
                          size_threshold_mb: float = 500.0,
                          extensions: List[str] = ['.txt', '.log']) -> None:
        """
        Clean up large files that may be consuming disk space.
        
        Args:
            directory: Directory to scan
            size_threshold_mb: Files larger than this will be flagged
            extensions: File extensions to check
        """
        dir_path = Path(directory)
        large_files = []
        
        for ext in extensions:
            for file_path in dir_path.glob(f'*{ext}'):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    if size_mb > size_threshold_mb:
                        large_files.append((file_path, size_mb))
        
        if large_files:
            logger.info("Large files found:")
            for file_path, size_mb in sorted(large_files, key=lambda x: x[1], reverse=True):
                logger.info(f"  {file_path.name}: {size_mb:.1f}MB")
        
        return large_files

def optimize_training_data_size(input_file: str, output_file: str, 
                               target_size_mb: float = 100.0) -> bool:
    """
    Create a smaller version of training data for memory-constrained environments.
    
    Args:
        input_file: Path to original training data
        output_file: Path to output optimized data
        target_size_mb: Target size in MB
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        target_bytes = target_size_mb * 1024 * 1024
        written_bytes = 0
        
        logger.info(f"Creating optimized training data: {target_size_mb}MB target")
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile):
                    if written_bytes >= target_bytes:
                        break
                    
                    # Sample every 5th line for diversity while keeping high quality
                    if line_num < 10000 or line_num % 5 == 0:
                        outfile.write(line)
                        written_bytes += len(line.encode('utf-8'))
                    
                    # Log progress
                    if line_num % 50000 == 0:
                        logger.info(f"Processed {line_num:,} lines, "
                                  f"{written_bytes/(1024*1024):.1f}MB written")
        
        actual_size_mb = written_bytes / (1024 * 1024)
        logger.info(f"Optimized data created: {output_file} ({actual_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize training data: {e}")
        return False

def cleanup_old_models(base_dir: str = ".", days_old: int = 7) -> None:
    """
    Clean up old model files to free disk space.
    
    Args:
        base_dir: Base directory to search
        days_old: Remove files older than this many days
    """
    import time
    from datetime import datetime, timedelta
    
    cutoff_time = time.time() - (days_old * 24 * 3600)
    base_path = Path(base_dir)
    
    model_dirs = ['bert_models', 'bert_50k_models', 'enhanced_models', 'large_models']
    removed_files = []
    
    for model_dir in model_dirs:
        dir_path = base_path / model_dir
        if dir_path.exists():
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_size_mb = file_path.stat().st_size / 1024 / 1024
                        file_path.unlink()
                        removed_files.append((str(file_path), file_size_mb))
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
    
    if removed_files:
        total_mb = sum(size for _, size in removed_files)
        logger.info(f"Cleaned up {len(removed_files)} old files, freed {total_mb:.1f}MB")
        for file_path, size_mb in removed_files:
            logger.info(f"  Removed: {file_path} ({size_mb:.1f}MB)")
    else:
        logger.info("No old model files found to clean up")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    optimizer = MemoryOptimizer(max_memory_gb=4.0)
    optimizer.log_memory_stats()
    
    # Check for large files
    large_files = optimizer.cleanup_large_files(".")
    
    # Clean up old models
    cleanup_old_models()
    
    # Optimize large training data if it exists
    if os.path.exists("enhanced_training_data_4.0gb.txt"):
        optimize_training_data_size(
            "enhanced_training_data_4.0gb.txt",
            "optimized_training_data.txt",
            target_size_mb=200.0
        )