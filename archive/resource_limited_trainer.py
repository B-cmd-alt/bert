import os
import psutil
import torch
import gc
from torch.utils.data import DataLoader

class ResourceManager:
    def __init__(self, max_cpu_percent=30, max_memory_gb=4):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        
    def limit_resources(self):
        # Set process priority to below normal
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
        
        # Limit CPU affinity to fewer cores
        cpu_count = psutil.cpu_count()
        limited_cores = max(1, cpu_count // 2)  # Use only half the cores
        self.process.cpu_affinity(list(range(limited_cores)))
        
    def check_resources(self):
        memory_usage = self.process.memory_info().rss
        cpu_usage = self.process.cpu_percent()
        
        if memory_usage > self.max_memory_bytes:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return memory_usage < self.max_memory_bytes and cpu_usage < self.max_cpu_percent

# Usage in your training scripts
def train_with_resource_limits():
    rm = ResourceManager(max_cpu_percent=25, max_memory_gb=3)
    rm.limit_resources()
    
    # Your existing training code here
    # Add periodic resource checks
    for epoch in range(epochs):
        if not rm.check_resources():
            print("Resource limit exceeded, pausing training...")
            time.sleep(5)
            continue
        # ... training code ...

if __name__ == "__main__":
    train_with_resource_limits()