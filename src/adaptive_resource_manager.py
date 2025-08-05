import psutil
import torch
import threading
import time
import os
import gc
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ResourceState:
    other_python_cpu: float
    other_python_memory: float
    system_cpu: float
    system_memory: float
    available_cpu: float
    available_memory: float

class AdaptiveResourceManager:
    def __init__(self, 
                 bert_process_name: str = "bert",
                 min_cpu_cores: int = 1,
                 max_cpu_cores: int = None,
                 min_memory_gb: float = 1.0,
                 max_memory_gb: float = 8.0,
                 monitor_interval: float = 2.0):
        
        self.bert_process_name = bert_process_name
        self.min_cpu_cores = min_cpu_cores
        self.max_cpu_cores = max_cpu_cores or psutil.cpu_count()
        self.min_memory_gb = min_memory_gb
        self.max_memory_gb = max_memory_gb
        self.monitor_interval = monitor_interval
        
        self.current_pid = os.getpid()
        self.current_process = psutil.Process(self.current_pid)
        
        # Resource tracking
        self.resource_history = deque(maxlen=10)
        self.current_limits = {
            'cpu_cores': min_cpu_cores,
            'memory_gb': min_memory_gb,
            'batch_size': 8,
            'learning_rate': 1e-5
        }
        
        # Control flags
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def add_callback(self, callback):
        """Add callback function to be called when resources change"""
        self.callbacks.append(callback)
        
    def get_other_python_processes(self) -> List[psutil.Process]:
        """Get all Python processes except current BERT process"""
        other_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if (proc.info['name'] == 'python.exe' and 
                    proc.info['pid'] != self.current_pid):
                    other_processes.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return other_processes
    
    def analyze_system_state(self) -> ResourceState:
        """Analyze current system resource usage"""
        other_processes = self.get_other_python_processes()
        
        # Calculate other Python processes usage
        other_cpu = sum(proc.cpu_percent() for proc in other_processes)
        other_memory = sum(proc.memory_info().rss for proc in other_processes) / (1024**3)  # GB
        
        # System-wide usage
        system_cpu = psutil.cpu_percent(interval=0.1)
        system_memory = psutil.virtual_memory().percent
        
        # Available resources (conservative estimation)
        available_cpu = max(0, 80 - system_cpu - other_cpu)  # Keep 20% buffer
        available_memory = max(0, (psutil.virtual_memory().total / (1024**3)) * 0.8 - other_memory)
        
        return ResourceState(
            other_python_cpu=other_cpu,
            other_python_memory=other_memory,
            system_cpu=system_cpu,
            system_memory=system_memory,
            available_cpu=available_cpu,
            available_memory=available_memory
        )
    
    def calculate_optimal_resources(self, state: ResourceState) -> dict:
        """Calculate optimal resource allocation based on current state"""
        # Adaptive CPU cores
        if state.other_python_cpu < 10:  # Other processes using < 10% CPU
            optimal_cores = min(self.max_cpu_cores, max(4, int(state.available_cpu / 20)))
        elif state.other_python_cpu < 30:  # Moderate usage
            optimal_cores = min(4, max(2, int(state.available_cpu / 25)))
        else:  # High usage by other processes
            optimal_cores = max(1, min(2, int(state.available_cpu / 30)))
        
        # Adaptive Memory
        if state.other_python_memory < 2:  # Other processes using < 2GB
            optimal_memory = min(self.max_memory_gb, max(4, state.available_memory * 0.6))
        elif state.other_python_memory < 4:  # Moderate usage
            optimal_memory = min(4, max(2, state.available_memory * 0.4))
        else:  # High usage
            optimal_memory = max(self.min_memory_gb, min(2, state.available_memory * 0.3))
        
        # Adaptive batch size and learning rate
        resource_ratio = (optimal_cores / self.max_cpu_cores + optimal_memory / self.max_memory_gb) / 2
        
        if resource_ratio > 0.7:  # High resources available
            batch_size = 32
            learning_rate = 2e-5
        elif resource_ratio > 0.4:  # Medium resources
            batch_size = 16
            learning_rate = 1.5e-5
        else:  # Low resources
            batch_size = 8
            learning_rate = 1e-5
        
        return {
            'cpu_cores': optimal_cores,
            'memory_gb': optimal_memory,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'resource_ratio': resource_ratio
        }
    
    def apply_resource_limits(self, limits: dict):
        """Apply resource limits to current process"""
        try:
            # Set CPU affinity
            available_cores = list(range(min(limits['cpu_cores'], psutil.cpu_count())))
            self.current_process.cpu_affinity(available_cores)
            
            # Set process priority based on resource availability
            if limits['resource_ratio'] > 0.6:
                priority = psutil.NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
            else:
                priority = psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10
            
            self.current_process.nice(priority)
            
            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return True
        except Exception as e:
            print(f"Failed to apply resource limits: {e}")
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("🔄 Starting adaptive resource monitoring...")
        
        while self.monitoring:
            try:
                # Analyze current state
                state = self.analyze_system_state()
                optimal_limits = self.calculate_optimal_resources(state)
                
                # Check if limits need updating
                if self._should_update_limits(optimal_limits):
                    print(f"📊 Resource adjustment:")
                    print(f"   Other Python CPU: {state.other_python_cpu:.1f}%")
                    print(f"   Other Python Memory: {state.other_python_memory:.1f}GB")
                    print(f"   → BERT CPU Cores: {optimal_limits['cpu_cores']}")
                    print(f"   → BERT Memory Limit: {optimal_limits['memory_gb']:.1f}GB")
                    print(f"   → Batch Size: {optimal_limits['batch_size']}")
                    print(f"   → Learning Rate: {optimal_limits['learning_rate']:.2e}")
                    
                    # Apply new limits
                    if self.apply_resource_limits(optimal_limits):
                        self.current_limits = optimal_limits.copy()
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                callback(optimal_limits)
                            except Exception as e:
                                print(f"Callback error: {e}")
                
                # Store history
                self.resource_history.append((time.time(), state, optimal_limits))
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(self.monitor_interval)
    
    def _should_update_limits(self, new_limits: dict) -> bool:
        """Check if limits should be updated (avoid thrashing)"""
        if not self.current_limits:
            return True
        
        # Only update if significant change
        cpu_change = abs(new_limits['cpu_cores'] - self.current_limits['cpu_cores'])
        memory_change = abs(new_limits['memory_gb'] - self.current_limits['memory_gb'])
        batch_change = abs(new_limits['batch_size'] - self.current_limits['batch_size'])
        
        return cpu_change >= 1 or memory_change >= 0.5 or batch_change >= 4
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Adaptive resource manager started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Adaptive resource manager stopped")
    
    def get_current_limits(self) -> dict:
        """Get current resource limits"""
        return self.current_limits.copy()
    
    def get_status(self) -> str:
        """Get current status as formatted string"""
        if not self.monitoring:
            return "❌ Not monitoring"
        
        limits = self.current_limits
        return f"""
🎯 Current BERT Resource Allocation:
   CPU Cores: {limits['cpu_cores']}/{self.max_cpu_cores}
   Memory Limit: {limits['memory_gb']:.1f}GB
   Batch Size: {limits['batch_size']}
   Learning Rate: {limits['learning_rate']:.2e}
   Status: {'🟢 High' if limits.get('resource_ratio', 0) > 0.6 else '🟡 Medium' if limits.get('resource_ratio', 0) > 0.3 else '🔴 Low'} resources
        """.strip()

# Global instance for easy access
adaptive_manager = AdaptiveResourceManager()

if __name__ == "__main__":
    # Test the adaptive manager
    manager = AdaptiveResourceManager()
    
    def on_resource_change(limits):
        print(f"🔄 Resources updated: {limits}")
    
    manager.add_callback(on_resource_change)
    manager.start_monitoring()
    
    try:
        # Keep running and show status every 10 seconds
        while True:
            time.sleep(10)
            print(manager.get_status())
    except KeyboardInterrupt:
        manager.stop_monitoring()
        print("👋 Stopped monitoring")