#!/usr/bin/env python3
"""
Interactive Adaptive BERT Training Controller
Monitors other Python processes and adjusts BERT resources dynamically
"""

import sys
import time
import signal
import threading
from pathlib import Path
from .adaptive_resource_manager import adaptive_manager
from .adaptive_bert_trainer import AdaptiveBERTTrainer

class InteractiveController:
    def __init__(self):
        self.trainer = None
        self.running = True
        self.training_thread = None
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Shutting down gracefully...")
        self.running = False
        adaptive_manager.stop_monitoring()
        if self.trainer:
            self.trainer.training_active = False
        sys.exit(0)
    
    def show_menu(self):
        """Display interactive menu"""
        print("\n" + "="*60)
        print("🤖 Adaptive BERT Training Controller")
        print("="*60)
        print("1. Start adaptive training")
        print("2. Show resource status")
        print("3. Show resource history")
        print("4. Pause/Resume training")
        print("5. Adjust resource limits")
        print("6. Monitor only (no training)")
        print("7. Exit")
        print("="*60)
    
    def show_resource_status(self):
        """Display current resource status"""
        print("\n📊 Current System Status:")
        print("-" * 40)
        
        # Get system state
        state = adaptive_manager.analyze_system_state()
        print(f"Other Python CPU Usage: {state.other_python_cpu:.1f}%")
        print(f"Other Python Memory: {state.other_python_memory:.1f}GB")
        print(f"System CPU: {state.system_cpu:.1f}%")
        print(f"System Memory: {state.system_memory:.1f}%")
        print(f"Available CPU: {state.available_cpu:.1f}%")
        print(f"Available Memory: {state.available_memory:.1f}GB")
        
        print(adaptive_manager.get_status())
    
    def show_resource_history(self):
        """Show recent resource history"""
        print("\n📈 Resource History (Last 10 readings):")
        print("-" * 60)
        
        if not adaptive_manager.resource_history:
            print("No history available yet.")
            return
        
        for timestamp, state, limits in list(adaptive_manager.resource_history)[-5:]:
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            print(f"{time_str} | Other CPU: {state.other_python_cpu:4.1f}% | "
                  f"BERT Cores: {limits['cpu_cores']} | "
                  f"BERT Memory: {limits['memory_gb']:.1f}GB | "
                  f"Batch: {limits['batch_size']}")
    
    def adjust_resource_limits(self):
        """Allow user to adjust resource limits"""
        print("\n⚙️ Adjust Resource Limits:")
        print(f"Current limits:")
        print(f"  Min CPU cores: {adaptive_manager.min_cpu_cores}")
        print(f"  Max CPU cores: {adaptive_manager.max_cpu_cores}")
        print(f"  Min Memory: {adaptive_manager.min_memory_gb}GB")
        print(f"  Max Memory: {adaptive_manager.max_memory_gb}GB")
        
        try:
            new_max_cpu = input(f"New max CPU cores ({adaptive_manager.max_cpu_cores}): ").strip()
            if new_max_cpu:
                adaptive_manager.max_cpu_cores = int(new_max_cpu)
            
            new_max_mem = input(f"New max memory GB ({adaptive_manager.max_memory_gb}): ").strip()
            if new_max_mem:
                adaptive_manager.max_memory_gb = float(new_max_mem)
            
            print("✅ Limits updated!")
        except ValueError:
            print("❌ Invalid input. Limits unchanged.")
    
    def start_training(self):
        """Start adaptive training"""
        if self.training_thread and self.training_thread.is_alive():
            print("⚠️ Training already running!")
            return
        
        print("🚀 Starting adaptive BERT training...")
        
        # Create trainer
        self.trainer = AdaptiveBERTTrainer()
        
        # Load or create sample data
        data_file = Path("training_data.txt")
        if data_file.exists():
            print(f"📂 Loading training data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print("📝 Using sample training data")
            texts = [
                "The adaptive BERT trainer automatically adjusts resources.",
                "Machine learning models benefit from dynamic resource allocation.",
                "System performance optimization improves overall efficiency.",
                "Natural language processing requires computational resources.",
                "Deep learning models adapt to available system capacity."
            ] * 50  # Create more samples
        
        # Prepare data
        self.trainer.prepare_data(texts)
        
        # Start training in separate thread
        def training_worker():
            try:
                self.trainer.train(num_epochs=10)
            except Exception as e:
                print(f"❌ Training error: {e}")
        
        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()
        print("✅ Training started in background")
    
    def monitor_only(self):
        """Start monitoring without training"""
        print("👁️ Starting resource monitoring...")
        adaptive_manager.start_monitoring()
        
        try:
            while self.running:
                print("\n" + "="*50)
                print(f"⏰ {time.strftime('%H:%M:%S')}")
                self.show_resource_status()
                print("Press Ctrl+C to stop monitoring")
                time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def run(self):
        """Main interactive loop"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🎯 Welcome to Adaptive BERT Training!")
        print("This system automatically adjusts BERT training resources")
        print("based on other Python processes running on your system.")
        
        # Start resource monitoring
        adaptive_manager.start_monitoring()
        
        while self.running:
            try:
                self.show_menu()
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "1":
                    self.start_training()
                elif choice == "2":
                    self.show_resource_status()
                elif choice == "3":
                    self.show_resource_history()
                elif choice == "4":
                    if self.trainer:
                        self.trainer.training_active = not self.trainer.training_active
                        status = "resumed" if self.trainer.training_active else "paused"
                        print(f"✅ Training {status}")
                    else:
                        print("⚠️ No training session active")
                elif choice == "5":
                    self.adjust_resource_limits()
                elif choice == "6":
                    self.monitor_only()
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                
                if choice not in ["6", "7"]:  # Don't pause for monitor-only or exit
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")
        
        # Cleanup
        adaptive_manager.stop_monitoring()
        print("🏁 Controller stopped")

if __name__ == "__main__":
    controller = InteractiveController()
    controller.run()