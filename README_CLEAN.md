# 🤖 Adaptive BERT Training System

An intelligent BERT training system that **automatically adjusts resource usage** based on other Python processes running on your system. When your other coding projects need resources, BERT training scales down. When resources are available, BERT scales up.

## 🎯 Key Features

- **Dynamic Resource Monitoring** - Tracks other Python processes in real-time
- **Adaptive Resource Allocation** - Automatically adjusts CPU, memory, batch size
- **Interactive Control** - Real-time adjustments and monitoring
- **Priority System** - Your other coding work always gets priority
- **Zero Configuration** - Works out of the box with smart defaults

## 🚀 Quick Start

```bash
# Option 1: Launch with GUI
scripts\quick_start.bat

# Option 2: Direct Python launch  
python scripts\launch_adaptive_bert.py

# Option 3: Monitor only (no training)
python src\adaptive_resource_manager.py
```

## 📊 How It Works

The system continuously monitors:
- **Other Python processes** CPU and memory usage
- **System resources** availability
- **BERT training** performance metrics

Based on this data, it dynamically adjusts:
- **CPU cores**: 1-8 cores based on availability  
- **Memory limits**: 1-8GB adaptive scaling
- **Batch size**: 8-32 adaptive sizing
- **Learning rate**: Optimized for current resources

## 📁 Project Structure

```
bert/
├── src/                    # Core adaptive system
│   ├── adaptive_resource_manager.py   # Resource monitoring
│   ├── adaptive_bert_trainer.py       # Adaptive trainer
│   ├── run_adaptive_bert.py          # Interactive controller
│   ├── bert-wordpiece-tokenizer/     # Tokenizer implementation
│   └── examples/                     # Usage examples
├── scripts/               # Launch scripts
│   ├── quick_start.bat             # Easy launcher
│   └── launch_adaptive_bert.py     # Python launcher  
├── models/               # Trained models
│   ├── 50k/             # 50k vocab models
│   ├── 15k/             # 15k vocab models
│   └── 100k/            # 100k vocab models
├── data/                # Training data
└── archive/             # Old files (can be deleted)
```

## 🎮 Interactive Controls

When you run the system, you get an interactive menu:

1. **Start adaptive training** - Begin training with resource monitoring
2. **Show resource status** - Real-time system status
3. **Show resource history** - Historical resource usage
4. **Pause/Resume training** - Manual control
5. **Adjust resource limits** - Customize limits
6. **Monitor only** - Watch resources without training
7. **Exit** - Clean shutdown

## 📈 Example Resource Adaptation

```
Other Python CPU < 10%  → BERT gets 4-8 cores, 4-8GB, batch=32
Other Python CPU 10-30% → BERT gets 2-4 cores, 2-4GB, batch=16  
Other Python CPU > 30%  → BERT gets 1-2 cores, 1-2GB, batch=8
```

## 💡 Usage Examples

### Basic Training
```python
from src.adaptive_bert_trainer import AdaptiveBERTTrainer

trainer = AdaptiveBERTTrainer()
trainer.prepare_data(["Your training texts..."])
trainer.train(num_epochs=5)
```

### Custom Resource Limits
```python  
from src.adaptive_resource_manager import AdaptiveResourceManager

manager = AdaptiveResourceManager(
    max_cpu_cores=4,
    max_memory_gb=6.0,
    monitor_interval=1.0
)
manager.start_monitoring()
```

## 🔧 Configuration

The system uses smart defaults but can be customized:

- **CPU Limits**: Set min/max cores available to BERT
- **Memory Limits**: Set min/max GB available to BERT  
- **Monitor Interval**: How often to check resources (default: 2 seconds)
- **Training Parameters**: Batch size, learning rate adaptation rules

## 📝 Dependencies

```
torch
transformers  
psutil
```

Install with: `pip install -r requirements.txt`

## 🧹 Cleanup Complete

The repository has been reorganized:
- ✅ **20+ redundant training scripts** moved to `archive/`
- ✅ **Model files organized** by vocabulary size
- ✅ **Core adaptive system** in clean `src/` structure
- ✅ **Easy launch scripts** in `scripts/`
- ✅ **Data files** organized in `data/`

You can safely delete the `archive/` folder if you don't need the old files.

## 🚀 Next Steps

1. Run `scripts\quick_start.bat` to try the system
2. Monitor how it adapts to your other processes
3. Customize resource limits as needed
4. Train your BERT models efficiently!

The system now prioritizes your other coding work while maximizing BERT training efficiency. 🎯