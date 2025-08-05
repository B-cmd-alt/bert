#!/usr/bin/env python3
"""
Launch script for the Adaptive BERT Training System
This script can be run from the root directory
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from run_adaptive_bert import InteractiveController

if __name__ == "__main__":
    controller = InteractiveController()
    controller.run()