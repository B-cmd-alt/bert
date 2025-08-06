@echo off
echo Setting up Modern BERT Environment
echo ================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python is available
echo.

REM Install required packages
echo Installing required packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install tokenizers>=0.14.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install evaluate>=0.4.0

echo.
echo Installation complete!
echo.

REM Run the demo
echo Starting Modern BERT demo...
python run_modern_bert.py

pause