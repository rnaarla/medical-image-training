#!/bin/bash
# Quick setup script for local development

set -e

echo "🚀 Setting up Medical Image Training Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "medical_training_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv medical_training_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source medical_training_env/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install basic dependencies first
echo "📦 Installing basic dependencies..."
pip install torch torchvision torchaudio

# Install other requirements
echo "📦 Installing additional requirements..."
pip install -r requirements.txt

echo "🧪 Testing installation..."

# Test PyTorch CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test imports
python3 -c "from src.core.model import get_model; print('✅ Model module working')"
python3 -c "from src.core.metrics import MetricsTracker; print('✅ Metrics module working')"

echo "🎉 Environment setup complete!"
echo "📋 Next steps:"
echo "   1. source medical_training_env/bin/activate"
echo "   2. python3 setup_custom_kernel.py build_ext --inplace"
echo "   3. python3 grayscale_wrapper.py"
echo "   4. python3 train.py --help"
