#!/bin/bash
# setup_env.sh - Environment setup for ASP Safety Layer project
set -e

echo "=========================================="
echo "ASP Safety Layer - Environment Setup"
echo "=========================================="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not installed"
    echo "Install from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "[1/4] Creating conda environment 'asp-safety'..."
conda create -n asp-safety python=3.10 -y

# Activate
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asp-safety

# Install PyTorch
echo "[3/4] Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other packages
echo "[4/4] Installing remaining packages..."
pip install clingo==5.7.1
pip install nuscenes-devkit
pip install numpy scipy pandas matplotlib seaborn
pip install pytest black flake8 mypy

# Verify
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import clingo; print(f'Clingo: {clingo.__version__}')"
python -c "import nuscenes; print('nuScenes: OK')"

echo ""
echo "Setup complete! Run: conda activate asp-safety"
