#!/bin/bash
# setup_env.sh - Environment setup for ASP Safety Layer project
# Run this on each team member's laptop

set -e  # Exit on error

echo "=========================================="
echo "ASP Safety Layer - Environment Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed."
    exit 1
fi

# Create conda environment
echo ""
echo "[1/6] Creating conda environment 'asp-safety' with Python 3.10..."
conda create -n asp-safety python=3.10 -y

# Activate environment
echo ""
echo "[2/6] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asp-safety

# Install PyTorch with CUDA 11.8
echo ""
echo "[3/6] Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install ASP Solver
echo ""
echo "[4/6] Installing Clingo 5.7.1 (ASP Solver)..."
pip install clingo==5.7.1

# Install nuScenes devkit and scientific computing
echo ""
echo "[5/6] Installing nuScenes devkit and scientific packages..."
pip install nuscenes-devkit
pip install numpy scipy pandas matplotlib seaborn

# Install development tools
echo ""
echo "[6/6] Installing development tools..."
pip install pytest black flake8 mypy

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

echo ""
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

echo ""
echo "Checking Clingo..."
python -c "import clingo; print(f'Clingo: {clingo.__version__}')"

echo ""
echo "Checking nuScenes devkit..."
python -c "import nuscenes; print('nuScenes devkit: OK')"

echo ""
echo "Checking scientific packages..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  conda activate asp-safety"
echo ""
echo "Next steps:"
echo "  1. Clone SparseDrive: git clone https://github.com/swc-17/SparseDrive.git"
echo "  2. Download nuScenes mini: https://www.nuscenes.org/download"
echo "  3. See IMPLEMENTATION_PLAN.md for detailed tasks"
