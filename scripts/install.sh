#\!/bin/bash

# HRM Installation Script - Production Version with Fast Health Check
# Automated installer with CUDA toolkit, C++ extensions, and verification

set -e
set -o pipefail

echo "========================================"
echo "HRM Production Installer v2"
echo "========================================"

# Configuration
CUDA_VERSION="12.8"
TORCH_VERSION="2.5.*"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Error handler
error_exit() {
    echo ""
    echo "\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!"
    echo "CATASTROPHIC FAILURE: $1"
    echo "Installation FAILED"
    echo "\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!"
    exit 1
}

# System dependencies
echo "[1/9] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg git build-essential dkms linux-headers-$(uname -r) tmux unzip || error_exit "Failed to install system packages"

# Check/Install CUDA Toolkit
echo "[2/9] Checking CUDA toolkit..."
source /etc/profile.d/cuda.sh 2>/dev/null || true
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
    
    tmux kill-session -t cuda_install 2>/dev/null || true
    echo "Starting CUDA installation in tmux (5-10 minutes)..."
    tmux new-session -d -s cuda_install "sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8 && echo DONE > /tmp/cuda_done"
    
    echo "Waiting for CUDA installation..."
    while ! [ -f /tmp/cuda_done ]; do
        sleep 10
        echo -n "."
    done
    echo ""
    rm -f /tmp/cuda_done
    tmux kill-session -t cuda_install 2>/dev/null || true
    
    echo 'export CUDA_HOME=/usr/local/cuda-12.8' | sudo tee /etc/profile.d/cuda.sh > /dev/null
    echo 'export PATH=$CUDA_HOME/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh > /dev/null
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}' | sudo tee -a /etc/profile.d/cuda.sh > /dev/null
    source /etc/profile.d/cuda.sh
    
    nvcc -V || error_exit "NVCC not found after CUDA installation"
    echo "✓ CUDA toolkit installed"
else
    echo "✓ CUDA toolkit already installed"
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.9"

# Create virtual environment
echo "[3/9] Setting up Python virtual environment..."
if [ ! -d /opt/venv ]; then
    python3 -m venv /opt/venv || error_exit "Failed to create virtual environment"
fi
source /opt/venv/bin/activate

# Install PyTorch 2.5
echo "[4/9] Installing PyTorch..."
pip install --upgrade pip setuptools wheel
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" || error_exit "Failed to install PyTorch"

# Install HRM requirements
echo "[5/9] Installing HRM dependencies..."
if [ ! -f /opt/HRM/requirements.txt ]; then
    error_exit "/opt/HRM/requirements.txt not found"
fi
pip install -r /opt/HRM/requirements.txt || error_exit "Failed to install requirements"
pip install numpy einops || error_exit "Failed to install additional dependencies"

# Build adam_atan2
echo "[6/9] Building adam_atan2 C++ extension..."
pip install ninja packaging
pip uninstall -y adam-atan2 2>/dev/null || true
pip install --no-cache-dir adam-atan2 --no-binary adam-atan2 || error_exit "Failed to build adam_atan2"

# Install flash_attn
echo "[7/9] Installing flash_attn..."
pip install --no-deps "${FLASH_ATTN_WHEEL}" || error_exit "Failed to install flash_attn"

# Generate dataset (optional, for full testing)

# Run health check
echo "[8/8] Running health check..."
cd /opt/linode-install/scripts || error_exit "Failed to change to scripts directory"
python3 healthcheck.py || error_exit "Health check failed"

echo "========================================"
echo "Installation completed successfully\!"
echo "========================================"
echo ""
echo "HRM has been installed with:"
echo "  - PyTorch 2.5 with CUDA 12.4"
echo "  - adam_atan2 C++ extension"
echo "  - flash_attn 2.8.2"
echo ""
echo "Health check passed successfully."
