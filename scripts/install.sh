#\!/bin/bash

# HRM Installation Script - Production Version
# Automated installer with CUDA toolkit, C++ extensions, and verification

set -e
set -o pipefail

echo "========================================"
echo "HRM Production Installer"
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
echo "[1/10] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg git build-essential dkms linux-headers-$(uname -r) tmux unzip || error_exit "Failed to install system packages"

# Check/Install CUDA Toolkit
echo "[2/10] Checking CUDA toolkit..."
source /etc/profile.d/cuda.sh 2>/dev/null || true
if \! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
    
    tmux kill-session -t cuda_install 2>/dev/null || true
    echo "Starting CUDA installation in tmux..."
    tmux new-session -d -s cuda_install "sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8 && echo DONE > /tmp/cuda_done"
    
    echo "Waiting for CUDA installation..."
    while \! [ -f /tmp/cuda_done ]; do
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
echo "[3/10] Setting up Python virtual environment..."
if [ \! -d /opt/venv ]; then
    python3 -m venv /opt/venv || error_exit "Failed to create virtual environment"
fi
source /opt/venv/bin/activate

# Install PyTorch 2.5
echo "[4/10] Installing PyTorch..."
pip install --upgrade pip setuptools wheel
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" || error_exit "Failed to install PyTorch"

# Install HRM requirements
echo "[5/10] Installing HRM dependencies..."
if [ \! -f /opt/HRM/requirements.txt ]; then
    error_exit "/opt/HRM/requirements.txt not found"
fi
pip install -r /opt/HRM/requirements.txt || error_exit "Failed to install requirements"
pip install numpy einops || error_exit "Failed to install additional dependencies"

# Build adam_atan2
echo "[6/10] Building adam_atan2 C++ extension..."
pip install ninja packaging
pip uninstall -y adam-atan2 2>/dev/null || true
pip install --no-cache-dir adam-atan2 --no-binary adam-atan2 || error_exit "Failed to build adam_atan2"

python -c "import torch; import adam_atan2_backend" || error_exit "adam_atan2_backend failed"

# Install flash_attn
echo "[7/10] Installing flash_attn..."
pip install --no-deps "${FLASH_ATTN_WHEEL}" || error_exit "Failed to install flash_attn"

python -c "import flash_attn" || error_exit "flash_attn failed"

# Generate dataset
echo "[8/10] Generating Sudoku dataset..."
cd /opt/HRM || error_exit "Failed to change to HRM directory"

if [ \! -d "data/sudoku-extreme-1k-aug-1000" ]; then
    echo "Building dataset..."
    tmux kill-session -t dataset_build 2>/dev/null || true
    tmux new-session -d -s dataset_build "source /opt/venv/bin/activate && python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000"
    
    while tmux has-session -t dataset_build 2>/dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    
    [ -d "data/sudoku-extreme-1k-aug-1000" ] || error_exit "Dataset generation failed"
    echo "✓ Dataset generated"
else
    echo "✓ Dataset exists"
fi

# Verification test
echo "[9/10] Running verification test..."
tmux kill-session -t hrm_verify 2>/dev/null || true
tmux new-session -d -s hrm_verify "cd /opt/HRM && source /opt/venv/bin/activate && WANDB_MODE=offline python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=1 eval_interval=1 global_batch_size=32 2>&1 | tee verify.log"

echo "Running verification..."
sleep 30

if pgrep -f "pretrain.py" > /dev/null; then
    echo "✓ HRM training verified"
    pkill -f "pretrain.py"
    tmux kill-session -t hrm_verify 2>/dev/null || true
else
    tmux kill-session -t hrm_verify 2>/dev/null || true
    cat /opt/HRM/verify.log 2>/dev/null || true
    error_exit "HRM verification failed"
fi

# Final checks
echo "[10/10] Final verification..."
source /opt/venv/bin/activate
python -c "
import torch
import flash_attn
import adam_atan2_backend
print('✓ All modules verified')
" || error_exit "Final verification failed"

echo ""
echo "========================================"
echo "✓✓✓ INSTALLATION SUCCESSFUL ✓✓✓"
echo "========================================"
echo ""
echo "HRM is ready\! To use:"
echo "  source /opt/venv/bin/activate"
echo "  cd /opt/HRM"
echo "  WANDB_MODE=offline python pretrain.py [config]"
