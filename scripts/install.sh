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
if \! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
    
    tmux kill-session -t cuda_install 2>/dev/null || true
    echo "Starting CUDA installation in tmux (5-10 minutes)..."
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
echo "[3/9] Setting up Python virtual environment..."
if [ \! -d /opt/venv ]; then
    python3 -m venv /opt/venv || error_exit "Failed to create virtual environment"
fi
source /opt/venv/bin/activate

# Install PyTorch 2.5
echo "[4/9] Installing PyTorch..."
pip install --upgrade pip setuptools wheel
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" || error_exit "Failed to install PyTorch"

# Install HRM requirements
echo "[5/9] Installing HRM dependencies..."
if [ \! -f /opt/HRM/requirements.txt ]; then
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
echo "[8/9] Generating Sudoku dataset..."
cd /opt/HRM || error_exit "Failed to change to HRM directory"

if [ \! -d "data/sudoku-extreme-1k-aug-1000" ]; then
    echo "Building dataset (1-2 minutes)..."
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

# Fast health check instead of long demo
echo "[9/9] Running health check (2 seconds)..."

# Create healthcheck script
cat > /opt/HRM/healthcheck.py << 'PY'
#\!/usr/bin/env python3
import os, sys, importlib
import torch
import torch.nn.functional as F

def fail(msg, code=1):
    print(f"❌ {msg}")
    sys.exit(code)

def ok(msg):
    print(f"✅ {msg}")

# CUDA + GPU present
if not torch.cuda.is_available():
    fail("CUDA is not available")

dev = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")

# Tiny GPU op
a = torch.randn(64, 64, device=dev, dtype=torch.float16)
b = torch.randn(64, 64, device=dev, dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
ok("Basic CUDA matmul")

# FlashAttention test
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
except Exception as e:
    fail(f"flash_attn import failed: {e}")

B, S, H, D = 1, 8, 4, 32
qkv = torch.randn(B, S, 3, H, D, device=dev, dtype=torch.float16)
out_fa = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)

q = qkv[:, :, 0].permute(0, 2, 1, 3)
k = qkv[:, :, 1].permute(0, 2, 1, 3)
v = qkv[:, :, 2].permute(0, 2, 1, 3)
out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
out_ref = out_ref.permute(0, 2, 1, 3).contiguous()

max_diff = (out_fa.float() - out_ref.float()).abs().max().item()
if max_diff >= 2e-2:
    fail(f"flash_attn mismatch: {max_diff:.3e}")
ok(f"flash_attn verified (diff={max_diff:.3e})")

# adam_atan2 check
try:
    import adam_atan2
    import adam_atan2_backend
    ok("adam_atan2 backend")
except Exception as e:
    fail(f"adam_atan2 failed: {e}")

print("\n🎉 HRM Ready\!")
sys.exit(0)
PY

# Run health check
python /opt/HRM/healthcheck.py 2>&1 | tee /var/log/hrm_healthcheck.log
HC_RC=${PIPESTATUS[0]}

if [ $HC_RC -ne 0 ]; then
    echo ""
    echo "\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!"
    echo "Health check FAILED (exit code: $HC_RC)"
    echo "See /var/log/hrm_healthcheck.log"
    echo "\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!"
    exit $HC_RC
fi

echo ""
echo "========================================"
echo "✓✓✓ INSTALLATION SUCCESSFUL ✓✓✓"
echo "========================================"
echo ""
echo "HRM is ready\! To use:"
echo "  source /opt/venv/bin/activate"
echo "  cd /opt/HRM"
echo "  WANDB_MODE=offline python pretrain.py [config]"
echo ""
echo "Health check passed in seconds\!"
echo "Log: /var/log/hrm_healthcheck.log"
