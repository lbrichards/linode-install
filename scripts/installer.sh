
#!/usr/bin/env bash
set -euo pipefail

echo "=== HRM Prod Installer v3 (safe CUDA/Torch match) ==="

# ---------- Detect GPU & compute capability ----------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || true)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 || true)  # e.g., 8.6
echo "GPU: ${GPU_NAME:-unknown}, CC: ${GPU_CC:-unknown}"

# ---------- Versions (pin exactly for reproducibility) ----------
PYVER=3.10
TORCH="2.5.1"                                  # pin
TORCH_IDX="https://download.pytorch.org/whl/cu124"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ---------- System deps ----------
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates build-essential tmux unzip python3-venv

# ---------- (Optional) CUDA toolkit for builds only ----------
# We need nvcc to build custom extensions. Using 12.8 is fine as long as we do not export LD_LIBRARY_PATH at runtime.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[CUDA] Installing toolkit 12.8 (build-time only)..."
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg
  echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y cuda-toolkit-12-8
fi
CUDA_HOME_BUILD=/usr/local/cuda-12.8    # used only while building

# ---------- Venv ----------
VENV=/opt/venv
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel ninja packaging

# ---------- Torch (cu124) ----------
pip install --index-url "$TORCH_IDX" "torch==${TORCH}"

# ---------- Dynamic arch list for extensions ----------
# Safe union if detection fails
ARCH_LIST_DEFAULT="7.5;8.0;8.6;8.9;9.0"
if [[ "$GPU_CC" =~ ^[0-9]+\.[0-9]+$ ]]; then
  export TORCH_CUDA_ARCH_LIST="$GPU_CC"
else
  export TORCH_CUDA_ARCH_LIST="$ARCH_LIST_DEFAULT"
fi
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ---------- HRM deps ----------
HRM_DIR=/opt/HRM
[ -f "$HRM_DIR/requirements.txt" ] || { echo "Missing $HRM_DIR/requirements.txt"; exit 1; }
pip install -r "$HRM_DIR/requirements.txt"
pip install numpy einops

# ---------- adam-atan2 (prefer binary; fall back to source) ----------
if ! python -c "import adam_atan2" >/dev/null 2>&1; then
  echo "Installing adam-atan2..."
  (
    export CUDA_HOME="$CUDA_HOME_BUILD"
    pip install adam_atan2 --no-cache-dir --verbose > adam_atan2_install.log 2>&1
    #pip install adam-atan2 || pip install --no-binary adam-atan2 adam-atan2
  )
fi

# ---------- flash-attn (install only if GPU supports it) ----------
USE_FLASH_ATTN=0
case "$GPU_CC" in
  8.0|8.6|8.9|9.0) USE_FLASH_ATTN=1 ;;
  *) USE_FLASH_ATTN=0 ;;
esac

if [[ "$USE_FLASH_ATTN" == "1" ]]; then
  echo "Installing flash-attn prebuilt wheel for Torch ${TORCH}..."
  pip install --no-deps "$FA_WHEEL" || { echo "flash-attn wheel failed; continuing without it"; USE_FLASH_ATTN=0; }
else
  echo "Skipping flash-attn (GPU CC=$GPU_CC not supported by prebuilt wheel)."
fi

# ---------- Unset build-time CUDA vars for runtime safety ----------
unset LD_LIBRARY_PATH || true
unset CUDA_HOME || true

# ---------- Health check (local) ----------
cat > "$HRM_DIR/healthcheck.py" <<'PY'
import torch, os, json, time
rep = {}
rep["torch_version"] = torch.__version__
rep["cuda_available"] = torch.cuda.is_available()
rep["cuda_device_count"] = torch.cuda.device_count()
rep["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
rep["torch_cuda_version"] = torch.version.cuda
# Try flash-attn if present
try:
    import flash_attn
    rep["flash_attn"] = "ok"
except Exception as e:
    rep["flash_attn"] = f"unavailable: {e.__class__.__name__}"
# Tiny tensor op on GPU if available
if rep["cuda_available"]:
    x = torch.randn(2,2, device="cuda")
    y = torch.mm(x, x)
    rep["mm_norm"] = float(y.norm().item())
print(json.dumps(rep, indent=2))
PY

python "$HRM_DIR/healthcheck.py"

echo "=== Install OK ==="
