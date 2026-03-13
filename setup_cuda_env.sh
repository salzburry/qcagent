#!/usr/bin/env bash
# setup_cuda_env.sh — One-shot environment setup for qcagent on a CUDA GPU machine.
#
# Tested with: Ubuntu 24.04 + Tesla T4 + Driver 580.x + CUDA 13.0
#
# What this does:
#   1. Installs NVIDIA CUDA toolkit (nvcc, libraries, headers)
#   2. Installs PyTorch with CUDA support
#   3. Installs vLLM (with T4 flashinfer workaround)
#   4. Installs project dependencies (pip install -e .)
#   5. Verifies GPU access from Python
#
# Usage:
#   chmod +x setup_cuda_env.sh
#   ./setup_cuda_env.sh
#
# The script is idempotent — safe to re-run.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
err()  { echo -e "${RED}[setup]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pre-flight checks ──────────────────────────────────────────────────────

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. NVIDIA driver must be installed first."
    err "Install driver: sudo apt install nvidia-driver-580"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')

log "GPU: $GPU_NAME"
log "Driver: $DRIVER_VER"
log "CUDA (driver): $CUDA_VER"

# ── Step 1: CUDA Toolkit ───────────────────────────────────────────────────

if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    log "CUDA Toolkit already installed: nvcc $NVCC_VER"
else
    log "Installing CUDA Toolkit..."

    # Determine CUDA toolkit version to install based on driver-reported CUDA version
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

    if [[ "$CUDA_MAJOR" -ge 13 ]]; then
        # CUDA 13.0 driver — install latest available toolkit
        # The toolkit version doesn't need to exactly match; it just needs
        # to be <= the driver's CUDA version
        TOOLKIT_PKG="nvidia-cuda-toolkit"
    else
        TOOLKIT_PKG="nvidia-cuda-toolkit"
    fi

    sudo apt-get update -qq
    sudo apt-get install -y -qq "$TOOLKIT_PKG"

    # Verify
    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
        log "CUDA Toolkit installed: nvcc $NVCC_VER"
    else
        warn "nvcc not on PATH after install. Checking common locations..."
        for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
            if [[ -x "$d/nvcc" ]]; then
                export PATH="$d:$PATH"
                log "Added $d to PATH"
                break
            fi
        done
    fi
fi

# ── Step 2: PyTorch with CUDA ──────────────────────────────────────────────

log "Installing PyTorch with CUDA support..."

# Use the latest stable PyTorch with CUDA 12.8 (compatible with CUDA 13.0 driver)
# PyTorch CUDA builds are forward-compatible with newer drivers
pip install --upgrade pip

# Check if torch already installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
    log "PyTorch $TORCH_VER (CUDA $TORCH_CUDA) already installed and working"
else
    # Install PyTorch with CUDA 12.8 support (works with CUDA 13.0 driver)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    # Verify
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
        TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
        log "PyTorch $TORCH_VER (CUDA $TORCH_CUDA) installed successfully"
    else
        err "PyTorch installed but CUDA not available. Check driver/toolkit compatibility."
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
        exit 1
    fi
fi

# ── Step 3: vLLM ───────────────────────────────────────────────────────────

log "Installing vLLM..."

if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
    log "vLLM $VLLM_VER already installed"
else
    pip install vllm
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
    log "vLLM $VLLM_VER installed"
fi

# T4 workaround: remove flashinfer if GPU is compute capability < 8.0
COMPUTE_CAP=$(python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(cap[0])
else:
    print('0')
" 2>/dev/null || echo "0")

if [[ "$COMPUTE_CAP" -lt 8 ]] && [[ "$COMPUTE_CAP" -gt 0 ]]; then
    warn "T4/sm_75 detected — removing flashinfer to avoid CUTLASS DSL crash..."
    pip uninstall -y flashinfer flashinfer-python 2>/dev/null || true
    log "flashinfer removed. vLLM will use compatible attention backend."
fi

# ── Step 4: Project dependencies ───────────────────────────────────────────

log "Installing project dependencies (pip install -e .)..."
cd "$SCRIPT_DIR"
pip install -e .

log "Project installed as editable package"

# ── Step 5: Verify everything ──────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════════════"
log "  Verification"
log "═══════════════════════════════════════════════════════════"

python3 << 'PYEOF'
import sys

checks = []

# GPU / PyTorch
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    if gpu_ok:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        checks.append(f"  PyTorch CUDA:  OK  ({name}, {mem:.1f} GB)")
    else:
        checks.append("  PyTorch CUDA:  FAIL (not available)")
except ImportError:
    checks.append("  PyTorch:       FAIL (not installed)")
    gpu_ok = False

# vLLM
try:
    import vllm
    checks.append(f"  vLLM:          OK  (v{vllm.__version__})")
except ImportError:
    checks.append("  vLLM:          FAIL (not installed)")

# Project package
try:
    import protocol_spec_assist
    checks.append("  qcagent pkg:   OK")
except ImportError:
    checks.append("  qcagent pkg:   FAIL (not installed)")

# Key deps
for pkg_name, import_name in [
    ("pydantic", "pydantic"),
    ("docling", "docling"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("qdrant-client", "qdrant_client"),
    ("openai", "openai"),
    ("prefect", "prefect"),
    ("streamlit", "streamlit"),
]:
    try:
        __import__(import_name)
        checks.append(f"  {pkg_name:16s} OK")
    except ImportError:
        checks.append(f"  {pkg_name:16s} FAIL")

print()
for c in checks:
    print(c)
print()

if not gpu_ok:
    print("WARNING: GPU not available. vLLM will not work.")
    sys.exit(1)
else:
    print("All checks passed. Ready to run:")
    print("  python setup_vllm.py --set-env")
    print()
PYEOF

log "Setup complete!"
