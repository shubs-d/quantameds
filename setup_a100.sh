#!/usr/bin/env bash
# setup_a100.sh — Bootstrap the Python environment on a fresh A100 instance.
#
# Run this FIRST on the remote machine:
#   bash setup_a100.sh 2>&1 | tee setup_a100.log
#
# Expected steps:
#   1. Check CUDA driver version via nvidia-smi
#   2. Install uv (fast pip replacement)
#   3. Create .venv with Python 3.11
#   4. Install CUDA-matched PyTorch wheel
#   5. Install project dependencies
#   6. Verify torch.cuda.is_available() == True
#   7. Check for hardcoded 'cpu' strings (audit only — we already patched them)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$REPO_ROOT/setup_a100.log"

echo "================================================================"
echo "  QuantaMED A100 Setup"
echo "  $(date)"
echo "  Repo: $REPO_ROOT"
echo "================================================================"

# ── Step 1: Check CUDA driver ──────────────────────────────────────
echo ""
echo "── Step 1: CUDA driver check ───────────────────────────────────"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
    CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Driver version: $CUDA_DRIVER"
else
    echo "[WARN] nvidia-smi not found. Is this actually a GPU instance?"
    echo "[WARN] Continuing anyway — will install CPU PyTorch as fallback."
fi

# Determine PyTorch CUDA wheel from driver/CUDA version
# CUDA 12.4+ → cu124, CUDA 12.1–12.3 → cu121, CUDA 11.8 → cu118
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
# Get actual CUDA toolkit version from nvcc if available
if command -v nvcc &>/dev/null; then
    CUDA_TOOLKIT=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    echo "CUDA toolkit version: $CUDA_TOOLKIT"
    MAJOR=$(echo "$CUDA_TOOLKIT" | cut -d. -f1)
    MINOR=$(echo "$CUDA_TOOLKIT" | cut -d. -f2)
    if [[ "$MAJOR" -ge 12 ]] && [[ "$MINOR" -ge 4 ]]; then
        TORCH_WHEEL="cu124"
    elif [[ "$MAJOR" -ge 12 ]]; then
        TORCH_WHEEL="cu121"
    else
        TORCH_WHEEL="cu118"
    fi
else
    # Fallback: check /usr/local/cuda symlink
    CUDA_LINK=$(readlink -f /usr/local/cuda 2>/dev/null || echo "")
    if [[ "$CUDA_LINK" == *"12.4"* ]] || [[ "$CUDA_LINK" == *"12.5"* ]] || [[ "$CUDA_LINK" == *"12.6"* ]]; then
        TORCH_WHEEL="cu124"
    elif [[ "$CUDA_LINK" == *"12"* ]]; then
        TORCH_WHEEL="cu121"
    else
        TORCH_WHEEL="cu118"
    fi
fi
echo "Selected PyTorch wheel: $TORCH_WHEEL"

# ── Step 2: Install uv ────────────────────────────────────────────
echo ""
echo "── Step 2: Install uv ───────────────────────────────────────────"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── Step 3: Create venv ───────────────────────────────────────────
echo ""
echo "── Step 3: Create .venv ─────────────────────────────────────────"
cd "$REPO_ROOT"
if [[ ! -d ".venv" ]]; then
    uv venv --python 3.11 .venv
    echo ".venv created."
else
    echo ".venv already exists — skipping creation."
fi
source .venv/bin/activate

# ── Step 4: Install PyTorch with CUDA ────────────────────────────
echo ""
echo "── Step 4: Install PyTorch ($TORCH_WHEEL) ───────────────────────"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_WHEEL}"
uv pip install \
    "torch>=2.2.0" \
    "torchvision>=0.17.0" \
    --index-url "$TORCH_INDEX"

# ── Step 5: Install project dependencies ─────────────────────────
echo ""
echo "── Step 5: Install project dependencies ────────────────────────"
# Core ML / science
uv pip install \
    pennylane \
    pennylane-lightning \
    "pennylane-lightning[gpu]" \
    scikit-learn \
    scipy \
    numpy \
    pandas \
    mlflow \
    tqdm \
    matplotlib \
    pillow \
    opencv-python-headless

# Install the project package itself (editable)
if [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]] || [[ -f "setup.cfg" ]]; then
    uv pip install -e .
else
    echo "[INFO] No pyproject.toml/setup.py found — project imported via sys.path."
fi

# ── Step 6: Verify CUDA ───────────────────────────────────────────
echo ""
echo "── Step 6: Verify torch.cuda ────────────────────────────────────"
python - <<'PYEOF'
import torch
cuda_ok = torch.cuda.is_available()
print(f"torch.cuda.is_available() = {cuda_ok}")
if cuda_ok:
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA version (torch): {torch.version.cuda}")
else:
    print("[WARN] CUDA not available — check driver + torch wheel compatibility.")
    import sys
    sys.exit(1)

# Quick forward pass to confirm GPU actually works
t = torch.randn(4, 3, 224, 224, device="cuda")
print(f"Test tensor on GPU: {t.shape} ✓")
PYEOF

# ── Step 7: PennyLane GPU backend check ──────────────────────────
echo ""
echo "── Step 7: Check PennyLane lightning.gpu ────────────────────────"
python - <<'PYEOF'
try:
    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=8)
    print(f"lightning.gpu available: YES")
except Exception as e:
    print(f"lightning.gpu not available: {e}")
    print("(Expected — we do NOT use lightning.gpu for 8-qubit VQC per design decision)")
PYEOF

# ── Step 8: Quick import check ────────────────────────────────────
echo ""
echo "── Step 8: Project import check ─────────────────────────────────"
python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from quantum_kc.config import config, get_device
device = get_device()
print(f"quantum_kc.config imported OK | device = {device}")
from quantum_kc.models.encoder import CornealEncoder
enc = CornealEncoder(in_channels=3, latent_dim=8)
param_count = sum(p.numel() for p in enc.parameters())
print(f"CornealEncoder: {param_count:,} params — imported OK")
PYEOF

echo ""
echo "================================================================"
echo "  Setup complete!  $(date)"
echo "  Activate with: source .venv/bin/activate"
echo "================================================================"
