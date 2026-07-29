#!/usr/bin/env bash
set -euo pipefail

# Simple installer for Colab / local use.
# - Installs PyTorch with a CUDA wheel if GPU is detected (cu118 by default)
# - Installs the rest from requirements.txt
# Usage:
#   bash install.sh
# If you need a different CUDA version, install torch manually following https://pytorch.org/get-started/locally/

REQ_FILE="requirements.txt"

echo "== Installing base requirements from ${REQ_FILE} =="

if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: ${REQ_FILE} not found in current directory."
  exit 1
fi

# Detect NVIDIA / GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected (nvidia-smi found). Installing CUDA-enabled PyTorch (cu118) via official wheels..."
  pip install --pre --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --upgrade
else
  echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing packages from ${REQ_FILE} ..."
pip install -r "${REQ_FILE}"

echo
echo "Verifying important optional package: bitsandbytes (may require CUDA and runtime restart)..."
python - <<'PY'
try:
    import importlib, sys
    bnb = importlib.import_module("bitsandbytes")
    print("bitsandbytes OK:", getattr(bnb, "__version__", "unknown"))
except Exception as e:
    print("bitsandbytes import failed or not installed:", e)
PY

echo
echo "Installation finished."
echo "Notes:"
echo "- If you installed/updated bitsandbytes in Google Colab, you may need to Restart runtime (Runtime → Restart runtime)."
echo "- If you need a different CUDA version for PyTorch, follow official instructions at https://pytorch.org/get-started/locally/ and then run: pip install -r requirements.txt"
