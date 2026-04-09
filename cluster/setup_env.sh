#!/bin/bash
# ===========================================================================
# setup_env.sh — One-time environment setup on UH GPU cluster
#
# Run this ONCE before submitting any SLURM jobs:
#   bash cluster/setup_env.sh
#
# What it does:
#   1. Loads required modules (CUDA, Python)
#   2. Creates a Python virtual environment
#   3. Installs all dependencies (PyTorch CUDA, Transformers, etc.)
#   4. Downloads NLTK data & sentence-transformers embedding model
#   5. Logs into HuggingFace (needed for gated models: Llama, Gemma)
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust these to match your cluster
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

# HuggingFace cache — point to scratch / fast storage to avoid quota issues
export HF_HOME="${PROJECT_DIR}/.hf_cache"

# CUDA version on your cluster (check with: module avail cuda)
CUDA_MODULE="cuda/12.1"            # ← adjust if needed
PYTHON_MODULE="python/3.11"        # ← adjust if needed

# PyTorch CUDA index (must match CUDA_MODULE version)
TORCH_INDEX="https://download.pytorch.org/whl/cu121"  # ← adjust if needed

echo "=============================================="
echo " StochasticExploration — Environment Setup"
echo "=============================================="
echo " Project dir:  ${PROJECT_DIR}"
echo " Venv dir:     ${VENV_DIR}"
echo " HF cache:     ${HF_HOME}"
echo ""

# ---------------------------------------------------------------------------
# 1. Load modules
# ---------------------------------------------------------------------------
echo "[1/6] Loading modules..."
if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load "${CUDA_MODULE}" 2>/dev/null || echo "  ⚠ Could not load ${CUDA_MODULE} — check 'module avail cuda'"
    module load "${PYTHON_MODULE}" 2>/dev/null || echo "  ⚠ Could not load ${PYTHON_MODULE} — check 'module avail python'"
    module list 2>&1 | head -10
else
    echo "  No module system detected — assuming CUDA and Python are in PATH."
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Creating virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "  Venv already exists — reusing."
else
    python3 -m venv "${VENV_DIR}"
    echo "  Created: ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA support
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing PyTorch (CUDA)..."
pip install torch --index-url "${TORCH_INDEX}" --quiet

echo "  PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available:  $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count:       $(python3 -c 'import torch; print(torch.cuda.device_count())')"

# ---------------------------------------------------------------------------
# 4. Install remaining dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing project dependencies..."
pip install -r "${PROJECT_DIR}/requirements.txt" --quiet

# ---------------------------------------------------------------------------
# 5. Download NLTK data + embedding model
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Downloading NLTK data and embedding model..."
python3 -c "
import nltk
nltk.download('punkt_tab', quiet=True)
print('  NLTK punkt_tab downloaded.')
"
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('  Embedding model downloaded and cached.')
"

# ---------------------------------------------------------------------------
# 6. HuggingFace login (required for gated models: Llama, Gemma)
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] HuggingFace authentication..."
echo ""
echo "  Some models (Llama 3, Gemma 2) are gated and require you to:"
echo "    1. Create a HuggingFace account at https://huggingface.co"
echo "    2. Accept the model license on each model page"
echo "    3. Create an access token at https://huggingface.co/settings/tokens"
echo ""

if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN is already set in environment. Logging in..."
    huggingface-cli login --token "${HF_TOKEN}"
else
    echo "  Option A: Run 'huggingface-cli login' interactively now."
    echo "  Option B: Set HF_TOKEN in your environment before submitting jobs."
    echo ""
    read -r -p "  Login interactively now? [y/N] " answer
    if [[ "${answer}" =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "  Skipped. Remember to export HF_TOKEN=<your-token> before running jobs."
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Make sure you have accepted model licenses on HuggingFace"
echo "   2. Submit jobs:  bash cluster/submit_all.sh"
echo "   3. Monitor:      squeue -u \$USER"
echo "=============================================="
