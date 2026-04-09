#!/bin/bash
# ===========================================================================
# setup_env.sh — One-time environment setup on UH GPU cluster
#
# Run this ONCE before submitting any SLURM jobs:
#   bash cluster/setup_env.sh
#
# Uses existing miniconda3 at /beegfs/general/kg23aay/miniconda3
# All outputs go to /beegfs/general/kg23aay/stochastic_exploration/
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths on beegfs
# ---------------------------------------------------------------------------
BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV_NAME="stochastic_exp"
OUTPUT_DIR="${BEEGFS_BASE}/stochastic_exploration"

# Caches — keep on beegfs to avoid home quota issues
export HF_HOME="${BEEGFS_BASE}/hf_cache"
export PIP_CACHE_DIR="${BEEGFS_BASE}/pip_cache"
export TMPDIR="${BEEGFS_BASE}/tmp"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

echo "=============================================="
echo " StochasticExploration — Environment Setup"
echo "=============================================="
echo " Project dir:   ${PROJECT_DIR}"
echo " Conda dir:     ${CONDA_DIR}"
echo " Conda env:     ${CONDA_ENV_NAME}"
echo " Output dir:    ${OUTPUT_DIR}"
echo " HF cache:      ${HF_HOME}"
echo " Pip cache:     ${PIP_CACHE_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Initialize conda
# ---------------------------------------------------------------------------
echo "[1/6] Initializing conda..."
source "${CONDA_DIR}/etc/profile.d/conda.sh"
echo "  Conda version: $(conda --version)"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Creating conda environment '${CONDA_ENV_NAME}'..."
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "  Environment already exists — reusing."
else
    conda create -n "${CONDA_ENV_NAME}" python=3.11 -y -q
    echo "  Created environment: ${CONDA_ENV_NAME}"
fi
conda activate "${CONDA_ENV_NAME}"
echo "  Python: $(python3 --version)"
echo "  Path:   $(which python3)"

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA support
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing PyTorch (CUDA)..."
pip install --upgrade pip setuptools wheel --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "  PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available:  $(python3 -c 'import torch; print(torch.cuda.is_available())')"

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
# 6. Create output directory + HuggingFace login
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Setting up output directories and HuggingFace auth..."

mkdir -p "${OUTPUT_DIR}/results/raw_responses"
mkdir -p "${OUTPUT_DIR}/results/metrics"
mkdir -p "${OUTPUT_DIR}/results/plots"
echo "  Output dirs created at: ${OUTPUT_DIR}"

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
echo " To activate later:  source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}"
echo ""
echo " Next steps:"
echo "   1. Accept model licenses on HuggingFace (Llama 3, Gemma 2)"
echo "   2. Submit jobs:  bash cluster/submit_all.sh"
echo "   3. Monitor:      squeue -u \$USER"
echo "   4. Results at:   ${OUTPUT_DIR}/results/"
echo "=============================================="
