#!/bin/bash
# ===========================================================================
# run_tier.sh — SLURM batch script for running a single tier of models.
#
# Submitted by submit_all.sh with the tier number as the first argument.
# Uses miniconda3 at /beegfs/general/kg23aay/miniconda3
# Outputs to /beegfs/general/kg23aay/stochastic_exploration/
#
# Usage (manual):
#   sbatch cluster/run_tier.sh 1   # small  models, 1 GPU
#   sbatch cluster/run_tier.sh 2   # medium models, 1 GPU
#   sbatch cluster/run_tier.sh 3   # large  models, 2 GPUs
#   sbatch cluster/run_tier.sh 4   # frontier models, 4 GPUs
# ===========================================================================

# ---------- Read tier from argument ----------
TIER=${1:?Usage: sbatch cluster/run_tier.sh <TIER 1-4>}

# ---------------------------------------------------------------------------
# Environment — beegfs paths + miniconda
# ---------------------------------------------------------------------------
set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV_NAME="stochastic_exp"

export HF_HOME="${BEEGFS_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${BEEGFS_BASE}/pip_cache"
export TMPDIR="${BEEGFS_BASE}/tmp"
export TOKENIZERS_PARALLELISM=false

# Activate conda
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# ---------------------------------------------------------------------------
# Log system info
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Stochastic Exploration — Tier ${TIER}"
echo "=============================================="
echo " Date:       $(date)"
echo " Node:       $(hostname)"
echo " User:       $(whoami)"
echo " CUDA:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo " GPU count:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo '0')"
echo " PyTorch:    $(python3 -c 'import torch; print(torch.__version__)')"
echo " CUDA avail: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo " Torch GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo " Output dir: ${BEEGFS_BASE}/stochastic_exploration/results/"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Run experiment for this tier
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

python3 main.py run --tier "${TIER}"

echo ""
echo "Tier ${TIER} complete at $(date)"
