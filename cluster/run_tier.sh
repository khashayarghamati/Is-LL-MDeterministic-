#!/bin/bash
# ===========================================================================
# run_tier.sh — SLURM batch script for running a single tier of models.
#
# This script is submitted by submit_all.sh with the tier number passed
# as the first argument.  It requests GPU resources appropriate for the
# model sizes in that tier.
#
# Usage (manual):
#   sbatch cluster/run_tier.sh 1   # small  models, 1 GPU
#   sbatch cluster/run_tier.sh 2   # medium models, 1 GPU
#   sbatch cluster/run_tier.sh 3   # large  models, 2 GPUs
#   sbatch cluster/run_tier.sh 4   # frontier models, 4 GPUs
# ===========================================================================

# ---------- Read tier from argument ----------
TIER=${1:?Usage: sbatch cluster/run_tier.sh <TIER 1-4>}

# =========================================================================
# SLURM directives — resources scale with tier
#
#   ┌──────────┬──────┬──────┬───────┬───────────┐
#   │  Tier    │ GPUs │ CPUs │  RAM  │ Wall time  │
#   ├──────────┼──────┼──────┼───────┼───────────┤
#   │ 1 small  │  1   │   8  │  32G  │  06:00:00  │
#   │ 2 medium │  1   │   8  │  64G  │  18:00:00  │
#   │ 3 large  │  2   │  16  │ 128G  │  36:00:00  │
#   │ 4 front. │  4   │  16  │ 256G  │  48:00:00  │
#   └──────────┴──────┴──────┴───────┴───────────┘
#
# ⚠  Adjust partition name, GPU type, and account to match your cluster!
# =========================================================================

#SBATCH --job-name=stoch-t${TIER}
#SBATCH --partition=gpu                        # ← change to your GPU partition
#SBATCH --account=your-account                 # ← change to your allocation
#SBATCH --output=logs/tier%a_%j.out
#SBATCH --error=logs/tier%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@herts.ac.uk     # ← change to your email

# GPU / CPU / memory / time are set dynamically below

# ---------------------------------------------------------------------------
# Resource allocation based on tier
# ---------------------------------------------------------------------------
case "${TIER}" in
    1)
        GPU_COUNT=1;  CPUS=8;  MEM="32G";  TIME="06:00:00" ;;
    2)
        GPU_COUNT=1;  CPUS=8;  MEM="64G";  TIME="18:00:00" ;;
    3)
        GPU_COUNT=2;  CPUS=16; MEM="128G"; TIME="36:00:00" ;;
    4)
        GPU_COUNT=4;  CPUS=16; MEM="256G"; TIME="48:00:00" ;;
    *)
        echo "ERROR: Invalid tier '${TIER}'. Must be 1-4."; exit 1 ;;
esac

# Re-submit ourselves with the right resources (SLURM doesn't expand
# variables inside #SBATCH, so we use srun or re-invoke via sbatch
# with --export).  Since this script is meant to be called by
# submit_all.sh which sets the resources, we just proceed here.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

# Load cluster modules
if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda/12.1 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# Log system info
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Stochastic Exploration — Tier ${TIER}"
echo "=============================================="
echo " Date:       $(date)"
echo " Node:       $(hostname)"
echo " GPUs:       ${GPU_COUNT}"
echo " CUDA:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo " PyTorch:    $(python3 -c 'import torch; print(torch.__version__)')"
echo " CUDA avail: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo " GPU count:  $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Run experiment for this tier
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

python3 main.py run --tier "${TIER}"

echo ""
echo "Tier ${TIER} complete at $(date)"
