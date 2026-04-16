#!/bin/bash
# ===========================================================================
# run_extended_tier.sh — SLURM batch script for running experiments with
#                        topic/temperature/experiment-name support.
#
# Usage:
#   sbatch cluster/run_extended_tier.sh <TIER> <TOPIC> <TEMP> <EXPERIMENT> <MODEL_SET>
#
# Example:
#   sbatch cluster/run_extended_tier.sh 1 climate 0.7 climate_T0.7 all
#   sbatch cluster/run_extended_tier.sh 2 healthcare 0.3 healthcare_T0.3 tempsweep
# ===========================================================================

TIER=${1:?Usage: sbatch cluster/run_extended_tier.sh <TIER> <TOPIC> <TEMP> <EXPERIMENT> <MODEL_SET>}
TOPIC=${2:-healthcare}
TEMP=${3:-0.7}
EXPERIMENT=${4:-""}
MODEL_SET=${5:-all}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="/home2/kg23aay/workspace/Is-LL-MDeterministic-"
CONDA_ENV_NAME="stochastic_exp"

# Load .env (HF_TOKEN)
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

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
echo " Stochastic Exploration — Extended Experiment"
echo "=============================================="
echo " Date:       $(date)"
echo " Node:       $(hostname)"
echo " Tier:       ${TIER}"
echo " Topic:      ${TOPIC}"
echo " Temperature:${TEMP}"
echo " Experiment: ${EXPERIMENT}"
echo " Model set:  ${MODEL_SET}"
echo " CUDA:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo " GPU count:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo '0')"
echo " PyTorch:    $(python3 -c 'import torch; print(torch.__version__)')"
echo " CUDA avail: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

CMD="python3 main.py run --tier ${TIER} --topic ${TOPIC} --temp ${TEMP} --model-set ${MODEL_SET}"
if [ -n "${EXPERIMENT}" ]; then
    CMD="${CMD} --experiment ${EXPERIMENT}"
fi

echo "Running: ${CMD}"
eval "${CMD}"

echo ""
echo "Tier ${TIER} (${TOPIC}, T=${TEMP}) complete at $(date)"
