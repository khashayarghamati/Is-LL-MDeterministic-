#!/bin/bash
# ===========================================================================
# run_extended_eval.sh — Evaluate + Analyze for a specific experiment.
#
# Usage:
#   sbatch cluster/run_extended_eval.sh <EXPERIMENT>
#
# Example:
#   sbatch cluster/run_extended_eval.sh climate_T0.7
# ===========================================================================

EXPERIMENT=${1:-""}

set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="/home2/kg23aay/workspace/Is-LL-MDeterministic-"
CONDA_ENV_NAME="stochastic_exp"

# Load .env
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi

export HF_HOME="${BEEGFS_BASE}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${BEEGFS_BASE}/pip_cache"
export TMPDIR="${BEEGFS_BASE}/tmp"
export TOKENIZERS_PARALLELISM=false

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"

echo "=============================================="
echo " Evaluate & Analyze — Experiment: ${EXPERIMENT:-default}"
echo " $(date)"
echo "=============================================="

CMD_EVAL="python3 main.py evaluate"
CMD_ANALYZE="python3 main.py analyze"

if [ -n "${EXPERIMENT}" ]; then
    CMD_EVAL="${CMD_EVAL} --experiment ${EXPERIMENT}"
    CMD_ANALYZE="${CMD_ANALYZE} --experiment ${EXPERIMENT}"
fi

echo "--- Running evaluation ---"
eval "${CMD_EVAL}"

echo ""
echo "--- Running analysis ---"
eval "${CMD_ANALYZE}"

echo ""
echo "All done at $(date)"
