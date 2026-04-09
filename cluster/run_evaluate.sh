#!/bin/bash
# ===========================================================================
# run_evaluate.sh — SLURM batch script for evaluation + analysis.
#
# Runs AFTER all tier jobs complete (submitted with --dependency by
# submit_all.sh).  Needs 1 GPU for the embedding model, minimal resources.
# Outputs to /beegfs/general/kg23aay/stochastic_exploration/results/
# ===========================================================================

set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="${BEEGFS_BASE}/Is-LL-MDeterministic-"
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

cd "${PROJECT_DIR}"

echo "=============================================="
echo " Stochastic Exploration — Evaluate & Analyze"
echo " $(date)"
echo " Output: ${BEEGFS_BASE}/stochastic_exploration/results/"
echo "=============================================="

# Step 1: Compute stochasticity metrics (needs GPU for embeddings)
echo ""
echo "--- Running evaluation ---"
python3 main.py evaluate

# Step 2: Statistical analysis and plots (CPU only)
echo ""
echo "--- Running analysis ---"
python3 main.py analyze

# Step 3: Show final status
echo ""
python3 main.py status

echo ""
echo "All done at $(date)"
echo "Results in: ${BEEGFS_BASE}/stochastic_exploration/results/"
