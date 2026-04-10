#!/bin/bash
# ===========================================================================
# run_eval_cpu.sh — Submit evaluation + analysis as a CPU-only SLURM job.
#
# Cancels any stuck eval job, then submits a new one without GPU.
#
# Usage:
#   bash cluster/run_eval_cpu.sh
# ===========================================================================

set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
CONDA_DIR="${BEEGFS_BASE}/miniconda3"
PROJECT_DIR="/home2/kg23aay/workspace/Is-LL-MDeterministic-"
CONDA_ENV_NAME="stochastic_exp"
LOG_DIR="${BEEGFS_BASE}/stochastic_exploration/logs"

mkdir -p "${LOG_DIR}"

# Cancel any stuck eval jobs
echo "Cancelling any stuck stoch-eval jobs..."
scancel --name=stoch-eval --user="$USER" 2>/dev/null || true

# Submit CPU-only eval+analysis job
JOB_ID=$(sbatch \
    --job-name="stoch-eval-cpu" \
    --partition="cs" \
    --account="cs" \
    --cpus-per-task=8 \
    --mem="16G" \
    --time="02:00:00" \
    --output="${LOG_DIR}/eval_cpu_%j.out" \
    --error="${LOG_DIR}/eval_cpu_%j.err" \
    --mail-type=END,FAIL \
    --mail-user="kg23aay@herts.ac.uk" \
    --export=ALL \
    --wrap="
set -euo pipefail

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

if [ -f ${PROJECT_DIR}/.env ]; then
    set -a; source ${PROJECT_DIR}/.env; set +a
fi

export HF_HOME=${BEEGFS_BASE}/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false

cd ${PROJECT_DIR}

echo '=============================================='
echo ' Stochastic Exploration — Evaluate & Analyze (CPU)'
echo ' \$(date)'
echo '=============================================='

echo ''
echo '--- Running evaluation ---'
python3 main.py evaluate

echo ''
echo '--- Running analysis ---'
python3 main.py analyze

echo ''
python3 main.py status

echo ''
echo 'All done at \$(date)'
" | awk '{print $NF}')

echo ""
echo "Submitted eval+analysis job: ${JOB_ID}"
echo "  Partition: cs (CPU only, no GPU)"
echo "  Logs: ${LOG_DIR}/eval_cpu_${JOB_ID}.out"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/eval_cpu_${JOB_ID}.out"
