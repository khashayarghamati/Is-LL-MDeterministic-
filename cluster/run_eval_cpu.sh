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
LOG_DIR="${BEEGFS_BASE}/stochastic_exploration/logs"

mkdir -p "${LOG_DIR}"

# Cancel any stuck eval jobs
echo "Cancelling any stuck stoch-eval jobs..."
scancel --name=stoch-eval --user="$USER" 2>/dev/null || true

# Submit CPU-only eval+analysis job using the existing run_evaluate.sh script
JOB_ID=$(sbatch \
    --job-name="stoch-eval-cpu" \
    --partition="gpu" \
    --account="cs" \
    --cpus-per-task=8 \
    --mem="16G" \
    --time="02:00:00" \
    --output="${LOG_DIR}/eval_cpu_%j.out" \
    --error="${LOG_DIR}/eval_cpu_%j.err" \
    --mail-type=END,FAIL \
    --mail-user="kg23aay@herts.ac.uk" \
    --export=ALL \
    cluster/run_evaluate.sh \
    | awk '{print $NF}')

echo ""
echo "Submitted eval+analysis job: ${JOB_ID}"
echo "  Partition: cs (CPU only, no GPU)"
echo "  Logs: ${LOG_DIR}/eval_cpu_${JOB_ID}.out"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/eval_cpu_${JOB_ID}.out"
