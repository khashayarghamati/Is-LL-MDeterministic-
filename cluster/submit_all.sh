#!/bin/bash
# ===========================================================================
# submit_all.sh — Submit all SLURM jobs for the full experiment.
#
# Submits 4 tier jobs (independent, run in parallel on different nodes)
# then 1 evaluation job that starts only after ALL tiers finish.
#
# All outputs go to /beegfs/general/kg23aay/stochastic_exploration/
#
# Usage:
#   bash cluster/submit_all.sh          # Submit all tiers + eval
#   bash cluster/submit_all.sh 1 2      # Submit only tiers 1 and 2 + eval
#
# Prerequisites:
#   1. Run  bash cluster/setup_env.sh   (one-time)
#   2. Accept model licenses on HuggingFace (Llama, Gemma)
#   3. Ensure HF_TOKEN is set or you've done huggingface-cli login
# ===========================================================================

set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BEEGFS_BASE}/stochastic_exploration/logs"
cd "${PROJECT_DIR}"

# Create log directory on beegfs
mkdir -p "${LOG_DIR}"

# =========================================================================
# Cluster configuration for UH
# =========================================================================
PARTITION="gpu"                          # GPU partition name
ACCOUNT="kg23aay"                        # SLURM account
GPU_TYPE=""                              # Leave empty if cluster doesn't need gpu type
EMAIL="kg23aay@herts.ac.uk"              # Notification email
# =========================================================================

# Tiers to submit (all by default, or from command-line args)
if [ $# -gt 0 ]; then
    TIERS=("$@")
else
    TIERS=(1 2 3 4)
fi

echo "=============================================="
echo " Submitting StochasticExploration Jobs"
echo " Partition: ${PARTITION}"
echo " Logs:      ${LOG_DIR}"
echo " Outputs:   ${BEEGFS_BASE}/stochastic_exploration/results/"
echo " Tiers:     ${TIERS[*]}"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Resource tables
# ---------------------------------------------------------------------------
declare -A TIER_GPUS=( [1]=1  [2]=1  [3]=2  [4]=4 )
declare -A TIER_CPUS=( [1]=8  [2]=8  [3]=16 [4]=16 )
declare -A TIER_MEM=(  [1]="32G" [2]="64G" [3]="128G" [4]="256G" )
declare -A TIER_TIME=( [1]="06:00:00" [2]="18:00:00" [3]="36:00:00" [4]="48:00:00" )
declare -A TIER_NAME=( [1]="small" [2]="medium" [3]="large" [4]="frontier" )

# Build --gres string (with or without gpu type)
_gres() {
    local count=$1
    if [ -n "${GPU_TYPE}" ]; then
        echo "gpu:${GPU_TYPE}:${count}"
    else
        echo "gpu:${count}"
    fi
}

# ---------------------------------------------------------------------------
# Submit tier jobs
# ---------------------------------------------------------------------------
TIER_JOB_IDS=()

for TIER in "${TIERS[@]}"; do
    JOB_ID=$(sbatch \
        --job-name="stoch-t${TIER}-${TIER_NAME[$TIER]}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres="$(_gres ${TIER_GPUS[$TIER]})" \
        --cpus-per-task="${TIER_CPUS[$TIER]}" \
        --mem="${TIER_MEM[$TIER]}" \
        --time="${TIER_TIME[$TIER]}" \
        --output="${LOG_DIR}/tier${TIER}_%j.out" \
        --error="${LOG_DIR}/tier${TIER}_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="${EMAIL}" \
        --export=ALL \
        cluster/run_tier.sh "${TIER}" \
        | awk '{print $NF}')

    TIER_JOB_IDS+=("${JOB_ID}")
    echo "  Tier ${TIER} (${TIER_NAME[$TIER]}): submitted as job ${JOB_ID}"
    echo "    GPUs=${TIER_GPUS[$TIER]}  CPUs=${TIER_CPUS[$TIER]}  Mem=${TIER_MEM[$TIER]}  Time=${TIER_TIME[$TIER]}"
done

# ---------------------------------------------------------------------------
# Submit evaluation job — depends on ALL tier jobs succeeding
# ---------------------------------------------------------------------------
DEPENDENCY_STR=$(IFS=:; echo "${TIER_JOB_IDS[*]}")

EVAL_JOB_ID=$(sbatch \
    --job-name="stoch-eval" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres="$(_gres 1)" \
    --cpus-per-task=8 \
    --mem="32G" \
    --time="04:00:00" \
    --output="${LOG_DIR}/evaluate_%j.out" \
    --error="${LOG_DIR}/evaluate_%j.err" \
    --mail-type=END,FAIL \
    --mail-user="${EMAIL}" \
    --dependency="afterok:${DEPENDENCY_STR}" \
    --export=ALL \
    cluster/run_evaluate.sh \
    | awk '{print $NF}')

echo ""
echo "  Evaluation: submitted as job ${EVAL_JOB_ID}"
echo "    Depends on: ${DEPENDENCY_STR}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " All jobs submitted!"
echo ""
echo " Monitor with:"
echo "   squeue -u \$USER"
echo "   tail -f ${LOG_DIR}/tier1_<jobid>.out"
echo ""
echo " After completion, results will be in:"
echo "   ${BEEGFS_BASE}/stochastic_exploration/results/"
echo "=============================================="
