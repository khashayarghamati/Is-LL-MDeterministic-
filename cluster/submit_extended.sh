#!/bin/bash
# ===========================================================================
# submit_extended.sh — Submit ALL extended experiments to SLURM.
#
# This submits three groups of experiments:
#
#   GROUP 1: Multi-topic (climate + software) with all models at T=0.7
#            → Tests generalizability across topics
#
#   GROUP 2: Temperature sweep (T=0.3, 0.5, 1.0) on healthcare prompts
#            with representative models (tempsweep set: 6 models)
#            → Shows how temperature modulates the scale–stochasticity curve
#
#   GROUP 3: New-generation models (Gemma 3, Qwen 3, Phi-4, Llama 4)
#            on healthcare at T=0.7
#            → Extends model coverage with latest architectures
#
# After all generation jobs finish, evaluation jobs run automatically.
#
# Usage:
#   bash cluster/submit_extended.sh              # Submit everything
#   bash cluster/submit_extended.sh topics        # Only multi-topic
#   bash cluster/submit_extended.sh tempsweep     # Only temperature sweep
#   bash cluster/submit_extended.sh newmodels     # Only new models
# ===========================================================================

set -euo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BEEGFS_BASE}/stochastic_exploration/logs"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# =========================================================================
# Cluster configuration
# =========================================================================
PARTITION="gpu"
ACCOUNT="cs"
EMAIL="kg23aay@herts.ac.uk"

# =========================================================================
# Resource tables (same as original — keyed by tier)
# =========================================================================
declare -A TIER_GPUS=( [1]=1  [2]=1  [3]=2  [4]=4 )
declare -A TIER_CPUS=( [1]=8  [2]=8  [3]=16 [4]=16 )
declare -A TIER_MEM=(  [1]="32G" [2]="64G" [3]="128G" [4]="256G" )
declare -A TIER_TIME=( [1]="06:00:00" [2]="18:00:00" [3]="36:00:00" [4]="4-00:00:00" )

_gres() { echo "gpu:$1"; }

# ---------------------------------------------------------------------------
# Helper: submit one tier job and return its job ID
# ---------------------------------------------------------------------------
submit_tier() {
    local tier=$1
    local topic=$2
    local temp=$3
    local experiment=$4
    local model_set=$5
    local job_name=$6

    local jid
    jid=$(sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres="$(_gres "${TIER_GPUS[$tier]}")" \
        --cpus-per-task="${TIER_CPUS[$tier]}" \
        --mem="${TIER_MEM[$tier]}" \
        --time="${TIER_TIME[$tier]}" \
        --output="${LOG_DIR}/${job_name}_%j.out" \
        --error="${LOG_DIR}/${job_name}_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="${EMAIL}" \
        --export=ALL \
        cluster/run_extended_tier.sh "${tier}" "${topic}" "${temp}" "${experiment}" "${model_set}" \
        | awk '{print $NF}')

    echo "${jid}"
}

# ---------------------------------------------------------------------------
# Helper: submit eval job with dependency on a list of job IDs
# ---------------------------------------------------------------------------
submit_eval() {
    local experiment=$1
    shift
    local dep_ids=("$@")
    local dep_str
    dep_str=$(IFS=:; echo "${dep_ids[*]}")
    local job_name="eval-${experiment}"

    local jid
    jid=$(sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --cpus-per-task=8 \
        --mem="16G" \
        --time="02:00:00" \
        --output="${LOG_DIR}/${job_name}_%j.out" \
        --error="${LOG_DIR}/${job_name}_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="${EMAIL}" \
        --dependency="afterany:${dep_str}" \
        --export=ALL \
        cluster/run_extended_eval.sh "${experiment}" \
        | awk '{print $NF}')

    echo "${jid}"
}

# ---------------------------------------------------------------------------
# Parse which groups to run
# ---------------------------------------------------------------------------
GROUPS=("$@")
if [ ${#GROUPS[@]} -eq 0 ]; then
    GROUPS=("topics" "tempsweep" "newmodels")
fi

echo "============================================================"
echo " Extended Experiment Submission"
echo " Groups: ${GROUPS[*]}"
echo " Partition: ${PARTITION}  Account: ${ACCOUNT}"
echo " Logs: ${LOG_DIR}"
echo "============================================================"
echo ""

ALL_EVAL_JOBS=()

# =========================================================================
# GROUP 1: Multi-topic experiments (climate + software, T=0.7, all models)
# =========================================================================
if [[ " ${GROUPS[*]} " =~ " topics " ]]; then
    echo "--- GROUP 1: Multi-topic (climate + software) ---"

    for TOPIC in climate software; do
        EXP="${TOPIC}_T0.7"
        TOPIC_JOBS=()
        for TIER in 1 2 3 4; do
            JID=$(submit_tier "${TIER}" "${TOPIC}" "0.7" "${EXP}" "original" "ext-${TOPIC}-t${TIER}")
            TOPIC_JOBS+=("${JID}")
            echo "  ${EXP} tier ${TIER}: job ${JID}"
        done
        EVAL_JID=$(submit_eval "${EXP}" "${TOPIC_JOBS[@]}")
        ALL_EVAL_JOBS+=("${EVAL_JID}")
        echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${TOPIC_JOBS[*]})"
        echo ""
    done
fi

# =========================================================================
# GROUP 2: Temperature sweep (healthcare, T=0.3/0.5/1.0, tempsweep models)
# =========================================================================
if [[ " ${GROUPS[*]} " =~ " tempsweep " ]]; then
    echo "--- GROUP 2: Temperature sweep (healthcare, T=0.3/0.5/1.0) ---"

    for TEMP in 0.3 0.5 1.0; do
        EXP="healthcare_T${TEMP}"
        TEMP_JOBS=()
        # tempsweep models span tiers 1-4
        for TIER in 1 2 3 4; do
            JID=$(submit_tier "${TIER}" "healthcare" "${TEMP}" "${EXP}" "tempsweep" "tsweep-T${TEMP}-t${TIER}")
            TEMP_JOBS+=("${JID}")
            echo "  ${EXP} tier ${TIER}: job ${JID}"
        done
        EVAL_JID=$(submit_eval "${EXP}" "${TEMP_JOBS[@]}")
        ALL_EVAL_JOBS+=("${EVAL_JID}")
        echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${TEMP_JOBS[*]})"
        echo ""
    done
fi

# =========================================================================
# GROUP 3: New-generation models (healthcare, T=0.7)
# =========================================================================
if [[ " ${GROUPS[*]} " =~ " newmodels " ]]; then
    echo "--- GROUP 3: New-generation models (healthcare T=0.7) ---"

    EXP="newmodels_healthcare_T0.7"
    NEW_JOBS=()
    for TIER in 1 2 3; do
        JID=$(submit_tier "${TIER}" "healthcare" "0.7" "${EXP}" "new" "newmod-t${TIER}")
        NEW_JOBS+=("${JID}")
        echo "  ${EXP} tier ${TIER}: job ${JID}"
    done
    EVAL_JID=$(submit_eval "${EXP}" "${NEW_JOBS[@]}")
    ALL_EVAL_JOBS+=("${EVAL_JID}")
    echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${NEW_JOBS[*]})"
    echo ""
fi

# =========================================================================
# Cross-experiment analysis (runs after ALL eval jobs finish)
# =========================================================================
if [ ${#ALL_EVAL_JOBS[@]} -gt 0 ]; then
    CROSS_DEP=$(IFS=:; echo "${ALL_EVAL_JOBS[*]}")
    CONDA_DIR="${BEEGFS_BASE}/miniconda3"
    CROSS_JID=$(sbatch \
        --job-name="cross-analysis" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --cpus-per-task=8 \
        --mem="16G" \
        --time="02:00:00" \
        --output="${LOG_DIR}/cross_analysis_%j.out" \
        --error="${LOG_DIR}/cross_analysis_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="${EMAIL}" \
        --dependency="afterany:${CROSS_DEP}" \
        --export=ALL \
        --wrap="source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate stochastic_exp && cd ${PROJECT_DIR} && python3 cross_analyzer.py" \
        | awk '{print $NF}')
    echo "--- Cross-experiment analysis: job ${CROSS_JID} ---"
    echo "    Depends on all eval jobs: ${ALL_EVAL_JOBS[*]}"
fi

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================================"
echo " All jobs submitted!"
echo ""
echo " Monitor:  squeue -u \$USER"
echo " Logs:     ${LOG_DIR}/"
echo ""
echo " Results will be in:"
echo "   ${BEEGFS_BASE}/stochastic_exploration/results/<experiment_name>/"
echo "============================================================"
