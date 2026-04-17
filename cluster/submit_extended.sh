#!/bin/bash
# ===========================================================================
# submit_extended.sh — Submit ALL extended experiments to SLURM.
#
# GROUP 1: Multi-topic (climate + software) with original models at T=0.7
# GROUP 2: Temperature sweep (T=0.3, 0.5, 1.0) on healthcare, 6 models
# GROUP 3: New-generation models (Gemma 3, Qwen 3, Phi-4, Llama 4)
#
# Usage:
#   bash cluster/submit_extended.sh              # Submit everything
#   bash cluster/submit_extended.sh topics        # Only multi-topic
#   bash cluster/submit_extended.sh tempsweep     # Only temperature sweep
#   bash cluster/submit_extended.sh newmodels     # Only new models
# ===========================================================================

set -eo pipefail

BEEGFS_BASE="/beegfs/general/kg23aay"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${BEEGFS_BASE}/stochastic_exploration/logs"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Working directory: ${PROJECT_DIR}"
echo "Log directory: ${LOG_DIR}"

# =========================================================================
# Cluster configuration
# =========================================================================
PARTITION="gpu"
ACCOUNT="cs"
EMAIL="kg23aay@herts.ac.uk"

# =========================================================================
# Resource tables
# =========================================================================
get_gpus()  { case $1 in 1) echo 1;; 2) echo 1;; 3) echo 2;; 4) echo 4;; esac; }
get_cpus()  { case $1 in 1) echo 8;; 2) echo 8;; 3) echo 16;; 4) echo 16;; esac; }
get_mem()   { case $1 in 1) echo "32G";; 2) echo "64G";; 3) echo "128G";; 4) echo "256G";; esac; }
get_time()  { case $1 in 1) echo "06:00:00";; 2) echo "18:00:00";; 3) echo "36:00:00";; 4) echo "4-00:00:00";; esac; }

# ---------------------------------------------------------------------------
# Helper: submit one tier job — prints job ID
# ---------------------------------------------------------------------------
submit_tier() {
    local tier=$1 topic=$2 temp=$3 experiment=$4 model_set=$5 job_name=$6

    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres="gpu:$(get_gpus "${tier}")" \
        --cpus-per-task="$(get_cpus "${tier}")" \
        --mem="$(get_mem "${tier}")" \
        --time="$(get_time "${tier}")" \
        --output="${LOG_DIR}/${job_name}_%j.out" \
        --error="${LOG_DIR}/${job_name}_%j.err" \
        --mail-type=END,FAIL \
        --mail-user="${EMAIL}" \
        --export=ALL \
        cluster/run_extended_tier.sh "${tier}" "${topic}" "${temp}" "${experiment}" "${model_set}" \
        | awk '{print $NF}'
}

# ---------------------------------------------------------------------------
# Helper: submit eval job with dependency
# ---------------------------------------------------------------------------
submit_eval() {
    local experiment=$1 dep_str=$2
    local job_name="eval-${experiment}"

    sbatch \
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
        | awk '{print $NF}'
}

# ---------------------------------------------------------------------------
# Decide which groups to run
# ---------------------------------------------------------------------------
RUN_TOPICS=false
RUN_TEMPSWEEP=false
RUN_NEWMODELS=false

if [ $# -eq 0 ]; then
    # No arguments → run everything
    RUN_TOPICS=true
    RUN_TEMPSWEEP=true
    RUN_NEWMODELS=true
else
    for arg in "$@"; do
        case "${arg}" in
            topics)    RUN_TOPICS=true ;;
            tempsweep) RUN_TEMPSWEEP=true ;;
            newmodels) RUN_NEWMODELS=true ;;
            *) echo "Unknown group: ${arg}. Use: topics, tempsweep, newmodels"; exit 1 ;;
        esac
    done
fi

echo ""
echo "============================================================"
echo " Extended Experiment Submission"
echo " topics=${RUN_TOPICS}  tempsweep=${RUN_TEMPSWEEP}  newmodels=${RUN_NEWMODELS}"
echo " Partition: ${PARTITION}  Account: ${ACCOUNT}"
echo "============================================================"
echo ""

ALL_EVAL_JOBS=""

# =========================================================================
# GROUP 1: Multi-topic (climate + software, T=0.7, original models)
# =========================================================================
if [ "${RUN_TOPICS}" = true ]; then
    echo "--- GROUP 1: Multi-topic (climate + software) ---"

    for TOPIC in climate software; do
        EXP="${TOPIC}_T0.7"
        TIER_JOBS=""
        for TIER in 1 2 3 4; do
            JID=$(submit_tier "${TIER}" "${TOPIC}" "0.7" "${EXP}" "original" "ext-${TOPIC}-t${TIER}")
            echo "  ${EXP} tier ${TIER}: job ${JID}"
            if [ -n "${TIER_JOBS}" ]; then
                TIER_JOBS="${TIER_JOBS}:${JID}"
            else
                TIER_JOBS="${JID}"
            fi
        done
        EVAL_JID=$(submit_eval "${EXP}" "${TIER_JOBS}")
        echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${TIER_JOBS})"
        if [ -n "${ALL_EVAL_JOBS}" ]; then
            ALL_EVAL_JOBS="${ALL_EVAL_JOBS}:${EVAL_JID}"
        else
            ALL_EVAL_JOBS="${EVAL_JID}"
        fi
        echo ""
    done
fi

# =========================================================================
# GROUP 2: Temperature sweep (healthcare, T=0.3/0.5/1.0, tempsweep models)
# =========================================================================
if [ "${RUN_TEMPSWEEP}" = true ]; then
    echo "--- GROUP 2: Temperature sweep (healthcare, T=0.3/0.5/1.0) ---"

    for TEMP in 0.3 0.5 1.0; do
        EXP="healthcare_T${TEMP}"
        TIER_JOBS=""
        for TIER in 1 2 3 4; do
            JID=$(submit_tier "${TIER}" "healthcare" "${TEMP}" "${EXP}" "tempsweep" "tsweep-T${TEMP}-t${TIER}")
            echo "  ${EXP} tier ${TIER}: job ${JID}"
            if [ -n "${TIER_JOBS}" ]; then
                TIER_JOBS="${TIER_JOBS}:${JID}"
            else
                TIER_JOBS="${JID}"
            fi
        done
        EVAL_JID=$(submit_eval "${EXP}" "${TIER_JOBS}")
        echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${TIER_JOBS})"
        if [ -n "${ALL_EVAL_JOBS}" ]; then
            ALL_EVAL_JOBS="${ALL_EVAL_JOBS}:${EVAL_JID}"
        else
            ALL_EVAL_JOBS="${EVAL_JID}"
        fi
        echo ""
    done
fi

# =========================================================================
# GROUP 3: New-generation models (healthcare, T=0.7)
# =========================================================================
if [ "${RUN_NEWMODELS}" = true ]; then
    echo "--- GROUP 3: New-generation models (healthcare T=0.7) ---"

    EXP="newmodels_healthcare_T0.7"
    TIER_JOBS=""
    for TIER in 1 2 3; do
        JID=$(submit_tier "${TIER}" "healthcare" "0.7" "${EXP}" "new" "newmod-t${TIER}")
        echo "  ${EXP} tier ${TIER}: job ${JID}"
        if [ -n "${TIER_JOBS}" ]; then
            TIER_JOBS="${TIER_JOBS}:${JID}"
        else
            TIER_JOBS="${JID}"
        fi
    done
    EVAL_JID=$(submit_eval "${EXP}" "${TIER_JOBS}")
    echo "  ${EXP} eval: job ${EVAL_JID} (depends on ${TIER_JOBS})"
    if [ -n "${ALL_EVAL_JOBS}" ]; then
        ALL_EVAL_JOBS="${ALL_EVAL_JOBS}:${EVAL_JID}"
    else
        ALL_EVAL_JOBS="${EVAL_JID}"
    fi
    echo ""
fi

# =========================================================================
# Cross-experiment analysis (runs after ALL eval jobs finish)
# =========================================================================
if [ -n "${ALL_EVAL_JOBS}" ]; then
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
        --dependency="afterany:${ALL_EVAL_JOBS}" \
        --export=ALL \
        --wrap="source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate stochastic_exp && cd ${PROJECT_DIR} && python3 cross_analyzer.py" \
        | awk '{print $NF}')
    echo "--- Cross-experiment analysis: job ${CROSS_JID} ---"
    echo "    Depends on: ${ALL_EVAL_JOBS}"
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
