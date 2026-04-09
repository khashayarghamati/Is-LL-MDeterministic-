#!/bin/bash
# ===========================================================================
# run_evaluate.sh — SLURM batch script for evaluation + analysis.
#
# Runs AFTER all tier jobs complete (submitted with --dependency by
# submit_all.sh).  Needs 1 GPU for the embedding model, minimal resources.
# ===========================================================================

#SBATCH --job-name=stoch-eval
#SBATCH --partition=gpu                        # ← change to your GPU partition
#SBATCH --account=your-account                 # ← change to your allocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@herts.ac.uk     # ← change to your email

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load cuda/12.1 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
fi

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

echo "=============================================="
echo " Stochastic Exploration — Evaluate & Analyze"
echo " $(date)"
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
echo "Results in: ${PROJECT_DIR}/results/"
