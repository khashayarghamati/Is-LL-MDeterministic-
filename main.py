#!/usr/bin/env python3
"""
LLM Stochasticity Exploration Experiment
=========================================

Measures how stochastic LLMs are, whether parameter count affects it,
and whether emergent properties change stochasticity patterns.

Backend: HuggingFace Transformers (GPU).  Designed for SLURM clusters.

Usage:
    python main.py run                    # Run all tiers
    python main.py run   --tier 1         # Run only small models (<3B)
    python main.py run   --tier 2         # Run medium models  (3-8B)
    python main.py run   --tier 3         # Run large models   (9-32B)
    python main.py run   --tier 4         # Run frontier models (70B+)
    python main.py evaluate               # Compute stochasticity metrics
    python main.py analyze                # Statistical analysis + plots
    python main.py all   --tier 1         # Run → Evaluate → Analyze for tier 1
    python main.py status                 # Check experiment progress

Options:
    --tier  N         Only run models in tier N  (1/2/3/4)
    --reps  N         Override repetitions per prompt  (default: 20)
    --temp  FLOAT     Sampling temperature             (default: 0.7)
    --models m1,m2    Comma-separated HuggingFace model IDs (overrides defaults)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import ExperimentConfig, ModelSpec, DEFAULT_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_config(args) -> ExperimentConfig:
    """Build ExperimentConfig from CLI arguments."""
    cfg = ExperimentConfig()

    if args.reps:
        cfg.num_repetitions = args.reps
    if args.temp is not None:
        cfg.temperature = args.temp
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
        default_map = {m.name: m for m in DEFAULT_MODELS}
        models = []
        for mid in model_ids:
            if mid in default_map:
                models.append(default_map[mid])
            else:
                params = _guess_params(mid)
                family = mid.split("/")[-1].split("-")[0].lower()
                tier = (1 if params < 3 else 2 if params < 9
                        else 3 if params <= 32 else 4)
                models.append(ModelSpec(mid, params, family, tier))
        cfg.models = models

    cfg.ensure_dirs()
    return cfg


def _guess_params(model_id: str) -> float:
    """Try to extract parameter count from model name like 'Qwen2.5-7B-Instruct'."""
    import re
    match = re.search(r"(\d+\.?\d*)[Bb]", model_id)
    if match:
        return float(match.group(1))
    return 0.0


def cmd_run(cfg: ExperimentConfig, tier: int | None):
    """Run the experiment: send all prompts to all models."""
    from runner import HuggingFaceRunner

    models = cfg.get_models_for_tier(tier) if tier else cfg.models

    logger.info("=" * 60)
    logger.info("STARTING EXPERIMENT")
    logger.info("  Tier:        %s", tier or "ALL")
    logger.info("  Models:      %d", len(models))
    logger.info("  Prompts:     20")
    logger.info("  Repetitions: %d per prompt per model", cfg.num_repetitions)
    logger.info("  Total calls: %d", len(models) * 20 * cfg.num_repetitions)
    logger.info("  Temperature: %.2f", cfg.temperature)
    logger.info("  Dtype:       %s", cfg.torch_dtype)
    logger.info("=" * 60)

    runner = HuggingFaceRunner(cfg)
    runner.run_experiment(tier=tier)


def cmd_evaluate(cfg: ExperimentConfig):
    """Evaluate saved responses with stochasticity metrics."""
    from evaluator import StochasticityEvaluator

    logger.info("=" * 60)
    logger.info("EVALUATING STOCHASTICITY METRICS")
    logger.info("=" * 60)

    evaluator = StochasticityEvaluator(cfg)
    results = evaluator.evaluate_all()

    print("\n--- Quick Summary ---")
    for model_name, data in sorted(results.items(),
                                   key=lambda x: x[1]["params_billion"]):
        agg = data.get("aggregate", {})
        si = agg.get("stochasticity_index", {})
        if si:
            print(f"  {model_name:45s} ({data['params_billion']:5.1f}B)  "
                  f"Stochasticity: {si['mean']:.4f} ± {si['std']:.4f}")


def cmd_analyze(cfg: ExperimentConfig):
    """Run statistical analysis and generate visualizations."""
    from analyzer import ResultAnalyzer

    logger.info("=" * 60)
    logger.info("RUNNING ANALYSIS & GENERATING PLOTS")
    logger.info("=" * 60)

    analyzer = ResultAnalyzer(cfg)
    report = analyzer.run_full_analysis()
    print("\n" + report)


def cmd_status(cfg: ExperimentConfig):
    """Check how much of the experiment has been completed."""
    raw_dir = Path(cfg.raw_responses_dir)
    if not raw_dir.exists():
        print("No results yet. Run: python main.py run")
        return

    total_expected = len(cfg.models)
    completed_files = list(raw_dir.glob("*.json"))
    print(f"\nExperiment Status")
    print(f"  Models configured: {total_expected}")
    print(f"  Result files:      {len(completed_files)}")
    print()

    for f in sorted(completed_files):
        with open(f) as fh:
            data = json.load(fh)
        model = data.get("model", "?")
        tier = data.get("tier", "?")
        responses = data.get("responses", {})
        total_reps = sum(
            len(p.get("repetitions", []))
            for p in responses.values()
        )
        expected_reps = 20 * cfg.num_repetitions
        pct = (total_reps / expected_reps * 100) if expected_reps > 0 else 0
        status = "DONE" if total_reps >= expected_reps else "PARTIAL"
        print(f"  [{status:7s}] T{tier}  {model:45s}  "
              f"{total_reps:4d}/{expected_reps} calls ({pct:.0f}%)")

    metrics_path = Path(cfg.metrics_dir) / "evaluation_results.json"
    print(f"\n  Evaluation:  {'DONE' if metrics_path.exists() else 'NOT YET'}")
    report_path = Path(cfg.results_dir) / "analysis_report.txt"
    print(f"  Analysis:    {'DONE' if report_path.exists() else 'NOT YET'}")


def cmd_all(cfg: ExperimentConfig, tier: int | None):
    """Run the complete pipeline: experiment → evaluation → analysis."""
    cmd_run(cfg, tier)
    cmd_evaluate(cfg)
    cmd_analyze(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Stochasticity Exploration Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["run", "evaluate", "analyze", "all", "status"],
        help="Which stage to execute",
    )
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Only process models in this tier (1=small, 4=frontier)")
    parser.add_argument("--reps", type=int, default=None,
                        help="Number of repetitions per prompt (default: 20)")
    parser.add_argument("--temp", type=float, default=None,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated HuggingFace model IDs (overrides defaults)")

    args = parser.parse_args()
    cfg = build_config(args)

    if args.command in ("run", "all"):
        {"run": cmd_run, "all": cmd_all}[args.command](cfg, args.tier)
    else:
        {"evaluate": cmd_evaluate, "analyze": cmd_analyze,
         "status": cmd_status}[args.command](cfg)


if __name__ == "__main__":
    main()
