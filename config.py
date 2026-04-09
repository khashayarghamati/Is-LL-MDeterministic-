"""
Configuration for the LLM Stochasticity Exploration Experiment.

Experiment: Measure how stochastic LLMs are and whether parameter count (B)
            affects stochasticity — including emergent-property thresholds.
Backend:    HuggingFace Transformers (runs on GPU cluster via SLURM).
Methodology: 20 prompts × 20 repetitions per model, stateless, structured JSON output.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    """Specification for a single model to test."""
    name: str               # HuggingFace model ID  (e.g. "Qwen/Qwen2.5-7B-Instruct")
    params_billion: float   # Approximate parameter count in billions
    family: str             # Model family for grouping  (e.g. "qwen2.5")
    tier: int               # 1-4 — maps to SLURM resource profiles

    def __repr__(self):
        return f"{self.name} ({self.params_billion}B, tier {self.tier})"

    @property
    def short_name(self) -> str:
        """Filesystem-safe short name derived from the HF ID."""
        return self.name.replace("/", "--").replace(":", "_")


# ---------------------------------------------------------------------------
# Default models — 0.5 B → 72 B across 4 tiers to capture emergent properties
#
# Tier 1  "Small"    : < 3 B   — pre-emergent baseline
# Tier 2  "Medium"   : 3–8 B   — early emergent behaviours
# Tier 3  "Large"    : 9–32 B  — instruction following, reasoning emerge
# Tier 4  "Frontier" : 70 B+   — full emergent capabilities
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    # ---- Tier 1: Small (<3B) — 1 GPU, fp16 ----
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct",       0.5,  "qwen2.5", 1),
    ModelSpec("meta-llama/Llama-3.2-1B-Instruct",  1.0,  "llama3",  1),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct",       1.5,  "qwen2.5", 1),
    ModelSpec("google/gemma-2-2b-it",              2.0,  "gemma2",  1),

    # ---- Tier 2: Medium (3–8B) — 1 GPU, fp16 ----
    ModelSpec("Qwen/Qwen2.5-3B-Instruct",         3.0,  "qwen2.5", 2),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct",  3.0,  "llama3",  2),
    ModelSpec("microsoft/Phi-3.5-mini-instruct",   3.8,  "phi3",    2),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct",         7.0,  "qwen2.5", 2),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct",  8.0,  "llama3",  2),

    # ---- Tier 3: Large (9–32B) — 1–2 GPUs, fp16 / 8-bit ----
    ModelSpec("google/gemma-2-9b-it",              9.0,  "gemma2",  3),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct",       14.0,  "qwen2.5", 3),
    ModelSpec("google/gemma-2-27b-it",            27.0,  "gemma2",  3),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct",       32.0,  "qwen2.5", 3),

    # ---- Tier 4: Frontier (70B+) — 4 GPUs, fp16 / 4-bit ----
    ModelSpec("meta-llama/Llama-3.1-70B-Instruct", 70.0, "llama3",  4),
    ModelSpec("Qwen/Qwen2.5-72B-Instruct",        72.0, "qwen2.5", 4),
]


# ---------------------------------------------------------------------------
# SLURM resource profiles per tier  (GPU count, RAM, wall-time, etc.)
# Tuned for a cluster with NVIDIA A100-80 GB nodes.
# ---------------------------------------------------------------------------
TIER_RESOURCES = {
    1: {"gpus": 1, "mem_gb": 32,  "cpus": 8,  "time": "06:00:00"},
    2: {"gpus": 1, "mem_gb": 64,  "cpus": 8,  "time": "18:00:00"},
    3: {"gpus": 2, "mem_gb": 128, "cpus": 16, "time": "36:00:00"},
    4: {"gpus": 4, "mem_gb": 256, "cpus": 16, "time": "48:00:00"},
}


@dataclass
class ExperimentConfig:
    """All tuneable knobs for the experiment."""

    # --- Models ---
    models: list = field(default_factory=lambda: list(DEFAULT_MODELS))

    # --- Experiment design ---
    num_repetitions: int = 20          # Times each prompt is sent per model
    temperature: float = 0.7           # Sampling temperature (>0 → stochastic)
    top_p: float = 0.9
    max_tokens: int = 1024
    seed: Optional[int] = None         # None → non-deterministic

    # --- HuggingFace settings ---
    torch_dtype: str = "float16"       # "float16", "bfloat16", "float32"
    # Auto-quantise large models that won't fit in fp16 on available VRAM
    quantize_above_b: float = 27.0     # Models > this use 8-bit
    quantize_4bit_above_b: float = 65.0  # Models > this use 4-bit
    trust_remote_code: bool = True

    # --- Embedding model for semantic metrics (sentence-transformers) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Paths (beegfs on UH cluster) ---
    beegfs_base: str = "/beegfs/general/kg23aay/stochastic_exploration"
    results_dir: str = "/beegfs/general/kg23aay/stochastic_exploration/results"
    raw_responses_dir: str = "/beegfs/general/kg23aay/stochastic_exploration/results/raw_responses"
    metrics_dir: str = "/beegfs/general/kg23aay/stochastic_exploration/results/metrics"
    plots_dir: str = "/beegfs/general/kg23aay/stochastic_exploration/results/plots"

    # --- Timeouts ---
    generation_timeout: int = 300      # seconds per single generation

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.beegfs_base, self.results_dir, self.raw_responses_dir,
                  self.metrics_dir, self.plots_dir]:
            os.makedirs(d, exist_ok=True)

    def get_models_for_tier(self, tier: int) -> list[ModelSpec]:
        """Return only models belonging to the given tier."""
        return [m for m in self.models if m.tier == tier]
