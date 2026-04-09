"""
Experiment Runner — HuggingFace Transformers backend.

Loads one model at a time, runs all prompts × repetitions (stateless),
saves results, frees GPU memory, then moves to the next model.
Supports automatic quantisation for models that exceed single-GPU VRAM.
"""

import gc
import json
import time
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from config import ExperimentConfig, ModelSpec
from prompts import get_all_prompts

logger = logging.getLogger(__name__)


class HuggingFaceRunner:
    """Manages model loading, generation, and GPU lifecycle."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.cfg.ensure_dirs()
        self._model = None
        self._tokenizer = None
        self._current_model_name: str | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _resolve_dtype(self):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.cfg.torch_dtype, torch.float16)

    def _quantization_config(self, spec: ModelSpec) -> BitsAndBytesConfig | None:
        """Pick quantisation strategy based on model size thresholds."""
        if spec.params_billion > self.cfg.quantize_4bit_above_b:
            logger.info("  Using 4-bit quantisation for %s (%.0fB params)",
                        spec.name, spec.params_billion)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._resolve_dtype(),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if spec.params_billion > self.cfg.quantize_above_b:
            logger.info("  Using 8-bit quantisation for %s (%.0fB params)",
                        spec.name, spec.params_billion)
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def load_model(self, spec: ModelSpec):
        """Load model and tokenizer onto available GPUs."""
        if self._current_model_name == spec.name:
            return  # Already loaded

        self.unload_model()

        logger.info("Loading model: %s …", spec.name)
        t0 = time.time()

        quant_cfg = self._quantization_config(spec)

        self._tokenizer = AutoTokenizer.from_pretrained(
            spec.name,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        # Many models lack a pad token — reuse eos
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = dict(
            torch_dtype=self._resolve_dtype(),
            device_map="auto",
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg

        self._model = AutoModelForCausalLM.from_pretrained(
            spec.name, **load_kwargs
        )
        self._model.eval()
        self._current_model_name = spec.name

        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs. Device map: %s",
                     elapsed, getattr(self._model, "hf_device_map", "single-device"))

    def unload_model(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------
    def _generate_one(self, prompt: str) -> dict:
        """Run a single stateless generation and return metadata + response."""
        # Build chat-format input
        messages = [{"role": "user", "content": prompt}]

        try:
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without a chat template
            input_text = prompt

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True,
        ).to(self._model.device)

        t0 = time.time()
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            elapsed = time.time() - t0

            # Decode only the newly generated tokens
            new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

            return {
                "response": response,
                "num_tokens_generated": len(new_token_ids),
                "wall_time_s": round(elapsed, 3),
                "error": None,
            }
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("Generation failed: %s", exc)
            return {
                "response": "",
                "num_tokens_generated": 0,
                "wall_time_s": round(elapsed, 3),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------
    def _result_path(self, model: ModelSpec) -> Path:
        return Path(self.cfg.raw_responses_dir) / f"{model.short_name}.json"

    def _load_existing(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_results(self, path: Path, model: ModelSpec, responses: dict):
        data = {
            "model": model.name,
            "params_billion": model.params_billion,
            "family": model.family,
            "tier": model.tier,
            "config": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_tokens": self.cfg.max_tokens,
                "num_repetitions": self.cfg.num_repetitions,
                "seed": self.cfg.seed,
                "torch_dtype": self.cfg.torch_dtype,
            },
            "responses": responses,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def run_experiment(self, tier: int | None = None):
        """
        Run the full experiment.

        Args:
            tier: If given, only run models in that tier (1-4).
                  None → run all models.
        """
        models = (self.cfg.get_models_for_tier(tier) if tier
                  else self.cfg.models)
        if not models:
            logger.warning("No models selected for tier=%s", tier)
            return

        all_prompts = get_all_prompts()
        total_models = len(models)

        for mi, model in enumerate(models, 1):
            result_path = self._result_path(model)
            existing = self._load_existing(result_path)

            logger.info(
                "=== Model %d/%d: %s (%.1fB, tier %d) ===",
                mi, total_models, model.name, model.params_billion, model.tier,
            )

            # Load model onto GPU(s)
            self.load_model(model)

            model_results = existing.get("responses", {})
            calls_made = 0
            calls_skipped = 0
            total_calls = len(all_prompts) * self.cfg.num_repetitions

            pbar = tqdm(total=total_calls, desc=model.short_name, unit="call")

            for prompt_id, angle, full_prompt in all_prompts:
                if prompt_id not in model_results:
                    model_results[prompt_id] = {
                        "angle": angle,
                        "prompt": full_prompt,
                        "repetitions": [],
                    }

                existing_reps = len(model_results[prompt_id]["repetitions"])
                needed = self.cfg.num_repetitions - existing_reps

                if needed <= 0:
                    pbar.update(self.cfg.num_repetitions)
                    calls_skipped += self.cfg.num_repetitions
                    continue

                if existing_reps > 0:
                    pbar.update(existing_reps)
                    calls_skipped += existing_reps

                for rep in range(needed):
                    result = self._generate_one(full_prompt)
                    result["repetition"] = existing_reps + rep + 1
                    model_results[prompt_id]["repetitions"].append(result)
                    calls_made += 1
                    pbar.update(1)

                    # Save incrementally every 10 calls
                    if calls_made % 10 == 0:
                        self._save_results(result_path, model, model_results)

            pbar.close()
            self._save_results(result_path, model, model_results)
            logger.info(
                "Model %s done — %d new calls, %d skipped (resumed).",
                model.name, calls_made, calls_skipped,
            )

            # Free GPU before loading the next model
            self.unload_model()

        logger.info("Experiment complete. Raw results in: %s",
                     self.cfg.raw_responses_dir)
