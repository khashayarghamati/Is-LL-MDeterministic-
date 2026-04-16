#!/usr/bin/env python3
"""
Cross-Experiment Analyzer — Compares stochasticity across topics & temperatures.

Produces:
  1. Topic generalizability analysis   (healthcare vs. climate vs. software)
  2. Temperature sensitivity analysis  (T=0.3, 0.5, 0.7, 1.0)
  3. New vs. old model generation comparison
  4. Combined scaling curve with confidence bands
  5. Summary report + plots

Usage:
    python cross_analyzer.py                              # Auto-discover experiments
    python cross_analyzer.py --base /path/to/results      # Custom base dir
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CrossExperimentAnalyzer:
    """Analyze and compare results across multiple experiments."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "cross_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_experiments(self) -> dict[str, Path]:
        """Find all experiment directories that have evaluation_results.json."""
        experiments = {}

        # Check flat layout (original experiment)
        flat_metrics = self.base_dir / "metrics" / "evaluation_results.json"
        if flat_metrics.exists():
            experiments["original_healthcare_T0.7"] = self.base_dir

        # Check subdirectories
        for subdir in sorted(self.base_dir.iterdir()):
            if subdir.is_dir() and subdir.name != "cross_analysis":
                metrics_file = subdir / "metrics" / "evaluation_results.json"
                if metrics_file.exists():
                    experiments[subdir.name] = subdir

        logger.info("Discovered %d experiments: %s",
                     len(experiments), list(experiments.keys()))
        return experiments

    def load_experiment(self, name: str, path: Path) -> pd.DataFrame:
        """Load aggregate metrics from one experiment."""
        metrics_path = path / "metrics" / "evaluation_results.json"
        with open(metrics_path) as f:
            results = json.load(f)

        rows = []
        for model_name, data in results.items():
            agg = data.get("aggregate", {})
            si = agg.get("stochasticity_index", {})
            if not si:
                continue

            row = {
                "experiment": name,
                "model": model_name,
                "params_B": data["params_billion"],
                "family": data["family"],
                "stochasticity_index": si.get("mean", np.nan),
                "si_std": si.get("std", np.nan),
            }
            # Add all other metric means
            for metric, vals in agg.items():
                if isinstance(vals, dict) and "mean" in vals:
                    row[f"{metric}_mean"] = vals["mean"]
            rows.append(row)

        return pd.DataFrame(rows)

    def load_all(self) -> pd.DataFrame:
        """Load and concatenate all experiments into a single DataFrame."""
        experiments = self.discover_experiments()
        dfs = []
        for name, path in experiments.items():
            df = self.load_experiment(name, path)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            logger.error("No experiment data found!")
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Combined data: %d rows across %d experiments",
                     len(combined), combined["experiment"].nunique())
        return combined

    # ------------------------------------------------------------------
    # Parse experiment name into topic + temperature
    # ------------------------------------------------------------------
    @staticmethod
    def parse_experiment_name(name: str) -> tuple[str, float]:
        """Extract topic and temperature from experiment name."""
        # Patterns: "climate_T0.7", "healthcare_T0.3", "original_healthcare_T0.7"
        parts = name.lower()
        temp = 0.7  # default
        topic = "healthcare"

        if "_t" in parts:
            idx = parts.rfind("_t")
            try:
                temp = float(parts[idx + 2:])
            except ValueError:
                pass
            topic_part = parts[:idx]
        else:
            topic_part = parts

        # Clean topic
        for t in ["healthcare", "climate", "software"]:
            if t in topic_part:
                topic = t
                break

        return topic, temp

    # ------------------------------------------------------------------
    # Analysis 1: Topic Generalizability
    # ------------------------------------------------------------------
    def analyze_topic_generalizability(self, df: pd.DataFrame):
        """Compare stochasticity patterns across topics."""
        # Filter to T=0.7 experiments
        topic_df = df.copy()
        topic_df["topic"], topic_df["temperature"] = zip(
            *topic_df["experiment"].map(self.parse_experiment_name))
        topic_df = topic_df[np.isclose(topic_df["temperature"], 0.7)]

        topics = topic_df["topic"].unique()
        if len(topics) < 2:
            logger.warning("Need ≥2 topics for generalizability analysis. Found: %s", topics)
            return

        logger.info("Topic generalizability: %s", topics)

        # Plot: overlaid scaling curves per topic
        fig, ax = plt.subplots(figsize=(12, 7))
        palette = sns.color_palette("Set1", len(topics))

        for i, topic in enumerate(sorted(topics)):
            sub = topic_df[topic_df["topic"] == topic].sort_values("params_B")
            ax.scatter(sub["params_B"], sub["stochasticity_index"],
                       color=palette[i], s=100, zorder=5, edgecolors="black",
                       linewidth=0.5, label=topic.title())

            # Regression line per topic
            mask = sub["stochasticity_index"].notna()
            if mask.sum() >= 3:
                x = sub.loc[mask, "params_B"].values
                y = sub.loc[mask, "stochasticity_index"].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "--", color=palette[i], alpha=0.6)

        ax.set_xlabel("Parameter Count (Billions)", fontsize=12)
        ax.set_ylabel("Stochasticity Index", fontsize=12)
        ax.set_title("Scaling vs. Stochasticity Across Topics (T=0.7)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "topic_generalizability.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Correlation per topic
        report_lines = ["TOPIC GENERALIZABILITY ANALYSIS", "=" * 50]
        for topic in sorted(topics):
            sub = topic_df[topic_df["topic"] == topic]
            mask = sub["stochasticity_index"].notna()
            if mask.sum() >= 3:
                rho, p_val = stats.spearmanr(
                    sub.loc[mask, "params_B"], sub.loc[mask, "stochasticity_index"])
                report_lines.append(
                    f"  {topic:15s}  Spearman ρ = {rho:+.4f}  (p = {p_val:.4f})  "
                    f"n = {mask.sum()}")

        # Cross-topic correlation of model rankings
        if len(topics) >= 2:
            report_lines.append("")
            report_lines.append("Cross-topic rank correlation:")
            pivot = topic_df.pivot_table(
                values="stochasticity_index", index="model", columns="topic")
            for t1, t2 in [(a, b) for i, a in enumerate(sorted(topics))
                           for b in sorted(topics)[i + 1:]]:
                if t1 in pivot.columns and t2 in pivot.columns:
                    shared = pivot[[t1, t2]].dropna()
                    if len(shared) >= 3:
                        rho, p_val = stats.spearmanr(shared[t1], shared[t2])
                        report_lines.append(
                            f"  {t1} ↔ {t2}:  ρ = {rho:+.4f}  (p = {p_val:.4f})  "
                            f"n = {len(shared)} shared models")

        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # Analysis 2: Temperature Sensitivity
    # ------------------------------------------------------------------
    def analyze_temperature_sensitivity(self, df: pd.DataFrame):
        """Show how temperature modulates stochasticity across scales."""
        temp_df = df.copy()
        temp_df["topic"], temp_df["temperature"] = zip(
            *temp_df["experiment"].map(self.parse_experiment_name))
        temp_df = temp_df[temp_df["topic"] == "healthcare"]

        temperatures = sorted(temp_df["temperature"].unique())
        if len(temperatures) < 2:
            logger.warning("Need ≥2 temperatures. Found: %s", temperatures)
            return

        logger.info("Temperature sensitivity: %s", temperatures)

        # Plot: scaling curve colored by temperature
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=min(temperatures), vmax=max(temperatures))

        for temp in temperatures:
            sub = temp_df[np.isclose(temp_df["temperature"], temp)].sort_values("params_B")
            color = cmap(norm(temp))
            ax.scatter(sub["params_B"], sub["stochasticity_index"],
                       color=color, s=100, zorder=5, edgecolors="black",
                       linewidth=0.5, label=f"T={temp}")

            mask = sub["stochasticity_index"].notna()
            if mask.sum() >= 3:
                x = sub.loc[mask, "params_B"].values
                y = sub.loc[mask, "stochasticity_index"].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "--", color=color, alpha=0.6)

        ax.set_xlabel("Parameter Count (Billions)", fontsize=12)
        ax.set_ylabel("Stochasticity Index", fontsize=12)
        ax.set_title("Temperature Sensitivity: Stochasticity vs. Scale", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "temperature_sensitivity.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Temperature × model interaction heatmap
        pivot = temp_df.pivot_table(
            values="stochasticity_index", index="model", columns="temperature",
            aggfunc="mean")
        if not pivot.empty:
            # Sort by average stochasticity
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

            fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                        linewidths=0.5, ax=ax)
            ax.set_title("Stochasticity: Model × Temperature", fontsize=13)
            ax.set_xlabel("Temperature")
            fig.tight_layout()
            fig.savefig(self.output_dir / "plots" / "temp_model_heatmap.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Report
        report_lines = ["TEMPERATURE SENSITIVITY ANALYSIS", "=" * 50]
        for temp in temperatures:
            sub = temp_df[np.isclose(temp_df["temperature"], temp)]
            mask = sub["stochasticity_index"].notna()
            if mask.sum() >= 3:
                rho, p_val = stats.spearmanr(
                    sub.loc[mask, "params_B"], sub.loc[mask, "stochasticity_index"])
                mean_si = sub["stochasticity_index"].mean()
                report_lines.append(
                    f"  T={temp:.1f}  mean SI={mean_si:.4f}  "
                    f"Spearman ρ={rho:+.4f} (p={p_val:.4f})  n={mask.sum()}")

        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # Analysis 3: New vs. Old Model Comparison
    # ------------------------------------------------------------------
    def analyze_model_generations(self, df: pd.DataFrame):
        """Compare stochasticity between original and new-generation models."""
        # Identify old vs new families
        old_families = {"qwen2.5", "llama3", "gemma2", "phi3"}
        new_families = {"qwen3", "gemma3", "phi4", "llama4"}

        gen_df = df.copy()
        gen_df["generation"] = gen_df["family"].apply(
            lambda f: "original" if f in old_families
            else "new" if f in new_families else "unknown")

        gen_df = gen_df[gen_df["generation"] != "unknown"]

        if gen_df["generation"].nunique() < 2:
            logger.warning("Need both old and new generation models for comparison.")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        for gen, marker, color in [("original", "o", "#2196F3"), ("new", "^", "#FF5722")]:
            sub = gen_df[gen_df["generation"] == gen].sort_values("params_B")
            ax.scatter(sub["params_B"], sub["stochasticity_index"],
                       color=color, marker=marker, s=120, zorder=5,
                       edgecolors="black", linewidth=0.5,
                       label=f"{gen.title()} gen models")

            mask = sub["stochasticity_index"].notna()
            if mask.sum() >= 3:
                x = sub.loc[mask, "params_B"].values
                y = sub.loc[mask, "stochasticity_index"].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "--", color=color, alpha=0.6)

        ax.set_xlabel("Parameter Count (Billions)", fontsize=12)
        ax.set_ylabel("Stochasticity Index", fontsize=12)
        ax.set_title("Model Generation Comparison: Old vs. New Architectures", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "generation_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Mann-Whitney U test
        report_lines = ["MODEL GENERATION COMPARISON", "=" * 50]
        old_si = gen_df[gen_df["generation"] == "original"]["stochasticity_index"].dropna()
        new_si = gen_df[gen_df["generation"] == "new"]["stochasticity_index"].dropna()

        if len(old_si) >= 3 and len(new_si) >= 3:
            u_stat, u_p = stats.mannwhitneyu(old_si, new_si, alternative="two-sided")
            report_lines.append(f"  Original gen: mean SI = {old_si.mean():.4f}  (n={len(old_si)})")
            report_lines.append(f"  New gen:      mean SI = {new_si.mean():.4f}  (n={len(new_si)})")
            report_lines.append(f"  Mann-Whitney U = {u_stat:.1f}  (p = {u_p:.4f})")

        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # Combined scaling curve with confidence bands
    # ------------------------------------------------------------------
    def plot_combined_scaling(self, df: pd.DataFrame):
        """Master plot: all models, all experiments, with LOESS-like trend."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Color by experiment
        experiments = df["experiment"].unique()
        palette = sns.color_palette("husl", len(experiments))
        exp_colors = dict(zip(experiments, palette))

        for exp in experiments:
            sub = df[df["experiment"] == exp].sort_values("params_B")
            ax.scatter(sub["params_B"], sub["stochasticity_index"],
                       color=exp_colors[exp], s=60, alpha=0.7,
                       edgecolors="black", linewidth=0.3, label=exp)

        # Overall trend (all data)
        all_valid = df.dropna(subset=["stochasticity_index"])
        if len(all_valid) >= 5:
            x = all_valid["params_B"].values
            y = all_valid["stochasticity_index"].values
            z = np.polyfit(np.log1p(x), y, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_smooth, p(np.log1p(x_smooth)), "-", color="black",
                    linewidth=2.5, alpha=0.8, label="Overall trend")

        ax.set_xlabel("Parameter Count (Billions)", fontsize=13)
        ax.set_ylabel("Stochasticity Index", fontsize=13)
        ax.set_title("Combined Scaling Curve: All Experiments", fontsize=15)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "combined_scaling.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> str:
        """Run all cross-experiment analyses."""
        df = self.load_all()
        if df.empty:
            return "No experiment data found."

        report_sections = []
        report_sections.append("=" * 70)
        report_sections.append("  CROSS-EXPERIMENT ANALYSIS REPORT")
        report_sections.append("=" * 70)
        report_sections.append(f"\nExperiments found: {df['experiment'].nunique()}")
        report_sections.append(f"Total model entries: {len(df)}")
        report_sections.append(f"Unique models: {df['model'].nunique()}")
        report_sections.append("")

        # Analysis 1: Topic generalizability
        result = self.analyze_topic_generalizability(df)
        if result:
            report_sections.append(result)
            report_sections.append("")

        # Analysis 2: Temperature sensitivity
        result = self.analyze_temperature_sensitivity(df)
        if result:
            report_sections.append(result)
            report_sections.append("")

        # Analysis 3: Model generations
        result = self.analyze_model_generations(df)
        if result:
            report_sections.append(result)
            report_sections.append("")

        # Combined scaling plot
        self.plot_combined_scaling(df)

        # Overall statistics
        report_sections.append("OVERALL STATISTICS")
        report_sections.append("=" * 50)
        mask = df["stochasticity_index"].notna()
        if mask.sum() >= 3:
            rho, p_val = stats.spearmanr(
                df.loc[mask, "params_B"], df.loc[mask, "stochasticity_index"])
            report_sections.append(
                f"  All data combined: Spearman ρ = {rho:+.4f}  (p = {p_val:.6f})  "
                f"n = {mask.sum()}")

        report_sections.append("")
        report_sections.append("=" * 70)
        report_sections.append(f"Plots saved in: {self.output_dir / 'plots'}")
        report_sections.append("=" * 70)

        report = "\n".join(report_sections)

        # Save report
        report_path = self.output_dir / "cross_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Save combined data
        df.to_csv(self.output_dir / "combined_results.csv", index=False)

        logger.info("Cross-analysis complete. Report: %s", report_path)
        return report


def main():
    parser = argparse.ArgumentParser(description="Cross-experiment analyzer")
    parser.add_argument("--base", type=str,
                        default="/beegfs/general/kg23aay/stochastic_exploration/results",
                        help="Base results directory")
    args = parser.parse_args()

    analyzer = CrossExperimentAnalyzer(args.base)
    report = analyzer.run_full_analysis()
    print(report)


if __name__ == "__main__":
    main()
