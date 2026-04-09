"""
Statistical Analysis & Visualization of stochasticity metrics.

Produces:
  - Scatter plot: parameter count vs. stochasticity index
  - Heatmap: all metrics × all models
  - Box plots: per-prompt stochasticity distribution
  - Radar chart: metric breakdown per model
  - Correlation analysis (Spearman, Pearson)
  - Kruskal-Wallis test across model size groups
  - Summary report (text + CSV)
"""

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

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze and visualize stochasticity evaluation results."""

    # Metrics to include in the comparative heatmap
    HEATMAP_METRICS = [
        "self_bleu", "rouge_l", "jaccard", "tfidf_cosine",
        "unique_ngram_ratio", "embedding_cosine", "semantic_entropy",
        "vendi_score", "template_adherence", "response_length_cv",
        "key_point_consistency", "confidence_consistency",
        "timeline_consistency", "stochasticity_index",
    ]

    # Readable labels
    METRIC_LABELS = {
        "self_bleu": "Self-BLEU",
        "rouge_l": "ROUGE-L",
        "jaccard": "Jaccard Sim.",
        "tfidf_cosine": "TF-IDF Cosine",
        "unique_ngram_ratio": "Unique N-gram %",
        "embedding_cosine": "Embed. Cosine",
        "semantic_entropy": "Semantic Entropy",
        "vendi_score": "Vendi Score",
        "template_adherence": "Template Adherence",
        "response_length_cv": "Length CV",
        "key_point_consistency": "Key-Point Consist.",
        "confidence_consistency": "Confidence Consist.",
        "timeline_consistency": "Timeline Consist.",
        "stochasticity_index": "STOCHASTICITY INDEX",
    }

    def __init__(self, config):
        self.cfg = config
        self.cfg.ensure_dirs()

    def load_results(self) -> dict:
        metrics_path = Path(self.cfg.metrics_dir) / "evaluation_results.json"
        with open(metrics_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    def _build_aggregate_df(self, results: dict) -> pd.DataFrame:
        """One row per model with aggregate metric means."""
        rows = []
        for model_name, data in results.items():
            agg = data.get("aggregate", {})
            row = {
                "model": model_name,
                "params_B": data["params_billion"],
                "family": data["family"],
            }
            for metric in self.HEATMAP_METRICS:
                if metric in agg:
                    row[metric] = agg[metric]["mean"]
                else:
                    row[metric] = np.nan
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("params_B").reset_index(drop=True)
        return df

    def _build_per_prompt_df(self, results: dict) -> pd.DataFrame:
        """One row per (model, prompt) with individual metric values."""
        rows = []
        for model_name, data in results.items():
            for prompt_id, pm in data.get("per_prompt", {}).items():
                row = {
                    "model": model_name,
                    "params_B": data["params_billion"],
                    "family": data["family"],
                    "prompt_id": prompt_id,
                    "angle": pm.get("angle", ""),
                }
                for metric in self.HEATMAP_METRICS:
                    row[metric] = pm.get(metric, np.nan)
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    def run_statistical_tests(self, df_agg: pd.DataFrame, df_prompt: pd.DataFrame) -> dict:
        """
        - Spearman correlation: params_B vs. stochasticity_index
        - Pearson correlation: same
        - Kruskal-Wallis: stochasticity across model size bins
        - Per-metric correlation with params_B
        """
        report = {}

        # 1. Overall correlation: param size vs stochasticity
        mask = df_agg["stochasticity_index"].notna()
        if mask.sum() >= 3:
            sp_corr, sp_p = stats.spearmanr(
                df_agg.loc[mask, "params_B"],
                df_agg.loc[mask, "stochasticity_index"],
            )
            pe_corr, pe_p = stats.pearsonr(
                df_agg.loc[mask, "params_B"],
                df_agg.loc[mask, "stochasticity_index"],
            )
            report["spearman"] = {"correlation": sp_corr, "p_value": sp_p}
            report["pearson"] = {"correlation": pe_corr, "p_value": pe_p}

        # 2. Per-metric correlation with param size
        metric_correlations = {}
        for metric in self.HEATMAP_METRICS:
            if metric in df_agg.columns:
                mask = df_agg[metric].notna()
                if mask.sum() >= 3:
                    corr, p = stats.spearmanr(
                        df_agg.loc[mask, "params_B"],
                        df_agg.loc[mask, metric],
                    )
                    metric_correlations[metric] = {"spearman_r": corr, "p_value": p}
        report["per_metric_correlation"] = metric_correlations

        # 3. Kruskal-Wallis: 4-tier bins aligned with emergent property thresholds
        if "stochasticity_index" in df_prompt.columns:
            df_prompt = df_prompt.copy()
            df_prompt["size_bin"] = pd.cut(
                df_prompt["params_B"],
                bins=[0, 3, 9, 32, 200],
                labels=["small (<3B)", "medium (3-9B)", "large (9-32B)", "frontier (>32B)"],
            )
            groups = [
                g["stochasticity_index"].dropna().values
                for _, g in df_prompt.groupby("size_bin", observed=True)
                if len(g["stochasticity_index"].dropna()) > 0
            ]
            if len(groups) >= 2:
                h_stat, kw_p = stats.kruskal(*groups)
                report["kruskal_wallis"] = {"H_statistic": h_stat, "p_value": kw_p}

        return report

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def plot_params_vs_stochasticity(self, df: pd.DataFrame, save_path: Path):
        """Scatter: parameter count vs. stochasticity index with regression."""
        fig, ax = plt.subplots(figsize=(10, 6))

        families = df["family"].unique()
        palette = sns.color_palette("husl", len(families))
        family_colors = dict(zip(families, palette))

        for _, row in df.iterrows():
            ax.scatter(
                row["params_B"], row["stochasticity_index"],
                color=family_colors[row["family"]],
                s=120, zorder=5, edgecolors="black", linewidth=0.5,
            )
            ax.annotate(
                row["model"], (row["params_B"], row["stochasticity_index"]),
                textcoords="offset points", xytext=(8, 5), fontsize=8,
            )

        # Regression line
        mask = df["stochasticity_index"].notna()
        if mask.sum() >= 2:
            x = df.loc[mask, "params_B"].values
            y = df.loc[mask, "stochasticity_index"].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.7, label="Linear fit")

        # Legend for families
        for fam, color in family_colors.items():
            ax.scatter([], [], color=color, label=fam, s=80, edgecolors="black", linewidth=0.5)
        ax.legend(loc="best")

        ax.set_xlabel("Parameter Count (Billions)", fontsize=12)
        ax.set_ylabel("Stochasticity Index", fontsize=12)
        ax.set_title("LLM Parameter Size vs. Stochasticity", fontsize=14)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def plot_metrics_heatmap(self, df: pd.DataFrame, save_path: Path):
        """Heatmap: models × metrics (aggregate means)."""
        metric_cols = [m for m in self.HEATMAP_METRICS if m in df.columns]
        labels = [self.METRIC_LABELS.get(m, m) for m in metric_cols]

        data = df.set_index("model")[metric_cols].astype(float)
        data.columns = labels

        fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 0.8)))
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
        )
        ax.set_title("Stochasticity Metrics Heatmap (0=low, 1=high)", fontsize=13)
        ax.set_ylabel("")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def plot_box_per_model(self, df_prompt: pd.DataFrame, save_path: Path):
        """Box plot: stochasticity index distribution per model."""
        fig, ax = plt.subplots(figsize=(12, 6))

        order = (
            df_prompt.groupby("model")["params_B"]
            .first()
            .sort_values()
            .index.tolist()
        )

        sns.boxplot(
            data=df_prompt, x="model", y="stochasticity_index",
            order=order, palette="viridis", ax=ax,
        )
        ax.set_xlabel("Model (sorted by param count)", fontsize=11)
        ax.set_ylabel("Stochasticity Index", fontsize=11)
        ax.set_title("Per-Prompt Stochasticity Distribution by Model", fontsize=13)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def plot_radar_chart(self, df: pd.DataFrame, save_path: Path):
        """Radar chart comparing metric profiles across models."""
        radar_metrics = [
            "self_bleu", "rouge_l", "jaccard", "embedding_cosine",
            "semantic_entropy", "vendi_score", "template_adherence",
            "key_point_consistency", "stochasticity_index",
        ]
        available = [m for m in radar_metrics if m in df.columns]
        if len(available) < 3:
            return

        labels = [self.METRIC_LABELS.get(m, m) for m in available]
        n_metrics = len(available)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        palette = sns.color_palette("husl", len(df))

        for idx, (_, row) in enumerate(df.iterrows()):
            values = [float(row[m]) if pd.notna(row[m]) else 0 for m in available]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=1.5, label=row["model"],
                    color=palette[idx], markersize=4)
            ax.fill(angles, values, alpha=0.05, color=palette[idx])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_title("Metric Profiles by Model", fontsize=13, y=1.08)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def plot_family_comparison(self, df_prompt: pd.DataFrame, save_path: Path):
        """Grouped bar: stochasticity by family and param size."""
        fig, ax = plt.subplots(figsize=(12, 6))

        summary = (
            df_prompt.groupby(["family", "params_B"])["stochasticity_index"]
            .mean()
            .reset_index()
            .sort_values("params_B")
        )
        summary["label"] = summary.apply(
            lambda r: f"{r['family']} ({r['params_B']}B)", axis=1
        )

        sns.barplot(
            data=summary, x="label", y="stochasticity_index",
            hue="family", dodge=False, palette="Set2", ax=ax,
        )
        ax.set_xlabel("Model Family (Param Size)", fontsize=11)
        ax.set_ylabel("Mean Stochasticity Index", fontsize=11)
        ax.set_title("Stochasticity by Model Family and Size", fontsize=13)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        ax.get_legend().remove()

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def plot_prompt_angle_heatmap(self, df_prompt: pd.DataFrame, save_path: Path):
        """Heatmap: stochasticity per prompt angle per model."""
        pivot = df_prompt.pivot_table(
            values="stochasticity_index", index="angle", columns="model",
            aggfunc="mean",
        )
        if pivot.empty:
            return

        # Sort columns by param size
        model_order = (
            df_prompt.groupby("model")["params_B"]
            .first()
            .sort_values()
            .index.tolist()
        )
        pivot = pivot[[c for c in model_order if c in pivot.columns]]

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title("Stochasticity by Prompt Angle × Model", fontsize=13)
        ax.set_ylabel("Prompt Angle")
        ax.set_xlabel("Model")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> str:
        """Run all analysis steps and return a summary report."""
        results = self.load_results()

        df_agg = self._build_aggregate_df(results)
        df_prompt = self._build_per_prompt_df(results)

        plots_dir = Path(self.cfg.plots_dir)

        # --- Statistical tests ---
        stat_tests = self.run_statistical_tests(df_agg, df_prompt)

        # --- Plots ---
        self.plot_params_vs_stochasticity(
            df_agg, plots_dir / "params_vs_stochasticity.png")
        self.plot_metrics_heatmap(
            df_agg, plots_dir / "metrics_heatmap.png")
        self.plot_box_per_model(
            df_prompt, plots_dir / "stochasticity_boxplot.png")
        self.plot_radar_chart(
            df_agg, plots_dir / "radar_chart.png")
        self.plot_family_comparison(
            df_prompt, plots_dir / "family_comparison.png")
        self.plot_prompt_angle_heatmap(
            df_prompt, plots_dir / "prompt_angle_heatmap.png")

        # --- Save CSVs ---
        df_agg.to_csv(Path(self.cfg.metrics_dir) / "aggregate_metrics.csv", index=False)
        df_prompt.to_csv(Path(self.cfg.metrics_dir) / "per_prompt_metrics.csv", index=False)

        # --- Text report ---
        report = self._generate_report(df_agg, df_prompt, stat_tests)
        report_path = Path(self.cfg.results_dir) / "analysis_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # --- Save stats JSON ---
        stats_path = Path(self.cfg.metrics_dir) / "statistical_tests.json"
        with open(stats_path, "w") as f:
            json.dump(stat_tests, f, indent=2, default=str)

        logger.info("Full analysis complete. Report: %s", report_path)
        return report

    def _generate_report(self, df_agg, df_prompt, stat_tests) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  LLM STOCHASTICITY EXPLORATION — ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Models tested
        lines.append("MODELS TESTED")
        lines.append("-" * 40)
        for _, row in df_agg.iterrows():
            si = row.get("stochasticity_index", float("nan"))
            lines.append(f"  {row['model']:25s}  {row['params_B']:5.1f}B  "
                         f"Stochasticity: {si:.4f}")
        lines.append("")

        # Ranking
        lines.append("STOCHASTICITY RANKING (most → least)")
        lines.append("-" * 40)
        ranked = df_agg.sort_values("stochasticity_index", ascending=False)
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            si = row.get("stochasticity_index", float("nan"))
            lines.append(f"  {rank}. {row['model']:25s}  Score: {si:.4f}")
        lines.append("")

        # Correlation analysis
        lines.append("CORRELATION: PARAMETER SIZE vs. STOCHASTICITY")
        lines.append("-" * 40)
        if "spearman" in stat_tests:
            sp = stat_tests["spearman"]
            lines.append(f"  Spearman ρ = {sp['correlation']:.4f}  "
                         f"(p = {sp['p_value']:.4f})")
        if "pearson" in stat_tests:
            pe = stat_tests["pearson"]
            lines.append(f"  Pearson  r = {pe['correlation']:.4f}  "
                         f"(p = {pe['p_value']:.4f})")

        if "spearman" in stat_tests:
            p = stat_tests["spearman"]["p_value"]
            r = stat_tests["spearman"]["correlation"]
            if p < 0.05:
                direction = "INCREASES" if r > 0 else "DECREASES"
                lines.append(f"\n  *** Statistically significant: Stochasticity {direction} "
                             f"with parameter count (p < 0.05) ***")
            else:
                lines.append(f"\n  No statistically significant relationship found (p = {p:.4f})")
        lines.append("")

        # Kruskal-Wallis
        if "kruskal_wallis" in stat_tests:
            kw = stat_tests["kruskal_wallis"]
            lines.append("KRUSKAL-WALLIS TEST (small vs medium vs large models)")
            lines.append("-" * 40)
            lines.append(f"  H-statistic = {kw['H_statistic']:.4f}  "
                         f"(p = {kw['p_value']:.4f})")
            if kw["p_value"] < 0.05:
                lines.append("  *** Significant difference across model size groups ***")
            else:
                lines.append("  No significant difference across groups")
            lines.append("")

        # Per-metric correlations
        lines.append("PER-METRIC CORRELATION WITH PARAMETER SIZE")
        lines.append("-" * 40)
        for metric, vals in stat_tests.get("per_metric_correlation", {}).items():
            label = self.METRIC_LABELS.get(metric, metric)
            sig = "*" if vals["p_value"] < 0.05 else " "
            lines.append(f"  {sig} {label:25s}  ρ={vals['spearman_r']:+.3f}  "
                         f"p={vals['p_value']:.4f}")
        lines.append("")

        # Summary
        lines.append("DETAILED METRIC AVERAGES")
        lines.append("-" * 40)
        for metric in self.HEATMAP_METRICS:
            if metric in df_agg.columns:
                label = self.METRIC_LABELS.get(metric, metric)
                vals = df_agg[metric].dropna()
                if len(vals) > 0:
                    lines.append(f"  {label:25s}  "
                                 f"mean={vals.mean():.4f}  "
                                 f"std={vals.std():.4f}  "
                                 f"range=[{vals.min():.4f}, {vals.max():.4f}]")
        lines.append("")
        lines.append("=" * 70)
        lines.append("Plots saved in: results/plots/")
        lines.append("Data  saved in: results/metrics/")
        lines.append("=" * 70)

        return "\n".join(lines)
