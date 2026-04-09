"""
Stochasticity Evaluator — State-of-the-art metrics to quantify LLM output variance.

Implements the following modern evaluation methods:

LEXICAL METRICS
  1. Self-BLEU            (Zhu et al., 2018 — "Texygen")
  2. ROUGE-L pairwise     (Lin, 2004)
  3. Jaccard Similarity    (token-level set overlap)
  4. Unique N-gram Ratio   (fraction of novel n-grams across responses)

SEMANTIC METRICS
  5. Embedding Cosine Similarity  (via sentence-transformers)
  6. Semantic Entropy              (Kuhn et al., ICLR 2023)
  7. Vendi Score                   (Friedman & Dieng, 2023)

STRUCTURAL METRICS
  8. Template Adherence Rate       (valid-JSON percentage)
  9. Response Length CV             (coefficient of variation)
 10. Key-Point Consistency         (Jaccard overlap of extracted key points)
 11. Confidence-Level Consistency  (mode frequency of discrete field)

COMPOSITE
 12. Stochasticity Index           (weighted aggregate, 0 = deterministic, 1 = chaotic)
"""

import json
import math
import logging
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import eigvalsh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

# ======================================================================
# Utility helpers
# ======================================================================

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _pairwise_avg(values_matrix: list[list[float]]) -> float:
    """Average of the upper triangle of a square matrix."""
    n = len(values_matrix)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += values_matrix[i][j]
            count += 1
    return total / count if count else 0.0


def _try_parse_json(text: str) -> dict | None:
    """Attempt to extract and parse JSON from model response."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON within markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ======================================================================
# LEXICAL METRICS
# ======================================================================

class LexicalMetrics:
    """Token-level and n-gram-level diversity/similarity measures."""

    @staticmethod
    def self_bleu(responses: list[str], max_n: int = 4) -> float:
        """
        Self-BLEU (Zhu et al., 2018): measures how similar generated texts
        are to each other. Lower Self-BLEU → more diverse → more stochastic.

        For each response, compute BLEU against all other responses as references,
        then average.
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        if len(responses) < 2:
            return 0.0

        tokenized = [_tokenize(r) for r in responses]
        smoothing = SmoothingFunction().method1
        weights = tuple(1.0 / max_n for _ in range(max_n))

        scores = []
        for i, hyp in enumerate(tokenized):
            refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
            if not hyp or not any(refs):
                scores.append(0.0)
                continue
            try:
                score = sentence_bleu(refs, hyp, weights=weights,
                                      smoothing_function=smoothing)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores))

    @staticmethod
    def rouge_l_pairwise(responses: list[str]) -> float:
        """Average pairwise ROUGE-L F1 between all response pairs."""
        from rouge_score.rouge_scorer import RougeScorer

        if len(responses) < 2:
            return 0.0

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for i, j in combinations(range(len(responses)), 2):
            result = scorer.score(responses[i], responses[j])
            scores.append(result["rougeL"].fmeasure)

        return float(np.mean(scores))

    @staticmethod
    def jaccard_pairwise(responses: list[str]) -> float:
        """Average pairwise Jaccard similarity of token sets."""
        if len(responses) < 2:
            return 0.0

        token_sets = [set(_tokenize(r)) for r in responses]
        scores = []
        for i, j in combinations(range(len(token_sets)), 2):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            scores.append(intersection / union if union > 0 else 0.0)

        return float(np.mean(scores))

    @staticmethod
    def unique_ngram_ratio(responses: list[str], n: int = 3) -> float:
        """
        Fraction of unique n-grams vs. total n-grams across all responses.
        Higher ratio → more diverse → more stochastic.
        """
        all_ngrams = []
        for r in responses:
            tokens = _tokenize(r)
            all_ngrams.extend(_ngrams(tokens, n))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)

    @staticmethod
    def tfidf_cosine_pairwise(responses: list[str]) -> float:
        """Average pairwise cosine similarity using TF-IDF vectors."""
        if len(responses) < 2:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(
                tokenizer=_tokenize, token_pattern=None, min_df=1
            )
            tfidf = vectorizer.fit_transform(responses)
            sim_matrix = sklearn_cosine(tfidf)
            # Extract upper triangle
            n = len(responses)
            scores = []
            for i in range(n):
                for j in range(i + 1, n):
                    scores.append(sim_matrix[i, j])
            return float(np.mean(scores))
        except ValueError:
            return 0.0


# ======================================================================
# SEMANTIC METRICS
# ======================================================================

class SemanticMetrics:
    """Embedding-based and semantic clustering metrics."""

    @staticmethod
    def embedding_cosine_similarity(embeddings: list[list[float]]) -> float:
        """
        Average pairwise cosine similarity of response embeddings.
        Uses pre-computed embeddings (from Ollama or sentence-transformers).
        """
        if len(embeddings) < 2:
            return 0.0

        emb_matrix = np.array(embeddings)
        # Normalize
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms

        sim_matrix = emb_matrix @ emb_matrix.T
        n = len(embeddings)
        scores = []
        for i in range(n):
            for j in range(i + 1, n):
                scores.append(sim_matrix[i, j])

        return float(np.mean(scores))

    @staticmethod
    def semantic_entropy(embeddings: list[list[float]],
                         similarity_threshold: float = 0.85) -> float:
        """
        Semantic Entropy (Kuhn et al., ICLR 2023).

        Cluster responses by semantic meaning, then compute entropy over
        cluster distribution.  Higher entropy → model produces semantically
        different answers → higher stochasticity.
        """
        n = len(embeddings)
        if n < 2:
            return 0.0

        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms

        # Distance = 1 - cosine_similarity
        sim_matrix = emb_matrix @ emb_matrix.T
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0, 2)

        # Agglomerative clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - similarity_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        # Compute Shannon entropy over cluster distribution
        counts = Counter(labels)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Normalize by log2(n) so result is in [0, 1]
        max_entropy = math.log2(n) if n > 1 else 1.0
        return entropy / max_entropy

    @staticmethod
    def vendi_score(embeddings: list[list[float]]) -> float:
        """
        Vendi Score (Friedman & Dieng, 2023).

        Measures effective diversity using the eigenvalues of the
        normalized similarity kernel.  VS = exp(H(eigenvalues)).
        Higher VS → more diverse responses → more stochastic.

        Returns normalized score in [0, 1] (divided by n).
        """
        n = len(embeddings)
        if n < 2:
            return 0.0

        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms

        # Kernel matrix (cosine similarity, clipped to [0,1])
        K = emb_matrix @ emb_matrix.T
        K = np.clip(K, 0, 1)
        K_normalized = K / n

        # Eigenvalues
        eigenvalues = eigvalsh(K_normalized)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Vendi Score = exp(Shannon entropy of eigenvalues)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        vs = math.exp(entropy)

        # Normalize: VS ∈ [1, n], map to [0, 1]
        return (vs - 1) / (n - 1) if n > 1 else 0.0


# ======================================================================
# STRUCTURAL METRICS
# ======================================================================

class StructuralMetrics:
    """Metrics about format adherence and structural consistency."""

    EXPECTED_KEYS = {
        "summary", "key_points", "challenges",
        "potential_impact", "confidence_level", "estimated_timeline",
    }
    VALID_CONFIDENCE = {"high", "medium", "low"}
    VALID_TIMELINE = {"short-term", "medium-term", "long-term"}

    @staticmethod
    def template_adherence_rate(responses: list[str]) -> float:
        """Fraction of responses that are valid JSON matching the template."""
        if not responses:
            return 0.0

        valid = 0
        for r in responses:
            parsed = _try_parse_json(r)
            if parsed is None:
                continue
            if StructuralMetrics.EXPECTED_KEYS.issubset(parsed.keys()):
                valid += 1

        return valid / len(responses)

    @staticmethod
    def response_length_cv(responses: list[str]) -> float:
        """Coefficient of variation of response lengths (chars)."""
        if len(responses) < 2:
            return 0.0

        lengths = [len(r) for r in responses]
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.0
        return float(np.std(lengths) / mean_len)

    @staticmethod
    def key_point_consistency(responses: list[str]) -> float:
        """
        Average pairwise Jaccard similarity of extracted key_points sets.
        Uses fuzzy matching: lowercase + strip, then word-set Jaccard.
        """
        key_point_sets = []
        for r in responses:
            parsed = _try_parse_json(r)
            if parsed and "key_points" in parsed:
                kp = parsed["key_points"]
                if isinstance(kp, list):
                    # Represent each key point as a set of words for fuzzy matching
                    word_bag = set()
                    for point in kp:
                        if isinstance(point, str):
                            word_bag.update(_tokenize(point))
                    key_point_sets.append(word_bag)

        if len(key_point_sets) < 2:
            return 0.0

        scores = []
        for i, j in combinations(range(len(key_point_sets)), 2):
            intersection = len(key_point_sets[i] & key_point_sets[j])
            union = len(key_point_sets[i] | key_point_sets[j])
            scores.append(intersection / union if union > 0 else 0.0)

        return float(np.mean(scores))

    @staticmethod
    def confidence_consistency(responses: list[str]) -> float:
        """How often the model outputs the same confidence_level value."""
        values = []
        for r in responses:
            parsed = _try_parse_json(r)
            if parsed and "confidence_level" in parsed:
                val = str(parsed["confidence_level"]).strip().lower()
                if val in StructuralMetrics.VALID_CONFIDENCE:
                    values.append(val)

        if not values:
            return 0.0

        counts = Counter(values)
        mode_freq = counts.most_common(1)[0][1]
        return mode_freq / len(values)

    @staticmethod
    def timeline_consistency(responses: list[str]) -> float:
        """How often the model outputs the same estimated_timeline value."""
        values = []
        for r in responses:
            parsed = _try_parse_json(r)
            if parsed and "estimated_timeline" in parsed:
                val = str(parsed["estimated_timeline"]).strip().lower()
                if val in StructuralMetrics.VALID_TIMELINE:
                    values.append(val)

        if not values:
            return 0.0

        counts = Counter(values)
        mode_freq = counts.most_common(1)[0][1]
        return mode_freq / len(values)


# ======================================================================
# COMPOSITE STOCHASTICITY INDEX
# ======================================================================

def compute_stochasticity_index(metrics: dict) -> float:
    """
    Weighted composite score mapping individual metrics to a single
    stochasticity index in [0, 1].

    0 = perfectly deterministic, 1 = maximally stochastic.

    Similarity-based metrics are inverted (1 - value) since high
    similarity = low stochasticity.
    """
    weights = {
        # Lexical (invert: high similarity → low stochasticity)
        "self_bleu":          (0.10, True),    # invert
        "rouge_l":            (0.08, True),    # invert
        "jaccard":            (0.07, True),    # invert
        "tfidf_cosine":       (0.07, True),    # invert
        "unique_ngram_ratio": (0.05, False),   # already: high = diverse

        # Semantic (invert similarity metrics)
        "embedding_cosine":   (0.12, True),    # invert
        "semantic_entropy":   (0.15, False),   # already: high = stochastic
        "vendi_score":        (0.10, False),   # already: high = diverse

        # Structural (invert consistency metrics)
        "template_adherence": (0.05, True),    # invert
        "response_length_cv": (0.06, False),   # already: high CV = stochastic
        "key_point_consistency": (0.07, True),  # invert
        "confidence_consistency": (0.04, True), # invert
        "timeline_consistency":  (0.04, True),  # invert
    }

    score = 0.0
    total_weight = 0.0

    for metric_name, (weight, invert) in weights.items():
        if metric_name in metrics and metrics[metric_name] is not None:
            val = metrics[metric_name]
            val = np.clip(val, 0.0, 1.0)
            if invert:
                val = 1.0 - val
            score += weight * val
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return score / total_weight


# ======================================================================
# MAIN EVALUATOR
# ======================================================================

class StochasticityEvaluator:
    """Orchestrates all metrics for a full experiment."""

    def __init__(self, config):
        self.cfg = config
        self._embed_model = None

    def _load_embed_model(self):
        """Lazy-load the sentence-transformers embedding model."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s",
                            self.cfg.embedding_model)
                self._embed_model = SentenceTransformer(
                    self.cfg.embedding_model)
            except Exception as exc:
                logger.warning("Could not load embedding model: %s", exc)
        return self._embed_model

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings via sentence-transformers."""
        model = self._load_embed_model()
        if model is None:
            return []
        try:
            embeddings = model.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except Exception as exc:
            logger.warning("Embedding computation failed: %s", exc)
            return []

    def evaluate_response_set(self, responses: list[str]) -> dict:
        """Compute all metrics for a set of responses to the same prompt."""
        metrics = {}

        # --- Lexical ---
        metrics["self_bleu"] = LexicalMetrics.self_bleu(responses)
        metrics["rouge_l"] = LexicalMetrics.rouge_l_pairwise(responses)
        metrics["jaccard"] = LexicalMetrics.jaccard_pairwise(responses)
        metrics["tfidf_cosine"] = LexicalMetrics.tfidf_cosine_pairwise(responses)
        metrics["unique_ngram_ratio"] = LexicalMetrics.unique_ngram_ratio(responses)

        # --- Semantic (require embeddings) ---
        embeddings = self._get_embeddings(responses)
        if embeddings:
            metrics["embedding_cosine"] = SemanticMetrics.embedding_cosine_similarity(embeddings)
            metrics["semantic_entropy"] = SemanticMetrics.semantic_entropy(embeddings)
            metrics["vendi_score"] = SemanticMetrics.vendi_score(embeddings)
        else:
            metrics["embedding_cosine"] = None
            metrics["semantic_entropy"] = None
            metrics["vendi_score"] = None

        # --- Structural ---
        metrics["template_adherence"] = StructuralMetrics.template_adherence_rate(responses)
        metrics["response_length_cv"] = StructuralMetrics.response_length_cv(responses)
        metrics["key_point_consistency"] = StructuralMetrics.key_point_consistency(responses)
        metrics["confidence_consistency"] = StructuralMetrics.confidence_consistency(responses)
        metrics["timeline_consistency"] = StructuralMetrics.timeline_consistency(responses)

        # --- Composite ---
        metrics["stochasticity_index"] = compute_stochasticity_index(metrics)

        return metrics

    def evaluate_model(self, model_data: dict) -> dict:
        """Evaluate all prompts for a single model."""
        responses_dict = model_data["responses"]
        prompt_metrics = {}

        for prompt_id, prompt_data in responses_dict.items():
            reps = prompt_data.get("repetitions", [])
            texts = [r["response"] for r in reps if r.get("response")]

            if len(texts) < 2:
                logger.warning("Prompt %s: only %d valid responses, skipping.", prompt_id, len(texts))
                continue

            logger.info("  Evaluating prompt %s (%d responses)...", prompt_id, len(texts))
            prompt_metrics[prompt_id] = self.evaluate_response_set(texts)
            prompt_metrics[prompt_id]["num_responses"] = len(texts)
            prompt_metrics[prompt_id]["angle"] = prompt_data.get("angle", "")

        # Aggregate across all prompts
        aggregate = self._aggregate_metrics(prompt_metrics)

        return {
            "model": model_data["model"],
            "params_billion": model_data["params_billion"],
            "family": model_data["family"],
            "per_prompt": prompt_metrics,
            "aggregate": aggregate,
        }

    def _aggregate_metrics(self, prompt_metrics: dict) -> dict:
        """Average, std, min, max across all prompt metrics."""
        if not prompt_metrics:
            return {}

        all_metric_names = set()
        for pm in prompt_metrics.values():
            all_metric_names.update(pm.keys())
        all_metric_names -= {"num_responses", "angle"}

        agg = {}
        for name in sorted(all_metric_names):
            values = [
                pm[name] for pm in prompt_metrics.values()
                if name in pm and pm[name] is not None
            ]
            if values:
                agg[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        return agg

    def evaluate_all(self) -> dict:
        """Evaluate all models from saved raw responses."""
        raw_dir = Path(self.cfg.raw_responses_dir)
        all_results = {}

        for result_file in sorted(raw_dir.glob("*.json")):
            logger.info("Loading %s ...", result_file.name)
            with open(result_file) as f:
                model_data = json.load(f)

            model_name = model_data["model"]
            logger.info("Evaluating model: %s", model_name)
            all_results[model_name] = self.evaluate_model(model_data)

        # Save metrics
        metrics_path = Path(self.cfg.metrics_dir) / "evaluation_results.json"
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation results saved to %s", metrics_path)
        return all_results
