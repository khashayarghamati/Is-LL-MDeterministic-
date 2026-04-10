# Scaling Tames Randomness: A Multi-Metric Investigation of Output Stochasticity Across Large Language Models from 0.5 B to 72 B Parameters

**Khashayar Ghamati**

School of Physics, Engineering and Computer Science, University of Hertfordshire, Hatfield AL10 9AB, UK; kg23aay@herts.ac.uk

---

## Abstract

Large language models (LLMs) generate text through probabilistic sampling, yet the degree to which their outputs vary -- and whether that variation changes systematically with model scale -- remains poorly quantified. We present a controlled empirical study that probes the stochasticity of 12 instruction-tuned LLMs spanning three architectural families (Qwen 2.5, LLaMA 3, Gemma 2) and a 144-fold range of parameter counts (0.5 B to 72 B). Each model was prompted 20 times with each of 20 thematically related but distinct questions on AI in healthcare, yielding 4800 stateless generations under identical decoding settings (temperature 0.7, top-p 0.9). We evaluate output variability through a battery of 13 complementary metrics drawn from lexical, semantic, and structural analysis, which we aggregate into a single composite Stochasticity Index. Our results reveal a statistically significant negative correlation between parameter count and stochasticity (Spearman rho = -0.767, p = 0.004): smaller models produce markedly more diverse -- and less predictable -- outputs, while larger models converge toward more deterministic response patterns. A Kruskal-Wallis test confirms that stochasticity differs significantly across four model-size tiers (H = 137.78, p < 10^-29). We further observe that this relationship is not driven by a single metric but is consistent across 11 of 13 individual measures, suggesting a fundamental shift in generation behaviour as models scale. These findings carry practical implications for reproducibility, evaluation methodology, and the deployment of LLMs in safety-critical domains.

**Keywords:** large language models; stochasticity; output variability; model scaling; emergent properties; reproducibility; natural language generation; evaluation metrics

---

## 1. Introduction

Every time a large language model is asked the same question, it may give a slightly -- or dramatically -- different answer. This fundamental property, known as *stochasticity*, arises from the probabilistic sampling mechanisms that underpin modern text generation. While some degree of variation is desirable (it prevents outputs from feeling robotic or repetitive), excessive unpredictability poses serious challenges in settings where consistency and reproducibility matter, such as clinical decision support, legal document generation, or automated scientific reasoning.

The past three years have witnessed an extraordinary expansion in LLM scale, with openly available models now ranging from sub-billion to over 70 billion parameters. Alongside this growth, researchers have documented *emergent capabilities* -- qualitative improvements in reasoning, instruction following, and factual accuracy that appear to materialise at certain parameter thresholds [1,2]. A natural yet largely unexplored question follows: does the stochastic behaviour of LLMs also change as models grow larger? If a 70-billion-parameter model generates more consistent answers than a 0.5-billion-parameter model under identical sampling conditions, this would have profound implications for how we benchmark, deploy, and trust these systems.

Despite its importance, stochasticity in LLMs has received surprisingly little systematic attention. Most evaluation benchmarks report a single score per model, implicitly assuming that the answer a model gives on one run is representative of all possible runs. Reproducibility studies have noted variation across runs [3], but few have attempted to *measure* that variation as a dependent variable and correlate it with architectural properties such as parameter count. Those that have tend to rely on a single metric (e.g., Self-BLEU or response length variance), capturing only one dimension of a multifaceted phenomenon.

In this paper, we address this gap through a carefully controlled experiment designed around four guiding questions:

1. **How stochastic are modern instruction-tuned LLMs?** We quantify the baseline variability of 12 models across three architectural families.
2. **Does parameter count predict stochasticity?** We test the hypothesis that larger models produce more consistent outputs.
3. **Is the relationship metric-dependent?** We evaluate whether the scaling effect holds across lexical, semantic, and structural dimensions of variation.
4. **Do emergent-property thresholds coincide with stochasticity shifts?** By sampling models at four size tiers aligned with known emergence boundaries, we probe whether transitions in capability correspond to transitions in output consistency.

To answer these questions, we adopt a multi-metric evaluation framework comprising 13 complementary measures -- from classical n-gram statistics to modern neural approaches such as Semantic Entropy [4] and the Vendi Score [5] -- aggregated into a single composite Stochasticity Index. Our experimental design ensures fair comparison: all models receive the same 20 prompts, each repeated 20 times without conversational memory, under identical decoding parameters. The result is a dataset of 4800 model responses that allows both aggregate and fine-grained analysis of output variability.

The remainder of this paper is organised as follows. Section 2 reviews related work on LLM evaluation, stochasticity, and emergent properties. Section 3 describes our experimental methodology, including model selection, prompt design, and metric definitions. Section 4 presents the results, moving from aggregate trends to per-metric and per-prompt analyses. Section 5 discusses the implications of our findings, considers limitations, and outlines directions for future research. Section 6 concludes.

---

## 2. Related Work

Understanding the variability of LLM outputs requires drawing on several intersecting lines of research: the scaling laws that govern model behaviour, the emergent capabilities that appear at certain scales, and the evaluation methodologies used to quantify text generation quality.

### 2.1. Scaling Laws and Emergent Capabilities

The relationship between model size and performance has been a central theme in LLM research since the seminal work of Kaplan et al. [6], who demonstrated smooth power-law improvements in loss as a function of parameter count, dataset size, and compute budget. Hoffmann et al. [7] refined these laws, showing that many models are undertrained relative to their size and that optimal scaling requires balancing parameters with data volume.

More intriguing than smooth improvements are the *emergent capabilities* documented by Wei et al. [1]: abilities that are absent in smaller models but appear -- sometimes abruptly -- once a model crosses a size threshold. Examples include chain-of-thought reasoning, instruction following, and multi-step arithmetic. Schaeffer et al. [8] later argued that some apparent emergences are artefacts of metric choice, but the consensus remains that qualitative shifts in model behaviour do accompany scale increases, particularly in the 3 B to 13 B parameter range for instruction-tuned models.

What has not been studied, however, is whether *output consistency* -- the degree to which a model gives similar answers to the same question across runs -- also changes with scale. Our work fills this gap by treating stochasticity itself as the dependent variable in a scaling analysis.

### 2.2. Measuring Text Diversity and Consistency

The NLP community has developed a rich toolkit for measuring how diverse or repetitive a set of generated texts is. Self-BLEU [9] adapts the classic BLEU metric to measure intra-set similarity: each text is scored against all others as references, with high Self-BLEU indicating low diversity. ROUGE-L [10] captures longest-common-subsequence overlap and is widely used for summarisation evaluation. TF-IDF cosine similarity [11] provides a term-frequency-weighted perspective on lexical overlap.

At the semantic level, embedding-based cosine similarity uses dense vector representations to capture meaning beyond surface forms. More recently, Kuhn et al. [4] introduced *Semantic Entropy*, which clusters responses by meaning and computes Shannon entropy over the cluster distribution. High semantic entropy indicates that a model produces genuinely different *answers*, not merely different *phrasings* of the same answer. Friedman and Dieng [5] proposed the *Vendi Score*, an eigenvalue-based diversity metric derived from the kernel similarity matrix, which elegantly captures the effective number of distinct outputs.

Most of these metrics have been applied in isolation. Our contribution is to deploy them as a *battery* and examine whether their signals converge -- testing whether the scaling effect on stochasticity is a robust phenomenon or an artefact of a particular measurement approach.

### 2.3. Reproducibility in LLM Evaluation

The reproducibility crisis in machine learning [12] is particularly acute for LLMs, where a single evaluation run may not reflect a model's typical behaviour. Zheng et al. [3] showed that LLM-as-a-judge scores vary across runs, and Biderman et al. [13] demonstrated that benchmark rankings can change depending on the random seed. These observations motivate our experimental design: by repeating each prompt 20 times, we can characterise the *distribution* of outputs, not just a point estimate, and quantify how that distribution changes with model scale.

---

## 3. Materials and Methods

Designing a fair comparison of stochasticity across models of vastly different sizes requires careful attention to experimental controls. In this section, we describe how we selected models, designed prompts, configured generation parameters, and defined our evaluation metrics, with the goal of isolating the effect of parameter count while minimising confounding factors.

### 3.1. Model Selection

We selected 12 instruction-tuned LLMs from three architectural families, spanning a 144-fold range of parameter counts from 0.5 billion to 72 billion. Table 1 summarises the models and their tier assignments.

**Table 1.** Models evaluated, organised by tier. All models are instruction-tuned variants accessed via HuggingFace Transformers.

| Tier | Label | Model | Parameters | Family |
|------|-------|-------|-----------|--------|
| 1 | Small | Qwen/Qwen2.5-0.5B-Instruct | 0.5 B | Qwen 2.5 |
| 1 | Small | meta-llama/Llama-3.2-1B-Instruct | 1.0 B | LLaMA 3 |
| 1 | Small | Qwen/Qwen2.5-1.5B-Instruct | 1.5 B | Qwen 2.5 |
| 1 | Small | google/gemma-2-2b-it | 2.0 B | Gemma 2 |
| 2 | Medium | Qwen/Qwen2.5-3B-Instruct | 3.0 B | Qwen 2.5 |
| 2 | Medium | meta-llama/Llama-3.2-3B-Instruct | 3.0 B | LLaMA 3 |
| 2 | Medium | Qwen/Qwen2.5-7B-Instruct | 7.0 B | Qwen 2.5 |
| 2 | Medium | meta-llama/Llama-3.1-8B-Instruct | 8.0 B | LLaMA 3 |
| 3 | Large | google/gemma-2-9b-it | 9.0 B | Gemma 2 |
| 3 | Large | Qwen/Qwen2.5-14B-Instruct | 14.0 B | Qwen 2.5 |
| 4 | Frontier | meta-llama/Llama-3.1-70B-Instruct | 70.0 B | LLaMA 3 |
| 4 | Frontier | Qwen/Qwen2.5-72B-Instruct | 72.0 B | Qwen 2.5 |

The tier boundaries were chosen to align with documented emergent-capability thresholds. Tier 1 (below 3 B) represents models that typically lack robust instruction-following abilities. Tier 2 (3--8 B) captures the range where instruction following and basic reasoning begin to emerge. Tier 3 (9--32 B) corresponds to models exhibiting more consistent reasoning and factual accuracy. Tier 4 (70 B and above) represents frontier models with full emergent capabilities.

We deliberately included multiple families at overlapping sizes (e.g., Qwen 3 B and LLaMA 3.2 3 B are both 3 B but from different architectures) to distinguish scale effects from family-specific effects.

### 3.2. Prompt Design

To ensure a controlled yet realistic evaluation, we designed 20 prompts centred on a single domain -- *Artificial Intelligence in Healthcare* -- but each approaching the topic from a distinct angle. The 20 angles span the breadth of the domain: general overview, medical diagnostics, drug discovery, patient data privacy, rural healthcare access, mental health, cost reduction, robotic surgery, personalised medicine, electronic health records, clinical trials, ethics, medical imaging, elderly care, pandemic response, medical education, billing and insurance, rare diseases, preventive care, and regulatory challenges.

Anchoring all prompts to a single domain serves two purposes. First, it ensures that all models are evaluated on comparable content, eliminating domain difficulty as a confound. Second, it allows us to probe whether certain sub-topics elicit more stochastic behaviour than others -- a nuance that would be lost in a multi-domain design.

Each prompt concludes with an identical structured output template requiring the model to return a JSON object with six fields: a two-to-three sentence summary, five key points, three challenges, a potential impact statement, a confidence level (high, medium, or low), and an estimated timeline (short-term, medium-term, or long-term). This structured format serves a dual role: it standardises outputs for fair comparison and enables *structural* metrics (e.g., template adherence, key-point consistency) alongside lexical and semantic ones.

### 3.3. Experimental Protocol

Each of the 20 prompts was sent to each model 20 times independently, yielding 400 generations per model and 4800 in total across all 12 models. Critically, each generation was *stateless*: no conversational history was carried between repetitions, and no system prompt was varied. This ensures that any observed variation arises solely from the sampling process, not from contextual differences.

All models were served through HuggingFace Transformers with identical decoding parameters:
- **Temperature:** 0.7 (moderate stochasticity -- not deterministic, not maximally random)
- **Top-p (nucleus sampling):** 0.9
- **Maximum tokens:** 1024
- **Random seed:** None (non-deterministic)

Models with fewer than 27 B parameters were loaded in FP16 precision. Models between 27 B and 65 B were quantised to 8-bit, and models above 65 B to 4-bit, using the BitsAndBytes library [14]. All experiments were executed on a SLURM-managed GPU cluster equipped with NVIDIA A100 80 GB GPUs at the University of Hertfordshire.

### 3.4. Evaluation Metrics

We assessed output stochasticity through 13 metrics spanning three categories: lexical, semantic, and structural. This multi-dimensional approach is motivated by the recognition that two sets of outputs can be lexically diverse yet semantically identical (paraphrasing) or lexically similar yet structurally inconsistent (e.g., differing JSON formatting). By evaluating all three dimensions, we obtain a more complete picture of stochastic behaviour.

#### 3.4.1. Lexical Metrics

**Self-BLEU** [9]. For each response in a set of 20, we compute BLEU-4 against all other responses as references, then average. High Self-BLEU indicates high inter-response similarity (low stochasticity). We employ Method 1 smoothing to handle short n-gram matches.

**ROUGE-L** (Pairwise) [10]. We compute the ROUGE-L F1 score for every pair of responses and average. ROUGE-L captures the longest common subsequence, making it sensitive to structural word-order similarity.

**Jaccard Similarity** (Pairwise). For each pair of responses, we compute the ratio of shared tokens to total unique tokens. This bag-of-words measure complements BLEU and ROUGE by ignoring word order.

**TF-IDF Cosine Similarity** (Pairwise). We vectorise responses using term frequency--inverse document frequency and compute pairwise cosine similarities. TF-IDF downweights common words, emphasising content-bearing term overlap.

**Unique N-gram Ratio**. We pool all trigrams from the 20 responses and compute the fraction that are unique. High uniqueness indicates high lexical diversity.

#### 3.4.2. Semantic Metrics

**Embedding Cosine Similarity**. We embed each response using the all-MiniLM-L6-v2 sentence transformer [15] and compute pairwise cosine similarities. Unlike lexical metrics, embedding similarity captures whether responses *mean* the same thing, even if they use different words.

**Semantic Entropy** [4]. Following Kuhn et al., we cluster response embeddings using agglomerative clustering with a cosine-distance threshold of 0.15, then compute normalised Shannon entropy over the cluster distribution. High entropy means the model produces semantically distinct answers; zero entropy means all responses fall into a single meaning cluster.

**Vendi Score** [5]. We construct the normalised cosine-similarity kernel matrix over the response embeddings and compute the exponential of the Shannon entropy of its eigenvalues. The Vendi Score approximates the effective number of distinct responses and is normalised to [0, 1] by dividing by *n*.

#### 3.4.3. Structural Metrics

**Template Adherence Rate**. The fraction of responses that parse as valid JSON and contain all six expected fields. This measures whether the model reliably follows the output format.

**Response Length CV**. The coefficient of variation (standard deviation divided by mean) of response character counts. High CV indicates inconsistent verbosity.

**Key-Point Consistency**. For responses that successfully parse as JSON, we extract the five key points, tokenise them into word bags, and compute pairwise Jaccard similarity. This measures whether the model emphasises the same themes across runs.

**Confidence-Level Consistency**. The frequency of the modal confidence-level value (high, medium, or low) across the 20 responses. A score of 1.0 means the model always picks the same confidence level.

**Timeline Consistency**. Analogous to confidence consistency, but for the estimated-timeline field.

#### 3.4.4. Composite Stochasticity Index

To synthesise the 13 individual metrics into a single score, we define a weighted Stochasticity Index (SI) as follows. Similarity-oriented metrics (Self-BLEU, ROUGE-L, Jaccard, TF-IDF cosine, embedding cosine, template adherence, key-point consistency, confidence consistency, timeline consistency) are inverted (1 - value) so that high values indicate high stochasticity. Diversity-oriented metrics (unique n-gram ratio, semantic entropy, Vendi Score, response length CV) are used directly. All values are clipped to [0, 1] and combined via a weighted average:

SI = (sum of w_i * s_i) / (sum of w_i)

where s_i is the (possibly inverted) metric value and w_i is its weight. The weights reflect the relative informativeness of each metric category: semantic metrics receive the highest weights (Semantic Entropy: 0.15, Embedding Cosine: 0.12, Vendi Score: 0.10) because they capture meaning-level variation; lexical metrics receive moderate weights (0.05--0.10); structural metrics receive lower weights (0.04--0.06) as they partly reflect prompt-following ability rather than pure stochasticity. The resulting SI ranges from 0 (perfectly deterministic) to 1 (maximally stochastic).

### 3.5. Statistical Analysis

We employ the following statistical tests to assess the significance of our findings:

- **Spearman rank correlation** between parameter count and Stochasticity Index, chosen because the relationship need not be linear and parameter counts are highly skewed.
- **Pearson correlation** for comparison, testing the linear association.
- **Kruskal-Wallis H-test** across the four model-size tiers, using per-prompt SI values (240 observations from 12 models times 20 prompts) to test whether at least one tier differs significantly from the others.
- **Per-metric Spearman correlations** to identify which individual metrics drive the aggregate trend.

All p-values are reported at the alpha = 0.05 significance level.

---

## 4. Results

We now present our findings, moving from the broadest view -- the overall relationship between model size and stochasticity -- to progressively finer-grained analyses of individual metrics, model families, and prompt angles.

### 4.1. Aggregate Stochasticity Across Model Scales

Figure 1 plots each model's aggregate Stochasticity Index against its parameter count, revealing a clear downward trend: smaller models are substantially more stochastic than larger ones.

**Table 2.** Stochasticity Index by model, sorted from most to least stochastic.

| Rank | Model | Parameters | Family | SI (mean +/- std) |
|------|-------|-----------|--------|-------------------|
| 1 | Qwen2.5-0.5B | 0.5 B | Qwen 2.5 | 0.341 +/- 0.021 |
| 2 | Qwen2.5-1.5B | 1.5 B | Qwen 2.5 | 0.317 +/- 0.019 |
| 3 | Qwen2.5-3B | 3.0 B | Qwen 2.5 | 0.312 +/- 0.015 |
| 4 | Llama-3.2-1B | 1.0 B | LLaMA 3 | 0.309 +/- 0.025 |
| 5 | Qwen2.5-14B | 14.0 B | Qwen 2.5 | 0.278 +/- 0.027 |
| 6 | Llama-3.2-3B | 3.0 B | LLaMA 3 | 0.271 +/- 0.021 |
| 7 | Qwen2.5-7B | 7.0 B | Qwen 2.5 | 0.270 +/- 0.017 |
| 8 | Gemma-2-2b | 2.0 B | Gemma 2 | 0.269 +/- 0.011 |
| 9 | Llama-3.1-8B | 8.0 B | LLaMA 3 | 0.245 +/- 0.023 |
| 10 | Qwen2.5-72B | 72.0 B | Qwen 2.5 | 0.233 +/- 0.017 |
| 11 | Llama-3.1-70B | 70.0 B | LLaMA 3 | 0.220 +/- 0.013 |
| 12 | Gemma-2-9b | 9.0 B | Gemma 2 | 0.205 +/- 0.014 |

The most stochastic model, Qwen2.5-0.5B (SI = 0.341), scores 67% higher than the least stochastic, Gemma-2-9b (SI = 0.205). The Stochasticity Index spans a range of 0.136, indicating meaningful variation across the model landscape.

The Spearman rank correlation between parameter count and Stochasticity Index is rho = -0.767 (p = 0.004), indicating a strong and statistically significant negative monotonic relationship. The Pearson correlation is r = -0.587 (p = 0.045), also significant, though weaker owing to the non-linear distribution of parameter counts (heavily skewed toward smaller models with two 70 B+ outliers).

### 4.2. Tier-Level Analysis: Do Size Groups Differ?

To test whether stochasticity differs across model-size groups aligned with emergent-capability thresholds, we conducted a Kruskal-Wallis test using per-prompt SI values grouped into four tiers. The result is highly significant: H = 137.78, p < 10^-29.

This finding confirms that the trend visible in Table 2 is not merely a smooth gradient; there are statistically meaningful *jumps* in output consistency between model-size categories. The median SI decreases monotonically from Tier 1 (small, median approximately 0.33) through Tier 2 (medium, approximately 0.28) and Tier 3 (large, approximately 0.24) to Tier 4 (frontier, approximately 0.23). The largest drop occurs between Tiers 1 and 2, suggesting that the transition from sub-3 B to 3--8 B models represents the most impactful scaling step for output consistency.

### 4.3. Per-Metric Correlations: A Consistent Signal

A natural concern is that the composite Stochasticity Index might be dominated by one or two metrics, masking disagreement among others. Table 3 addresses this by reporting the Spearman correlation of each individual metric with parameter count.

**Table 3.** Per-metric Spearman correlations with parameter count. Asterisks denote significance at p < 0.05.

| Metric | Category | Spearman rho | p-value | Direction |
|--------|----------|-------------|---------|-----------|
| Self-BLEU | Lexical | +0.771* | 0.003 | Larger models more self-similar |
| ROUGE-L | Lexical | +0.676* | 0.016 | Larger models more overlapping |
| Jaccard | Lexical | +0.708* | 0.010 | Larger models share more tokens |
| TF-IDF Cosine | Lexical | +0.522 | 0.082 | Trend, not significant |
| Unique N-gram % | Lexical | -0.701* | 0.011 | Larger models use fewer unique n-grams |
| Embedding Cosine | Semantic | +0.644* | 0.024 | Larger models semantically closer |
| Semantic Entropy | Semantic | -0.618* | 0.032 | Larger models lower entropy |
| Vendi Score | Semantic | -0.644* | 0.024 | Larger models less diverse |
| Template Adherence | Structural | +0.650* | 0.022 | Larger models follow format better |
| Length CV | Structural | -0.781* | 0.003 | Larger models more consistent length |
| Key-Point Consistency | Structural | +0.658* | 0.020 | Larger models repeat key points |
| Confidence Consistency | Structural | +0.032 | 0.923 | No relationship |
| Timeline Consistency | Structural | +0.806* | 0.002 | Larger models repeat timeline |

The results are striking: 11 of 13 metrics show a significant correlation with parameter count, and all in the direction that larger models are less stochastic. The two exceptions are TF-IDF cosine similarity (trending but not significant, p = 0.082) and confidence-level consistency (effectively flat, rho = 0.032, p = 0.923).

The consistency across metric categories is particularly compelling. Whether we measure variation through lexical overlap (Self-BLEU, rho = +0.771), semantic clustering (Semantic Entropy, rho = -0.618), or structural reliability (Length CV, rho = -0.781), the conclusion is the same: larger models produce more consistent outputs.

The lone holdout -- confidence-level consistency -- is readily explained. Most models, regardless of size, assign "medium" confidence to healthcare AI topics, yielding near-ceiling consistency scores (mean 0.915) with almost no variance to correlate against. This is a floor effect rather than evidence against the scaling trend.

### 4.4. Metric Profiles: A Deeper Look

Beyond correlations, the absolute metric values reveal the *character* of stochasticity at different scales. Table 4 presents summary statistics for each metric.

**Table 4.** Descriptive statistics of evaluation metrics across all 12 models.

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Self-BLEU | 0.640 | 0.096 | 0.493 | 0.805 |
| ROUGE-L | 0.429 | 0.059 | 0.354 | 0.539 |
| Jaccard Similarity | 0.390 | 0.046 | 0.321 | 0.478 |
| TF-IDF Cosine | 0.542 | 0.091 | 0.389 | 0.676 |
| Unique N-gram Ratio | 0.568 | 0.099 | 0.396 | 0.709 |
| Embedding Cosine | 0.916 | 0.013 | 0.892 | 0.941 |
| Semantic Entropy | 0.015 | 0.016 | 0.000 | 0.050 |
| Vendi Score | 0.032 | 0.006 | 0.022 | 0.043 |
| Template Adherence | 0.925 | 0.164 | 0.483 | 1.000 |
| Length CV | 0.135 | 0.055 | 0.086 | 0.244 |
| Key-Point Consistency | 0.282 | 0.050 | 0.206 | 0.360 |
| Confidence Consistency | 0.915 | 0.080 | 0.705 | 0.983 |
| Timeline Consistency | 0.838 | 0.130 | 0.613 | 1.000 |

Several patterns merit attention. First, embedding cosine similarity is uniformly high (mean 0.916, range 0.892--0.941), indicating that even the most stochastic models rarely produce *semantically contradictory* answers -- the variation is primarily in *phrasing* and *emphasis*, not in *meaning*. Second, semantic entropy is near zero for most models (mean 0.015), confirming that responses almost always cluster into a single semantic group. This means that the stochasticity we observe is predominantly *surface-level*: models say roughly the same thing in different ways, rather than saying fundamentally different things.

Third, template adherence varies dramatically at the small end. The 3 B LLaMA model achieves only 48.3% template adherence, while most models above 7 B achieve 99--100%. This aligns with the emergent-capability literature: reliably following a structured output format is itself an ability that scales with model size.

### 4.5. Family-Level Analysis

Including three architectural families allows us to separate scaling effects from family-specific effects. Figure 5 (family comparison) shows that within each family, stochasticity consistently decreases with size:

- **Qwen 2.5:** SI drops from 0.341 (0.5 B) to 0.233 (72 B) -- a 32% reduction.
- **LLaMA 3:** SI drops from 0.309 (1 B) to 0.220 (70 B) -- a 29% reduction.
- **Gemma 2:** SI drops from 0.269 (2 B) to 0.205 (9 B) -- a 24% reduction (with a narrower size range).

The cross-family consistency reinforces our conclusion that the scaling--stochasticity relationship is a general property of language models, not an artefact of one particular architecture.

Interestingly, at matched parameter counts, the families differ. At 3 B, Qwen 2.5 (SI = 0.312) is notably more stochastic than LLaMA 3 (SI = 0.271). This suggests that training data, alignment procedure, and architectural details all modulate stochasticity independently of scale.

### 4.6. Prompt-Level Analysis: Does Topic Angle Matter?

Figure 6 (prompt-angle heatmap) reveals that stochasticity varies not only across models but also across prompt topics, though the model-size effect dominates. Several patterns emerge:

- **High-stochasticity prompts** (consistently red across models): "General overview" (P01), "Ethics" (P12), "Robotic surgery" (P08), and "Regulatory challenges" (P20). These are topics with broad scope and multiple valid framings, inviting diverse responses.
- **Low-stochasticity prompts** (consistently blue): "Preventive care" (P19) and "Elderly care" (P14). These are more concrete, narrower topics where models converge on similar content.

The interaction between prompt angle and model size is also informative. Small models show high variability across *all* topics (most cells in the 0.30--0.37 range), while large models show lower variability overall but with greater *relative* sensitivity to topic difficulty (e.g., Gemma-2-9b ranges from 0.17 on "Elderly care" to 0.27 on "Ethics"). This suggests that as models scale, they develop more nuanced topic sensitivity while simultaneously becoming more consistent in their baseline behaviour.

---

## 5. Discussion

Our results paint a coherent picture: as language models grow larger, they become less stochastic across virtually every dimension of output variability we measured. This section interprets these findings in the broader context of LLM science, identifies practical implications, and acknowledges limitations.

### 5.1. Why Do Larger Models Exhibit Lower Stochasticity?

We propose three complementary explanations for the observed scaling--stochasticity relationship.

**Sharper probability distributions.** Larger models, having been trained on more data with more parameters, develop sharper conditional probability distributions over the next token. When the probability mass is concentrated on fewer tokens at each generation step, the sampling process has less room for variation, even at the same temperature setting. In effect, a temperature of 0.7 *feels* more deterministic to a 72 B model than to a 0.5 B model because the underlying distribution is already more peaked.

**Better instruction following.** Larger models are more adept at following the structured output template, which constrains the space of valid outputs. A model that reliably produces a JSON object with five key points and three challenges has less room for variation than one that sometimes outputs free text, sometimes partial JSON, and sometimes valid JSON.

**Richer internal representations.** Larger models have more capacity to build robust, stable representations of concepts. When asked about "AI in drug discovery," a 72 B model likely activates a well-defined cluster of knowledge, leading to consistent outputs. A 0.5 B model, with sparser and noisier internal representations, is more susceptible to the randomness introduced by sampling.

### 5.2. The Semantic Consistency Paradox

One of our most striking findings is the contrast between high lexical stochasticity and near-zero semantic stochasticity. Even the smallest models produce responses with embedding cosine similarities above 0.89, and semantic entropy values near zero. This means that the *information content* of responses is remarkably stable across runs -- what varies is the *expression* of that content.

This finding has important implications for evaluation methodology. If we evaluate models using lexical metrics alone (as many benchmarks implicitly do through exact-match or F1 scoring), we may dramatically overestimate a model's inconsistency. Semantic evaluation methods are essential for distinguishing genuine knowledge instability from harmless paraphrasing.

### 5.3. Emergent Consistency: A New Perspective on Emergence

Our tier-level analysis suggests that output consistency may itself be an emergent property. The largest drop in stochasticity occurs between Tier 1 (below 3 B) and Tier 2 (3--8 B), coinciding with the parameter range where instruction-following capabilities are known to emerge. This suggests a tantalising possibility: the same architectural changes that enable a model to reliably follow instructions also cause it to produce more consistent outputs.

If confirmed by future work, this would add "output consistency" to the list of emergent capabilities -- capabilities that do not improve gradually with scale but instead appear relatively abruptly once a threshold is crossed.

### 5.4. Practical Implications

Our findings carry several practical implications for practitioners deploying LLMs.

**For reproducibility.** Researchers evaluating small models should report results averaged over multiple runs, as single-run evaluations may be unrepresentative. For models above 9 B, single-run evaluations are more (though not perfectly) reliable.

**For safety-critical applications.** In domains where consistency is paramount -- clinical decision support, legal reasoning, financial analysis -- our results favour larger models, which produce more predictable outputs. However, even the most consistent model in our study (Gemma-2-9b, SI = 0.205) exhibits non-trivial variability, suggesting that deterministic generation settings (temperature = 0) or ensemble methods may be necessary for high-stakes applications.

**For model selection.** Practitioners face a trade-off: smaller models are cheaper and faster but less consistent. Our data provides quantitative guidance for this trade-off, showing approximately a 0.014-point reduction in SI per doubling of parameter count.

**For prompt engineering.** Our prompt-level analysis shows that stochasticity varies by topic, suggesting that prompt design can partially mitigate inconsistency. Narrow, concrete prompts elicit more consistent responses than broad, open-ended ones.

### 5.5. Limitations

Several limitations constrain the generalisability of our findings.

**Single domain.** All 20 prompts concern AI in healthcare. While the within-domain variation (20 distinct angles) provides breadth, we cannot confirm that the scaling--stochasticity relationship holds for fundamentally different domains (e.g., creative writing, mathematics, code generation). Creative tasks, in particular, may exhibit different scaling dynamics, as a certain degree of stochasticity is desirable.

**Fixed decoding parameters.** We used a single temperature (0.7) and top-p (0.9) setting. The interaction between decoding parameters and model size is an important open question: it is possible that the scaling effect disappears at temperature 0 (deterministic decoding) or reverses at temperature 1.0.

**One model excluded.** Microsoft Phi-3.5-mini-instruct (3.8 B) was included in the experimental design but failed all 400 generation attempts due to a library incompatibility. Its exclusion does not affect our conclusions, as Tier 2 retains three other models, but it does reduce coverage of the 3--4 B parameter range.

**Quantisation effects.** Models above 65 B were quantised to 4-bit precision. While quantisation is standard practice for serving large models, it introduces a confound: we cannot fully separate the effect of model size from the effect of reduced precision. Future work should evaluate the same model at multiple quantisation levels.

**Limited frontier representation.** Only two models represent the 70 B+ tier. While both show consistent results, a larger sample at this scale would strengthen the frontier analysis.

### 5.6. Future Work

This study opens several directions for future investigation. First, extending the analysis to multiple domains and generation tasks (creative writing, code generation, reasoning) would test the generalisability of the scaling--stochasticity relationship. Second, varying decoding parameters systematically (temperature, top-p, top-k) would reveal how model size interacts with sampling strategy. Third, analysing individual token-level probability distributions -- rather than completed outputs -- could elucidate the mechanistic basis of the scaling effect. Fourth, longitudinal studies tracking stochasticity across model versions (e.g., Llama 2 vs. 3 vs. 3.1) could disentangle the effects of scale from the effects of training data and alignment improvements. Finally, fine-tuning experiments could test whether post-training alignment (RLHF, DPO) independently affects output consistency.

---

## 6. Conclusions

We have presented what is, to our knowledge, the first systematic multi-metric study of how LLM output stochasticity varies with model scale. By evaluating 12 instruction-tuned models from three architectural families across a 144-fold range of parameter counts, we establish a clear and robust empirical finding: **larger language models are significantly less stochastic than smaller ones** (Spearman rho = -0.767, p = 0.004).

This relationship is not a measurement artefact. It holds across 11 of 13 evaluation metrics, spanning lexical, semantic, and structural dimensions. It persists within each of the three model families tested. And it is confirmed by a Kruskal-Wallis test showing statistically significant differences across four model-size tiers (H = 137.78, p < 10^-29).

At the same time, our analysis reveals an important nuance: the stochasticity that does exist is predominantly *surface-level*. Even the most variable models in our study produce semantically consistent responses (embedding cosine similarity > 0.89), varying in phrasing and structure rather than in meaning. This finding should temper concerns about LLM reliability: when a small model gives "different" answers to the same question, it is usually saying the same thing in different words.

These results have practical implications for reproducibility (multi-run evaluation is essential for small models), model selection (the consistency cost of choosing a smaller model is now quantifiable), and safety-critical deployment (even frontier models require additional controls for high-stakes use). They also contribute to the theoretical understanding of emergence in language models, suggesting that output consistency may itself be an emergent property that crystallises in the 3--8 B parameter range.

As LLMs increasingly serve as the backbone of real-world applications, understanding not just *what* they say but *how reliably* they say it becomes essential. Our work provides a methodological framework and empirical baseline for this investigation, and we hope it inspires further research into the fundamental nature of variability in AI-generated text.

---

## Author Contributions

K.G. conceived and designed the experiment, implemented the evaluation framework, conducted all computational experiments, performed the statistical analysis, and wrote the manuscript.

## Funding

This research was supported by computational resources provided by the University of Hertfordshire GPU cluster.

## Data Availability Statement

The complete dataset of 4800 model responses, evaluation metrics, and analysis code is available at: https://github.com/khashayarghamati/Is-LL-MDeterministic-

## Conflicts of Interest

The author declares no conflicts of interest.

---

## References

1. Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.; Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler, D.; et al. Emergent Abilities of Large Language Models. *Trans. Mach. Learn. Res.* **2022**.
2. Srivastava, A.; Rastogi, A.; Rao, A.; Shoeb, A.A.M.; Abid, A.; Fisch, A.; Brown, A.R.; Santoro, A.; Gupta, A.; Garriga-Alonso, A.; et al. Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. *arXiv* **2022**, arXiv:2206.04615.
3. Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.P.; et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Adv. Neural Inf. Process. Syst.* **2023**, *36*.
4. Kuhn, L.; Gal, Y.; Farquhar, S. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. In Proceedings of the *International Conference on Learning Representations (ICLR)*, Kigali, Rwanda, 1--5 May 2023.
5. Friedman, D.; Dieng, A.B. The Vendi Score: A Diversity Evaluation Metric for Machine Learning. *Trans. Mach. Learn. Res.* **2023**.
6. Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T.B.; Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; Amodei, D. Scaling Laws for Neural Language Models. *arXiv* **2020**, arXiv:2001.08361.
7. Hoffmann, J.; Borgeaud, S.; Mensch, A.; Buchatskaya, E.; Cai, T.; Rutherford, E.; Casas, D.d.L.; Hendricks, L.A.; Welbl, J.; Clark, A.; et al. Training Compute-Optimal Large Language Models. *Adv. Neural Inf. Process. Syst.* **2022**, *35*, 30016--30030.
8. Schaeffer, R.; Miranda, B.; Koyejo, S. Are Emergent Abilities of Large Language Models a Mirage? *Adv. Neural Inf. Process. Syst.* **2023**, *36*.
9. Zhu, Y.; Lu, S.; Zheng, L.; Guo, J.; Zhang, W.; Wang, J.; Yu, Y. Texygen: A Benchmarking Platform for Text Generation Models. In Proceedings of the *41st International ACM SIGIR Conference on Research and Development in Information Retrieval*, Ann Arbor, MI, USA, 8--12 July 2018; pp. 1097--1100.
10. Lin, C.-Y. ROUGE: A Package for Automatic Evaluation of Summaries. In *Text Summarization Branches Out*; Association for Computational Linguistics: Barcelona, Spain, 2004; pp. 74--81.
11. Salton, G.; Buckley, C. Term-Weighting Approaches in Automatic Text Retrieval. *Inf. Process. Manag.* **1988**, *24*, 513--523.
12. Pineau, J.; Vincent-Lamarre, P.; Sinha, K.; Lariviere, V.; Beygelzimer, A.; d'Alche-Buc, F.; Fox, E.; Larochelle, H. Improving Reproducibility in Machine Learning Research. *J. Mach. Learn. Res.* **2021**, *22*, 1--20.
13. Biderman, S.; Schoelkopf, H.; Anthony, Q.; Bradley, H.; O'Brien, K.; Hallahan, E.; Khan, M.A.; Purber, S.; Prashanth, U.S.; Raff, E.; et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. In Proceedings of the *International Conference on Machine Learning (ICML)*, Honolulu, HI, USA, 23--29 July 2023.
14. Dettmers, T.; Pagnoni, A.; Holtzman, A.; Zettlemoyer, L. QLoRA: Efficient Finetuning of Quantized Language Models. *Adv. Neural Inf. Process. Syst.* **2023**, *36*.
15. Reimers, N.; Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the *2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Hong Kong, China, 3--7 November 2019.
