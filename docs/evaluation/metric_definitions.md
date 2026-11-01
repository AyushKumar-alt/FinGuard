# FinGuard Gold-Evidence Relevance Benchmark: Metric Definitions

This document provides formal mathematical definitions, boundary conditions, and evaluation protocols for all retrieval metrics computed by the FinGuard evaluation harness ([src/benchmark_evaluator.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/benchmark_evaluator.py)).

---

## 1. Relevance Data Model

For an evaluation claim $c \in C$, let:
- $\mathcal{E}$ be the collection of all candidate evidence units in the canonical corpus ($|\mathcal{E}| = 12,941$).
- $R_c \subseteq \mathcal{E}$ be the set of ground-truth relevant evidence items for claim $c$.
- $g(e) \in \{0, 1, 2, 3, \text{None}\}$ be the relevance grade assigned to evidence item $e$ for claim $c$:
  - **3:** Directly decisive evidence
  - **2:** Substantially relevant evidence
  - **1:** Contextually relevant evidence
  - **0:** Judged not relevant
  - **None / Unjudged:** Unknown relevance (preserved explicitly)
- $\tau \in \{1, 2, 3\}$ be the configurable binary relevance threshold (default: $\tau = 2$). An evidence item is binary-relevant if $g(e) \ge \tau$.
- $\mathbb{I}(\cdot)$ denote the indicator function.
- $r_{c, 1}, r_{c, 2}, \dots, r_{c, K}$ be the top-$K$ evidence units retrieved by a system for claim $c$.

---

## 2. Metric Formulations

### 2.1 Precision at Rank K (P@K)

Precision at rank $K$ measures the proportion of retrieved documents in the top-$K$ that are judged relevant:

$$\text{P@K}(c) = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(g(r_{c, i}) \ge \tau)$$

- **Boundary Conditions:** $\text{P@K} \in [0.0, 1.0]$. If $K = 0$, $\text{P@K} = 0.0$.
- **Macro-Averaging:**
  $$\text{P@K} = \frac{1}{|C|} \sum_{c \in C} \text{P@K}(c)$$

---

### 2.2 Recall at Rank K (R@K)

Recall at rank $K$ measures the fraction of total known relevant items for claim $c$ that appear in the top-$K$ retrieved list:

$$\text{R@K}(c) = \begin{cases} 
\frac{1}{|R_c|} \sum_{i=1}^K \mathbb{I}(g(r_{c, i}) \ge \tau) & \text{if } |R_c| > 0 \\ 
0.0 & \text{if } |R_c| = 0 
\end{cases}$$

- **Boundary Conditions:** $\text{R@K} \in [0.0, 1.0]$.
- **Macro-Averaging:** Averaged over all claims where $|R_c| > 0$.

---

### 2.3 Mean Reciprocal Rank (MRR)

Reciprocal Rank evaluates the rank of the *first* relevant evidence unit retrieved:

$$\text{RR}(c) = \begin{cases} 
\frac{1}{\min \{ i \in [1, K] \mid g(r_{c, i}) \ge \tau \}} & \text{if } \exists i \le K \text{ with } g(r_{c, i}) \ge \tau \\ 
0.0 & \text{otherwise} 
\end{cases}$$

$$\text{MRR} = \frac{1}{|C|} \sum_{c \in C} \text{RR}(c)$$

- **Interpretation:** Rewards systems that position an authoritative evidence sentence at rank 1 ($\text{RR}=1.0$) rather than rank 5 ($\text{RR}=0.20$) or rank 10 ($\text{RR}=0.10$).

---

### 2.4 Mean Average Precision (MAP)

Average Precision (AP) rewards systems for placing relevant documents higher in the ranking, weighted across all recall levels. FinGuard uses the standard TREC formulation where the normalizer is the total number of relevant documents $|R_c|$:

$$\text{AP@K}(c) = \begin{cases} 
\frac{1}{|R_c|} \sum_{i=1}^K \text{P@i}(c) \cdot \mathbb{I}(g(r_{c, i}) \ge \tau) & \text{if } |R_c| > 0 \\ 
0.0 & \text{if } |R_c| = 0 
\end{cases}$$

$$\text{MAP} = \frac{1}{|C_{\text{rel}}|} \sum_{c \in C_{\text{rel}}} \text{AP@K}(c)$$

where $C_{\text{rel}} = \{c \in C \mid |R_c| > 0\}$.

- **Important Methodological Note:** Unlike heuristic implementations that divide by $\min(|R_c|, K)$—which artificially inflates MAP when $K < |R_c|$—our harness divides by $|R_c|$, preserving standard TREC benchmark comparability.

---

### 2.5 Graded Normalized Discounted Cumulative Gain (nDCG@K)

nDCG accounts for graded relevance ($g \in \{0, 1, 2, 3\}$), using exponential gain weighting:

$$\text{gain}(g) = 2^g - 1$$

$$\text{DCG@K}(c) = \sum_{i=1}^K \frac{2^{g(r_{c, i})} - 1}{\log_2(i + 1)}$$

Let $g^*_{c, 1} \ge g^*_{c, 2} \ge \dots \ge g^*_{c, |R_c|}$ be the relevance grades of all known relevant items for claim $c$, sorted in descending order. The Ideal DCG (IDCG) is:

$$\text{IDCG@K}(c) = \sum_{i=1}^{\min(K, |R_c|)} \frac{2^{g^*_{c, i}} - 1}{\log_2(i + 1)}$$

$$\text{nDCG@K}(c) = \begin{cases} 
\frac{\text{DCG@K}(c)}{\text{IDCG@K}(c)} & \text{if } \text{IDCG@K}(c) > 0 \\ 
0.0 & \text{if } \text{IDCG@K}(c) = 0 
\end{cases}$$

$$\text{nDCG@K} = \frac{1}{|C|} \sum_{c \in C} \text{nDCG@K}(c)$$

---

### 2.6 Judged Coverage at Rank K (Coverage@K)

In pooled retrieval benchmarks, not all retrieved items have been manually judged. Judged Coverage measures the fraction of top-$K$ candidates that possess explicit human or silver annotations:

$$\text{Coverage@K}(c) = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(r_{c, i} \in \mathcal{J}_c)$$

where $\mathcal{J}_c$ is the set of judged candidates for claim $c$.
Reporting coverage alongside Precision and nDCG allows researchers to identify when retrieval systems retrieve novel, unannotated documents rather than verified negatives.

---

### 2.7 Reference Evidence Retrieval Rate (RefRate@K)

Measures the fraction of original dataset-provided fact-checker reference evidence items retrieved in the top-$K$:

$$\text{RefRate@K}(c) = \frac{|\{r_{c, 1}, \dots, r_{c, K}\} \cap \text{RefEvidence}_c|}{|\text{RefEvidence}_c|}$$

---

### 2.8 Paired Bootstrap Confidence Intervals (95% CI)

To determine statistical significance without parametric normality assumptions:
1. Resample $|C|$ evaluation claims with replacement $B = 500$ times.
2. For each bootstrap iteration $b \in [1, B]$, compute the macro-averaged metric $\bar{M}_b$.
3. Compute the empirical 2.5th and 97.5th percentiles of $\{\bar{M}_1, \dots, \bar{M}_B\}$ to establish the 95% Confidence Interval $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$.
