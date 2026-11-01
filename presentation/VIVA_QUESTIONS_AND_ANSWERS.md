# FinGuard: Comprehensive Viva Voce Questions & Defensible Answers

**Project Title:** FinGuard: Financial Claim Verification Using NLP and Evidence Retrieval  
**Academic Target:** B.Tech Final Year Capstone Project Defense  
**Audience:** Faculty Examiners, Departmental Reviewers, and Viva Voce Evaluators  

This document contains 25 rigorous technical questions and academically defensible answers covering the complete FinGuard system. Every answer is grounded in the actual codebase, empirical findings, and mathematical foundations of the repository.

---

### Q1: Why is financial fact-checking a distinct research problem compared to general-domain fact-checking?
**Answer:**  
"General-domain fact-checking datasets like FEVER focus on encyclopedic, qualitative statements (e.g., *'Albert Einstein was born in Germany'*), where verifying factual truth relies on standard entity linking and biographical prose. 

In sharp contrast, financial claim verification introduces unique domain hardships:
1. **High Numerical and Ratio Sensitivity:** Financial statements hinge on exact figures, currencies, basis points, and relative proportions (e.g., *'revenue increased 14%'* vs. *'operating profit fell 14%'*).
2. **Strict Directional Polarity:** Small lexical tokens—such as *up, down, surplus, deficit, missed, exceeded*—completely reverse market veracity.
3. **Temporal Sensitivity:** Financial facts are valid only within specific reporting windows (e.g., Q2 FY2024 vs. Q2 FY2023). A statement that is true for one quarter is false for another.
4. **Non-Standard Lexical Entities:** Corporate tickers (e.g., `$NVDA`, `BRK.A`), currencies (`₹16,811 crore`, `$90.8B`), and accounting acronyms (`EBITDA`, `GAAP`) cause standard NLP tokenizers to fail.
5. **Authoritative Evidence Requirement:** Verification cannot rely on generic news or opinion articles; it requires retrieving verifiable sentences from SEC filings, earnings releases, or certified audit reports."

---

### Q2: Why did you choose the FinFact dataset for this investigation?
**Answer:**  
"FinFact is an academic benchmark designed specifically for financial claim verification. It contains **3,369 fact-checked financial claims** paired with **18,270 evidence passages**, fact-checker justifications, and citation URLs gathered from reputable financial news and corporate disclosures.

Crucially, FinFact provides a realistic 3-class distribution:
- **1,492 False claims (44.3%)**
- **1,275 True claims (37.8%)**
- **602 Not Enough Information (NEI) claims (17.9%)**

Unlike synthetic or balanced toy datasets, FinFact reflects real-world market ambiguity: many claims are partially true, lack authoritative corroboration, or are speculative forward-looking statements that fall under NEI. Furthermore, FinFact's metadata (source URLs, claim entities, justifications) allowed us to construct a rigorous, leakage-free source-grouped evaluation benchmark."

---

### Q3: Why is specialized preprocessing necessary for financial text? Why not rely on standard tokenizers like NLTK `word_tokenize` or spaCy?
**Answer:**  
"Standard off-the-shelf tokenizers are trained on general English (e.g., Penn Treebank, news corpora) and make catastrophic assumptions on financial strings:
1. **Currency and Punctuation Fragmentation:** In standard NLTK or spaCy, a string like `$90.8B` is fragmented into four separate tokens: `['$', '90', '.', '8B']`. In Indian financial text, `₹16,811 crore` is split into `['₹', '16', ',', '811', 'crore']`. This destroys numerical entity cohesion.
2. **Ticker Symbol Splitting:** Ticker symbols like `$AAPL` or `$TSLA` have the dollar sign stripped as punctuation, turning a financial entity into a common noun or mistagged word.
3. **Fiscal Period Destruction:** Fiscal representations like `Q2 2024` or `FY2025` are split into isolated letters and digits.

In FinGuard (`src/p1_tokenizer.py`), we engineered a domain-specific regular expression tokenizer that explicitly matches financial tokens as single atomic units: `(?:\$[\d,]+(?:\.\d+)?(?:[BMKbmk]| billion| million)?|₹[\d,]+(?:\.\d+)?(?: crore| lakh)?|\d+(?:\.\d+)?%|\$[A-Z]{1,5}|[A-Z]{1,5}\b|\bFY\d{2,4}\b|\bQ[1-4]\b)`. This guarantees that financial values and entities remain intact throughout the pipeline."

---

### Q4: Why did you preserve negation and polarity words instead of removing all standard stopwords?
**Answer:**  
"In traditional information retrieval, stopwords (words like *the, is, at, which*) are stripped to reduce index size and improve search efficiency. However, standard stopword lists—such as NLTK's 179 English stopwords—contain critical polarity markers: *not, no, nor, against, above, below, down, up, off, over, under*.

In financial fact-checking, removing these words produces disastrous false equivalences:
- *'Profits did not increase'* becomes *'Profits increase'*, entirely inverting claim veracity.
- *'Inflation is above 5%'* becomes *'Inflation 5%'*, discarding the relational operator.

In FinGuard (`src/p1_stopwords.py`), we designed a polarity-preserving filter. Out of the standard NLTK list, we filter 214 high-frequency non-informative words (such as articles, auxiliary verbs, and pronouns), but explicitly preserve **58 domain polarity terms**, including negation operators (*not, no, never, neither*), directional adverbs (*up, down, above, below, under, over*), and financial sentiment cues (*loss, profit, gain, deficit*)."

---

### Q5: How does POS-guided lemmatization work, and why is it superior to naive lemmatization in finance?
**Answer:**  
"Naive lemmatization (such as calling NLTK's `WordNetLemmatizer.lemmatize(word)` without a POS tag) assumes every token is a noun by default. This causes serious morphological corruption:
- The verb *'operating'* (as in *operating profit*) is left unreduced, while *'surged'* remains *'surged'* because WordNet does not know it is a past-tense verb.
- Worse, corporate acronyms or capitalized abbreviations can be erroneously converted to lowercase base nouns.

In FinGuard (`src/p1_lemmatizer.py`), we implement **POS-Guided Lemmatization**:
1. We run Penn Treebank Part-of-Speech tagging over the sentence.
2. We map Penn Treebank tags to WordNet POS constants: `JJ*` maps to `ADJ`, `VB*` maps to `VERB`, `NN*` maps to `NOUN`, and `RB*` maps to `ADV`.
3. Before invoking WordNet, we check an **entity protection bypass**: recognized stock tickers, currency symbols, and fiscal markers are protected from any morphological transformation.
4. WordNet is then invoked with the precise syntactic category (e.g., `lemmatize('surged', pos='v')` correctly yields `surge`).

This produces canonical lemma representations across both claims and evidence sentences without corrupting financial entities."

---

### Q6: How does your Positional Inverted Index work internally? What data structures are used?
**Answer:**  
"Implemented in `src/p1_indexer.py`, our positional inverted index extends a classical inverted index by recording not only the document IDs containing each term, but also the exact token positions (word offsets) within each document.

**Internal Structure:**
```python
index: dict[str, dict[int, list[int]]]
# term -> { doc_id: [pos_0, pos_1, pos_2, ...] }
```
- The keys are the 32,845 unique vocabulary terms.
- The value for each term is a postings list dictionary mapping each `doc_id` to a sorted list of integer offsets where the term appears in that document.

**Corpus Scale:**
Our index was built over **23,568 segmented units** (3,369 claims + 18,270 evidence sentences + 3,367 fact-checker justifications) and stores **393,323 total posting entries**. It is serialized via Python's `pickle` format and verified using cryptographic SHA-256 hashes, enabling sub-millisecond in-memory lookups without requiring an external database."

---

### Q7: How does your exact phrase search algorithm work? What is its time complexity?
**Answer:**  
"Implemented in `src/p1_retrieval.py` (`search_phrase`), exact phrase search verifies that a multi-word sequence appears contiguously in the exact word order.

**Algorithm:**
1. Tokenize and lemmatize the query phrase $w_1, w_2, \dots, w_m$.
2. Retrieve the positional postings lists for each term: $P_1, P_2, \dots, P_m$. If any term is not in the index, return empty results immediately.
3. Find the candidate documents via intersection: $D = \bigcap_{i=1}^m \text{keys}(P_i)$.
4. For each candidate document $d \in D$, verify position adjacency:
   - For every start offset $p \in P_1[d]$:
   - Check if $p + 1 \in P_2[d]$, $p + 2 \in P_3[d]$, ..., $p + (m - 1) \in P_m[d]$.
   - Because postings lists are sorted integers, this check executes in $O(m \log L)$ or $O(m + L)$ via two-pointer merge, where $L$ is the average postings length per document.
5. If all offsets match, document $d$ is added to the result set.

*Architectural Boundary:* This algorithm guarantees exact adjacency ($k=1$). It does not support arbitrary variable-distance proximity ($pos_{i+1} - pos_i \le k$ for $k > 1$)."

---

### Q8: How does BM25 retrieval work, and how did you configure it for FinGuard?
**Answer:**  
"Okapi BM25 is a non-linear term frequency-inverse document frequency ranking function that addresses the term-frequency saturation problem of standard TF-IDF.

**BM25 Scoring Formula:**
$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Where:
- $f(q, D)$ is the frequency of query term $q$ in document $D$.
- $|D|$ is document length, and $\text{avgdl}$ is the average document length across the corpus.
- $k_1$ controls term frequency saturation limits (as $f \to \infty$, score asymptotes). In FinGuard, $k_1 = 1.5$.
- $b$ controls document length normalization penalty. In FinGuard, $b = 0.75$.
- $\text{IDF}(q) = \ln\left(\frac{N - n(q) + 0.5}{n(q) + 0.5} + 1\right)$, where $N$ is total evidence sentences (12,943) and $n(q)$ is the number of sentences containing $q$.

Implemented in `src/p2_retrieval.py` (`BM25Retriever`), BM25 acts as our primary candidate retrieval engine, retrieving the Top-15 most lexically relevant candidate passages per claim."

---

### Q9: What is the role of query expansion in Phase 2? Does it improve or hurt retrieval?
**Answer:**  
"In `src/p2_query_builder.py`, query expansion addresses the **vocabulary mismatch problem** between colloquial claims and formal financial reporting. 

For example, a user claim might state: *'Apple made $90B last quarter'*. Corporate earnings filings, however, use formal terminology like *'Apple Inc. reported quarterly net sales of $90.8 billion for the fiscal second quarter'*.

**Our Implementation:**
1. Extracts financial named entities (organizations, metrics, money, fiscal periods).
2. Maps company names to known stock tickers and full legal names (e.g., `Apple` $\to$ `Apple Inc. AAPL`).
3. Maps metric keywords to standardized financial reporting synonyms (e.g., `profit` $\to$ `net income, net earnings, operating margin`).
4. Binds these tokens to formulate an expanded query string passed to BM25.

**Empirical Finding:**
While query expansion successfully bridged synonyms, on unseen source URLs it occasionally caused query drift: expanding a ticker or company name across general news articles retrieved unrelated corporate press releases, contributing to the 22% entity mismatch rate observed in our 50-case error audit."

---

### Q10: Why did you introduce a neural Cross-Encoder reranker instead of relying solely on BM25?
**Answer:**  
"BM25 is a bag-of-words lexical retrieval method. It suffers from two major limitations:
1. **Inability to Capture Semantic Similarity:** If a claim says *'workforce reduction'* and the passage says *'layoffs'*, BM25 assigns zero score to that term because the surface forms do not match.
2. **Ignorance of Word Order and Syntax:** BM25 treats terms independently and cannot distinguish *'Company A acquired Company B'* from *'Company B acquired Company A'*.

In `src/p2_reranker.py`, we deploy **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (22.7 million parameters). The Cross-Encoder takes the claim and passage concatenated as a single sequence: `[CLS] Claim [SEP] Passage [SEP]`. It computes all-to-all cross-attention across all transformer layers, evaluating deep contextual interactions between every claim token and every evidence token. This allows it to rerank the Top-15 BM25 candidates and select the Top-4 most semantically relevant passages for verification."

---

### Q11: What is the architectural difference between a Bi-Encoder and a Cross-Encoder? When should each be used?
**Answer:**  
"The fundamental difference lies in how cross-attention is computed between the query (claim) and the candidate passage:

| Dimension | Bi-Encoder (e.g., `all-MiniLM-L6-v2`) | Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`) |
| :--- | :--- | :--- |
| **Architecture** | Dual independent encoders: $\mathbf{u} = \text{BERT}(Q)$, $\mathbf{v} = \text{BERT}(P)$ | Single shared encoder: $\text{score} = \text{BERT}(Q \oplus P)$ |
| **Attention Mechanism** | Token self-attention only within $Q$ and within $P$; no cross-attention between $Q$ and $P$. | Full cross-attention between every token in $Q$ and every token in $P$ across all layers. |
| **Similarity Metric** | Cosine similarity or dot product: $\cos(\mathbf{u}, \mathbf{v})$ | Multi-layer perceptron head over the `[CLS]` token output. |
| **Inference Speed** | Very fast ($O(1)$ lookup via pre-computed embeddings and vector indices). | Slow ($O(N)$ forward passes through the full transformer). |
| **Semantic Accuracy** | Moderate (loses fine-grained token interaction). | State-of-the-art (captures subtle grammatical and numerical nuances). |

**Pipeline Integration:** In FinGuard, we use lexical retrieval (BM25) or Bi-Encoders for fast candidate retrieval over thousands of documents, and use the Cross-Encoder exclusively as a stage-two reranker over the Top-15 candidates where deep attention is computationally feasible."

---

### Q12: What is Natural Language Inference (NLI), and how is it used for claim verification?
**Answer:**  
"Natural Language Inference is the task of determining whether a natural language hypothesis $H$ logically follows from a premise $P$.

In FinGuard (`src/p2_verifier.py`), we map fact-checking directly into the NLI framework:
- **Premise ($P$):** The retrieved financial evidence sentence.
- **Hypothesis ($H$):** The financial claim to be verified.

The NLI model (`cross-encoder/nli-distilroberta-base`) passes $(P, H)$ through DistilRoBERTa and outputs three raw logits, which are converted to probabilities via softmax:
1. $p(\text{Entailment})$: Probability that $P$ proves $H$ is True.
2. $p(\text{Contradiction})$: Probability that $P$ proves $H$ is False.
3. $p(\text{Neutral})$: Probability that $P$ neither proves nor disproves $H$ (insufficient evidence / NEI).

This formulation makes verification principled: a claim is verified True if explicit evidence entails it, and False if explicit evidence contradicts it."

---

### Q13: What is the formal difference between Entailment, Contradiction, and Neutral? Give financial examples of each.
**Answer:**  
"Given Premise $P$ and Hypothesis $H$:

1. **Entailment:** $H$ is necessarily true whenever $P$ is true.
   - *Premise:* 'Apple Inc. reported fiscal Q2 2024 revenue of $90.75 billion, beating consensus expectations.'
   - *Hypothesis:* 'Apple generated approximately $90.8 billion in revenue in Q2 2024.'
   - *Result:* **Entailment** ($p \ge 0.65$). The premise mathematically proves the claim.

2. **Contradiction:** $H$ is necessarily false whenever $P$ is true.
   - *Premise:* 'Tesla delivered 386,810 vehicles in Q1 2024, down 8.5% year-over-year.'
   - *Hypothesis:* 'Tesla deliveries grew by 8.5% in the first quarter of 2024.'
   - *Result:* **Contradiction** ($p \ge 0.60$). The premise directly refutes the claim.

3. **Neutral:** The truth value of $H$ cannot be determined from $P$ alone.
   - *Premise:* 'Amazon invested $4 billion in artificial intelligence startup Anthropic.'
   - *Hypothesis:* 'Amazon is the majority shareholder of Anthropic.'
   - *Result:* **Neutral**. The investment is documented, but majority ownership is neither confirmed nor refuted."

---

### Q14: Why did you use pretrained zero-shot models in Phase 2 instead of fine-tuning on the FinFact training set?
**Answer:**  
"We made this explicit architectural choice for two reasons:
1. **Scientific Baseline Investigation:** Off-the-shelf zero-shot transfer is widely promoted in industry and recent literature as a plug-and-play solution for fact-checking. We wanted to rigorously test whether general-purpose language models (trained on MS-MARCO and MNLI/SNLI) can actually generalize zero-shot to complex financial discourse.
2. **Isolating the Negative Result:** By deploying `nli-distilroberta-base` in zero-shot mode, our experiments precisely isolated and quantified the **neutral collapse phenomenon** (55.1% neutral predictions across 2,700 evaluated pairs). 

While PyTorch fine-tuning scaffolding exists in `src/transformer_model.py`, no fine-tuned weights were deployed in our configured evaluation pipeline. We explicitly document this in our presentation and audit, ensuring absolute academic honesty."

---

### Q15: Why did Phase 2 (the neural evidence-based pipeline) underperform Phase 1's classical baseline (TF-IDF + Logistic Regression)?
**Answer:**  
"This is the most critical scientific finding of our defense. System A (baseline) achieved **55.56% Accuracy** and **48.97% Macro-F1**, while System B (evidence-grounded pipeline) achieved **40.30% Accuracy** and **31.07% Macro-F1**.

Our root-cause investigation identified three distinct mechanisms:
1. **Evidence Threshold Gap (Measured Fact):** In our test set, **243 of 255 gold-true claims (95.3%) did not receive a supporting passage above the evidence aggregation threshold ($p_{\text{entail}} \ge 0.65$)**. Only 12 claims received at least one passage above the threshold. Without supporting evidence exceeding the decision threshold, the verifier cannot output True.
2. **Neutral Collapse in General NLI (Measured Fact):** General NLI models require strict logical deduction. Pretrained DistilRoBERTa labeled **1,489 of 2,700 pairs (55.1%) as Neutral**, causing True Recall to collapse to 2.75%.
3. **50-Case Error Taxonomy (Measured Observation):** Auditing 50 incorrect pipeline executions revealed that the first three failure modes accounted for **44 of the 50 cases (88.0%)**: Irrelevant Retrieval (23 cases, 46%), Wrong Entity Mismatch (11 cases, 22%), and Insufficient Evidence Pool (10 cases, 20%).
4. **Baseline Immunity (Structural Advantage):** The classical baseline (TF-IDF + Logistic Regression) classified claims directly from their lexical surface patterns without needing external evidence. It was completely immune to retrieval failures, allowing it to maintain a 59.22% True Recall."

---

### Q16: What is a source-grouped evaluation split, and why did you use it instead of a standard random train/test split?
**Answer:**  
"In fact-checking datasets, multiple claims often originate from the same news outlet, SEC filing, or corporate website domain (e.g., `reuters.com`, `cnbc.com`, `sec.gov`).

If we perform a **random split**, claims sharing the same source URL appear in both the training set and the test set. The model can simply memorize source-specific stylistic tokens, reporting templates, or URL-associated vocabulary, leading to **data leakage** and artificially inflated test metrics (often exceeding 70-80%).

In our **source URL-grouped split** (`data/grouped_splits/`):
- All 531 source URLs in the test set ($N=675$) are **100% mutually exclusive** from the 1,864 source URLs in training and 267 in validation.
- Zero source URLs overlap between splits.
- This strictly evaluates how the system generalizes to completely unseen financial publications and reporting entities, reflecting real-world deployment."

---

### Q17: What specific steps did you take to prevent data leakage in your experiments?
**Answer:**  
"We enforced four strict leakage prevention protocols:
1. **Mutually Exclusive Source Partitioning:** We verified that `len(set(train_urls) & set(test_urls)) == 0`.
2. **Corpus Isolation:** The evidence retrieval corpus (`outputs/experiments/aligned_grouped_verification/retrieval_corpus_train_val.json`) was constructed strictly from training and validation records (2,560 records, 13,088 evidence sentences). All 1,253 evidence sentences linked to test-set claims were strictly excluded from the corpus.
3. **Verdict Cue Stripping:** Fact-checker justification texts often contain explicit verdict leakages (e.g., *'This claim is false because...'*, *'We rate this statement as mostly true'*). We implemented `strip_verdict_cues()`, removing phrases containing *'true'*, *'false'*, *'rating'*, and *'verdict'* from all evidence text before indexing.
4. **Segregated Review Queue:** Anonymized evaluation IDs were segregated from private ground-truth mappings (`data/restricted_evaluation/private_evaluation_mapping.csv`)."

---

### Q18: Why did System B have a True Recall of only 2.75% (7 out of 255 claims)?
**Answer:**  
"The collapse of True Recall to 2.75% was driven by the compound interaction of our decision thresholds and evidence availability:
1. **High Entailment Bar ($T \ge 0.65$):** To predict True, at least one retrieved passage must produce $p(\text{entailment}) \ge 0.65$, and supporting passages must outnumber contradicting passages.
2. **Threshold Non-Attainment:** Out of 255 gold-true claims, **243 claims did not receive any supporting passage above the 0.65 aggregation threshold**. Only 12 claims received at least one passage above the threshold.
3. **Defaulting to False and NEI:** For the 243 claims where no entailment was found, 120 claims were predicted as False (because background sentences contained conflicting dates or figures that triggered the 0.60 contradiction threshold) and 128 claims were predicted as NEI.
4. **Final Resolution:** Of the 12 claims with retrieved support, 5 encountered conflicting contradiction signals that caused them to be downgraded. Exactly 7 claims survived aggregation as True ($7 / 255 = 2.75\%$).

*Crucial Scientific Clarification:* We report that 243/255 claims did not receive a supporting passage above the evidence aggregation threshold. We do not claim BM25 retrieved zero supporting sentences across all 15 candidates because retrieval-level ground truth was not manually annotated for all 10,125 candidates."

---

### Q19: Why are the evaluation results for System B and System C completely identical in your report?
**Answer:**  
"In our experimental design:
- **System B** was defined as the evidence-grounded neural pipeline using BM25 retrieval (Top-15), Cross-Encoder reranking (Top-4), Zero-Shot NLI verification, and threshold-based evidence aggregation.
- **System C** was defined as our configured pipeline execution (`FinGuardPipeline.run()`) without parameter overrides.

When we inspected `FinGuardPipeline.run()` in `src/p2_pipeline.py`, we verified that its default keyword arguments are:
`retrieval_mode='bm25'`, `top_k_retrieve=15`, `top_k_rerank=4`, `enable_reranker=True`, and `enable_verifier=True`. 

Because System C's configured defaults execute the exact same algorithmic branch, with identical random seeds and model checkpoints as System B, they represent the identical mathematical execution path. Rather than inventing artificial differences, our audit honestly documented that System C produced identical predictions across all 675 claims."

---

### Q20: What is Reciprocal Rank Fusion (RRF), and how does it combine BM25 and Dense Retrieval?
**Answer:**  
"Reciprocal Rank Fusion is an unsupervised ranking algorithm that merges ranked lists from multiple independent retrieval models without requiring score calibration or normalization.

**RRF Formula:**
$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where:
- $M = \{\text{BM25}, \text{Dense}\}$ is the set of retrieval models.
- $r_m(d)$ is the integer rank position of document $d$ in the output of model $m$ (1-indexed).
- $k$ is a constant smoothing parameter (typically $k = 60$).

**Advantages:**
- Lexical BM25 and neural dense models output scores on completely incompatible scales (BM25 scores are unbounded positive numbers $[0, \infty)$; cosine similarities are bounded $[-1, 1]$).
- RRF ignores raw scores and relies strictly on rank order, preventing one retriever from dominating the other.
- Documents retrieved near the top of both lists receive significantly higher fused scores."

---

### Q21: What is dense retrieval, and how is it implemented in FinGuard?
**Answer:**  
"Dense retrieval replaces lexical keyword matching with continuous vector space search.

In FinGuard (`src/p2_dense_retrieval.py`):
1. **Embedding Generation:** Each evidence sentence is encoded into a fixed-dimensional dense vector using the `all-MiniLM-L6-v2` transformer. Mean pooling over token embeddings produces a normalized 384-dimensional vector $\mathbf{v} \in \mathbb{R}^{384}$.
2. **Dense Indexing:** Embeddings for all 12,943 evidence sentences are indexed in memory.
3. **Query Search:** Given claim $q$, the model computes query embedding $\mathbf{u} = \text{MiniLM}(q)$ and calculates inner dot products (cosine similarities):
   $$\text{sim}(\mathbf{u}, \mathbf{v}_i) = \frac{\mathbf{u} \cdot \mathbf{v}_i}{\|\mathbf{u}\| \|\mathbf{v}_i\|}$$
4. The Top-$k$ passages with the highest cosine similarities are returned as candidate evidence."

---

### Q22: What is the exact difference between MiniLM dense embeddings and Phase 1 SVD semantic embeddings?
**Answer:**  
"Our repository contains two completely different dense representation techniques:

| Dimension | Phase 1 Latent Semantic Analysis (SVD) | Phase 2 Dense Sentence Transformer (MiniLM) |
| :--- | :--- | :--- |
| **Source Module** | `src/p1_dense_search.py` | `src/p2_dense_retrieval.py` |
| **Model** | Truncated Singular Value Decomposition (SVD) over TF-IDF matrix | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimensionality** | **128 dimensions** | **384 dimensions** |
| **Architecture** | Linear matrix factorization: $X \approx U_k \Sigma_k V_k^T$ | 6-layer deep bidirectional transformer with self-attention |
| **Contextuality** | Static, bag-of-words: polysemy is compressed linearly; word order is completely lost. | Deeply contextual: token representations depend on surrounding sentence context. |
| **Training** | Fit strictly on the local FinGuard vocabulary | Pretrained on 1 billion sentence pairs across diverse web corpora |

Examiners should note that these represent two distinct paradigms: classical linear dimensionality reduction (128-dim) versus deep transformer representations (384-dim)."

---

### Q23: How does your evidence aggregator make decisions, and what are its current limitations?
**Answer:**  
"Implemented in `src/p2_aggregator.py`, the aggregator processes the Top-4 passages scored by NLI:

**Current Decision Logic:**
1. A passage is counted as **Supporting** if $p(\text{entailment}) \ge 0.65$.
2. A passage is counted as **Contradicting** if $p(\text{contradiction}) \ge 0.60$.
3. If $N_{\text{support}} > N_{\text{contradict}}$, verdict is **TRUE**.
4. If $N_{\text{contradict}} > N_{\text{support}}$, verdict is **FALSE**.
5. If $N_{\text{support}} == N_{\text{contradict}} == 0$, or if signals conflict equally, verdict defaults to **NOT ENOUGH INFORMATION (NEI)**.
6. The reported confidence is a heuristic score representing the maximum predicted probability among decisive passages.

**Current Limitations:**
- **No Probability Calibration:** We do not perform temperature scaling or Platt calibration; the raw softmax maximum is reported as a heuristic score.
- **Majority Voting Vulnerability:** It treats all retrieved passages with equal weight, regardless of reranker relevance score.
- **Fixed Hard Thresholds:** Fixed cuts at 0.65 and 0.60 do not adapt to noisy domains.
- **Absence of Numerical Logic:** The aggregator does not verify whether numbers match; it relies entirely on NLI logits."

---

### Q24: What capabilities are explicitly NOT implemented in the current codebase?
**Answer:**  
"To maintain absolute scientific integrity, we explicitly document what is not implemented:
1. **Variable-Distance Proximity Search:** Phase 1 supports exact adjacent phrase matching ($k=1$), but does not support arbitrary $k$-word proximity windows (e.g., finding term A within 5 words of term B).
2. **Learned Neural Named Entity Recognition:** Phase 2 claim understanding uses rule-based regex patterns and keyword dictionaries; it is not a fine-tuned BERT or spaCy NER model.
3. **Fine-Tuned Domain Transformers:** All neural models (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, `nli-distilroberta-base`) operate in pretrained zero-shot mode without fine-tuning on FinFact in the evaluation pipeline.
4. **Structured Tabular Fact-Checking:** We evaluate unstructured textual sentences; we do not parse or reason over raw HTML/XML corporate balance sheet tables.
5. **Probability Calibration:** The confidence score is a heuristic maximum probability; formal calibration (e.g. Platt scaling) was not performed."

---

### Q25: If you had another semester to extend FinGuard, what specific technical improvements would you implement?
**Answer:**  
"With an additional semester, I would focus on four high-impact technical priorities:
1. **Financial Domain NLI Fine-Tuning:** Fine-tune compact transformers (e.g., `deberta-v3-small` or `MiniLM`) directly on financial claim-evidence pairs using contrastive loss and hard negative mining. This would directly eliminate the 55.1% neutral collapse.
2. **Numerical and Tabular Verification Engine:** Integrate a hybrid table-text parser (such as TAPAS) to verify quantitative assertions directly against corporate 10-K financial tables and cash flow statements.
3. **Adaptive Proximity Indexing:** Implement variable-distance positional indexing ($pos_{i+1} - pos_i \le k$) to capture related financial entities separated by modifier clauses.
4. **Bayesian Probability Calibration:** Replace heuristic decision thresholds with Bayesian evidence fusion and Platt scaling to produce statistically calibrated veracity probabilities."

---

*(End of Viva Voce Preparation Guide)*
