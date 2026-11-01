# FinGuard: Second-Stage Research Validation Audit

**Author:** Senior NLP Researcher & Information Retrieval Engineer  
**Date:** November 1, 2026  
**Target Repository:** FinGuard NLP System  
**Audit Document:** `reports/FINGuard_RESEARCH_VALIDATION_AUDIT.md`  
**Prior Reference Audit:** `reports/FINGuard_NLP_METHODOLOGY_AUDIT.md`  

---

## 1. Verified Phase 1 Architecture & Implementation Audit

Phase 1 provides the classical lexical, positional, and statistical baseline foundation of FinGuard. Every component was inspected at the file, class, function, test, and output level:

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                   PHASE 1 IMPLEMENTATION VERIFICATION                                                    |
+------------------------------+---------------------------+-----------------------------------+--------------------+-----------+---------+
| Method / Component           | Source File & Class       | Function Name                     | Called by Default? | Tested?   | Eval'd? |
+------------------------------+---------------------------+-----------------------------------+--------------------+-----------+---------+
| Finance-Aware Tokenizer      | src/claim_understanding.py| finance_aware_tokenize(text)      | YES (via step 1)   | YES (2)   | YES (Q) |
| Polarity-Preserving Stops    | src/p1_stopwords.py       | filter_tokens(tokens)             | NO (p1_lemma only) | YES (1)   | YES (I) |
| POS-Guided Lemmatization     | src/p1_lemmatizer.py      | POSGuidedLemmatizer.lemmatize_*   | NO (p1_lemma only) | YES (1)   | YES (I) |
| Positional Inverted Index    | src/positional_index.py   | PositionalInvertedIndex.build_*   | NO (app & p1 only) | YES (1)   | YES (I) |
| Boolean & Phrase Retrieval   | src/positional_index.py   | BooleanQueryParser.parse()        | NO (app only)      | YES (1)   | NO (IR) |
| TF-IDF Cosine Retrieval      | src/retrieval.py          | EvidenceRetrievalBenchmark.retr_* | NO (optional mode) | YES (1)   | YES (B) |
| Logistic Regression Baseline | src/baseline_models.py    | evaluate_models_on_split()        | YES (as fallback)  | YES (2)   | YES (B) |
| Linear SVM / NB / RF Bases   | src/baseline_models.py    | build_models()                    | NO (ablation only) | YES (1)   | YES (B) |
+------------------------------+---------------------------+-----------------------------------+--------------------+-----------+---------+
*Notes: (Q) = Qualitative / Regex unit validation; (I) = Indexed corpus statistics verified; (IR) = IR metric evaluation (P/R/MAP); (B) = Benchmark evaluated on test split.
```

### Detailed Component Verification:

1. **Finance-Aware Tokenization:**
   - **Source:** [`src/claim_understanding.py:69-90`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/claim_understanding.py#L69-L90), `finance_aware_tokenize(text: str) -> List[str]`.
   - **Input:** Raw text string.
   - **Output:** List of string tokens preserving currency (`$11B`, `₹5,000`), percentages (`10.5%`), fiscal periods (`FY2025`, `Q4`), and vehicles (`401(k)`, `S&P 500`).
   - **Execution Status:** **Called in default execution.** Called inside `extract_financial_entities()` on Line 93 of [`src/pipeline.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/pipeline.py#L93).
   - **Unit Tests:** [`tests/test_claim_understanding.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_claim_understanding.py) and [`tests/test_p1_tokenizer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_p1_tokenizer.py).
   - **Evaluation Status:** Experimentally validated via unit tests for exact token preservation.

2. **Polarity-Preserving Stopword Filtering:**
   - **Source:** [`src/p1_stopwords.py:57-59`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_stopwords.py#L57-L59), `filter_tokens(tokens: list) -> list`.
   - **Input:** Token list.
   - **Output:** Filtered token list where 214 stop words are removed, but financial directions (`increased`, `fell`), polar negations (`not`, `never`, `zero`), and scales (`billion`) are strictly retained.
   - **Execution Status:** **NOT called by the default pipeline.** Default `FinGuardPipeline` relies on `src/retrieval.py` which uses scikit-learn standard English stop words. `filter_tokens` is only executed when `retrieval_mode="p1_lemma"` or using `src/p1_pipeline.py`.
   - **Unit Tests:** [`tests/test_p1_stopwords.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_p1_stopwords.py).

3. **POS-Guided Lemmatization:**
   - **Source:** [`src/p1_lemmatizer.py:49-113`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_lemmatizer.py#L49-L113), `POSGuidedLemmatizer.lemmatize_tokens(tokens: list) -> list`.
   - **Input:** Token list.
   - **Output:** Lemmatized tokens using NLTK `WordNetLemmatizer` guided by `averaged_perceptron_tagger_eng`. Currency symbols, percentages, and acronyms pass through untouched via `_is_protected()`.
   - **Execution Status:** **NOT called by the default pipeline.** Only called when `retrieval_mode="p1_lemma"`.
   - **Unit Tests:** [`tests/test_p1_lemmatizer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_p1_lemmatizer.py).

4. **Positional Inverted Indexing:**
   - **Source:** [`src/positional_index.py:30-113`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/positional_index.py#L30-L113), `PositionalInvertedIndex.build_index()`, and [`src/p1_index.py:21-149`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_index.py#L21-L149), `Pipeline1Index`.
   - **Input:** Document corpus list.
   - **Output:** Serialized dictionary mapping `term -> {doc_id: [offset_0, offset_1, ...]}`.
   - **Execution Status:** **NOT queried during default verification.** The default pipeline queries `BM25Okapi` in `src/retrieval.py`. The positional index is queried in the Streamlit search tab ([`app.py:load_positional_search`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/app.py#L112)) and `src/p1_pipeline.py`.
   - **Verified Index Statistics (Extracted from [`outputs/p1_positional_index.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/p1_positional_index.json)):**
     - Total indexed evidence sentences: **23,568**
     - Vocabulary size: **32,845** terms
     - Total positional postings: **393,323**
     - Input CSV record count: 3,369 claims
     - Raw sentences extracted: 36,590
     - Duplicates excluded: 9,180
     - Short sentences excluded ($< 4$ tokens): 3,842
     - Empty sentences excluded: 104
     - Unique source URLs indexed: 2,527

5. **Boolean and Phrase Retrieval:**
   - **Source:** [`src/positional_index.py:141-303`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/positional_index.py#L141-L303), `BooleanQueryParser.parse(query: str) -> Set[int]`.
   - **Input:** String query containing `AND`, `OR`, `NOT`, parentheses, and quotes.
   - **Output:** Set of matching document IDs.
   - **Execution Status:** **NOT called in default pipeline.**
   - **Unit Tests:** [`tests/test_positional_index.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_positional_index.py).
   - **Evaluation Status:** Verified functionally in unit tests for operator precedence and phrase offsets. Has **never** been evaluated against an IR test collection with Precision/Recall/MAP metrics.

6. **TF-IDF Retrieval:**
   - **Source:** [`src/retrieval.py:153-157`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/retrieval.py#L153-L157), `retrieve_tfidf(query: str, top_k: int) -> list`.
   - **Input:** Query string, integer `top_k`.
   - **Output:** List of `(doc_id, cosine_score)` tuples.
   - **Execution Status:** Optional mode (`retrieval_mode="tfidf"`). Default pipeline uses BM25.
   - **Evaluation Status:** Fully evaluated in `outputs/retrieval_metrics.json` (Recall@10 = 0.3834, MRR = 0.5800) and `outputs/gold_benchmark_results.json` (Recall@10 = 0.2841, MRR = 0.4653).

7. **Supervised Classification Baselines:**
   - **Source:** [`src/baseline_models.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/baseline_models.py) and [`src/run_source_grouped_baseline.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/run_source_grouped_baseline.py).
   - **Input:** Claim + Evidence concatenated string.
   - **Output:** Predicted class label (`false`, `true`, `NEI`), probability distribution.
   - **Execution Status:** **Active in production.** The trained artifact [`outputs/best_baseline_model.pkl`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/best_baseline_model.pkl) ($630\text{ KB}$) is loaded on startup by `FinancialClaimVerifier` and serves as the calibrated fallback verifier.
   - **Evaluation Status:** Thoroughly evaluated under both random splits ($N=506$) and source-grouped splits ($N=675$).

---

## 2. Phase 2 Execution Path: Line-by-Line Pipeline Trace

Tracing [`src/pipeline.py:FinGuardPipeline.run()`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/pipeline.py#L68-L201) line by line reveals the exact default behavior:

```python
def run(self, claim: str,
        retrieval_mode: str = "bm25",      # DEFAULT: "bm25"
        use_query_expansion: bool = True,  # DEFAULT: True
        use_reranker: bool = True,         # DEFAULT: True
        use_nli: bool = True,              # DEFAULT: True
        top_k_retrieve: int = 15,          # DEFAULT: 15
        top_k_rerank: int = 4)             # DEFAULT: 4
```

### Execution Flow Trace:

1. **Lines 78–90 (Empty Input Check):**
   If `claim` is empty or whitespace, execution terminates immediately. Returns `verdict = "NEI"`, `confidence = 0.0`, `is_uncertain = True`.
2. **Lines 92–93 (Claim Understanding):**
   `parsed_claim = extract_financial_entities(claim)` executes [`src/claim_understanding.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/claim_understanding.py). Runs regex extraction across 8 categories (organizations, financial metrics, monetary amounts, percentages, fiscal periods, directional assertions, negations, comparisons).
3. **Lines 95–98 (Query Construction & Expansion):**
   `query_info = self.query_builder.build_queries(claim, parsed_claim)` executes [`src/query_construction.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/query_construction.py). Because `use_query_expansion=True` by default, `search_query` is set to `query_info["combined_query"]`, which appends high-precision entity-metric anchors and approved synonyms to the claim text.
4. **Lines 99–131 (Retrieval Execution):**
   - Condition check: `used_mode = retrieval_mode.lower()`.
   - By default, `used_mode == "bm25"`.
   - **Does Dense Retrieval execute? NO.**
   - **Does Hybrid Retrieval execute? NO.**
   - **Is Reciprocal Rank Fusion (RRF) called? NO.**
   - Execution branches to Line 131:
     `retrieved_pairs = self.retriever.retrieve_bm25(search_query, top_k=15)`.
     BM25Okapi scores the 12,943 candidate sentences and returns the top 15 `(doc_id, score)` pairs.
5. **Lines 133–157 (Candidate Document Assembly):**
   Pulls the 15 documents from `self.corpus`. Attaches `text`, `score`, `source_urls`, and `parent_claim_ids`.
6. **Lines 158–164 (Cross-Encoder Joint Reranking):**
   Because `use_reranker=True` and `self.reranker.is_available=True`:
   `selected_docs = self.reranker.rerank(claim, candidate_docs, top_k=4)`.
   Executes [`src/reranker.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/reranker.py) using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Scores all 15 `[CLS] claim [SEP] evidence [SEP]` pairs with full cross-attention. Sorts candidates by sigmoid probability and truncates to the top 4 passages (`top_k_rerank=4`). `reranker_active` is set to `True`.
7. **Lines 166–172 (Claim Verification Invocation):**
   Because `use_nli=True` and `self.verifier.has_transformer=True`:
   `engine_type = "transformer_nli"`.
   Calls `self.verifier.verify_claim(claim=claim, evidence_docs=selected_docs, engine_choice="transformer_nli")`.
8. **Verifier Execution Inside [`src/verifier.py:145-230`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/verifier.py#L145-L230):**
   - Iterates through the top 4 selected passages.
   - For each passage, calls `predict_nli_single_pair(claim, sentence)`.
   - `cross-encoder/nli-distilroberta-base` processes `premise = evidence`, `hypothesis = claim`.
   - Obtains softmax probabilities for `entailment`, `contradiction`, and `neutral`.
   - Partitions evidence into `supporting_evidence` ($\ge 0.50$ entailment), `contradicting_evidence` ($\ge 0.50$ contradiction), and `neutral_evidence` ($> 0.50$ neutral).
9. **Aggregation, Conflict, and Threshold Gating:**
   - **Gating Rule 1 (Missing Evidence):** If no evidence was retrieved $\implies$ Verdict = `"NEI"`, confidence = 0.50.
   - **Gating Rule 2 (Conflict):** If `supporting_evidence` is non-empty AND `contradicting_evidence` is non-empty $\implies$ `is_conflicting = True`, Verdict = `"NEI"`, confidence = 0.55.
   - **Gating Rule 3 (Uncertainty):** If the highest aggregated class probability $< 0.60$ $\implies$ `is_uncertain = True`, Verdict = `"NEI"`, confidence = highest probability.
   - **Gating Rule 4 (Direct Verdict):** If highest class probability $\ge 0.60$ and not conflicting $\implies$ Verdict = `True`, `False`, or `NEI`.

### Default Execution Path Summary:
The default pipeline executes:
$$\text{Claim} \xrightarrow{\text{Regex}} \text{Entities} \xrightarrow{\text{Expansion}} \text{BM25 (Top 15)} \xrightarrow{\text{MiniLM-CE}} \text{Rerank (Top 4)} \xrightarrow{\text{DistilRoBERTa-NLI}} \text{Gated Verdict}$$
Dense retrieval, Hybrid RRF, and Pipeline 1 lemmatization are **implemented but dormant in the default pipeline**, executing only when the user explicitly overrides `retrieval_mode`.

---

## 3. Benchmark Methodology Audit

The benchmark suite was audited across manifests, protocols, and evaluation engines:

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                             BENCHMARK METHODOLOGY AUDIT                                                       |
+--------------------------+-------------------------------------+--------------------------------------------------------------+
| Dimension                | Implementation in Code / Data       | Scientific Validity Verdict                                  |
+--------------------------+-------------------------------------+--------------------------------------------------------------+
| Candidate Pool Source    | data/benchmark/v1/ and v2/          | 4,891 pooled pairs across 75 test claims (depth 20)           |
| Ground Truth Generation  | finfact_reference (v1)              | Silver reference sentences from original fact-check articles |
| Automatic Labeling       | src/judgment_automation.py (v2)     | Dual-assessor (MiniLM cross-encoder + entity rule rubric)    |
| Independence of Labels   | System predictions used as labels   | **COMPROMISED (Circular Bias in v2)**                        |
| Source Disjointness      | Stratified sample of test.csv       | **LEAKED (117 shared URLs between train and test.csv)**      |
| Gold Justification Leak  | Excluded from corpus                | **CLEAN (Only evidence sentences indexed)**                  |
| Metric Reproducibility   | src/benchmark_evaluator.py          | **REPRODUCIBLE (Exact numbers recomputed from manifests)**   |
+--------------------------+-------------------------------------+--------------------------------------------------------------+
```

### Critical Findings on Benchmark Construction:

1. **How Relevance Labels Were Created:**
   - **In Benchmark v1 (`data/benchmark/v1/`):** Exactly **410 pairs** are labeled as relevant (`relevance_grade=3`). These were **NOT human-labeled during this project**; they are the "silver reference" evidence sentences extracted directly from `finfact.json` (`annotation_source: "finfact_reference"`). The remaining **4,480 candidate pairs in the pool were left unjudged (`relevance_grade=None`)**.
   - **In Benchmark v2 (`data/benchmark/v2/`):** Relevance labels were generated by an **automated dual-assessor pipeline** ([`src/judgment_automation.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/judgment_automation.py)). Assessor 1 was `cross-encoder/ms-marco-MiniLM-L-6-v2`; Assessor 2 was a rule-based rubric checking entity/metric overlap. Pairs where both agreed were automatically assigned grades (`judgment_status: "automatically_accepted"`). Only 107 disputed pairs were queued for human review.
2. **Relevance Judgment Independence:**
   - **The benchmark relevance judgments were NOT created independently of system predictions.**
   - In v2, one of the primary assessors assigning ground-truth relevance grades was `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is the **exact same model checkpoint** evaluated as the system under test in Configuration G. This circularity guarantees inflated scores for neural cross-encoder systems.
3. **Gold Justification & Evidence Separation:**
   - Gold justifications (`justification`) and digests (`sci_digest`) were **strictly excluded** from the candidate retrieval corpus. The corpus strictly contains evidence sentences.
4. **Unseen Sources Audit:**
   - The 75 benchmark claims were sampled from `data/test.csv`.
   - Because `data/test.csv` originates from the random split, **it shares 117 URLs with training data**. The gold benchmark does **not** evaluate generalization to unseen financial publishers.
5. **Metric Reproducibility & Mathematical Invariants:**
   - In v1, because the only graded relevant items were the 410 silver references, **Recall@10 and Reference Evidence Retrieval Rate (RefRate@10) are mathematically identical across all systems**.
   - The reported `Coverage@10` of $\sim 97.6\% - 100\%$ in early Phase 2 reports was an evaluation bug: `claim_judged_set` counted documents whose grade was `None` (unjudged). Under true definitive judgments, true judged coverage is only $19\% - 24\%$.

---

## 4. Reconciled Results Table across All Experiments

To eliminate confusion from previous conflicting reports, every experimental result across all output files is reconciled below:

```
+----------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                          RECONCILED RESULTS MATRIX                                                                 |
+--------------------------+---------------------+-------------------+-------+----------+----------+-------------+-----------------------------------+
| Experiment Name          | Model Architecture  | Split Type        | N     | Accuracy | Macro-F1 | Valid?      | Source File                       |
+--------------------------+---------------------+-------------------+-------+----------+----------+-------------+-----------------------------------+
| Exp A: Random Baseline   | TF-IDF + LogReg     | Standard Random   | 506   | 74.70%   | 0.7195   | Weak (Leak) | outputs/baseline_metrics.json     |
| Exp A: Random SVM        | TF-IDF + LinearSVC  | Standard Random   | 506   | 72.53%   | 0.6822   | Weak (Leak) | outputs/baseline_metrics.json     |
| Exp A: Random NaiveBayes | TF-IDF + MultinomNB | Standard Random   | 506   | 68.77%   | 0.5915   | Weak (Leak) | outputs/baseline_metrics.json     |
| Exp A: Random RandForest | TF-IDF + RandForest | Standard Random   | 506   | 71.94%   | 0.6784   | Weak (Leak) | outputs/baseline_metrics.json     |
| Exp B: Claim-Only        | TF-IDF + LogReg     | Standard Random   | 506   | 64.03%   | 0.6028   | Prior Test  | outputs/baseline_metrics.json     |
| Exp C: Cleaned Evidence  | TF-IDF + LogReg     | Standard Random   | 506   | 73.72%   | 0.7064   | Ablation    | outputs/baseline_metrics.json     |
| Exp D: Grouped (C=1.0)   | TF-IDF + LogReg     | Source-Grouped    | 675   | 59.70%   | 0.5528   | Defensible  | outputs/baseline_metrics.json     |
| Grouped Pilot (Raw)      | TF-IDF + LogReg     | Source-Grouped    | 675   | 57.63%   | 0.5400   | Defensible  | outputs/experiments/.../res.json  |
| Grouped Pilot (Cleaned)  | TF-IDF + LogReg     | Source-Grouped    | 675   | 57.78%   | 0.5411   | Defensible  | outputs/experiments/.../res.json  |
| System 1: Baseline Test  | TF-IDF + LogReg     | Standard Random   | 506   | 74.51%   | 0.7178   | Weak (Leak) | outputs/verification_metrics.json |
| System 2: Zero-Shot NLI  | DistilRoBERTa NLI   | Standard Random   | 100   | 34.00%   | 0.2985   | Asymmetric  | outputs/verification_metrics.json |
| System 3: Full Pipeline  | BM25+CE+DistilRoB   | Standard Random   | 100   | 34.00%   | 0.3210   | Asymmetric  | outputs/verification_metrics.json |
+--------------------------+---------------------+-------------------+-------+----------+----------+-------------+-----------------------------------+
```

### Retrieval Benchmark Results Reconciled:
*Source File:* [`outputs/retrieval_metrics.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/retrieval_metrics.json) (Held-Out Test Queries $N=150$, Silver Reference Target)

```
+----------------------------------------------------------------------------------------------------------------------+
|                                            RECONCILED RETRIEVAL BENCHMARK (N=150)                                    |
+--------------------------+-------------------------------------+-----------+------------+--------+--------+----------+
| Config ID                | Retrieval System                    | Recall@5  | Recall@10  | P@10   | MRR    | MAP      |
+--------------------------+-------------------------------------+-----------+------------+--------+--------+----------+
| Config A                 | BM25 (Inverted Index)               | 0.4089    | 0.4775     | 0.2253 | 0.7680 | 0.3969   |
| Config B                 | TF-IDF Cosine                       | 0.2962    | 0.3834     | 0.1893 | 0.5800 | 0.2716   |
| Config C                 | SVD Semantic (128-dim Dense)        | 0.0412    | 0.0707     | 0.0407 | 0.1291 | 0.0394   |
| Config D                 | Hybrid RRF (BM25 + TF-IDF)          | 0.3510    | 0.4350     | 0.2173 | 0.6711 | 0.3304   |
| Config E                 | Dense Transformer (all-MiniLM-L6-v2)| 0.4491    | 0.5302     | 0.2707 | 0.8046 | 0.4673   |
| Config F                 | Hybrid RRF (BM25 + Dense)           | 0.4752    | 0.5646     | 0.2767 | 0.8267 | 0.4711   |
| Config G                 | BM25 + Cross-Encoder Reranking      | 0.4671    | 0.5076     | 0.2547 | 0.8713 | 0.4803   |
| Config H                 | Expanded Query BM25                 | 0.3803    | 0.4500     | 0.2167 | 0.7592 | 0.3710   |
+--------------------------+-------------------------------------+-----------+------------+--------+--------+----------+
```

---

## 5. Formal Methodology Status Classification

Every major technique in the FinGuard repository is categorized according to strict research integrity criteria:

### Category A: Implemented and Executed in Default Pipeline
*Methods that actually run when a claim is processed by default:*
- **Finance-Aware Tokenization Regex** ([`src/claim_understanding.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/claim_understanding.py)).
- **Financial Entity & Metric Extraction** ([`src/claim_understanding.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/claim_understanding.py)).
- **Direction-Preserving Query Construction & Expansion** ([`src/query_construction.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/query_construction.py)).
- **BM25Okapi Lexical Retrieval** ([`src/retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/retrieval.py)).
- **Pretrained Cross-Encoder Joint Reranker (`ms-marco-MiniLM-L-6-v2`)** ([`src/reranker.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/reranker.py)).
- **Pretrained Cross-Encoder NLI Verifier (`nli-distilroberta-base`)** ([`src/verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/verifier.py)).
- **Multi-Sentence Conflict and Uncertainty Gating** ([`src/verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/verifier.py)).

### Category B: Implemented but Not Used by Default Pipeline
*Fully implemented methods that remain dormant unless explicitly requested via flags:*
- **Polarity-Preserving Custom Stopword Filtering** ([`src/p1_stopwords.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_stopwords.py)).
- **WordNet POS-Guided Financial Lemmatizer** ([`src/p1_lemmatizer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_lemmatizer.py)).
- **Positional Inverted Indexing & Postings** ([`src/positional_index.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/positional_index.py), [`src/p1_index.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_index.py)).
- **Pretrained Dense Transformer Retrieval (`all-MiniLM-L6-v2`)** ([`src/dense_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/dense_retrieval.py)).
- **Reciprocal Rank Fusion (RRF)** ([`src/pipeline.py:105-118`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/pipeline.py#L105-L118)).
- **TruncatedSVD LSA Dense Retrieval** ([`src/retrieval.py:145-151`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/retrieval.py#L145-L151)).

### Category C: Implemented but Not Experimentally Validated
*Code exists and passes unit tests, but has never been evaluated on a benchmark:*
- **Boolean Query Parser & AST Shunting-Yard Evaluator** ([`src/positional_index.py:141-303`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/positional_index.py#L141-L303)): Verified on synthetic test queries; never benchmarked for document retrieval accuracy.
- **Exact Phrase Matching with Consecutive Offset Verification** ([`src/positional_index.py:77-113`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/positional_index.py#L77-L113)): Verified functionally; IR performance unmeasured.

### Category D: Experimentally Evaluated Under a Weak Protocol
*Evaluated, but results suffer from source leakage, sample asymmetry, or circular labeling:*
- **Standard Random-Split Baseline (74.7% Acc / 0.7195 Macro-F1):** Flawed due to 117 source URLs overlapping between training and test sets.
- **Zero-Shot NLI Verifier vs. Baseline Comparison (34.00% vs. 74.51%):** Mismatched sample sizes ($N=100$ vs. $N=506$).
- **Benchmark v2 Retrieval Evaluation:** Circular evaluation bias because `ms-marco-MiniLM-L-6-v2` served as both relevance judge and system under test.

### Category E: Scientifically Validated Under a Defensible Protocol
*Results derived from leakage-free, mutually exclusive data with reproducible controls:*
- **Source-Grouped Logistic Regression Baseline:**
  - Evaluated on [`data/grouped_test.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_test.csv) ($N=675$, exactly 0 URL overlap).
  - Accuracy: **57.63% (Raw) / 57.78% (Cleaned)**.
  - Macro-F1: **0.5400 (Raw) / 0.5411 (Cleaned)**.
  - Verifies that true out-of-domain baseline accuracy is $\approx 58\%$, not $75\%$.
- **Controlled Retrieval Comparison (Configs A-H on 150 Held-Out Queries):**
  - All 8 retrieval configurations queried the exact same 12,943 candidate sentences for the exact same 150 test claims against silver references.
  - Conclusively establishes that Hybrid RRF (BM25 + Dense) achieves the highest recall ($0.5646$) and Cross-Encoder Reranking achieves the highest MRR ($0.8713$).

### Category F: Planned or Absent
*Features documented or conceptualized that do not exist in the codebase:*
- **Arbitrary Proximity Search ($\text{NEAR}/k$):** Absent from boolean parser.
- **Transformer Fine-Tuning on FinFact:** Zero fine-tuned checkpoints exist.
- **Deterministic Numerical & Currency Verification Layer:** Absent.
- **Statistical Named Entity Recognition Model:** Absent (regex/dictionary only).

---

## 6. Unsupported Claims That Must Not Appear in a Research Presentation

1. **DO NOT CLAIM: *"FinGuard fine-tuned a transformer or cross-encoder verifier on financial data."***
   *Fact:* No transformer models were fine-tuned. The densitometer, reranker, and NLI models are off-the-shelf zero-shot HuggingFace checkpoints.
2. **DO NOT CLAIM: *"The neural verifier outperforms the baseline with 75% accuracy."***
   *Fact:* The pretrained NLI model achieved only **34.00% accuracy** on financial test claims. The 74.7% figure belongs strictly to the TF-IDF Logistic Regression baseline evaluated on an overlapping random split.
3. **DO NOT CLAIM: *"The baseline achieves 74.7% out-of-domain generalization."***
   *Fact:* The 74.7% accuracy relies on publisher leakage (117 shared URLs). On unseen publishers (source-grouped split), baseline accuracy is **57.78%**.
4. **DO NOT CLAIM: *"The system uses a 27,617-document retrieval corpus in production."***
   *Fact:* The production retrieval engine queries **12,943 unique evidence sentences**. The 27,617 figure is a theoretical sum that includes raw claims, justifications, and digests from metadata manifests.
5. **DO NOT CLAIM: *"Benchmark v2 is an independent human gold-standard benchmark."***
   *Fact:* Benchmark v2 relevance labels were automatically generated by `ms-marco-MiniLM-L-6-v2` and rule heuristics; only 1 pair in v1 was human-reviewed.

---

## 7. Prioritized Roadmap for Defensible Next Experiments

```
+----------------------------------------------------------------------------------------------------+
|                                    DEFENSIBLE EXPERIMENT ROADMAP                                   |
+----------+-------------------------------------+---------------------------------------------------+
| Priority | Experiment Objective                | Technical Implementation Details                  |
+----------+-------------------------------------+---------------------------------------------------+
| 1 (P0)   | Aligned 675-Claim Verification Test | Evaluate DistilRoBERTa NLI and Full Pipeline on   |
|          |                                     | all N=675 source-grouped claims (match baseline). |
| 2 (P0)   | Deterministic Arithmetic Verifier   | Implement rule-based numerical & date pre-check   |
|          |                                     | to intercept currency, %, and FY discrepancies.   |
| 3 (P1)   | True Compact NLI Pilot (22.7M)      | Evaluate MoritzLaurer/MiniLM-L6-mnli (H=384,      |
|          |                                     | 22.7M params) zero-shot on source-grouped test.   |
| 4 (P1)   | Human Adjudication of v2 Queue      | Adjudicate 107 disputed pairs in human review     |
|          |                                     | queue to remove circular cross-encoder bias.      |
| 5 (P2)   | Sentence-Level NLI Fine-Tuning      | Fine-tune compact NLI on verified claim-sentence  |
|          |                                     | pairs from reference_evidence.json.               |
+----------+-------------------------------------+---------------------------------------------------+
```

---

## 8. Final Synthesis: What Has FinGuard Honestly Achieved, and What Remains Unproven?

### What FinGuard Has Honestly Achieved Today:
1. **A Highly Functional, Documented Multi-Strategy IR Engine:**
   FinGuard has successfully implemented, indexed, and evaluated 8 distinct retrieval architectures across 12,943 evidence sentences. It demonstrably proves that fusing BM25 with Dense Transformer embeddings via Reciprocal Rank Fusion improves Recall@10 by $+47.3\%$ over TF-IDF ($0.3834 \to 0.5646$), and that Cross-Encoder joint reranking improves MRR to $0.8713$.
2. **A Complete, End-to-End Operational Application:**
   The repository contains a fully working, interactive pipeline orchestrating entity parsing, query expansion, candidate retrieval, cross-encoder reranking, multi-sentence aggregation, conflict gating, a Streamlit dashboard, a Python HTTP server, and a Next.js web application.
3. **Rigorous Linguistic Preprocessing Tools:**
   The custom finance-aware tokenizer, 214-word polarity-preserving stopword dictionary, and POS-guided lemmatizer have been verified through exhaustive automated unit test suites.
4. **An Audited, Scientifically Defensible Baseline Benchmark:**
   The project has surfaced and eliminated critical data leakage risks (source URL overlap and verdict cues), proving that classical TF-IDF models achieve a real out-of-domain baseline of **57.78% accuracy and 0.5411 Macro-F1**.

### What Remains Unproven Today:
1. **Evidence-Grounded Neural Verification Superiority:**
   It is completely unproven that neural cross-encoders outperform classical models on FinFact claim verification. The zero-shot NLI verifier collapsed to $34\%$ accuracy because journalistic text triggers neutral classifications.
2. **Transformer Fine-Tuning Feasibility:**
   No transformer has been fine-tuned on FinFact. Whether fine-tuning an 82M or 22M parameter transformer on claim-level labels without injecting sentence-level noise is feasible remains unproven.
3. **Independent Human Relevance Benchmark:**
   The retrieval benchmark has not been validated against an extensive human-annotated ground-truth corpus. Benchmark v1 is based on fact-checker reference sentences, and v2 relies on automated cross-encoder consensus.
4. **Fine-Grained Numerical Reasoning:**
   The neural pipeline cannot reliably verify numeric scaling (e.g., crores vs. billions, percentage point differences, fiscal reporting periods) without a dedicated deterministic arithmetic component.

