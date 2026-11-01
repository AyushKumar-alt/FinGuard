# FinGuard: Phase Completion & Validation Matrix

This matrix establishes the implementation status, classification, unit test validation, pipeline integration, and scientific performance of every component across **Phase 1** and **Phase 2**.

---

## 1. Component Classification Framework

Every component is categorized into one of five standard operational classifications:
- **DEFAULT PIPELINE**: Executed by default in the end-to-end production verification flow (`FinGuardPipeline.run()`).
- **IMPLEMENTED OPTIONAL COMPONENT**: Fully implemented, tested, and operational via explicit pipeline arguments or standalone invocation.
- **EXPERIMENTAL**: Research code, evaluation benchmarks, or exploratory modules not active in production.
- **NOT IMPLEMENTED**: Conceptual capability or planned extension not implemented in the current codebase.
- **LEGACY**: Superseded or deprecated modules retained for historical reproducibility.

---

## 2. Phase 1 Matrix: NLP Foundations & Classical Retrieval

| Component | Operational Classification | Unit Test Status | Source Code Path | Results & Artifacts | Empirical Metric / Functionality |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Finance-Aware Text Cleaning** | DEFAULT PIPELINE | Passing (`test_audit_and_splits.py`) | `phase1/src/preprocessing/text_cleaner.py` | `outputs/dataset_audit.json` | Strips HTML and removes verdict cues (`strip_verdict_cues`) to prevent data leakage. |
| **Domain-Aware Tokenizer** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_p1_tokenizer.py`) | `phase1/src/tokenization/tokenizer.py` | `phase1/results/metrics/` | Preserves currency amounts (`$90B`, `₹16,811 crore`), percentages (`33.5%`), tickers, fiscal periods. |
| **Polarity Stopword Filter** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_p1_stopwords.py`) | `phase1/src/tokenization/stopwords.py` | `phase1/results/metrics/` | 214 stopwords; preserves 58 directional & polarity terms (*profit, loss, up, down, increased*). |
| **POS-Guided Lemmatization** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_p1_lemmatizer.py`) | `phase1/src/lemmatization/lemmatizer.py` | `phase1/results/metrics/` | WordNet + POS tags; protects financial acronyms and entities from morphological distortion (Pipeline A). |
| **Positional Inverted Index** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_positional_index.py`, `test_p1_index.py`) | `phase1/src/indexing/positional_index.py` | `outputs/p1_positional_index.json` | 23,568 docs, 32,845 terms, 393,323 postings; tracks exact term word-offset positions. |
| **Corpus Builder** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_p1_corpus.py`) | `phase1/src/indexing/corpus_builder.py` | `outputs/corpus_provenance_sample.json` | Segments 3,369 claims, 18,270 evidence sentences, 3,367 justifications. |
| **Exact Phrase Search** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_positional_index.py`) | `phase1/src/indexing/positional_index.py` | `phase1/results/metrics/` | Deterministic positional verification requiring $pos_{i+1} = pos_i + 1$. |
| **Proximity Search (Distance k)** | **NOT IMPLEMENTED** | None | `phase1/src/indexing/positional_index.py` | N/A | Arbitrary within-$k$-words proximity search is not implemented; exact phrase search ($k=1$) is supported. |
| **Boolean Query Parser** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_positional_index.py`) | `phase1/src/indexing/positional_index.py` | `phase1/results/metrics/` | Shunting-Yard parser supporting `AND`, `OR`, `NOT`, and parenthetical sub-expressions. |
| **TF-IDF Lexical Retrieval** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_p1_retriever.py`) | `phase1/src/retrieval/tfidf/tfidf_retriever.py` | `outputs/retrieval_metrics.json` | Sublinear TF scaling with cosine ranking across segmented units. |
| **Supervised ML Baselines** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_source_grouped_baseline.py`) | `phase1/src/classical_ml/baseline_models.py` | `outputs/best_baseline_model.pkl` | Logistic Regression (Balanced): **67.00% Accuracy**, **59.98% Macro-F1** (Random Split). |
| **Source-Grouped Baseline** | EXPERIMENTAL | Passing (`test_source_grouped_baseline.py`) | `phase1/src/classical_ml/source_grouped_baseline.py` | `outputs/experiments/source_grouped_baseline/` | Logistic Regression: **55.56% Accuracy**, **48.97% Macro-F1**, **59.22% True Recall** (Held-Out Sources). |

---

## 3. Phase 2 Matrix: Neural Retrieval, Reranking & Verification

| Component | Operational Classification | Unit Test Status | Source Code Path | Results & Artifacts | Empirical Metric / Functionality |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Financial NER & Understanding** | DEFAULT PIPELINE | Passing (`test_claim_understanding.py`) | `phase2/src/claim_understanding/claim_understanding.py` | `phase2/results/metrics/` | Rule & regex extraction of ORG, MONEY, PERCENT, DATES, METRICS, and directionality cues. |
| **Structured Query Construction** | DEFAULT PIPELINE | Passing (`test_query_expansion.py`) | `phase2/src/query_construction/query_builder.py` | `phase2/results/metrics/` | Expands raw claims into entity-metric sub-queries for precision candidate retrieval. |
| **Inverted BM25 Retrieval** | DEFAULT PIPELINE | Passing (`test_retrieval.py`) | `phase2/src/retrieval/bm25/bm25_retriever.py` | `outputs/retrieval_metrics.json` | Okapi BM25 inverted index over 12,943 candidate factual evidence sentences. |
| **Dense Vector Retrieval** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_retrieval.py`) | `phase2/src/retrieval/dense/dense_retriever.py` | `outputs/dense_evidence_embeddings.npz` | `all-MiniLM-L6-v2` dense embeddings; 128-dim TruncatedSVD fallback. |
| **Reciprocal Rank Fusion (RRF)** | IMPLEMENTED OPTIONAL COMPONENT | Passing (`test_pipeline_retrieval_modes.py`) | `phase2/src/pipeline/finguard_pipeline.py` | `outputs/retrieval_metrics.json` | Fuses BM25 and Dense rankings using $RRF(d) = \sum \frac{1}{60 + r(d)}$. |
| **Neural Cross-Encoder Reranking** | DEFAULT PIPELINE | Passing (`test_leakage_and_reranker.py`) | `phase2/src/retrieval/reranking/cross_encoder_reranker.py` | `outputs/retrieval_metrics.json` | `cross-encoder/ms-marco-MiniLM-L-6-v2` evaluates joint claim-passage cross-attention. |
| **NLI Claim Verification** | DEFAULT PIPELINE | Passing (`test_verifier.py`, `test_pipeline_and_nli.py`) | `phase2/src/verification/nli/nli_verifier.py` | `outputs/verification_metrics.json` | `cross-encoder/nli-distilroberta-base` (zero-shot); premise entailment/contradiction inference. |
| **End-to-End FinGuard Pipeline** | DEFAULT PIPELINE | Passing (`test_pipeline_and_nli.py`) | `phase2/src/pipeline/finguard_pipeline.py` | `outputs/experiments/aligned_grouped_verification/` | Connects understanding $\to$ BM25 $\to$ Reranker $\to$ NLI verifier. |
| **Source-Grouped Audit Evaluation** | EXPERIMENTAL | Passing (`test_audit_and_splits.py`) | `src/run_aligned_grouped_verification.py` | `outputs/experiments/aligned_grouped_verification/results.json` | Accuracy **40.30%**, Macro-F1 **31.07%**, True Recall **2.75%** on held-out source URLs ($N=675$). |
| **Transformer Fine-Tuning Module** | EXPERIMENTAL | Untested in production | `phase2/src/verification/nli/transformer_model.py` | N/A | PyTorch fine-tuning scaffolding; production verifier uses pretrained zero-shot checkpoint. |
| **Relevance Automation & Gold Benchmark** | EXPERIMENTAL | Passing (`test_gold_benchmark.py`, `test_judgment_automation.py`) | `shared/evaluation/judgment_automation.py` | `data/benchmark/v2/` | Cross-encoder automated pseudo-labeling and human review queue generator. |
| **Interactive Interfaces** | DEFAULT PIPELINE | Validated | `phase2/src/interfaces/api_bridge.py`, `app.py`, `frontend/` | Web Dashboard & REST API | Interactive claim verification with evidence provenance and probability gauges. |

---

## 4. Known Limitations & Documented Scientific Findings

1. **The Evidence-Grounded Performance Gap**:
   - On the strictly held-out source URL-grouped test set ($N=675$), the claim-only classical baseline (55.56% accuracy, 48.97% macro-F1) outperformed the evidence-based neural pipeline (40.30% accuracy, 31.07% macro-F1).
2. **Neutral Collapse in Zero-Shot NLI (Observed & Tested)**:
   - Zero-shot `nli-distilroberta-base` achieved only **2.75% True-class recall** (recognizing only 7 of 255 true claims, while classifying 128 as NEI and 120 as False).
   - Reason: Pretrained general-domain NLI models treat financial disclaimers, forward guidance, and conditional phrasing as unprovable (NEI / Neutral), creating severe conservative classification bias.
3. **Retrieval Domain Shift (Diagnostic Observation)**:
   - Across unseen source URLs, lexical BM25 retrieval frequently retrieved background context rather than direct verifications, depriving the NLI verifier of conclusive evidence.
4. **Computational Latency Profiles**:
   - Classical TF-IDF + Logistic Regression: < 2 ms latency per claim on CPU.
   - Evidence Retrieval + Cross-Encoder Reranking + NLI: ~ 250 - 450 ms latency per claim on CPU.
