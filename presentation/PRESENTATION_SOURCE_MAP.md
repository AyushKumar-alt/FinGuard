# FinGuard: Presentation Slide-to-Source Mapping

This document provides a precise, cross-referenced mapping for every slide in [`presentation/FinGuard_Phase2_Defense.pptx`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/presentation/FinGuard_Phase2_Defense.pptx). 

For every slide, it specifies:
1. **Implementation Files**: Source code modules powering the described components.
2. **Result & Artifact Files**: Empirical metrics, index files, or experimental records.
3. **Documentation Files**: Comprehensive technical guides and audit reports.
4. **Live Terminal Commands**: Exact, copy-paste commands to demonstrate that slide's claims in front of evaluation committees.

---

## Slide 1: Title & Project Overview

### Concept & Claims
- Project Overview: FinGuard financial claim verification using NLP and evidence retrieval.
- Two-Phase investigation contrasting classical linguistic information retrieval with neural architectures.
- Academic integrity: Zero data leakage, strict evaluation governance.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Documentation** | [`README.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/README.md) | Master repository overview and architecture index |
| **Documentation** | [`docs/PROFESSOR_DEMO_GUIDE.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/PROFESSOR_DEMO_GUIDE.md) | Faculty defense navigation guide |
| **Documentation** | [`docs/POST_REORGANIZATION_AUDIT.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/POST_REORGANIZATION_AUDIT.md) | Comprehensive technical audit report |
| **Demo Command** | `python phase1/scripts/run_demo.py` | Runs Phase 1 foundational pipeline demonstration |
| **Demo Command** | `python phase2/scripts/run_demo.py` | Runs Phase 2 end-to-end neural verification demonstration |

---

## Slide 2: Problem Statement & Dataset

### Concept & Claims
- Five domain challenges: financial terminology, numerical precision, directional polarity/negation, temporal sensitivity, and evidence relevance gap.
- FinFact dataset statistics: 3,369 claims (1,492 False, 1,275 True, 602 NEI).
- 3-Way verification task: TRUE / FALSE / NOT ENOUGH INFORMATION (NEI), returning a predicted confidence score based on the highest-scoring decisive passage.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Master Dataset** | [`data/raw/finfact.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/raw/finfact.json) | Full 3,369-record raw FinFact dataset |
| **Grouped Splits** | [`data/grouped_splits/train.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_splits/train.csv) | Training partition (2,357 claims; 1,044 False, 892 True, 421 NEI) |
| **Grouped Splits** | [`data/grouped_splits/val.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_splits/val.csv) | Validation partition (337 claims; 149 False, 128 True, 60 NEI) |
| **Grouped Splits** | [`data/grouped_splits/test.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_splits/test.csv) | Held-out test partition (675 claims; 299 False, 255 True, 121 NEI) |
| **Split Generator** | [`src/create_source_grouped_splits.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/create_source_grouped_splits.py) | Source URL grouping logic ensuring zero overlap |
| **Verification Command** | `python -c "import pandas as pd; df=pd.read_csv('data/grouped_splits/test.csv'); print(df['label'].value_counts())"` | Inspects test set class distribution directly |

---

## Slide 3: Two-Phase Architecture

### Concept & Claims
- Progression from Phase 1 classical linguistic foundations to Phase 2 multi-stage neural verification.
- Phase 1 scope: Tokenizer, preserved stopwords, POS lemmatization, positional index, classical search, TF-IDF + Logistic Regression baseline.
- Phase 2 scope: Financial entity extraction, query expansion, BM25 retrieval, Cross-Encoder reranking, zero-shot NLI, conflict-aware evidence aggregation.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Phase 1 Guide** | [`phase1/README.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase1/README.md) | Phase 1 component documentation |
| **Phase 2 Guide** | [`phase2/README.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/README.md) | Phase 2 component documentation |
| **System Matrix** | [`docs/PHASE_COMPLETION_MATRIX.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/PHASE_COMPLETION_MATRIX.md) | Cross-phase feature completion comparison |
| **File Mapping** | [`docs/FILE_ORGANIZATION_MAP.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/FILE_ORGANIZATION_MAP.md) | Structural mirror alignment table |
| **Test Suite Command** | `python -m unittest discover -s tests -p "test_*.py"` | Runs all 100 unit tests spanning both phases |

---

## Slide 4: Phase 1 Implementation

### Concept & Claims
- Domain-aware regex tokenization preserving tickers, currencies, percentages, and dates.
- Preserved 58 domain polarity stopwords; removed 214 high-frequency non-informative stopwords.
- POS-guided WordNet lemmatizer with protected entity and acronym bypass.
- Positional Inverted Index: 23,568 corpus units, 32,845 terms, 393,323 positional postings.
- Exact phrase search ($k=1$) and Shunting-Yard Boolean parsing.
- Explicit note: Arbitrary variable-distance proximity search is NOT implemented.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Tokenizer** | [`src/p1_tokenizer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_tokenizer.py) | `FinancialTokenizer.tokenize` regex patterns |
| **Stopwords** | [`src/p1_stopwords.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_stopwords.py) | `FinancialStopwords.filter` (58 polarity terms) |
| **Lemmatizer** | [`src/p1_lemmatizer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_lemmatizer.py) | `POSGuidedLemmatizer.lemmatize_tokens` |
| **Positional Index** | [`src/p1_indexer.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_indexer.py) | `PositionalInvertedIndex.build_index` |
| **Retrieval Engine** | [`src/p1_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_retrieval.py) | `search_phrase` (exact $k=1$ verification) |
| **Boolean Engine** | [`src/p1_boolean_search.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_boolean_search.py) | Shunting-Yard Boolean query parser |
| **Serialized Index** | `data/indices/positional_index.pkl` | Binary serialized positional inverted index |
| **Index Verification** | `python -c "import pickle; idx=pickle.load(open('data/indices/positional_index.pkl','rb')); print(len(idx))"` | Verifies vocabulary size (32,845 terms) |
| **Phase 1 Demo** | `python phase1/scripts/run_demo.py` | Demonstrates tokenization, phrase search, and Boolean logic |

---

## Slide 5: Phase 2 Architecture & Execution Flow

### Concept & Claims
- Verified default execution trace: Claim $\to$ Rule-based entity extraction $\to$ Query expansion $\to$ BM25 candidate retrieval (Top-15) $\to$ Cross-Encoder reranking (Top-4) $\to$ Zero-Shot NLI verification $\to$ Aggregation $\to$ Final Verdict.
- Entity extraction is rule-based (regex/dictionaries), NOT learned neural NER.
- Clear distinction between configured default execution path and optional modules (Dense Retrieval, RRF Hybrid).

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Pipeline Runner** | [`src/p2_pipeline.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_pipeline.py) | `FinGuardPipeline.run` default execution trace |
| **Entity Extractor** | [`src/p2_entity_extractor.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_entity_extractor.py) | `FinancialEntityExtractor.extract_entities` |
| **Query Expansion** | [`src/p2_query_builder.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_query_builder.py) | `FinancialQueryBuilder.build_query` |
| **BM25 Retrieval** | [`src/p2_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_retrieval.py) | `BM25Retriever.retrieve` (Okapi BM25) |
| **Cross-Encoder** | [`src/p2_reranker.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_reranker.py) | `CrossEncoderReranker.rerank` |
| **NLI Verifier** | [`src/p2_verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_verifier.py) | `NLIVerifier.verify` |
| **Aggregator** | [`src/p2_aggregator.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_aggregator.py) | `EvidenceAggregator.aggregate` |
| **Optional Dense** | [`src/p2_dense_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_dense_retrieval.py) | `DenseRetriever` (384-dim MiniLM) |
| **Optional Hybrid** | [`src/p2_hybrid_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_hybrid_retrieval.py) | `HybridRetriever` (Reciprocal Rank Fusion) |
| **Phase 2 Demo** | `python phase2/scripts/run_demo.py` | Live step-by-step pipeline execution on test claim |

---

## Slide 6: Technical Implementation

### Concept & Claims
- Okapi BM25 ($k_1=1.5, b=0.75$) over 12,943 factual evidence sentences.
- Classical dense semantic index: Phase 1 SVD (128-dim) vs Phase 2 MiniLM (384-dim).
- Neural Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters).
- Zero-Shot NLI Verifier: `cross-encoder/nli-distilroberta-base` (82M parameters).
- Operational integrity guarantee: Pretrained zero-shot mode without fine-tuning on FinFact in the evaluation pipeline.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Phase 1 SVD Dense** | [`src/p1_dense_search.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p1_dense_search.py) | 128-dimensional Truncated SVD over TF-IDF |
| **Phase 2 MiniLM Dense**| [`src/p2_dense_retrieval.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_dense_retrieval.py) | 384-dimensional `all-MiniLM-L6-v2` embeddings |
| **Cross-Encoder Model**| [`src/p2_reranker.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_reranker.py) | `ms-marco-MiniLM-L-6-v2` invocation |
| **NLI Verifier Model** | [`src/p2_verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_verifier.py) | `nli-distilroberta-base` invocation |
| **Decision Logic** | [`src/p2_aggregator.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/p2_aggregator.py) | Entailment threshold (0.65), Contradiction (0.60) |
| **Model Verification** | `python -c "from src.p2_verifier import NLIVerifier; v=NLIVerifier(); print(v.model_name)"` | Confirms exact HuggingFace model checkpoint |

---

## Slide 7: Experimental Evaluation & Results

### Concept & Claims
- Strict source URL-grouped test benchmark ($N=675$, 531 unique held-out source URLs, 0 overlap).
- System A (TF-IDF + LR): Accuracy 55.56%, Macro-F1 48.97%, True Recall 59.22%, False Recall 66.22%, NEI Recall 21.49%.
- System B (Evidence Neural): Accuracy 40.30%, Macro-F1 31.07%, True Recall 2.75%, False Recall 71.57%, NEI Recall 42.15%.
- System C (Configured Pipeline): Functionally identical to System B because `FinGuardPipeline.run()` default parameters match System B's configuration exactly.
- Honest scientific presentation of the baseline superiority.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Master Result JSON** | [`outputs/experiments/aligned_grouped_verification/results.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/experiments/aligned_grouped_verification/results.json) | Sole authoritative record of all benchmark metrics |
| **Benchmark Script** | [`src/run_aligned_grouped_verification.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/run_aligned_grouped_verification.py) | Evaluation runner executing System A, B, and C |
| **Phase 1 Metrics** | [`phase1/results/metrics/baseline_performance.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase1/results/metrics/baseline_performance.json) | Phase 1 baseline evaluation metrics |
| **Phase 2 Metrics** | [`phase2/results/metrics/neural_pipeline_performance.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/results/metrics/neural_pipeline_performance.json) | Phase 2 neural pipeline evaluation metrics |
| **Metric Inspection** | `python -c "import json; d=json.load(open('outputs/experiments/aligned_grouped_verification/results.json')); print('Sys A Acc:', d['metrics']['A']['accuracy'], 'Sys B Acc:', d['metrics']['B']['accuracy'])"` | Dumps exact test accuracies to terminal |

---

## Slide 8: Error Analysis & Limitations

### Concept & Claims
- NLI output distribution across 2,700 evaluated pairs: 1,489 Neutral (55.1%), 1,145 Contradiction (42.4%), 66 Entailment (2.4%).
- 243 of 255 gold-true claims did not receive a supporting passage above the evidence aggregation threshold ($p_{\text{entail}} \ge 0.65$).
- 50-case audited error taxonomy: Irrelevant Retrieval 23 (46.0%), Wrong Entity 11 (22.0%), Insufficient Evidence 10 (20.0%)—these top 3 categories account for **44/50 cases (88.0%)**. Specific other errors account for 6 cases (12.0%).
- Clear scientific demarcation: Observed Facts vs Working Hypotheses.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Audit Logs** | [`outputs/experiments/aligned_grouped_verification/results.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/experiments/aligned_grouped_verification/results.json) | Contains `error_analysis` and `manual_review` objects |
| **Error Triage Script** | `src/run_aligned_grouped_verification.py#L380-L520` | Implements automated error classification and triage |
| **Verdict Cue Filter** | `src/run_aligned_grouped_verification.py` (`strip_verdict_cues`) | Code that strips rating text to prevent label leakage |
| **Audit Verification** | `python -c "import json; d=json.load(open('outputs/experiments/aligned_grouped_verification/results.json')); print(d['error_analysis']['manual_review']['categories'])"` | Displays the exact audited 50-case error counts |

---

## Slide 9: Conclusion & Future Work

### Concept & Claims
- Core takeaway: Evidence-grounded verification is strictly bounded by retrieval quality and verifier domain adaptation.
- Five actionable future directions: Domain NLI fine-tuning, tabular/numerical reasoning, variable-distance proximity search, hybrid retrieval calibration, and Bayesian multi-source probability calibration.

### Source Mapping Table
| Resource Type | File Path / Command | Description / Key Reference |
| :--- | :--- | :--- |
| **Conclusion Summary** | [`presentation/README.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/presentation/README.md) | Slide outline and talking points |
| **Training Scaffolding**| [`src/transformer_model.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/transformer_model.py) | PyTorch model scaffolding for future fine-tuning |
| **Viva Voce Answers** | [`presentation/VIVA_QUESTIONS_AND_ANSWERS.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/presentation/VIVA_QUESTIONS_AND_ANSWERS.md) | In-depth answers for faculty viva questions 24 and 25 |
| **Speaker Notes** | [`presentation/SPEAKER_NOTES.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/presentation/SPEAKER_NOTES.md) | Word-for-word spoken defense script for Slide 9 |

---

*(End of Presentation Source Map)*
