# FinGuard: Repository File Organization & Canonical Artifact Map

This document establishes the verified inventory, classification, canonical location, purpose, and operational status of every important file in the FinGuard project.

---

## 1. Master Source Code, Dataset, Model, and Output Inventory

| Original path | Current path | Classification | Canonical copy? | Purpose | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `src/preprocess.py` | `src/preprocess.py` | Phase 1 Preprocessing | **Yes (Canonical)** | Core text cleaning, regex HTML stripping, verdict cue removal | Operational |
| `src/preprocess.py` | `phase1/src/preprocessing/text_cleaner.py` | Phase 1 Preprocessing | No (Phase 1 Mirror) | Phase 1 self-contained copy of text cleaner | Operational |
| `src/p1_tokenizer.py` | `src/p1_tokenizer.py` | Phase 1 Tokenization | **Yes (Canonical)** | Finance-aware regex tokenization preserving currency, %, fiscal years | Operational |
| `src/p1_tokenizer.py` | `phase1/src/tokenization/tokenizer.py` | Phase 1 Tokenization | No (Phase 1 Mirror) | Phase 1 self-contained copy of finance tokenizer | Operational |
| `src/p1_stopwords.py` | `src/p1_stopwords.py` | Phase 1 Stopwords | **Yes (Canonical)** | 214 stopwords preserving 58 financial polarity cues (*loss, profit, up, down*) | Operational |
| `src/p1_stopwords.py` | `phase1/src/tokenization/stopwords.py` | Phase 1 Stopwords | No (Phase 1 Mirror) | Phase 1 self-contained copy of stopword filter | Operational |
| `src/p1_lemmatizer.py` | `src/p1_lemmatizer.py` | Phase 1 Lemmatization | **Yes (Canonical)** | POS-guided WordNet lemmatizer with financial entity bypass (Pipeline A) | Operational |
| `src/p1_lemmatizer.py` | `phase1/src/lemmatization/lemmatizer.py` | Phase 1 Lemmatization | No (Phase 1 Mirror) | Phase 1 self-contained copy of POS lemmatizer | Operational |
| `src/positional_index.py` | `src/positional_index.py` | Phase 1 Indexing & Search | **Yes (Canonical)** | Positional inverted index, exact phrase search, Boolean parser | Operational |
| `src/positional_index.py` | `phase1/src/indexing/positional_index.py` | Phase 1 Indexing & Search | No (Phase 1 Mirror) | Phase 1 self-contained copy of positional index | Operational |
| `src/p1_corpus_builder.py` | `src/p1_corpus_builder.py` | Phase 1 Corpus Builder | **Yes (Canonical)** | Segments FinFact into 23,568 retrieval units | Operational |
| `src/p1_corpus_builder.py` | `phase1/src/indexing/corpus_builder.py` | Phase 1 Corpus Builder | No (Phase 1 Mirror) | Phase 1 self-contained copy of corpus builder | Operational |
| `src/p1_index.py` | `src/p1_index.py` | Phase 1 Index Serialization | **Yes (Canonical)** | JSON index cache serializer and metadata manager | Operational |
| `src/p1_index.py` | `phase1/src/indexing/p1_index.py` | Phase 1 Index Serialization | No (Phase 1 Mirror) | Phase 1 self-contained copy of index serializer | Operational |
| `src/p1_retriever.py` | `src/p1_retriever.py` | Phase 1 Retrieval | **Yes (Canonical)** | Sublinear TF-IDF natural language evidence retriever | Operational |
| `src/p1_retriever.py` | `phase1/src/retrieval/tfidf/tfidf_retriever.py` | Phase 1 Retrieval | No (Phase 1 Mirror) | Phase 1 self-contained copy of TF-IDF retriever | Operational |
| `src/baseline_models.py` | `src/baseline_models.py` | Phase 1 Baselines | **Yes (Canonical)** | Classical ML models (Logistic Regression, LinearSVM, RF, Naive Bayes) | Operational |
| `src/baseline_models.py` | `phase1/src/classical_ml/baseline_models.py` | Phase 1 Baselines | No (Phase 1 Mirror) | Phase 1 self-contained copy of baseline models | Operational |
| `src/run_source_grouped_baseline.py` | `src/run_source_grouped_baseline.py` | Phase 1 Baseline Runner | **Yes (Canonical)** | Leakage-resistant source-grouped baseline training runner | Operational |
| `src/run_source_grouped_baseline.py` | `phase1/src/classical_ml/source_grouped_baseline.py` | Phase 1 Baseline Runner | No (Phase 1 Mirror) | Phase 1 self-contained copy of grouped baseline | Operational |
| `src/p1_pipeline.py` | `src/p1_pipeline.py` | Phase 1 Pipeline | **Yes (Canonical)** | Pipeline 1 standalone orchestrator for lemmatized retrieval | Operational |
| `src/p1_pipeline.py` | `phase1/src/pipeline.py` | Phase 1 Pipeline | No (Phase 1 Mirror) | Phase 1 self-contained copy of Pipeline 1 | Operational |
| `src/claim_understanding.py` | `src/claim_understanding.py` | Phase 2 Claim Understanding | **Yes (Canonical)** | Financial entity extraction (ORG, MONEY, PERCENT, DATES, METRICS) | Operational |
| `src/claim_understanding.py` | `phase2/src/claim_understanding/claim_understanding.py` | Phase 2 Claim Understanding | No (Phase 2 Mirror) | Phase 2 self-contained copy of claim understanding | Operational |
| `src/query_construction.py` | `src/query_construction.py` | Phase 2 Query Expansion | **Yes (Canonical)** | Structured financial query builder with entity-metric bindings | Operational |
| `src/query_construction.py` | `phase2/src/query_construction/query_builder.py` | Phase 2 Query Expansion | No (Phase 2 Mirror) | Phase 2 self-contained copy of query builder | Operational |
| `src/retrieval.py` | `src/retrieval.py` | Phase 2 Retrieval | **Yes (Canonical)** | BM25Okapi inverted index, TF-IDF cosine, and 128-dim SVD | Operational |
| `src/retrieval.py` | `phase2/src/retrieval/bm25/bm25_retriever.py` | Phase 2 Retrieval | No (Phase 2 Mirror) | Phase 2 self-contained copy of BM25 retriever | Operational |
| `src/dense_retrieval.py` | `src/dense_retrieval.py` | Phase 2 Dense Retrieval | **Yes (Canonical)** | Pretrained dense bi-encoder (`all-MiniLM-L6-v2`) / SVD fallback | Operational |
| `src/dense_retrieval.py` | `phase2/src/retrieval/dense/dense_retriever.py` | Phase 2 Dense Retrieval | No (Phase 2 Mirror) | Phase 2 self-contained copy of dense retriever | Operational |
| `src/reranker.py` | `src/reranker.py` | Phase 2 Reranking | **Yes (Canonical)** | Neural Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) | Operational |
| `src/reranker.py` | `phase2/src/retrieval/reranking/cross_encoder_reranker.py` | Phase 2 Reranking | No (Phase 2 Mirror) | Phase 2 self-contained copy of cross-encoder reranker | Operational |
| `src/verifier.py` | `src/verifier.py` | Phase 2 NLI Verifier | **Yes (Canonical)** | Zero-shot NLI claim verifier (`nli-distilroberta-base`) & baseline fallback | Operational |
| `src/verifier.py` | `phase2/src/verification/nli/nli_verifier.py` | Phase 2 NLI Verifier | No (Phase 2 Mirror) | Phase 2 self-contained copy of NLI verifier | Operational |
| `src/transformer_model.py` | `src/transformer_model.py` | Phase 2 Transformer Training | **Yes (Canonical)** | Transformer fine-tuning scaffolding and evaluation code | Experimental |
| `src/transformer_model.py` | `phase2/src/verification/nli/transformer_model.py` | Phase 2 Transformer Training | No (Phase 2 Mirror) | Phase 2 self-contained copy of transformer model | Experimental |
| `src/pipeline.py` | `src/pipeline.py` | Phase 2 Pipeline Orchestrator | **Yes (Canonical)** | `FinGuardPipeline` end-to-end multi-stage architecture | Operational |
| `src/pipeline.py` | `phase2/src/pipeline/finguard_pipeline.py` | Phase 2 Pipeline Orchestrator | No (Phase 2 Mirror) | Phase 2 self-contained copy of FinGuard pipeline | Operational |
| `src/api_bridge.py` | `src/api_bridge.py` | Phase 2 Interfaces | **Yes (Canonical)** | REST API bridge powering the Next.js frontend | Operational |
| `src/api_bridge.py` | `phase2/src/interfaces/api_bridge.py` | Phase 2 Interfaces | No (Phase 2 Mirror) | Phase 2 self-contained copy of API bridge | Operational |
| `app.py` | `app.py` | Phase 2 Web UI | **Yes (Canonical)** | Streamlit interactive web dashboard | Operational |
| `app.py` | `phase2/src/interfaces/app.py` | Phase 2 Web UI | No (Phase 2 Mirror) | Phase 2 self-contained copy of Streamlit app | Operational |
| `src/gold_benchmark_schema.py` | `shared/schemas/gold_benchmark_schema.py` | Shared Benchmark Schemas | **Yes (Canonical)** | Dataclass schemas for benchmark evaluation | Operational |
| `src/reference_evidence.py` | `shared/corpus/reference_evidence.py` | Shared Benchmark Corpus | **Yes (Canonical)** | Gold reference evidence alignment & extraction | Operational |
| `src/pooling.py` | `shared/corpus/pooling.py` | Shared Benchmark Corpus | **Yes (Canonical)** | Multi-retriever candidate evidence pooling | Operational |
| `src/benchmark_evaluator.py` | `shared/evaluation/benchmark_evaluator.py` | Shared IR Evaluation | **Yes (Canonical)** | IR metrics evaluator (Precision, Recall, MRR, MAP, nDCG) | Operational |
| `src/freeze_benchmark.py` | `shared/evaluation/freeze_benchmark.py` | Shared Benchmark Freezing | **Yes (Canonical)** | Deterministic benchmark dataset freezing | Operational |
| `src/judgment_automation.py` | `shared/evaluation/judgment_automation.py` | Shared Relevance Automation | **Yes (Canonical)** | Automated cross-encoder candidate relevance annotation | Operational |
| `src/adjudication.py` | `shared/evaluation/adjudication.py` | Shared Adjudication | **Yes (Canonical)** | Human vs automated judgment reconciliation | Operational |
| `src/evaluate_verifiers.py` | `shared/evaluation/evaluate_verifiers.py` | Shared Verifier Evaluation | **Yes (Canonical)** | Verifier accuracy & calibration evaluator | Operational |
| `src/evaluate_all_retrieval.py` | `shared/evaluation/evaluate_all_retrieval.py` | Shared Retrieval Evaluation | **Yes (Canonical)** | Multi-engine retrieval evaluator | Operational |
| `src/dataset_audit.py` | `shared/utils/dataset_audit.py` | Shared Auditing | **Yes (Canonical)** | Overlap, duplicate & leakage detection auditor | Operational |
| `src/eda.py` | `shared/utils/eda.py` | Shared EDA | **Yes (Canonical)** | Exploratory data analysis & figure generator | Operational |
| `src/run_aligned_grouped_verification.py` | `src/run_aligned_grouped_verification.py` | Phase 2 Experiment Runner | **Yes (Canonical)** | Aligned source-grouped evaluation benchmark runner | Operational |
| `src/run_aligned_grouped_verification.py` | `phase2/experiments/run_aligned_grouped_verification.py` | Phase 2 Experiment Runner | No (Phase 2 Mirror) | Phase 2 self-contained experiment runner copy | Operational |

---

## 2. Demonstration Scripts

| Original path | Current path | Classification | Canonical copy? | Purpose | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| N/A (New) | `phase1/scripts/run_demo.py` | Phase 1 Demonstration | **Yes (Canonical)** | Self-contained, runnable CLI demo of all Phase 1 capabilities | Operational |
| N/A (New) | `phase2/scripts/run_demo.py` | Phase 2 Demonstration | **Yes (Canonical)** | Self-contained, runnable CLI demo of all Phase 2 capabilities | Operational |

---

## 3. Master Datasets & Data Splits

| Original path | Current path | Classification | Canonical copy? | Purpose | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `finfact.json` | `finfact.json` | Master Dataset | **Yes (Canonical)** | Master FinFact dataset (3,369 fact-checked claims) | Read-Only |
| `finfact.json` | `data/raw/finfact.json` | Raw Dataset | No (Directory Mirror) | Identical raw copy in data hierarchy | Read-Only |
| `data/processed_finfact.csv` | `data/processed_finfact.csv` | Processed Dataset | **Yes (Canonical)** | Preprocessed FinFact claims with normalized fields | Operational |
| `data/processed_finfact.csv` | `data/processed/processed_finfact.csv` | Processed Dataset | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/train.csv` | `data/train.csv` | Standard Splits | **Yes (Canonical)** | Stratified 70% training split (N = 2,358 claims) | Operational |
| `data/train.csv` | `data/splits/train.csv` | Standard Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/val.csv` | `data/val.csv` | Standard Splits | **Yes (Canonical)** | Stratified 15% validation split (N = 505 claims) | Operational |
| `data/val.csv` | `data/splits/val.csv` | Standard Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/test.csv` | `data/test.csv` | Standard Splits | **Yes (Canonical)** | Stratified 15% test split (N = 506 claims) | Operational |
| `data/test.csv` | `data/splits/test.csv` | Standard Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/grouped_train.csv` | `data/grouped_train.csv` | Source-Grouped Splits | **Yes (Canonical)** | Source URL-grouped training split (N = 2,357 claims) | Operational |
| `data/grouped_train.csv` | `data/splits/grouped_train.csv` | Source-Grouped Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/grouped_val.csv` | `data/grouped_val.csv` | Source-Grouped Splits | **Yes (Canonical)** | Source URL-grouped validation split (N = 337 claims) | Operational |
| `data/grouped_val.csv` | `data/splits/grouped_val.csv` | Source-Grouped Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/grouped_test.csv` | `data/grouped_test.csv` | Source-Grouped Splits | **Yes (Canonical)** | Source URL-grouped test split (N = 675 claims) | Operational |
| `data/grouped_test.csv` | `data/splits/grouped_test.csv` | Source-Grouped Splits | No (Directory Mirror) | Identical copy in data hierarchy | Operational |
| `data/restricted_evaluation/private_evaluation_mapping.csv` | `data/restricted_evaluation/private_evaluation_mapping.csv` | Private Evaluation Data | **Yes (Canonical)** | Sensitive evaluation mapping for integrity auditing | **RESTRICTED / SECURE** |

---

## 4. Trained Model Checkpoints

| Original path | Current path | Classification | Canonical copy? | Purpose | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `outputs/best_baseline_model.pkl` | `outputs/best_baseline_model.pkl` | Trained Baseline | **Yes (Canonical)** | Best TF-IDF + Logistic Regression (Random Split, Acc: 67.00%) | Operational |
| `outputs/best_baseline_model.pkl` | `models/trained/best_baseline_model.pkl` | Trained Baseline | No (Directory Mirror) | Model registry copy | Operational |
| `outputs/experiments/source_grouped_baseline/experiment_a_raw_model.joblib` | `outputs/experiments/source_grouped_baseline/experiment_a_raw_model.joblib` | Trained Baseline | **Yes (Canonical)** | Source-grouped baseline (Raw Claims, Acc: 55.56%) | Operational |
| `outputs/experiments/source_grouped_baseline/experiment_a_raw_model.joblib` | `models/trained/experiment_a_raw_model.joblib` | Trained Baseline | No (Directory Mirror) | Model registry copy | Operational |
| `outputs/experiments/source_grouped_baseline/experiment_b_cleaned_model.joblib` | `outputs/experiments/source_grouped_baseline/experiment_b_cleaned_model.joblib` | Trained Baseline | **Yes (Canonical)** | Source-grouped baseline (Cleaned Claims, Acc: 54.96%) | Operational |
| `outputs/experiments/source_grouped_baseline/experiment_b_cleaned_model.joblib` | `models/trained/experiment_b_cleaned_model.joblib` | Trained Baseline | No (Directory Mirror) | Model registry copy | Operational |

---

## 5. Primary Reports

| Original path | Current path | Classification | Canonical copy? | Purpose | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `reports/FINGuard_NLP_METHODOLOGY_AUDIT.md` | `reports/FINGuard_NLP_METHODOLOGY_AUDIT.md` | Phase 1 Methodology Audit | **Yes (Canonical)** | Full linguistic and algorithmic audit of Phase 1 | Audited |
| `reports/FINGuard_NLP_METHODOLOGY_AUDIT.md` | `phase1/reports/FINGuard_NLP_METHODOLOGY_AUDIT.md` | Phase 1 Methodology Audit | No (Phase 1 Mirror) | Phase 1 local copy | Audited |
| `reports/PHASE2_IMPLEMENTATION_REPORT.md` | `reports/PHASE2_IMPLEMENTATION_REPORT.md` | Phase 2 Architecture Report | **Yes (Canonical)** | Full neural pipeline implementation report | Audited |
| `reports/PHASE2_IMPLEMENTATION_REPORT.md` | `phase2/reports/PHASE2_IMPLEMENTATION_REPORT.md` | Phase 2 Architecture Report | No (Phase 2 Mirror) | Phase 2 local copy | Audited |
| `reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md` | `reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md` | Benchmark Audit Report | **Yes (Canonical)** | Source-grouped aligned verification benchmark report | Audited |
| `reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md` | `phase2/reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md` | Benchmark Audit Report | No (Phase 2 Mirror) | Phase 2 local copy | Audited |
| `reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md` | `reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md` | Scientific Investigation Report | **Yes (Canonical)** | Root-cause analysis of NLI neutral collapse and domain shift | Audited |
| `reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md` | `phase2/reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md` | Scientific Investigation Report | No (Phase 2 Mirror) | Phase 2 local copy | Audited |
| `reports/FINGuard_RESEARCH_VALIDATION_AUDIT.md` | `reports/FINGuard_RESEARCH_VALIDATION_AUDIT.md` | Research Validation Report | **Yes (Canonical)** | Independent scientific audit across all components | Audited |
| `reports/FINGuard_RESEARCH_VALIDATION_AUDIT.md` | `phase2/reports/FINGuard_RESEARCH_VALIDATION_AUDIT.md` | Research Validation Report | No (Phase 2 Mirror) | Phase 2 local copy | Audited |

---

## 6. Authoritative Resolution Policy

To avoid fragmentation and guarantee backward compatibility:
1. **Source Code**: Root `src/` remains the canonical source of truth for runtime execution, unit tests, and imports. The `phase1/src/`, `phase2/src/`, and `shared/` directories provide a clean, self-contained educational hierarchy for inspection and demonstration.
2. **Datasets**: Root `finfact.json` and `data/*.csv` are the canonical data files referenced by all pipelines. `data/raw/` and `data/splits/` contain identical mirrors for navigational clarity.
3. **Model Weights**: `outputs/best_baseline_model.pkl` and `outputs/experiments/source_grouped_baseline/*.joblib` are the canonical trained checkpoints loaded by production pipelines.
4. **Experimental Results**: `outputs/experiments/aligned_grouped_verification/results.json` is the single canonical source of truth for all source-grouped evaluation figures.
5. **Reports**: Root `reports/` is the canonical master repository of all technical reports; phase-specific mirrors are provided in `phase1/reports/` and `phase2/reports/`.
