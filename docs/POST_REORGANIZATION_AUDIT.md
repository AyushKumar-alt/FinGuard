# FinGuard: Comprehensive Post-Reorganization Audit & Verification Report

**Date of Audit**: November 1, 2026  
**Auditor**: Senior NLP & IR Verification Auditor  
**Repository Working Directory**: `c:\Users\calpo\OneDrive\Pictures\final project of NLP`

---

## Executive Summary

Following the restructuring of the FinGuard repository into dedicated `phase1/`, `phase2/`, `shared/`, `data/`, and `models/` hierarchies, this audit was conducted to verify that:
1. Original implementation code, datasets, evaluation splits, and trained model weights were strictly preserved without deletion or corruption.
2. Root-level backward-compatibility import bridges remain operational, ensuring zero regressions across existing test suites and interfaces.
3. Every quantitative metric and architectural statement in project documentation is backed by empirical artifacts on disk.
4. Phase 1 and Phase 2 demonstration scripts execute cleanly and deterministically without hard-coded shortcuts or network dependencies.
5. Sensitive evaluation mapping materials remain isolated and protected.

---

## 1. Git Repository Status

A diagnostic inspection of the version control environment was executed using standard Git plumbing and porcelain commands:

```powershell
git rev-parse --show-toplevel
git status --short --branch
git log -1 --oneline
```

### Audit Findings
- **Execution Result**: All three commands exited with **Exit Code 1**, returning:
  ```text
  fatal: not a git repository (or any of the parent directories): .git
  ```
- **Root Cause**: The workspace directory is an uninitialized local directory and is not enclosed within any active `.git` repository hierarchy.
- **Verification Guarantee**: No Git commits, tags, branch manipulations, or remote pushes were executed during the reorganization or this audit. All file operations were executed via direct filesystem tools preserving historical files.

---

## 2. Master File Integrity & Cryptographic Checksums

To verify that historical datasets, splits, trained models, and benchmark evaluation artifacts were not modified or corrupted, SHA-256 cryptographic hashes and byte counts were captured:

| File Path | File Size (Bytes) | SHA-256 Checksum | Integrity Status |
| :--- | :---: | :--- | :---: |
| `finfact.json` | 29,972,446 | `5e36d32a0e0149f5d86b1f81ab6fd85144136cb7975374364060f8fcf43629fd` | Verified Intact |
| `data/train.csv` | 30,775,828 | `5f27121566e6e98354142d895ddef94d60356e83bc00e3e3cae9df8a6b0c0b5b` | Verified Intact |
| `data/val.csv` | 6,484,861 | `f5a884d52757e25c7b85f9002d1af98cd382aac0417443621834032c1809d819` | Verified Intact |
| `data/test.csv` | 6,594,504 | `5cdc205b8f929ff606ed3b067fa33aabf18f4e04dbb955093c1c62c12ccc9442` | Verified Intact |
| `data/grouped_train.csv` | 29,700,663 | `c31f02b18d7393552a7f6e8ab1df83cba4ada1d85d74f4150581a0cfff1797e7` | Verified Intact |
| `data/grouped_val.csv` | 4,829,918 | `dc994f2993fc42b24ac1885927e9b2017f2d7bff9225db7da8b476caeb711660` | Verified Intact |
| `data/grouped_test.csv` | 9,324,612 | `7c28fbdb6a4f039080fec0d46585ea528947c723731207c92108ce3a003ac03a` | Verified Intact |
| `data/processed_finfact.csv` | 43,854,827 | `610201755e20444551e3117136e0782f732399e0c87939c98590f4d04e7b6500` | Verified Intact |
| `outputs/best_baseline_model.pkl` | 630,820 | `3396dda4dc3e971fc145b52618de96160aa87dd4cfe5f246793ee3b4b1753415` | Verified Intact |
| `outputs/p1_positional_index.json` | 17,413,191 | `808698f518c1289b4950c72e343da78ffaa0d6c18d697015fd2259b4a3c0d016` | Verified Intact |
| `outputs/retrieval_metrics.json` | 2,078 | `2131e80b2901d460628ac48106369c915551d0ef402e474e270a179b2e291601` | Verified Intact |
| `outputs/verification_metrics.json` | 18,017 | `0d26bd7d2f463a546673894cd728ee8290dc429eda76c51bdc78e35d5ca60a17` | Verified Intact |
| `outputs/experiments/aligned_grouped_verification/results.json` | 44,182 | `d4a07e0ddfd80459bf85db284dc4cf37499d61b34dc3dd781fa1a9de9e6a53d8` | Verified Intact |
| `outputs/experiments/source_grouped_baseline/results.json` | 6,040 | `ee36cb5a983311b85832ac3cc35684b311d034bcfb3d4003f8ee1808e80b5af5` | Verified Intact |
| `data/restricted_evaluation/private_evaluation_mapping.csv` | 380,141 | `d491b1aba09960d257412df9c352beddf9dd50b9338ddaf4d939d3509db275fa` | Verified Intact |

---

## 3. Actual Directory Structure & Duplicate Inventory

An automated SHA-256 tree audit identified 226 total files and 113 duplicate clusters across the project hierarchy.

### Findings on Duplicate Copies:
1. **Source Code**: Root `src/` retains all 32 original Python modules. Organized copies reside in `phase1/src/` (11 modules), `phase2/src/` (10 modules), and `shared/` (11 modules). Root `src/` modules remain operational as the unified runtime source.
2. **Datasets**: Root `finfact.json` and `data/*.csv` are mirrored into `data/raw/` and `data/splits/`.
3. **Checkpoints**: Root `outputs/best_baseline_model.pkl` and `outputs/experiments/source_grouped_baseline/*.joblib` are mirrored into `models/trained/`.
4. **Metrics**: Output JSONs in root `outputs/` are mirrored into `phase1/results/metrics/` and `phase2/results/metrics/`.
5. **Reports**: Root `reports/` contains all 5 master reports; phase-relevant copies reside in `phase1/reports/` (1 report) and `phase2/reports/` (4 reports).
6. **Tests**: Root `tests/` contains all 19 test files (100 test cases); partitioned copies reside in `phase1/tests/` (9 files, 58 cases) and `phase2/tests/` (10 files, 42 cases).

### Authoritative Resolution Policy:
- **Canonical Runtime**: Root `src/` is the canonical source of truth for runtime execution, automated test discovery, and import stability.
- **Canonical Demonstration**: `phase1/scripts/run_demo.py` and `phase2/scripts/run_demo.py` are the authoritative demonstration entry points.
- **Canonical Evaluation Output**: `outputs/experiments/aligned_grouped_verification/results.json` is the sole authoritative record for source-grouped verification benchmarks.

---

## 4. Phase 1 Implementation Verification

Each Phase 1 component was audited directly in source code:

| Component | Source File | Class / Function | Input | Output | Independently Executable? | Used in Other Pipelines? | Relevant Test File | Result Artifact |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Text Preprocessing** | `src/preprocess.py` | `clean_text`, `strip_verdict_cues` | Raw string | Normalized string | Yes | Yes (Pipeline 1 & FinGuard) | `test_audit_and_splits.py` | `outputs/dataset_audit.json` |
| **Financial Tokenization** | `src/p1_tokenizer.py` | `finance_aware_tokenize` | Text string | `list[str]` tokens | Yes | Yes (Pipeline 1) | `test_p1_tokenizer.py` | `phase1/results/metrics/` |
| **Polarity Stopwords** | `src/p1_stopwords.py` | `STOP_WORDS`, `is_stop_word` | Token string | Boolean | Yes | Yes (Pipeline 1) | `test_p1_stopwords.py` | `phase1/results/metrics/` |
| **POS Lemmatization** | `src/p1_lemmatizer.py` | `POSGuidedLemmatizer.lemmatize_tokens` | `list[str]` tokens | `list[str]` lemmas | Yes | Yes (Pipeline 1) | `test_p1_lemmatizer.py` | `phase1/results/metrics/` |
| **Positional Inverted Index** | `src/positional_index.py` | `PositionalInvertedIndex` | Token lists with positions | Term $\to$ Doc $\to$ Positions | Yes | Yes (Pipeline 1) | `test_positional_index.py` | `outputs/p1_positional_index.json` |
| **Corpus Builder** | `src/p1_corpus_builder.py` | `P1CorpusBuilder` | FinFact CSV / JSON | Dict of 23,568 units | Yes | Yes (Pipeline 1) | `test_p1_corpus.py` | `outputs/corpus_provenance_sample.json` |
| **Exact Phrase Search** | `src/positional_index.py` | `search_phrase` | Phrase string | `set[int]` doc IDs | Yes | Yes (Index query engine) | `test_positional_index.py` | `phase1/results/metrics/` |
| **Proximity Search** | `src/positional_index.py` | **Not Implemented** | N/A | N/A | No | No | None | N/A (Only exact phrase $k=1$ supported) |
| **Boolean Retrieval** | `src/positional_index.py` | `BooleanQueryParser.evaluate` | Infix Boolean string | `set[int]` doc IDs | Yes | Yes (Index query engine) | `test_positional_index.py` | `phase1/results/metrics/` |
| **TF-IDF Retrieval** | `src/p1_retriever.py` | `NaturalLanguageEvidenceRetriever` | Query string | `list[tuple[int, float]]` | Yes | Yes (Pipeline 1) | `test_p1_retriever.py` | `outputs/retrieval_metrics.json` |
| **Classical ML Baselines** | `src/baseline_models.py` | `train_baseline_models` | Text arrays & labels | Scikit-learn estimators | Yes | Yes (Production fallback verifier) | `test_source_grouped_baseline.py` | `outputs/best_baseline_model.pkl` |

**Proximity Search Clarification**: General proximity search with variable distance constraints ($k > 1$ tokens between arbitrary words) is **NOT implemented**. Only exact phrase matching ($k=1$ positional adjacency check) is implemented.

---

## 5. Phase 2 Implementation & Default Pipeline Call Trace

Tracing `FinGuardPipeline.run()` (`src/pipeline.py`, lines 77–225) establishes the default production call chain:

```
[Input Claim]
      │
      ▼ (Line 102)
1. Claim Understanding: extract_financial_entities(claim)
      │
      ▼ (Line 105)
2. Query Construction: FinancialQueryBuilder.build_queries(claim, parsed_claim)
      │
      ▼ (Line 150)
3. Candidate Retrieval (DEFAULT: retrieval_mode="bm25")
   • Calls: EvidenceRetrievalBenchmark.retrieve_bm25(search_query, top_k=15)
   • Corpus: 12,943 candidate factual evidence sentences
      │
      ▼ (Line 182)
4. Neural Reranking (DEFAULT: use_reranker=True)
   • Calls: PretrainedCrossEncoderReranker.rerank(claim, candidate_docs, top_k=4)
   • Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (cached locally, 22.7M params)
      │
      ▼ (Line 191)
5. NLI Verification (DEFAULT: use_nli=True)
   • Calls: FinancialClaimVerifier.verify_claim(claim, evidence_docs, engine_choice="transformer_nli")
   • Model: cross-encoder/nli-distilroberta-base (cached locally, 82.0M params)
   • Premise = Evidence passage; Hypothesis = Claim
      │
      ▼ (Lines 207-224)
6. Decision Aggregation
   • Output Verdict: True | False | NEI
   • Calibrated Confidence Score, Class Probabilities, Evidence Passages with URLs
```

### Component Classification Summary

- **DEFAULT PIPELINE**:
  - `extract_financial_entities` (Claim understanding)
  - `FinancialQueryBuilder` (Query expansion)
  - `BM25Okapi` (Lexical candidate retrieval)
  - `PretrainedCrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`)
  - `FinancialClaimVerifier` (`nli-distilroberta-base`)
  - Conflict / Confidence Decision Aggregator
  - `api_bridge.py` & `app.py` UI Interfaces
- **IMPLEMENTED OPTIONAL COMPONENT**:
  - Dense Bi-Encoder Retrieval (`sentence-transformers/all-MiniLM-L6-v2` / 128-dim SVD)
  - Reciprocal Rank Fusion (`_reciprocal_rank_fusion`, $k=60$)
  - TF-IDF Cosine Retrieval (`retrieve_tfidf`)
  - SVD/LSA Retrieval (`retrieve_dense`)
  - Pipeline 1 Lemmatized Retrieval (`p1_lemma`)
  - Calibrated Logistic Regression Fallback Verifier (`predict_baseline`)
- **EXPERIMENTAL**:
  - Source-Grouped Aligned Verification Harness (`run_aligned_grouped_verification.py`)
  - Transformer Fine-Tuning Scaffolding (`transformer_model.py` — audited: production models are pretrained zero-shot, not fine-tuned on FinFact)
  - Gold Benchmark Generation & Adjudication (`data/benchmark/`, `judgment_automation.py`, `adjudication.py`)
- **NOT IMPLEMENTED**:
  - Variable-distance proximity search ($k > 1$)
  - Learned financial NER (rule/regex heuristics are used)
  - Numerical formula consistency verifier
  - Live external web search integration

---

## 6. Evaluation Metric Verification (Source-Grouped Benchmark)

The quantitative results of the aligned source-grouped benchmark were audited directly from `outputs/experiments/aligned_grouped_verification/results.json`:

### Test Split Specification
- **Split Type**: Source URL-grouped held-out test split (partitioned strictly by `source_url` so that no source URL in the test set appears in training or validation).
- **Test Size**: $N = 675$ claims (from 531 unique source URLs).
- **Test Label Distribution**:
  - False: **299** claims (44.30%)
  - True: **255** claims (37.78%)
  - NEI: **121** claims (17.93%)

### Empirical Performance Comparison

| Metric / Dimension | System A: Claim-Only Baseline (TF-IDF + Logistic Reg) | System B: Evidence-Grounded FinGuard (BM25 + Cross-Encoder + NLI) | System C: FinGuard Calibrated Thresholds |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **55.56%** (375 / 675) | **40.30%** (272 / 675) | **40.30%** (272 / 675) |
| **Macro-Precision** | 49.19% | 38.94% | 38.94% |
| **Macro-Recall** | 48.97% | 38.82% | 38.82% |
| **Macro-F1** | **48.97%** | **31.07%** | **31.07%** |
| **Weighted-F1** | 54.98% | 33.83% | 33.83% |
| **False Class Recall** | 66.22% (198 / 299) | **71.57%** (214 / 299) | **71.57%** (214 / 299) |
| **True Class Recall** | **59.22%** (151 / 255) | **02.75%** (7 / 255) | **02.75%** (7 / 255) |
| **NEI Class Recall** | 21.49% (26 / 121) | **42.15%** (51 / 121) | **42.15%** (51 / 121) |

### Documentation Correction Note
An earlier summary draft reported System A True Recall as `54.00%`. The verified ground-truth value extracted directly from `results.json` (`metrics.A.per_class.true.recall`) is **59.22%** (151 / 255). All project documentation has been corrected to reflect this exact value.

---

## 7. Root-Cause Analysis: Evidence vs Hypotheses

The root-cause investigation (`reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md`) was reviewed to distinguish verified facts from diagnostic observations:

1. **Directly Observed Evidence (Verified Facts)**:
   - Across all 2,700 production candidate pairs evaluated on the test set, NLI model outputs yielded: **1,489 neutral**, **1,145 contradiction**, and only **66 entailment** predictions.
   - For 243 of the 255 gold-true claims, not a single supporting passage exceeded the verifier's confidence threshold.
   - The NLI input configuration (`tokenizer(evidence, claim)`) and label mapping (`0: contradiction`, `1: entailment`, `2: neutral`) were audited and verified correct.
2. **Diagnostic Observations (Screening Signals)**:
   - Automated entity mismatch heuristics flagged entity discrepancies on 304 claims; however, the heuristic extractor has known noise on conversational claims and is an automated screening signal rather than a gold human judgment.
3. **Hypotheses (Plausible Explanations Subject to Ongoing Research)**:
   - *Hypothesis 1 (Neutral Collapse)*: Pretrained general NLI models (`nli-distilroberta-base`) treat financial hedging, conditional guidance, and fiscal disclaimers as unproven (NEI) rather than factual entailment.
   - *Hypothesis 2 (Retrieval Domain Shift)*: BM25 retrieval over unseen source URLs frequently retrieves topical background sentences rather than decisive verification sentences.
4. **Current Scientific Limitations**:
   - Human relevance adjudication for the review queue (`data/human_evaluation/`) is awaiting multi-annotator completion. Causality between retrieval rank degradation and NLI errors remains an observed correlation, not a mathematical certainty.

---

## 8. Test Duplication Analysis & Test Coverage

An automated Abstract Syntax Tree (AST) inspection of all test files across the repository was conducted:

| Test Directory | Test Files | Total Test Methods | Test Scope |
| :--- | :---: | :---: | :--- |
| `tests/` (Root) | 19 | **100** | Full regression test suite covering all modules |
| `phase1/tests/` | 9 | **58** | Exclusively Phase 1 components (tokenization, index, baselines) |
| `phase2/tests/` | 10 | **42** | Exclusively Phase 2 components (NER, query, retrieval, NLI) |

### Mathematical Verification of Partition:
- Intersection of files between `phase1/tests/` and `phase2/tests/`: **0 files** (Mutually exclusive).
- Union of files in `phase1/tests/` and `phase2/tests/`: **19 files** (Identical to `tests/`).
- Sum of test cases: $58 + 42 = \mathbf{100}$.
- **Finding**: There are **100 unique test cases** in the repository. The phase directories house a clean, non-redundant partition of the test suite.
- **Execution Verification**: Running the complete test suite exited with **Exit Code 0** (`Ran 100 tests in 124.128s - OK`).

---

## 9. Demonstration Script Execution Verification

Both demonstration scripts were executed end-to-end from the project root:

### 1. Phase 1 Demo: `python phase1/scripts/run_demo.py`
- **Exit Code**: **0 (Clean Success)**
- **Pipeline Stages Verified**:
  - Text normalization (`clean_text`, `strip_verdict_cues`)
  - Financial tokenization (preserving `$90.8B`, `₹16,811 crore`, `33.5%`)
  - Polarity-preserving stopwords (audit of 58 protected terms)
  - POS-guided WordNet lemmatization with entity protection
  - Positional inverted index term lookup
  - Exact phrase search (`"net profit"`)
  - Boolean search parser (`Apple AND profit`, `profit AND NOT loss`)
  - Production index stats verification (`outputs/p1_positional_index.json`: 23,568 docs, 32,845 terms, 393,323 postings)
  - Baseline classifier inference (`outputs/best_baseline_model.pkl`)

### 2. Phase 2 Demo: `python phase2/scripts/run_demo.py`
- **Exit Code**: **0 (Clean Success)**
- **Offline Configuration**: Verified `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` ensure instant local cache loading with zero network timeout latency.
- **Trace Walkthrough Demonstrated**:
  1. Claim Input: `"Apple Inc. reported quarterly revenue of $90.8 billion in Q2 FY2024, beating analyst expectations."`
  2. Financial Claim Understanding: Extracted organizations `['Apple', 'Apple Inc']`, metrics `['revenue']`, polarity cues.
  3. Query Construction: Built expanded queries (`Apple Apple Inc revenue Q2 FY2024`, etc.).
  4. BM25 Candidate Retrieval: Retrieved top-3 candidate evidence passages from FinFact.
  5. Neural Cross-Encoder Reranking: Scored claim-passage pairs via `ms-marco-MiniLM-L-6-v2`.
  6. NLI Verification: Generated premise-hypothesis probability distribution.
  7. End-to-End Output: Produced aggregated verdict, confidence score, and citations.
  8. Empirical Benchmark Table: Reported verified source-grouped evaluation figures.

---

## 10. Restricted Evaluation Data Protection Audit

The confidentiality and segregation of sensitive human evaluation materials was audited:
1. **Location Verification**: `data/restricted_evaluation/private_evaluation_mapping.csv` is present strictly in its restricted folder (`data/restricted_evaluation/`).
2. **Access Isolation**: Neither demonstration script, public report, presentation file, nor documentation artifact references or exposes rows from `private_evaluation_mapping.csv`.
3. **Public Reviewer Files**: Inspection of `data/human_evaluation/reviewer_1.csv` and `reviewer_2.csv` confirmed that all records use anonymized identifiers (`anonymous_claim_id`). No original claim URLs, gold veracity labels, or model predictions are present.
4. **Filesystem Security Note**: File isolation is enforced via project path compartmentalization; standard Windows NTFS workspace permissions apply. No elevated operating-system access control lists (ACLs) have been modified.

---

## 11. Documentation Corrections Summary

The following documentation files were updated and synchronized with empirical evidence:

1. **`README.md`**:
   - Corrected System A True Recall from 54.00% to **59.22%**.
   - Updated terminology from "unseen publishers" to "source URL-grouped split".
   - Verified links to Phase 1, Phase 2, demonstration scripts, and faculty guides.
2. **`phase1/README.md`**:
   - Documented exact positional index statistics (23,568 units, 32,845 terms, 393,323 postings).
   - Documented exact phrase search capabilities and the non-implementation of arbitrary proximity search.
3. **`phase2/README.md`**:
   - Synchronized the source-grouped benchmark table with ground-truth per-class recall figures (False: 71.57%, True: 2.75%, NEI: 42.15%).
   - Clarified that zero-shot pretrained models were used without FinFact fine-tuning.
4. **`docs/PROFESSOR_DEMO_GUIDE.md`**:
   - Updated Question 8 with verified System A recall (59.22%) and honest scientific explanations of NLI neutral collapse and retrieval domain shift.
5. **`docs/FILE_ORGANIZATION_MAP.md`**:
   - Re-structured to match the required 6-column specification (`Original path | Current path | Classification | Canonical copy? | Purpose | Status`).
   - Categorized all 32 Python modules, datasets, models, demos, and reports.
6. **`docs/PHASE_COMPLETION_MATRIX.md`**:
   - Adopted strict operational classifications (`DEFAULT PIPELINE`, `IMPLEMENTED OPTIONAL COMPONENT`, `EXPERIMENTAL`, `NOT IMPLEMENTED`, `LEGACY`).
   - Added explicit entry noting that arbitrary proximity search ($k > 1$) is not implemented.
7. **`docs/decisions/README.md` & `presentation/README.md`**:
   - Updated split terminology to "held-out source URLs".

---

## 12. Faculty Defense & Presentation Quick Reference

When presenting the project to faculty or examiners, use the following exact reference points:

### What is Verified (Defensible with Code & Artifacts):
- **Phase 1 NLP Foundations**: Preprocessing, finance regex tokenization, 58 preserved polarity stopwords, POS-guided lemmatization, and positional inverted index over 23,568 segmented units are fully implemented, tested, and demonstrated.
- **Phase 2 Pipeline Integration**: The call chain from Claim Understanding $\to$ Query Expansion $\to$ BM25 Retrieval $\to$ Cross-Encoder Reranking $\to$ Zero-Shot NLI Verifier $\to$ Aggregated Verdict is operational and verified.
- **Empirical Accuracy**: The claim-only baseline (55.56% accuracy, 48.97% macro-F1) outperformed the evidence-grounded neural system (40.30% accuracy, 31.07% macro-F1) on held-out source URLs.
- **NLI Neutral Collapse**: Across 2,700 evaluated evidence pairs, 1,489 were predicted neutral and only 66 entailment, causing True-class recall to collapse to 2.75% (7 of 255 true claims recognized).
- **Test Integrity**: All 100 automated unit tests pass cleanly.

### What is Partially Verified / Diagnostic:
- **Retrieval Domain Shift**: Observed lexical divergence across unseen source URLs is a diagnostic observation supported by entity mismatch heuristics, not a mathematically closed causal proof.
- **Cross-Encoder Contribution**: Disabling reranking reduced accuracy from 40.30% to 39.41%; independent validation against gold human relevance judgments remains pending final review queue completion.

### What Remains Incomplete / Future Work:
- Variable-distance proximity search ($k > 1$).
- Fine-tuning NLI models directly on financial claim-evidence pairs.
- Arithmetic numerical verification and live external web retrieval.

---

### Commands for Professor Demonstration

```powershell
# 1. Run the Complete Phase 1 Demonstration
python phase1/scripts/run_demo.py

# 2. Run the Complete Phase 2 Demonstration
python phase2/scripts/run_demo.py

# 3. Run the Full Test Suite (100 Tests)
python -m unittest discover -s tests -p "test_*.py"

# 4. Run Phase 1 Tests Exclusively (58 Tests)
python -m unittest discover -s phase1/tests -p "test_*.py"

# 5. Run Phase 2 Tests Exclusively (42 Tests)
python -m unittest discover -s phase2/tests -p "test_*.py"

# 6. Launch the Interactive Web Dashboard
streamlit run app.py
```

