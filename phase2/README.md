# FinGuard — Phase 2: Neural Retrieval, Reranking & Claim Verification

## Executive Overview

**Phase 2** scales FinGuard from classical lexical matching into a multi-stage neural fact-checking architecture. It introduces domain-specific financial claim understanding, structured query expansion, candidate retrieval (BM25Okapi, Dense Bi-Encoder representations, and Reciprocal Rank Fusion), neural Cross-Encoder passage reranking, and Natural Language Inference (NLI) claim verification.

Importantly, Phase 2 features rigorous scientific auditing on **source-grouped splits**, revealing critical empirical findings regarding retrieval domain shift and zero-shot NLI neutral collapse in financial domains.

---

## Directory Structure

```
phase2/
├── README.md                           # This comprehensive guide
├── src/                                # Core Phase 2 implementations
│   ├── claim_understanding/            # Financial entity & semantic extraction
│   │   └── claim_understanding.py      # NER: ORG, MONEY, PERCENT, DATES, METRICS, direction
│   ├── query_construction/             # Financial query expansion
│   │   └── query_builder.py            # Generates entity-expanded & metric-focused sub-queries
│   ├── retrieval/                      # Multi-modal retrieval engines
│   │   ├── bm25/                       # BM25Okapi lexical retrieval
│   │   ├── dense/                      # Dense Bi-Encoder (all-MiniLM-L6-v2 / 128-dim SVD)
│   │   ├── hybrid/                     # Reciprocal Rank Fusion (RRF, k=60)
│   │   └── reranking/                  # Neural Cross-Encoder (ms-marco-MiniLM-L-6-v2)
│   ├── verification/                   # Claim verification engines
│   │   ├── nli/                        # NLI Cross-Encoder (nli-distilroberta-base)
│   │   └── aggregation/                # Evidence aggregation, confidence scoring & thresholding
│   ├── pipeline/                       # End-to-end architecture orchestrator
│   │   └── finguard_pipeline.py        # FinGuardPipeline class connecting all components
│   └── interfaces/                     # Application entry points
│       ├── api_bridge.py               # REST API bridge for Next.js frontend
│       └── app.py                      # Interactive Streamlit dashboard
├── scripts/
│   └── run_demo.py                     # Self-contained Phase 2 demonstration script
├── experiments/                        # Phase 2 experimental runners
├── results/                            # Metrics, figures, and logs
│   ├── metrics/                        # Quantitative evaluation records
│   │   ├── retrieval_metrics.json
│   │   ├── verification_metrics.json
│   │   ├── gold_benchmark_results.json
│   │   ├── aligned_grouped_verification_results.json
│   │   └── error_analysis.csv
│   ├── figures/                        # Performance curves & confusion matrices
│   │   ├── retrieval_comparison.png
│   │   └── confusion_matrices/         # System A, B, C confusion matrices
│   └── logs/                           # Complete execution logs
│       ├── aligned_grouped_experiment.log
│       └── experiment.log
├── reports/                            # In-depth scientific audit reports
│   ├── PHASE2_IMPLEMENTATION_REPORT.md
│   ├── FINGuard_ALIGNED_GROUPED_VERIFICATION.md
│   ├── FINGuard_ROOT_CAUSE_INVESTIGATION.md
│   └── FINGuard_RESEARCH_VALIDATION_AUDIT.md
└── tests/                              # Dedicated Phase 2 unit test suite (42 tests)
    ├── test_claim_understanding.py
    ├── test_query_expansion.py
    ├── test_retrieval.py
    ├── test_pipeline_retrieval_modes.py
    ├── test_leakage_and_reranker.py
    ├── test_verifier.py
    ├── test_pipeline_and_nli.py
    ├── test_gold_benchmark.py
    ├── test_judgment_automation.py
    └── test_audit_and_splits.py
```

---

## Architectural Stages

```
Financial Claim Input
       │
       ▼
[Stage 1: Claim Understanding]
  • Extract entities (ORG, MONEY, PERCENT, DATES, METRICS)
  • Identify sentiment / polarity cues (increase, drop, loss)
       │
       ▼
[Stage 2: Query Construction]
  • Expand claim into structured sub-queries with entity-metric bindings
       │
       ▼
[Stage 3: Evidence Retrieval]
  • BM25Okapi Lexical Index (default, candidate pool: 12,943 sentences)
  • Dense Semantic Bi-Encoder (all-MiniLM-L6-v2)
  • Reciprocal Rank Fusion (RRF, k=60)
       │
       ▼
[Stage 4: Neural Reranking]
  • Cross-Encoder (ms-marco-MiniLM-L-6-v2) scores (claim, passage) joint interactions
  • Filters top-4 candidate evidence passages
       │
       ▼
[Stage 5: Claim Verification]
  • Zero-Shot NLI Cross-Encoder (nli-distilroberta-base)
  • Fallback: Calibrated balanced Logistic Regression
       │
       ▼
[Stage 6: Decision Aggregation & Output]
  • Verdict: TRUE | FALSE | NEI
  • Calibrated confidence score & provenance citations
```

---

## Empirical Benchmark & Scientific Findings

To prevent source URL and evidence leakage, FinGuard was evaluated on a strictly held-out **Source URL-Grouped Split** ($N=675$ test claims across mutually exclusive source URLs).

### Source-Grouped Benchmark Comparison ($N=675$)

| Architecture | Pipeline Mode | Accuracy | Macro-F1 | False Recall | True Recall | NEI Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **System A** | Claim-Only Baseline (TF-IDF + LogReg) | **55.56%** | **48.97%** | 66.22% | **59.22%** | 21.49% |
| **System B** | FinGuard Default (BM25 + Cross-Encoder + NLI) | **40.30%** | **31.07%** | **71.57%** | **02.75%** | **42.15%** |
| **System C** | FinGuard Calibrated (Adjusted Thresholds) | 40.30% | 31.07% | 71.57% | 02.75% | 42.15% |

### Critical Scientific Findings & Root Causes
1. **The Baseline Anomaly**:
   The claim-only classical baseline outperforms the complex evidence-grounded neural system (55.56% vs 40.30% accuracy) on held-out source URLs.
2. **Neutral Collapse in General NLI (Observed & Tested)**:
   The pretrained NLI model (`cross-encoder/nli-distilroberta-base`, zero-shot, not fine-tuned on FinFact) achieved only **2.75% True-class recall** (7 of 255 true claims). General-domain NLI models interpret financial hedging, forward-looking statements, and numerical qualifications as *neutral* (NEI) rather than *entailment*.
3. **Retrieval Domain Shift (Diagnostic Observation)**:
   When evaluating on unseen sources, BM25 retrieval frequently suffers lexical mismatch, retrieving non-informative background passages rather than direct verifications.
   When evaluating on unseen sources, BM25 retrieval suffers lexical vocabulary mismatch, retrieving non-informative or tangential background passages rather than direct verifications.

---

## Quickstart & Execution

### 1. Run the Complete Phase 2 Demonstration
```powershell
python phase2/scripts/run_demo.py
```

### 2. Run Phase 2 Unit Tests (42 Tests)
```powershell
python -m unittest discover -s phase2/tests -p "test_*.py"
```

### 3. Launch the Interactive Interfaces
- **Streamlit App**:
  ```powershell
  streamlit run app.py
  ```
- **Next.js Web UI**:
  ```powershell
  cd frontend
  npm run dev
  ```

