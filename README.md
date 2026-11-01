# FinGuard: Evidence-Grounded Financial Claim Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture Phase 1](https://img.shields.io/badge/Phase%201-NLP%20Foundations-green.svg)](phase1/README.md)
[![Architecture Phase 2](https://img.shields.io/badge/Phase%202-Neural%20Verification-blueviolet.svg)](phase2/README.md)
[![Tests Passing](https://img.shields.io/badge/Tests-100%2F100%20Passing-brightgreen.svg)](tests/)
[![Faculty Demo Guide](https://img.shields.io/badge/Docs-Professor%20Demo%20Guide-orange.svg)](docs/PROFESSOR_DEMO_GUIDE.md)

FinGuard is a modular, multi-stage natural language processing system for verifying factual claims in the financial domain using the **FinFact** fact-checking benchmark. FinGuard is architected and organized into two distinct, independently navigable and executable phases:
- **Phase 1**: Domain-Adapted Natural Language Processing Foundations, Inverted Positional Indexing, Boolean/Phrase Search, and Supervised Classical Baselines.
- **Phase 2**: Neural Financial Claim Understanding, Structured Query Construction, Multi-Modal Candidate Retrieval (BM25 & Dense Bi-Encoder), Neural Cross-Encoder Reranking, and Zero-Shot Natural Language Inference (NLI) Claim Verification.

---

## Quick Navigation & Faculty Defense Links

| Deliverable / Guide | Target Path | Description |
| :--- | :--- | :--- |
| **Faculty Presentation Guide** | [`docs/PROFESSOR_DEMO_GUIDE.md`](docs/PROFESSOR_DEMO_GUIDE.md) | **Step-by-step navigation sheet** answering every faculty question with exact file paths and commands |
| **Phase 1 Complete Guide** | [`phase1/README.md`](phase1/README.md) | Preprocessing, tokenization, polarity stopwords, POS-lemmatization, positional index, and baselines |
| **Phase 2 Complete Guide** | [`phase2/README.md`](phase2/README.md) | Financial NER, query expansion, BM25, dense retrieval, Cross-Encoder reranking, and NLI verification |
| **File Organization Map** | [`docs/FILE_ORGANIZATION_MAP.md`](docs/FILE_ORGANIZATION_MAP.md) | Exhaustive file-by-file mapping table from original paths to organized folders |
| **Phase Completion Matrix** | [`docs/PHASE_COMPLETION_MATRIX.md`](docs/PHASE_COMPLETION_MATRIX.md) | Component-by-component implementation status, test validation, and scientific results |
| **Shared Infrastructure** | [`shared/README.md`](shared/README.md) | Shared schemas, dataset loaders, benchmark evaluation harnesses, and EDA utilities |
| **Model Zoo Registry** | [`models/README.md`](models/README.md) | Checkpoint registry for trained baselines and pretrained Hugging Face neural models |

---

## Complete Repository Architecture

```
finfact-finguard/
├── phase1/                             # Phase 1: NLP Foundations & Classical Retrieval
│   ├── README.md                       # Comprehensive Phase 1 documentation
│   ├── src/                            # Modular implementations (preprocessing, indexing, retrieval)
│   ├── scripts/run_demo.py             # Self-contained Phase 1 demonstration runner
│   ├── results/                        # Baseline metrics, confusion matrices, figures, logs
│   ├── reports/                        # Phase 1 methodology audit report
│   └── tests/                          # 58 dedicated Phase 1 unit tests
├── phase2/                             # Phase 2: Neural Retrieval, Reranking & Verification
│   ├── README.md                       # Comprehensive Phase 2 documentation
│   ├── src/                            # Claim understanding, query expansion, reranking, NLI verifier
│   ├── scripts/run_demo.py             # Self-contained Phase 2 demonstration runner
│   ├── results/                        # Retrieval & verification metrics, ablation studies, error logs
│   ├── reports/                        # Source-grouped benchmark & root-cause reports
│   └── tests/                          # 42 dedicated Phase 2 unit tests
├── shared/                             # Shared Utilities, Schemas & Data Loaders
│   ├── README.md                       # Shared module documentation
│   ├── config/                         # System constants and environment parameters
│   ├── data_access/                    # Safe dataset loaders
│   ├── corpus/                         # Evidence pooling and gold reference alignment
│   ├── schemas/                        # Dataclass schemas for evaluation benchmarks
│   ├── evaluation/                     # Metric calculators (Precision, Recall, MRR, MAP, nDCG)
│   └── utils/                          # Dataset auditing, leakage detection, and EDA tools
├── data/                               # Dataset Storage & Splits
│   ├── raw/                            # Raw FinFact dataset (finfact.json)
│   ├── processed/                      # Cleaned and processed dataset
│   ├── splits/                         # Random (70/15/15) and Source-Grouped splits
│   ├── benchmark/                      # Frozen evaluation benchmarks (v1, v2)
│   ├── human_evaluation/               # Reviewer adjudication sheets
│   └── restricted_evaluation/          # Compartmentalized private evaluation mapping
├── models/                             # Model Registry & Checkpoints
│   ├── README.md                       # Model specifications and parameters
│   ├── trained/                        # Trained supervised baseline checkpoints (.pkl, .joblib)
│   └── pretrained/                     # Hugging Face neural model specifications & cache
├── outputs/                            # Historical and Organized Output Artifacts
├── docs/                               # Architectural & Methodological Documentation
│   ├── PROFESSOR_DEMO_GUIDE.md         # Faculty defense questions, paths, and commands
│   ├── FILE_ORGANIZATION_MAP.md        # Complete file migration and organization table
│   ├── PHASE_COMPLETION_MATRIX.md      # Feature implementation and validation matrix
│   ├── architecture/                   # High-level architecture blueprints
│   ├── methodology/                    # Annotation and preprocessing protocols
│   ├── evaluation/                     # Benchmark definitions and IR metrics
│   ├── setup/                          # Environment configuration guides
│   └── decisions/                      # Architecture Decision Records (ADRs)
├── presentation/                       # Presentation decks and defense talking points
├── archive/                            # Historical logs and superseded experiment runs
├── src/                                # Backward-Compatible Import Shims (preserves all legacy calls)
├── frontend/                           # Interactive Next.js web application
├── app.py                              # Interactive Streamlit dashboard
├── finfact.json                        # Master benchmark dataset (3,369 claims)
├── pyproject.toml                      # Modern Python packaging configuration
└── requirements.txt                    # Project Python dependencies
```

---

## Live Demonstrations

FinGuard includes dedicated, self-contained demonstration scripts for each phase. Both scripts execute deterministically in offline environments without external dependencies.

### 1. Phase 1 Demonstration
Demonstrates text normalization, finance-aware tokenization, polarity stopword preservation, POS-guided WordNet lemmatization, positional index term lookups, exact phrase search, boolean query parsing (`AND`, `OR`, `NOT`), and inference with the trained classical baseline classifier.
```powershell
python phase1/scripts/run_demo.py
```

### 2. Phase 2 Demonstration
Demonstrates financial claim understanding (NER), structured query expansion, candidate evidence retrieval via Inverted BM25, neural Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), zero-shot NLI claim verification (`nli-distilroberta-base`), end-to-end `FinGuardPipeline` execution, and empirical benchmark comparisons.
```powershell
python phase2/scripts/run_demo.py
```

---

## Empirical Benchmark Findings

To evaluate real-world generalization and prevent URL leakage, FinGuard was subjected to an independent audit on a strictly held-out **Source URL-Grouped Split** ($N=675$ claims partitioned by source URLs so no test source URL appears in train or validation).

```
========================================================================================
                      SOURCE-GROUPED EVALUATION BENCHMARK (N = 675)
========================================================================================
System Architecture                      Accuracy       Macro-F1       True-Class Recall
----------------------------------------------------------------------------------------
System A: TF-IDF + Logistic Regression    55.56%         48.97%             59.22%
System B: Evidence-Based FinGuard         40.30%         31.07%              2.75%
System C: FinGuard (Calibrated Threshold) 40.30%         31.07%              2.75%
========================================================================================
```

### Key Scientific Findings & Root Causes
1. **The Classical Baseline Outperformed Neural Evidence Grounding**:
   The claim-only classical baseline achieved 55.56% accuracy versus 40.30% for the full evidence-grounded neural system on held-out source URLs.
2. **Neutral Collapse in General-Domain NLI (Observed & Tested)**:
   Pretrained zero-shot general-domain NLI models (`cross-encoder/nli-distilroberta-base`, not fine-tuned on FinFact) achieved only **2.75% True-class recall** (recognizing only 7 of 255 true claims). General NLI models interpret financial hedging, forward guidance, and numerical qualifications as unprovable (NEI / Neutral) rather than entailment.
3. **Retrieval Domain Shift (Diagnostic Observation)**:
   Across unseen source URLs, lexical BM25 retrieval frequently retrieved background contextual text rather than direct verifications, depriving the NLI engine of conclusive premises.
4. **Complete Investigation**:
   Read the full scientific post-mortem in [`reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md`](reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md) and [`reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md`](reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md).

---

## Unit Test Suite & Validation

The entire repository is covered by an automated test suite comprising **100 tests**, all of which pass cleanly:

```powershell
# Run the complete test suite (100 tests)
python -m unittest discover -s tests -p "test_*.py"

# Run Phase 1 tests exclusively (58 tests)
python -m unittest discover -s phase1/tests -p "test_*.py"

# Run Phase 2 tests exclusively (42 tests)
python -m unittest discover -s phase2/tests -p "test_*.py"
```

---

## User Interfaces

FinGuard provides two interactive user interfaces for exploring financial claim verification:

### Streamlit Dashboard
```powershell
streamlit run app.py
```

### Next.js Modern Web Interface
```powershell
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` to inspect candidate retrieval, gold benchmark auditing, and live claim verification.
