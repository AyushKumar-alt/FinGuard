# FinGuard — Shared Utilities, Schemas & Benchmark Infrastructure

## Overview

The `shared/` directory contains modular utilities, dataset schemas, benchmark evaluation harnesses, and data loaders utilized jointly across **Phase 1** and **Phase 2**.

---

## Directory Organization

```
shared/
├── README.md                   # This documentation
├── config/                     # Shared system configuration and constants
├── data_access/                # Structured access to raw FinFact and split datasets
├── corpus/                     # Corpus management & candidate pooling
│   ├── pooling.py              # Multi-retriever candidate pooling
│   └── reference_evidence.py   # Gold & silver reference evidence extraction
├── schemas/                    # Pydantic / dataclass schemas
│   └── gold_benchmark_schema.py# Gold evidence benchmark specification
├── evaluation/                 # Benchmark evaluators and adjudication
│   ├── benchmark_evaluator.py  # Precision, Recall, MRR, MAP, nDCG computation
│   ├── freeze_benchmark.py     # Deterministic benchmark dataset freezing
│   ├── judgment_automation.py  # Automated relevance annotation
│   ├── adjudication.py         # Human & automated judgment reconciliation
│   ├── evaluate_all_retrieval.py # Multi-engine retrieval evaluator
│   └── evaluate_verifiers.py   # Verifier accuracy & calibration evaluator
└── utils/                      # Exploratory data analysis & auditing
    ├── dataset_audit.py        # Leakage, duplicates & class balance auditor
    └── eda.py                  # Distribution & length visualizer
```

---

## Security & Privacy Notice
All private human evaluation mapping files (`data/restricted_evaluation/private_evaluation_mapping.csv`) are strictly compartmentalized and must never be exposed or committed to public documentation or external repositories.

