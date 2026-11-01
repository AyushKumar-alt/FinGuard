# FinGuard — Model Zoo & Checkpoints Registry

## Overview

This directory indexes all trained and pretrained machine learning models utilized in FinGuard across Phase 1 and Phase 2.

---

## Directory Organization

```
models/
├── README.md               # Model specifications and usage notes
├── trained/                # Trained supervised model checkpoints
│   ├── best_baseline_model.pkl             # Best TF-IDF + Logistic Regression (Random Split)
│   ├── experiment_a_raw_model.joblib       # Source-grouped baseline (Raw Claims)
│   └── experiment_b_cleaned_model.joblib   # Source-grouped baseline (Cue-Stripped Claims)
└── pretrained/             # HuggingFace pretrained model specifications & cache paths
```

---

## Model Specifications

### 1. Trained Supervised Baselines (`models/trained/`)

| Model File | Architecture | Features | Training Split | Performance |
| :--- | :--- | :--- | :--- | :--- |
| `best_baseline_model.pkl` | Logistic Regression (Balanced) | Sublinear TF-IDF (word+char 1-3) | Random 70/15/15 | Acc: **67.00%**, F1: **59.98%** |
| `experiment_a_raw_model.joblib` | Logistic Regression (Balanced) | Word+Char n-grams (1-2) | Source-Grouped Train | Acc: **55.56%**, F1: **48.97%** |
| `experiment_b_cleaned_model.joblib` | Logistic Regression (Balanced) | Word+Char n-grams (1-2, stripped) | Source-Grouped Train | Acc: **54.96%**, F1: **48.43%** |

### 2. Pretrained Neural Models (`models/pretrained/`)

| Model Identifier | Role | Parameters | Format / Framework | Local Disk Size |
| :--- | :--- | :---: | :--- | :---: |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural Reranker | 22.7M | PyTorch / Transformers | ~90 MB |
| `cross-encoder/nli-distilroberta-base` | NLI Claim Verifier | 82.0M | PyTorch / Transformers | ~330 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | Dense Vector Retriever | 22.7M | PyTorch / SentenceTransformers | ~90 MB |

---

## Loading Examples

### Loading Classical Baseline
```python
import joblib
model = joblib.load("models/trained/best_baseline_model.pkl")
preds = model.predict(["Apple reported record revenue."])
```

### Loading Neural Reranker
```python
from src.reranker import PretrainedCrossEncoderReranker
reranker = PretrainedCrossEncoderReranker()
reranked = reranker.rerank(claim, candidate_docs, top_k=3)
```

