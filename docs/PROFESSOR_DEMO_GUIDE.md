# FinGuard: Faculty Demonstration & Defense Guide

This guide is designed for presentation to faculty, evaluators, and academic committees. It provides exact answers, file paths, commands, and expected outputs for each question a professor or reviewer might ask.

---

## Part 1: Phase 1 Inquiries

### Question 1: "Show me what you implemented in Phase 1."
**Code Locations**:
- Core Preprocessing: `phase1/src/preprocessing/text_cleaner.py` (or `src/preprocess.py`)
- Financial Tokenization: `phase1/src/tokenization/tokenizer.py` (or `src/p1_tokenizer.py`)
- Polarity Stopwords: `phase1/src/tokenization/stopwords.py` (or `src/p1_stopwords.py`)
- POS-Guided Lemmatization: `phase1/src/lemmatization/lemmatizer.py` (or `src/p1_lemmatizer.py`)
- Positional Inverted Index: `phase1/src/indexing/positional_index.py` (or `src/positional_index.py`)
- Query Engines: `phase1/src/retrieval/` (boolean, phrase, tfidf)
- Classical ML Baselines: `phase1/src/classical_ml/` (or `src/baseline_models.py`)

**What to explain**:
> "In Phase 1, we implemented the foundational NLP pipeline and classical retrieval system adapted specifically for financial text. We solved key financial domain challenges: standard tokenizers split currency amounts and fiscal years, standard stopword lists delete critical directional terms like 'profit', 'loss', 'increase', 'above', and standard lemmatizers distort financial acronyms. We also built an inverted positional index tracking exact term-offsets across 23,568 segmented units."

---

### Question 2: "Show me the preprocessing and tokenization results."
**Execution Command**:
```powershell
python -c "from src.p1_tokenizer import finance_aware_tokenize; print(finance_aware_tokenize('HDFC Bank net profit surged 33.5% to ₹16,811 crore in FY2024; stock at $145.50.'))"
```
**Expected Output**:
```python
['HDFC', 'Bank', 'net', 'profit', 'surged', '33.5%', 'to', '₹16,811 crore', 'in', 'FY2024', 'stock', 'at', '$145.50']
```
**Key Point**: Currency symbols (`₹`, `$`), comma-separated amounts (`₹16,811 crore`), percentages (`33.5%`), and fiscal years (`FY2024`) are preserved as single semantic tokens.

---

### Question 3: "Show me the positional index and search implementation."
**Execution Command**:
```powershell
python -c "from src.positional_index import PositionalInvertedIndex, BooleanQueryParser; idx = PositionalInvertedIndex([{'doc_id': 1, 'text': 'Apple record profit'}, {'doc_id': 2, 'text': 'Tesla record loss'}]); print('Profit docs:', idx.search_term('profit')); print('Boolean Apple AND profit:', BooleanQueryParser(idx).evaluate('Apple AND profit'))"
```
**Corpus Scale**:
- Document Count: **23,568 units** (Claims: 3,369, Evidence Sentences: 18,270, Justifications: 3,367)
- Vocabulary Size: **32,845 unique lemmas**
- Positional Postings: **393,323**
- Checkpoint: `outputs/p1_positional_index.json`

---

### Question 4: "Show me the Phase 1 evaluation results."
**Execution Command**:
```powershell
python -c "import json; res=json.load(open('phase1/results/metrics/baseline_metrics.json')); print('Logistic Regression:', res.get('Logistic Regression', res))"
```
**Results Table**:
- **Logistic Regression (Best)**: Accuracy **67.00%**, Macro-F1 **59.98%** (Random Split)
- **Source-Grouped Baseline**: Accuracy **55.56%**, Macro-F1 **48.97%** (Held-Out Sources)
- Visual Artifact: `phase1/results/figures/best_baseline_confusion_matrix.png`

---

### Run the Full Phase 1 Demonstration
```powershell
python phase1/scripts/run_demo.py
```

---

## Part 2: Phase 2 Inquiries

### Question 5: "Now show me what changed in Phase 2."
**Code Locations**:
- Claim Understanding (NER): `phase2/src/claim_understanding/claim_understanding.py`
- Query Construction: `phase2/src/query_construction/query_builder.py`
- Lexical BM25: `phase2/src/retrieval/bm25/bm25_retriever.py`
- Dense Vector Retrieval: `phase2/src/retrieval/dense/dense_retriever.py`
- Cross-Encoder Reranking: `phase2/src/retrieval/reranking/cross_encoder_reranker.py`
- NLI Verification: `phase2/src/verification/nli/nli_verifier.py`
- End-to-End Orchestrator: `phase2/src/pipeline/finguard_pipeline.py`

**What to explain**:
> "Phase 2 transforms FinGuard from a document lookup tool into a multi-stage evidence-grounded verification system. It parses financial claims into structured entity profiles, expands queries to retrieve candidate evidence passages via BM25 and Dense embeddings, reranks candidates using a Cross-Encoder (ms-marco-MiniLM-L-6-v2), and verifies claims using Natural Language Inference (cross-encoder/nli-distilroberta-base)."

---

### Question 6: "Show me the claim understanding and query expansion in action."
**Execution Command**:
```powershell
python -c "from src.claim_understanding import extract_financial_entities; from src.query_construction import FinancialQueryBuilder; claim = 'Apple Q2 FY2024 revenue surged 14% to $90.8B'; parsed = extract_financial_entities(claim); print('Entities:', parsed); print('Query:', FinancialQueryBuilder().build_queries(claim, parsed)['expanded_queries'])"
```

---

### Question 7: "Show me the final verification pipeline executing on a live claim."
**Execution Command**:
```powershell
python -c "from src.pipeline import FinGuardPipeline; p = FinGuardPipeline(load_heavy_models=False); res = p.run('Apple reported record quarterly revenue of $90.8 billion.'); print('Verdict:', res['verdict_display'], '| Confidence:', res['confidence'])"
```

---

### Question 8: "Show me the experimental results and explain why the evidence-based system underperformed the baseline."
**What to explain**:
> "We conducted a rigorous, leakage-resistant benchmark on a source-grouped held-out test split ($N=675$ claims partitioned strictly by source URLs so no test source URL appears in training or validation). The results revealed a critical empirical finding:
> - **System A (Claim-Only Baseline)**: Accuracy **55.56%**, Macro-F1 **48.97%**, True Recall **59.22%** (151/255)
> - **System B (Evidence-Based FinGuard)**: Accuracy **40.30%**, Macro-F1 **31.07%**, True Recall **2.75%** (7/255)
>
> **Root Cause Analysis (Observations & Hypotheses)**:
> 1. **Zero-Shot NLI Neutral Collapse (Observed & Tested)**: Pretrained general NLI models (`cross-encoder/nli-distilroberta-base`, zero-shot, not fine-tuned on FinFact) exhibit conservative classification on financial language. They interpret forward-looking statements, fiscal qualifications, and hedging as NEI (neutral) rather than entailment, causing True-class recall to collapse to 2.75% (only 7 of 255 True claims were recognized, while 128 were labeled NEI and 120 False).
> 2. **Retrieval Domain Shift (Diagnostic Observation)**: Across unseen source URLs, lexical BM25 retrieval frequently retrieved background financial context rather than direct verifications, depriving the NLI engine of decisive premises."

**Where to show the evidence**:
- Comprehensive Audit Report: `reports/FINGuard_ALIGNED_GROUPED_VERIFICATION.md`
- Root Cause Report: `reports/FINGuard_ROOT_CAUSE_INVESTIGATION.md`
- Data Artifact: `outputs/experiments/aligned_grouped_verification/results.json`

---

### Run the Full Phase 2 Demonstration
```powershell
python phase2/scripts/run_demo.py
```

