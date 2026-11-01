# FinGuard: Gold-Evidence Relevance Benchmark & System Implementation Audit

**Audit Date:** 2026-10-27  
**Auditor:** Antigravity AI Research Agent  
**Dataset:** FinFact (`finfact.json`, $N=3,369$ records)  
**Corpus Units:** $12,941$ canonical candidate evidence sentences  
**Automated Tests:** 33 / 33 Unit Tests Passing (`python -m unittest discover tests`)

---

## 1. Executive Implementation Audit Report

This audit independently investigates the repository code, data structures, evaluation routines, and leakage boundaries to answer the 10 core questions specified in Task 1.

---

### Question 1: What is the actual dataset schema and local record count?
- **Dataset File:** `finfact.json` (local size: 29,972,446 bytes).
- **Exact Record Count:** Exactly **3,369** claim records.
- **Top-Level Record Keys:**
  ```python
  ['url', 'claim', 'author', 'posted', 'sci_digest', 'justification', 'issues', 'image_data', 'evidence', 'label']
  ```
- **Field Data Types:**
  - `url`: String (e.g., source fact-checking article URL on Snopes or PolitiFact).
  - `claim`: String (natural language financial claim).
  - `author`: String (journalist / fact-checker author name).
  - `posted`: String (publication date, format `MM/DD/YYYY`).
  - `sci_digest`: List of strings (summary digest extracted by original authors).
  - `justification`: Long string (investigative narrative, typically 2,000–10,000 characters).
  - `issues`: List of strings (financial topic categories e.g. `['debt']`, `['tax']`).
  - `image_data`: List of dicts (`image_src`, `image_caption`).
  - `evidence`: List of dicts (`{'sentence': str, 'hrefs': list[str]}`).
  - `label`: String (`'false'`, `'true'`, `'NEI'`).

---

### Question 2: What are the original claim-label values and their frequencies?
The ground-truth label distribution across all 3,369 records is:
- **`false`**: **1,492** records (**44.29%**)
- **`true`**: **1,275** records (**37.85%**)
- **`NEI` (Not Enough Information)**: **602** records (**17.87%**)
- **Total:** 3,369 records (100.0%)

*Provenance of "Mixture":* In raw fact-checking sources, nuanced verdicts such as "Half True" or "Mixture" were mapped by the FinFact benchmark creators into the standard FEVER-compatible 3-class schema as `NEI`. Our implementation preserves these three labels without remapping.

---

### Question 3: Where are evidence texts, evidence URLs, justifications, and claim IDs stored?
- **Evidence Texts:** Stored in the `evidence` field as a list of dicts: `item["evidence"][i]["sentence"]`. In preprocessed tabular splits (`data/train.csv`, `data/val.csv`, `data/test.csv`), they are joined into `evidence_text`.
- **Evidence URLs:** Stored in `item["evidence"][i]["hrefs"]` as a list of hyperlink URLs cited in each evidence sentence.
- **Justifications:** Stored in `item["justification"]` as a single long-form narrative.
- **Claim IDs:** In raw `finfact.json`, there were no explicit primary keys. In `src/preprocess.py`, each record was assigned an integer `id` (0 to 3368) matching its array index. In the newly implemented gold benchmark schema ([src/gold_benchmark_schema.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/gold_benchmark_schema.py)), stable canonical identifiers `claim_XXXX` (`claim_0000` to `claim_3368`) are formalized.

---

### Question 4: Does each claim have one or multiple evidence items?
- **Multiple evidence items.**
- **Distribution:**
  - Minimum: 0 evidence sentences (101 claims, 3.0%).
  - Maximum: 40 evidence sentences.
  - Mean: **5.45** evidence sentences per claim.
  - Total evidence sentence instances across all 3,369 claims: **18,357**.

---

### Question 5: How are candidate evidence units constructed?
In [src/retrieval.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/retrieval.py) and [src/reference_evidence.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/reference_evidence.py):
1. Sentences shorter than 4 words are filtered out as non-informative.
2. Identical evidence sentences across different claims are deduplicated into a single canonical entry with aggregated `parent_claim_ids` and combined `source_urls`.
3. This yields **12,941 unique canonical candidate evidence sentences** (or 12,943 in raw unnormalized text).
4. Each candidate evidence unit is assigned a stable ID `ev_XXXXX` (e.g. `ev_00000` to `ev_12940`) and stored with its text, normalized text, SHA-256 hash, and provenance links in [data/benchmark/v1/corpus_manifest.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/benchmark/v1/corpus_manifest.json).

---

### Question 6: How are relevant documents or sentences currently identified during evaluation?
In the prior evaluation scripts:
- A document `doc_id` was considered "relevant" for query `q_id` if and only if `doc_id in claim_to_doc_ids[q_id]`.
- This meant that **only the sentences scraped under that specific claim's record** were marked relevant.
- All other 12,941 sentences in the corpus—even if they were from the same source, discussed the exact same event, or provided valid evidence—were treated as non-relevant (grade 0).
- In the newly implemented benchmark, we distinguish:
  - **Silver Reference Evidence:** Dataset-provided sentences explicitly associated with the claim (Grade 3).
  - **Unjudged Candidates:** Retrieved candidates that lack manual annotations (explicitly represented as `None` / `unjudged`, rather than assumed to be 0).
  - **Adjudicated Judgments:** Formally reviewed evidence items with assigned grades (0, 1, 2, 3) and support relations (`supports`, `refutes`, `mixed`, `context_only`).

---

### Question 7: Are retrieval metrics computed against explicit evidence references, string matches, URLs, claim labels, or another heuristic?
- Prior retrieval was computed strictly against candidate document IDs (`doc_id in gold_doc_ids`), with one string heuristic:
  ```python
  retrieved_ids = [d_id for d_id in retrieved_ids if self.corpus[d_id]["text"].lower() != claim_text.lower()]
  ```
  to prevent exact claim-text identity matching.
- It did **not** use graded relevance, did **not** compute nDCG, and did **not** distinguish unjudged candidates from irrelevant candidates.
- The new benchmark harness ([src/benchmark_evaluator.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/benchmark_evaluator.py)) computes:
  - Binary metrics: Precision@K, Recall@K, MRR, MAP (against configurable threshold $\tau = 2$).
  - Graded metrics: nDCG@5, nDCG@10 using graded gains ($2^g - 1$).
  - Incomplete judgment metrics: Judged Coverage@K and Reference Evidence Retrieval Rate@K.

---

### Question 8: Are any test labels, justifications, gold evidence, or verdict-related text accidentally exposed to retrieval?
- **Inference-Time Isolation Confirmed:**
  - The search query submitted to retrievers is strictly `claim_text` (or expanded query derived purely from `claim_text`).
  - The searchable candidate corpus contains only evidence sentences and source URLs; it does **not** contain justifications, digests, or claim labels.
  - Test labels and gold justifications are never passed to the retrieval functions.
- **Corpus Scope:** The candidate corpus is a collection of 12,941 evidence sentences pooled across the dataset. The query retriever searches this collection without knowing which sentence belongs to which claim.

---

### Question 9: Are existing retrieval metrics mathematically correct and reproducible?
- **Finding:**
  1. Precision@K, Recall@K, and MRR were mathematically standard and reproducible.
  2. MAP in the prior script used a bounded denominator:
     `ap = ap / min(len(gold_doc_ids), top_k)`
     In classic information retrieval (TREC `trec_eval`), the denominator of Average Precision is strictly $|R_c|$ (total relevant items for that query in the collection), not $\min(|R_c|, K)$. Dividing by $\min(|R_c|, K)$ artificially inflates MAP when $K < |R_c|$.
  3. nDCG was absent.
  4. Unjudged retrieved candidates were silently treated as negative (0), penalizing retrievers that discovered novel, unannotated relevant evidence.
- **Correction:** [src/benchmark_evaluator.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/benchmark_evaluator.py) implements the standard TREC MAP formula, exponential graded nDCG, coverage tracking, and paired bootstrap 95% confidence intervals.

---

### Question 10: Which components can be reused without modification?
- **`src/dense_retrieval.py`:** Pretrained dense retriever with cached normalized embeddings `outputs/dense_evidence_embeddings.npz` (reused directly).
- **`src/reranker.py`:** Cross-encoder joint reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (reused directly).
- **`src/claim_understanding.py`:** Financial entity and metric parser (reused directly).
- **`src/query_construction.py`:** Direction-preserving query builder (reused directly).
- **`src/positional_index.py`:** Positional inverted index and Boolean query parser (reused directly).
- **`src/retrieval.py`:** Okapi BM25 and TF-IDF vectorizer (reused directly).
- **`src/verifier.py`:** Calibrated logistic regression and NLI verifier (reused directly).

---

## 2. Deliverables Summary

| Deliverable | File Path | Status |
| :--- | :--- | :---: |
| **Benchmark Documentation** | [docs/gold_evidence_benchmark.md](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/gold_evidence_benchmark.md) | `COMPLETE` |
| **Annotation Guidelines** | [docs/annotation_guidelines.md](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/annotation_guidelines.md) | `COMPLETE` |
| **Metric Definitions Document** | [docs/metric_definitions.md](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/docs/metric_definitions.md) | `COMPLETE` |
| **Benchmark Configuration** | [configs/gold_evidence_benchmark.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/configs/gold_evidence_benchmark.json) | `COMPLETE` |
| **Versioned Claims Manifest** | [data/benchmark/v1/claims_manifest.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/benchmark/v1/claims_manifest.json) | `COMPLETE` |
| **Versioned Corpus Manifest** | [data/benchmark/v1/corpus_manifest.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/benchmark/v1/corpus_manifest.json) | `COMPLETE` |
| **Reference Evidence Mapping** | [data/benchmark/v1/reference_evidence.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/benchmark/v1/reference_evidence.json) | `COMPLETE` |
| **Pooled Candidates Annotation CSV** | `data/benchmark/v1/pooled_candidates_annotation.csv` (4,891 pairs) | `COMPLETE` |
| **Pooled Candidates Annotation JSONL**| `data/benchmark/v1/pooled_candidates_annotation.jsonl` (4,891 pairs) | `COMPLETE` |
| **Relevance Judgments Store** | `data/benchmark/v1/relevance_judgments.json` & `.csv` | `COMPLETE` |
| **Frozen Benchmark Manifest** | `data/benchmark/v1/benchmark_manifest.json` | `COMPLETE` |
| **Candidate Pooler** | [src/pooling.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/pooling.py) | `COMPLETE` |
| **Adjudication Engine** | [src/adjudication.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/adjudication.py) | `COMPLETE` |
| **Shared Benchmark Evaluator** | [src/benchmark_evaluator.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/benchmark_evaluator.py) | `COMPLETE` |
| **Benchmark Results (JSON & Markdown)**| [outputs/gold_benchmark_results.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/gold_benchmark_results.json), [outputs/gold_benchmark_results.md](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/outputs/gold_benchmark_results.md) | `COMPLETE` |
| **Automated Test Suite** | [tests/test_gold_benchmark.py](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/tests/test_gold_benchmark.py) (33 tests total) | `COMPLETE` |

---

## 3. Pilot Benchmark Evaluation Results (75 Held-Out Test Claims)

Evaluated under identical frozen candidate pools and non-leakage controls:

| Retrieval Architecture | MRR (95% CI) | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 | Precision@5 | Precision@10 | Judged Coverage@10 | RefRate@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cross-Encoder Rerank** | **0.6686** ([0.564, 0.764]) | 0.3372 | 0.3920 | **0.4641** | 0.4366 | 0.3520 | 0.2133 | 100.0% | 0.3920 | 683.8 ms |
| **Dense Transformer (`all-MiniLM-L6-v2`)** | 0.6469 ([0.532, 0.741]) | 0.3130 | 0.4100 | 0.4492 | 0.4375 | 0.3600 | **0.2400** | 100.0% | 0.4100 | **18.7 ms** |
| **Hybrid RRF (BM25 + Dense)** | 0.6445 ([0.540, 0.737]) | **0.3476** | **0.4212** | 0.4597 | **0.4405** | **0.3707** | 0.2347 | 97.6% | **0.4212** | 65.8 ms |
| **BM25 (Inverted Index)** | 0.5661 ([0.455, 0.660]) | 0.3041 | 0.3636 | 0.3880 | 0.3719 | 0.3067 | 0.1933 | 100.0% | 0.3636 | 53.0 ms |
| **Query Expansion BM25** | 0.5462 ([0.428, 0.637]) | 0.2775 | 0.3411 | 0.3627 | 0.3534 | 0.2880 | 0.1880 | 100.0% | 0.3411 | 57.8 ms |
| **Hybrid RRF (BM25 + TF-IDF)** | 0.5136 ([0.402, 0.603]) | 0.2546 | 0.3315 | 0.3453 | 0.3406 | 0.2747 | 0.1853 | 100.0% | 0.3315 | 79.2 ms |
| **TF-IDF Cosine** | 0.4653 ([0.364, 0.559]) | 0.2289 | 0.2841 | 0.3032 | 0.2925 | 0.2427 | 0.1547 | 100.0% | 0.2841 | 36.7 ms |
| **SVD Semantic (128-dim)** | 0.0928 ([0.046, 0.147]) | 0.0424 | 0.0547 | 0.0498 | 0.0504 | 0.0427 | 0.0280 | 100.0% | 0.0547 | 24.6 ms |
