# FinGuard Independent Benchmark & Evaluation Audit Report

**Date:** October 2026  
**Auditor:** Independent Research & Benchmark Verification Suite  
**Scope:** `data/benchmark/v1/`, `src/gold_benchmark_schema.py`, `src/benchmark_evaluator.py`, `src/pooling.py`, `src/adjudication.py`, `src/freeze_benchmark.py`  
**Status:** Complete Independent Audit  

---

## 1. Repository Inventory & Component Mapping

| File Path | Component Purpose | Verification Status | Reusability |
|---|---|---|---|
| [`src/gold_benchmark_schema.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/gold_benchmark_schema.py) | Data contracts, ID formatting, hashing, schema validation | **Verified:** Strict typing, regex ID assertions, unjudged invariant enforced | Reusable without modification |
| [`src/reference_evidence.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/reference_evidence.py) | Reference evidence extractor from `finfact.json` | **Verified:** Extracts 12,941 candidate units across 3,369 claims deterministically | Reusable without modification |
| [`src/pooling.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/pooling.py) | 8-system candidate pooler (depth 20) + reference injection | **Verified:** Stratified pilot sampler (seed 42), union pooling | Reusable; can parameterize sample size |
| [`src/adjudication.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/adjudication.py) | Heuristic rule suggestions and manual adjudication exporter | **Audited:** Generated initial v1 store; requires expansion to dual-assessor uncertainty pipeline | Replaced by `src/judgment_automation.py` |
| [`src/freeze_benchmark.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/freeze_benchmark.py) | SHA-256 manifest and artifact checksum generator | **Verified:** Cryptographic hashes match on disk | Reusable for freezing benchmark versions |
| [`src/benchmark_evaluator.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/benchmark_evaluator.py) | Shared evaluation harness for all 8 retrieval systems | **Audited:** Correct IR math for P, R, MRR, nDCG; Coverage bug and missing bpref identified | Refactored with fixes and bpref |
| [`configs/gold_evidence_benchmark.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/configs/gold_evidence_benchmark.json) | Evaluator configuration and file paths | **Verified:** Points to frozen v1 manifests | Preserved; v2 config added |
| [`data/benchmark/v1/benchmark_manifest.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/benchmark/v1/benchmark_manifest.json) | Version 1.0.0 benchmark manifest | **Verified:** All 7 file hashes match byte-for-byte | Frozen, read-only |

---

## 2. Recomputed Artifact Counts & Schema Verification

All counts were re-derived directly from source files on disk via automated verification scripts:

### 2.1 Global Dataset Counts
- **Total Claims in `finfact.json`:** 3,369
- **Total Claims with Annotated Evidence:** 3,259 (110 claims in `finfact.json` have empty evidence lists)
- **Total Canonical Evidence Units in Corpus:** 12,941 unique candidate sentences
- **Held-Out Test Claims Pool (`data/test.csv`):** 506 claims
- **Evaluated Pilot Subset:** 75 claims (stratified sample with random seed 42)
  - `false`: 33 claims (44.0%)
  - `true`: 28 claims (37.3%)
  - `NEI`: 14 claims (18.7%)
  - Zero training overlap confirmed.

### 2.2 Benchmark Candidate Pool Counts (`data/benchmark/v1/relevance_judgments.json`)
- **Total Pooled Candidate Pairs:** 4,891 pairs
- **Total Unique Evidence Units in Pool:** 3,839 units
- **Grade Distribution:**
  - `None` (Unjudged): **4,480 pairs** (91.60%)
  - `Grade 3` (Directly decisive / silver reference): **410 pairs** (8.38%)
  - `Grade 1` (Contextual match): **1 pair** (0.02%)
  - `Grade 2` (Substantial): **0 pairs**
  - `Grade 0` (Confirmed not relevant): **0 pairs**
- **Judgment Status Distribution:**
  - `unjudged`: 4,480 pairs
  - `silver_reference`: 410 pairs
  - `reviewed`: 1 pair
- **Annotation Source Distribution:**
  - `unjudged_pool`: 4,480 pairs
  - `finfact_reference`: 410 pairs
  - `model_assisted`: 1 pair

---

## 3. Independent Audit of Evaluation Logic & Mathematical Correctness

### 3.1 Investigation 1: Why Reported Judged Coverage Was ~97.6%–100% Despite 4,480 Unjudged Pairs

In the initial Phase 2 benchmark evaluation, all systems reported `Coverage@10` between **97.6%** and **100.0%**. An audit of `src/benchmark_evaluator.py` revealed the exact cause:

```python
# In src/benchmark_evaluator.py (Lines 59-71)
for d in judgments_list:
    cid = d["claim_id"]
    eid = d["evidence_id"]
    self.judgments[(cid, eid)] = d
    self.claim_judged_set[cid].add(eid)   # <-- FLAW: Added all pooled entries!
    grade = d.get("relevance_grade")
    if grade is not None and int(grade) >= self.binary_threshold:
        self.claim_relevance[cid][eid] = int(grade)
```

**Finding:**  
`self.claim_judged_set[cid].add(eid)` added every document in `relevance_judgments.json` to the judged set, including the **4,480 records where `relevance_grade is None`** and `judgment_status == "unjudged"`.  
Consequently, `Coverage@K` was testing:  
*"Is this retrieved document in the pooled candidate set?"*  
rather than:  
*"Does this retrieved document have an actual definitive relevance judgment?"*

Because pooling collected the top-20 documents from all 8 systems, the top-10 results of any single retriever were almost guaranteed to be in the pool, producing artificial coverage scores near 100%.

**Correction Required:**  
`self.claim_judged_set[cid]` must ONLY include evidence IDs where `relevance_grade is not None` (or `judgment_status != "unjudged"`). Under the true definitive judgments of v1 (where only 410 reference items are judged), the true `Judged Coverage@10` is:
- **BM25:** ~19.3%
- **Dense Transformer:** ~24.0%
- **Hybrid BM25+Dense:** ~24.1%
- **SVD:** ~5.5%

This distinction is crucial for scientific validity and will be formally enforced in `src/benchmark_evaluator.py`.

---

### 3.2 Investigation 2: Why Reference-Evidence Retrieval Rate Was Identical to Recall@10

In the reported metrics table, `Recall@10` and `RefRate@10` were identical across all 8 retrieval systems:
- `BM25`: 0.3636 vs 0.3636
- `Dense Transformer`: 0.4100 vs 0.4100
- `Hybrid BM25+Dense`: 0.4212 vs 0.4212
- `SVD`: 0.0547 vs 0.0547

**Mathematical Explanation:**  
In benchmark v1, the ONLY candidate evidence units assigned a relevance grade $\ge 2$ were the **silver reference items** extracted from `finfact.json` (`relevance_grade=3`).  
Therefore:
$$\text{RelMap}_c \equiv \text{RefEvidence}_c$$
For any retrieval ranking $r_{c, 1}, \dots, r_{c, 10}$:
$$\text{Recall@10}(c) = \frac{|\{r_{c, 1}, \dots, r_{c, 10}\} \cap \text{RelMap}_c|}{|\text{RelMap}_c|}$$
$$\text{RefRate@10}(c) = \frac{|\{r_{c, 1}, \dots, r_{c, 10}\} \cap \text{RefEvidence}_c|}{|\text{RefEvidence}_c|}$$

Because the two target sets were strictly identical, the two formulas evaluated to the exact same numerical result for every single claim.  
Once additional pooled candidates are judged relevant (Grade 2 or 3) during Phase 3, the ground-truth relevant set expands beyond the dataset-provided reference evidence, causing `Recall@10` (overall recall of all relevant facts) and `RefRate@10` (retrieval rate of original fact-checker references) to diverge.

---

### 3.3 Investigation 3: MAP Denominator Convention

Average Precision is computed in `src/benchmark_evaluator.py` as:
$$\text{AP@K}(c) = \frac{1}{|R_c|} \sum_{i=1}^K \text{P@i}(c) \cdot \mathbb{I}(g(r_{c, i}) \ge \tau)$$
where $K=20$.

**Methodological Implications:**
1. Dividing by $|R_c|$ (total known relevant documents for the claim) rather than by $\min(|R_c|, K)$ adheres strictly to standard TREC MAP definitions.
2. If a claim has $|R_c| = 25$ relevant documents, a top-20 evaluation can achieve a maximum possible AP of $\frac{20}{25} = 0.80$. In our 75-claim pilot, the maximum number of reference items for any single claim is 12, so $|R_c| \le 20$ for all pilot claims.
3. However, under incomplete judgments, conventional MAP treats unjudged documents as non-relevant ($g=0$), depressing precision at subsequent ranks. To provide an unbiased comparison, `bpref` (Buckley & Voorhees 2004) must be reported alongside MAP.

---

## 4. Fair System Comparison Audit

All 8 retrieval systems in the benchmark were audited for parity:

1. **Candidate Corpus & Vocabulary:** All systems query the exact same 12,941 candidate evidence sentences.
2. **Evaluation Claims:** All systems evaluate the exact same 75 stratified held-out test claims.
3. **Retrieval Depth:** All systems return top-20 candidates using their standard scoring functions.
4. **Non-Leakage Verification:**
   - Queries are constructed strictly from `claim_text`.
   - Test labels (`label`), justifications (`justification`), digests (`sci_digest`), and gold evidence texts are NOT accessed by the retrieval engines during inference.
   - Reference evidence injection occurred only into the annotation pool, NOT into retriever query processing.
5. **Latency Measurement Parity:**
   - All systems are timed using `time.perf_counter()` over the exact query execution window.
   - Dense transformer embeddings are precomputed once on corpus initialization, matching production inference conditions where corpus vectors are indexed offline.
   - Mean and median latencies are recorded.

---

## 5. Audit Conclusions & Prescribed Actions

1. **Benchmark v1 is Immutable:** The original v1 files and checksums in `data/benchmark/v1/` remain untouched.
2. **Create Benchmark v2:** The research-grounded automation pipeline (`src/judgment_automation.py`) will generate `data/benchmark/v2/` containing audited judgments and an active human review queue.
3. **Enhance Evaluator:** Refactor `src/benchmark_evaluator.py` to:
   - Fix `Coverage@K` to count only definitive, non-null judgments.
   - Implement `bpref` (Buckley & Voorhees 2004) for incomplete-information evaluation.
   - Implement sensitivity analysis across judgment assumptions.
4. **Automated Unit Testing:** Add unit tests validating schema invariants, metric formulas, and lack of leakage.
