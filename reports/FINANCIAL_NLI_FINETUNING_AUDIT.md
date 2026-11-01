# FinGuard Phase 2: Financial NLI Fine-Tuning Feasibility & Dataset Audit

**Document Status:** Complete Scientific Audit  
**Author:** FinGuard Research Team  
**Date:** November 2026  
**Target Architecture:** Evidence-Grounded Financial NLI Claim Verification (Phase 2)  
**Pretrained Starting Checkpoint:** `cross-encoder/nli-distilroberta-base` (82M parameters, RoBERTa architecture)  

---

## Executive Summary

This audit establishes the empirical and methodological foundations for fine-tuning the Natural Language Inference (NLI) component of FinGuard Phase 2. Currently, the FinGuard pipeline utilizes `cross-encoder/nli-distilroberta-base` in a zero-shot capacity, trained on general-domain NLI benchmarks (MNLI, SNLI). While effective for general semantic inference, the zero-shot model suffers from financial domain mismatch, leading to the known **neutral-collapse phenomenon** (where financial evidence is overwhelmingly classified as Neutral with low confidence, causing 64.30% of gold-true claims to be defaulted to NEI).

This audit rigorously inspects the FinFact dataset, existing repository splits, evidence structure, and available supervision signals to assess the scientific feasibility of financial domain fine-tuning.

---

## 1. Pretrained NLI Model Inspection & Architecture

### 1.1 Model Initialization & Checkpoint Specs
- **Model Identifier:** `cross-encoder/nli-distilroberta-base`
- **Location in Codebase:** Initialized in [`src/verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/verifier.py) (`FinancialClaimVerifier`) and mirrored in [`phase2/src/verifier/verifier.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/src/verifier/verifier.py).
- **Execution Mode:** Offline-capable via `AutoTokenizer` and `AutoModelForSequenceClassification` with explicit environment guards (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- **Hardware Profile:** Executed on CPU (`PyTorch 2.12.1+cpu`, 0 CUDA devices detected). All training and inference routines must be optimized for CPU memory footprint and latency.

### 1.2 Premise-Hypothesis Pair Construction
In the production pipeline ([`src/verifier.py#L78-L83`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/src/verifier.py#L78-L83)), premise-hypothesis pairs are passed to the tokenizer as:
$$\text{Input} = [\text{CLS}] \; \text{Premise (Evidence Passage)} \; [\text{SEP}] \; [\text{SEP}] \; \text{Hypothesis (Financial Claim)} \; [\text{SEP}]$$
- **Premise:** Candidate evidence sentence retrieved and reranked from the corpus.
- **Hypothesis:** The financial claim string under scrutiny.
- **Maximum Sequence Length:** Truncated and padded at `max_length = 160` (or `max_length = 192`), ensuring full retention of both hypothesis and premise tokens.

### 1.3 Label Mapping Verification
Direct programmatic inspection of `AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base").config` confirms:
- **`id2label`:**
  - `0`: `"contradiction"`
  - `1`: `"entailment"`
  - `2`: `"neutral"`
- **`label2id`:**
  - `{"contradiction": 0, "entailment": 1, "neutral": 2}`
- **`num_labels`:** 3

```python
# Programmatic Verification Result:
Config id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
Config label2id: {'contradiction': 0, 'entailment': 1, 'neutral': 2}
```

### 1.4 Downstream Evidence Aggregation Logic
The verifier predicts class probabilities via $\text{Softmax}(\mathbf{z})$ over the 3 logits:
- **Contradiction Threshold:** $p_{\text{contra}} \ge 0.50 \implies$ appended to `contradicting_evidence`.
- **Entailment Threshold:** $p_{\text{entail}} \ge 0.50 \implies$ appended to `supporting_evidence`.
- **Neutral / Context:** Otherwise appended to `neutral_evidence`.
- **Decision Engine:**
  - If $\text{len}(\text{supporting}) > 0$ and $\text{len}(\text{contradicting}) > 0 \implies \text{Verdict} = \text{NEI}$ (Conflict detected across sources).
  - Else if $\text{len}(\text{contradicting}) > 0$ and $\max(p_{\text{contra}}) \ge 0.60 \implies \text{Verdict} = \text{FALSE}$.
  - Else if $\text{len}(\text{supporting}) > 0$ and $\max(p_{\text{entail}}) \ge 0.60 \implies \text{Verdict} = \text{TRUE}$.
  - Else $\implies \text{Verdict} = \text{NEI}$ (Uncertain / Insufficient evidence).

---

## 2. FinFact Dataset Schema & Inspection

### 2.1 Storage Locations & Formats
1. **Raw Master File:** [`data/raw/finfact.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/raw/finfact.json) (29.97 MB, 195,017 JSON lines, 3,369 claims).
2. **Processed Tabular Formats:**
   - [`data/grouped_train.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_train.csv) (2,357 claims)
   - [`data/grouped_val.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_val.csv) (337 claims)
   - [`data/grouped_test.csv`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/data/grouped_test.csv) (675 claims)

### 2.2 Schema of Raw FinFact Record
Each entry in `finfact.json` contains:
| Field Name | Type | Description | Sample / Presence |
| :--- | :--- | :--- | :--- |
| `url` | string | Canonical source URL of the fact-check article | 3,369 / 3,369 (100%) |
| `claim` | string | Natural language assertion under factual review | 3,369 / 3,369 (100%) |
| `label` | string | Ground-truth veracity verdict (`"false"`, `"true"`, `"NEI"`) | 3,369 / 3,369 (100%) |
| `author` | string | Fact-checker or journalist byline | Present in ~95% |
| `posted` | string | Publication date (`MM/DD/YYYY`) | Present in ~98% |
| `sci_digest` | list[str] | Scientific or topical summary digest | Present in ~40% |
| `justification` | string | Full-text investigative journalistic analysis | 3,365 / 3,369 (99.9%) |
| `issues` | list[str] | Financial topics (e.g., `["debt"]`, `["taxes"]`, `["inflation"]`) | Present in ~85% |
| `image_data` | list[dict] | URLs and metadata for embedded images/charts | Present in ~50% |
| `evidence` | list[dict] | List of evidence sentences and source hyperlinks | 3,268 / 3,369 (97.0%) |

### 2.3 Structure of `evidence` Field
Within each claim record, `evidence` is a list of dictionary elements:
```json
{
  "sentence": "In July and August, Facebook users shared posts that contained the following text...",
  "hrefs": ["https://archive.is/eqnFJ", "https://archive.is/vBHE0"]
}
```
- Total evidence sentences across all 3,369 claims: **18,357 sentences**.
- In the training split (`grouped_train.csv`): **12,484 sentences** across 2,357 claims.
- Mean sentence length: **43 words** (median), 75th percentile: **63 words**, 90th percentile: **84 words**.

---

## 3. Critical Scientific Findings: Pair-Level Supervision Analysis

### 3.1 The Pair-Level Supervision Gap
**Crucial Finding:** The FinFact dataset provides **claim-level labels only** (`true`, `false`, `NEI`). It does **not** provide ground-truth sentence-level NLI stance labels (`entailment`, `contradiction`, `neutral`).

### 3.2 Risks of Naive Heuristic Labeling
A naive or reckless label assignment strategy would cause severe label noise and distort model training:
1. **The "FALSE Claim $\implies$ Contradiction" Fallacy:**
   - Fact-checking articles for FALSE claims frequently quote the rumor or false assertion verbatim in the introductory sentences (e.g., *"Social media posts claimed that Amazon is giving away free AirPods"*).
   - If every sentence in a FALSE claim's evidence list is labeled as `contradiction`, the model is actively trained to classify identical text or quotes as a contradiction, severely corrupting its semantic representations.
   - Background sentences (e.g., *"Amazon was founded in 1994 in Bellevue, Washington"*) would be erroneously labeled as contradictions rather than neutral context.
2. **The "NEI Claim $\implies$ Neutral" Fallacy:**
   - Articles investigating unconfirmed claims often contain verified factual assertions (e.g., *"The company filed Form 10-K on March 15"*), which are factual entailments of sub-components, even though the primary claim cannot be fully established.
   - Forcing all sentences in NEI claims to be labeled `neutral` introduces semantic inconsistency.
3. **The Unrelated Random Sampling Fallacy:**
   - Pairing random unrelated passages from arbitrary claims as "neutral" examples creates trivial negative pairs with zero lexical or entity overlap. The model learns to rely on surface keywords rather than evaluating semantic sufficiency.

### 3.3 Feasibility Assessment of Supervised NLI Fine-Tuning
Supervised fine-tuning is **scientifically feasible**, provided that pair-level training data is constructed through **rigorous, transparent, and verifiable rules** with strict provenance:

1. **High-Confidence Entailment ($\mathcal{L} = 1$):**
   - Derived from `grouped_train.csv` claims where `gold_label == "true"`.
   - Selected evidence sentences must exhibit strong semantic and entity concordance with the claim (e.g., verified via lexical overlap + financial entity alignment), confirming that the sentence directly substantiates the claim.
2. **High-Confidence Contradiction ($\mathcal{L} = 0$):**
   - Derived from `grouped_train.csv` claims where `gold_label == "false"`.
   - Selected evidence sentences must contain explicit refutation cues, factual negations, or contradictory assertions (e.g., *"did not"*, *"no evidence"*, *"contrary to"*, *"debunked"*, *"false"*), directly refuting the claim premise.
3. **High-Confidence Contextual Neutral ($\mathcal{L} = 2$):**
   - Derived from hard-negative pairs: evidence passages from the same financial article or topically retrieved candidates (sharing entity/topic names) that describe contextual background, different reporting periods, or adjacent metrics, but do not substantiate or refute the specific claim hypothesis.
4. **Weakly Supervised / Ambiguous Queue:**
   - Any pair failing high-confidence validation criteria is flagged as weakly labeled with explicit rule provenance, or routed to an annotation-ready queue.
5. **Leakage Isolation:**
   - Pairs are derived strictly from `grouped_train.csv` for training, and `grouped_val.csv` for validation.
   - The 675 held-out test claims (`grouped_test.csv`) and the restricted private evaluation mapping remain 100% untouched.

---

## 4. Dataset Splits & Leakage Prevention Protocol

| Split | Number of Claims | False Claims | True Claims | NEI Claims | Base URL Grouping | Leakage Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 2,357 | 1,044 (44.3%) | 892 (37.8%) | 421 (17.9%) | Strict domain isolation | Clean |
| **Validation** | 337 | 149 (44.2%) | 128 (38.0%) | 60 (17.8%) | Strict domain isolation | Clean |
| **Test (Held-Out)** | 675 | 299 (44.3%) | 255 (37.8%) | 121 (17.9%) | Strict domain isolation | Frozen & Protected |

### Integrity Rules
1. **Zero Test Contamination:** No test claims or test evidence sentences will be used in pair construction, training, hyperparameter tuning, or early stopping.
2. **Claim-Level Clumping:** All pairs derived from claim $C_i$ are assigned exclusively to the split containing $C_i$.
3. **No Private Mapping Access:** The restricted evaluation mapping in `data/restricted_evaluation/` is strictly barred from training pipelines.

---

## 5. Decision & Next Steps

Based on this audit:
1. **Feasibility Verdict:** Supervised domain-specific NLI fine-tuning is feasible and methodologically sound using a high-precision, rule-verified pair extraction pipeline with explicit provenance.
2. **Proceed to Phase 2:** Implement [`phase2/scripts/build_financial_nli_dataset.py`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/scripts/build_financial_nli_dataset.py) to construct `phase2/data/financial_nli/train.jsonl` and `validation.jsonl`.
3. **Proceed to Phase 3 & 4:** Execute fine-tuning with early stopping on validation macro-F1 and CPU-optimized sequence configurations.

