# FinGuard: Presentation & Faculty Defense Guide

This directory contains the presentation slide deck outline, talking points, and live demonstration script for defending the FinGuard project before faculty and academic evaluation committees.

---

## Slide-by-Slide Defense Presentation Outline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FINGUARD PROJECT DEFENSE STRUCTURE                              │
│                                                                                        │
│  Slide 1: Problem & Benchmark    ──►  Slide 2: Two-Phase Architecture                  │
│  Slide 3: Phase 1 NLP Pipeline   ──►  Slide 4: Positional Index & Search Engine        │
│  Slide 5: Phase 2 Neural Verifier ──►  Slide 6: Source-Grouped Benchmark Results        │
│  Slide 7: Scientific Root-Cause  ──►  Slide 8: Live Terminal Demonstrations             │
│  Slide 9: Anticipated Q&A & Viva Defense Talking Points                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Slide 1: Problem Statement & Motivation
- **Context**: Financial claims contain subtle numeric nuances, currency amounts, percentages, and temporal markers where a single token changes market meaning (e.g., *"revenue surged 14%"* vs *"net profit fell 14%"*).
- **Benchmark**: FinFact dataset containing 3,369 fact-checked financial claims categorized into three classes: **True**, **False**, and **Not Enough Information (NEI)**.
- **Objective**: Build an evidence-grounded verification system that retrieves supporting/refuting context and predicts claim veracity with an output confidence score.

---

### Slide 2: Two-Phase Architectural Philosophy
- **Phase 1 (Linguistic & Classical Retrieval Foundations)**:
  - Custom finance-aware text preprocessing and tokenization.
  - Polarity-preserving stopword filtering (keeping directional cues).
  - Part-of-speech (POS)-guided WordNet lemmatization with protected entity bypass.
  - Positional inverted index (word-offset tracking) supporting phrase & boolean search.
  - Supervised classical machine learning baselines (TF-IDF + Logistic Regression).
- **Phase 2 (Neural Verification & Multi-Stage Architecture)**:
  - Rule & regex financial Named Entity Recognition (NER).
  - Structured query expansion with entity-metric bindings.
  - Candidate evidence retrieval via Inverted BM25 (pool: 12,943 factual evidence sentences).
  - Neural Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`, 22.7M parameters).
  - Natural Language Inference (NLI) claim verification (`cross-encoder/nli-distilroberta-base`, 82M parameters).
  - Conflict-aware evidence aggregation producing final verdicts (True / False / NEI).

---

### Slide 3: Phase 1 — Solving Financial Domain Linguistic Challenges
- **Why Standard NLP Fails on Finance**:
  - *Standard Tokenizers* split currencies and dates (e.g., splitting `₹16,811` into 4 tokens).
  - *Standard Stopword Lists* remove words like *profit, loss, up, down, increased, above, below, not*, reversing claim polarity.
  - *Standard Lemmatizers* corrupt fiscal markers (`FY2025` $\to$ `fy2025`, `401(k)` $\to$ `401k`).
- **FinGuard Phase 1 Solutions**:
  - Custom regex tokenizer preserving currencies (`$90.8B`, `₹16,811 crore`), percentages (`33.5%`), and fiscal periods (`FY2024`).
  - Preserved **58 domain polarity terms** while filtering 214 high-frequency non-informative stopwords.
  - POS-guided lemmatizer protecting financial acronyms and entities from morphological alteration.

---

### Slide 4: Phase 1 — Positional Inverted Index & Search Engine
- **Corpus Scale**:
  - **Indexed Units**: 23,568 segmented units (3,369 claims + 18,270 evidence sentences + 3,367 justifications).
  - **Vocabulary**: 32,845 unique lemmas.
  - **Positional Postings**: 393,323 term-offset entries.
- **Search Capabilities**:
  - *Exact Phrase Search*: Positional verification ensuring adjacent offsets ($pos_{i+1} = pos_i + 1$).
  - *Boolean Query Engine*: Shunting-Yard parser supporting `AND`, `OR`, `NOT`, and parenthetical precedence.
  - *TF-IDF Retrieval*: Sublinear term-frequency weighting with cosine ranking.

---

### Slide 5: Phase 2 — Multi-Stage Neural Verification Call Chain
Walk through the verified default execution trace:
```
Financial Claim: "Apple Q2 FY2024 revenue surged 14% to $90.8 billion."
       │
       ▼
[1. Financial NER] ────► Extracted: ORG=['Apple'], METRIC=['revenue'], MONEY=['$90.8B'], PCT=['14%']
       │
       ▼
[2. Query Expansion] ──► Generated: "Apple Apple Inc revenue Q2 FY2024 $90.8B"
       │
       ▼
[3. BM25 Retrieval] ───► Retrieved: Top-15 candidate evidence sentences from 12,943 pool
       │
       ▼
[4. Neural Reranking] ─► Cross-Encoder (MiniLM) scores (claim, passage) joint interaction (Top 4)
       │
       ▼
[5. NLI Verification] ─► RoBERTa NLI scores (Premise=Evidence, Hypothesis=Claim)
       │
       ▼
[6. Aggregation] ──────► Verdict: TRUE | Confidence: 0.587 | Evidence: 3 cited passages
```

---

### Slide 6: Empirical Findings — Source-Grouped Benchmark ($N=675$)
- **Experimental Design**: Evaluated on a strictly held-out source URL-grouped test set ($N=675$ claims from 531 unique source URLs; no test URL appears in training or validation).
- **Ground-Truth Comparison**:

| System Architecture | Test Accuracy | Macro-F1 | False Recall | True Recall | NEI Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System A: Claim-Only Baseline (TF-IDF + LogReg)** | **55.56%** | **48.97%** | 66.22% | **59.22%** (151/255) | 21.49% |
| **System B: Evidence-Based FinGuard (Neural)** | **40.30%** | **31.07%** | **71.57%** | **02.75%** (7/255) | **42.15%** |
| **System C: Configured Pipeline (Default)** | **40.30%** | **31.07%** | **71.57%** | **02.75%** (7/255) | **42.15%** |

---

### Slide 7: Scientific Root-Cause Post-Mortem
Present this with scientific confidence — honesty about real-world performance is the hallmark of great research:
1. **The Classical Baseline Outperformed Neural Evidence Grounding**:
   - The claim-only baseline achieved 55.56% accuracy vs 40.30% for evidence-grounded verification.
2. **Neutral Collapse in General NLI (Observed & Quantified)**:
   - Across 2,700 evaluated candidate pairs, the zero-shot NLI model predicted **1,489 neutral**, **1,145 contradiction**, and only **66 entailment** outputs.
   - For 243 of 255 gold-true claims, not a single supporting passage exceeded the evidence aggregation threshold. Pretrained general NLI treats financial hedging, forward guidance, and conditional phrasing as unprovable (NEI).
3. **Retrieval Domain Shift & Error Taxonomy**:
   - Over unseen source URLs, lexical BM25 retrieved tangential background sentences rather than direct verifications, depriving the NLI verifier of conclusive premises.
   - In our 50-case error audit, the top 3 categories (Irrelevant Retrieval: 23, Wrong Entity: 11, Insufficient Evidence: 10) accounted for 44/50 cases (88.0%).

---

### Slide 8: Live Demonstration Guide

Execute these live in front of the committee:

```powershell
# 1. Phase 1 Demonstration (Text cleaning, tokenization, lemmatizer, positional index, phrase/boolean search, baseline inference)
python phase1/scripts/run_demo.py

# 2. Phase 2 Demonstration (Live claim parsing, query expansion, BM25 retrieval, cross-encoder reranking, NLI scoring, aggregated verdict)
python phase2/scripts/run_demo.py

# 3. Automated Test Suite (Proves all 100 unit tests pass cleanly)
python -m unittest discover -s tests -p "test_*.py"
```

---

### Slide 9: Anticipated Committee Questions & Defensible Answers

#### Q1: "Why did you build Phase 1 if Phase 2 uses neural models?"
> **Answer**: *"Phase 1 established essential domain foundations. General NLP tools fail on financial text: standard tokenizers fragment amounts like `$90.8B` or `₹16,811 crore`, standard stopword lists delete directional polarity cues like `loss` and `profit`, and general lemmatizers alter financial acronyms. Phase 1 provided domain-adapted tokenization, positional indexing over 23,568 units, and our classical baseline benchmark against which Phase 2 was rigorously compared."*

#### Q2: "Why did your evidence-based neural pipeline score lower than the claim-only baseline on held-out sources?"
> **Answer**: *"This is our most significant empirical research finding. On our strictly held-out source URL-grouped split ($N=675$), the classical baseline achieved 55.56% accuracy while evidence-based verification achieved 40.30%. We conducted an extensive root-cause audit:
> 1. Pretrained zero-shot NLI models suffer from severe 'neutral collapse' on financial text: general NLI models treat forward guidance, conditional projections, and numerical qualifications as unprovable (NEI), recognizing only 7 of 255 True claims (2.75% True recall).
> 2. Lexical BM25 retrieval across unseen source URLs experiences domain shift, retrieving background context rather than direct fact-checking verifications. A claim-only baseline is immune to retrieval failure because it does not depend on premise passage quality."*

#### Q3: "Did you fine-tune the transformer models on FinFact?"
> **Answer**: *"No, and we explicitly document this in our audit. FinGuard uses pretrained zero-shot checkpoints (`cross-encoder/ms-marco-MiniLM-L-6-v2` for passage reranking and `cross-encoder/nli-distilroberta-base` for NLI verification). The PyTorch training scaffolding exists in `transformer_model.py`, but no fine-tuned weights were used in the evaluation pipeline. This intentional design choice directly revealed how zero-shot general NLI models fail when confronted with financial domain text."*

#### Q4: "How do you guarantee that your evaluation did not leak training data?"
> **Answer**: *"We enforced three strict leakage prevention safeguards:
> 1. Evaluated on a source URL-grouped split where all 531 source URLs in the test set are mutually exclusive from training and validation.
> 2. Removed fact-checker verdict cues (`strip_verdict_cues`) from all evidence text so labels were not leaked in retrieved sentences.
> 3. Segregated private evaluation mappings in `data/restricted_evaluation/private_evaluation_mapping.csv`, ensuring public review queues contain only anonymized identifiers."*
