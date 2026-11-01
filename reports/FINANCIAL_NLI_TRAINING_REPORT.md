# Financial Domain NLI Fine-Tuning: Training Dynamics & Methodology Report

**Project:** FinGuard — Financial Claim Verification Pipeline  
**Component:** Natural Language Inference (NLI) Verification Engine  
**Experiment Date:** November 2026  
**Artifacts Generated:** [`models/finetuned_nli/`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/models/finetuned_nli/), [`phase2/data/financial_nli/`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/data/financial_nli/)

---

## 1. Executive Summary

This report documents the domain-specific fine-tuning of the Natural Language Inference (NLI) cross-encoder in FinGuard Phase 2. Prior to this intervention, FinGuard utilized an off-the-shelf general-domain cross-encoder ([`cross-encoder/nli-distilroberta-base`](https://huggingface.co/cross-encoder/nli-distilroberta-base)) operating zero-shot. As demonstrated in [`reports/FINANCIAL_NLI_FINETUNING_AUDIT.md`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/reports/FINANCIAL_NLI_FINETUNING_AUDIT.md), the zero-shot model suffered from acute **neutral collapse**, failing to identify domain-specific financial entailment and contradiction, and collapsing 73.4% of true financial entailments into Neutral.

To address this failure mode, we constructed a leakage-free, rule-verified premise-hypothesis training corpus directly from the FinFact dataset and domain evidence corpus, strictly preserving claim-source grouping. We fine-tuned the model under early stopping on validation Macro-F1.

### Key Results
- **Validation Macro-F1:** Improved from **27.94% (Zero-Shot)** to **64.64% (Epoch 1 Fine-Tuned)**, an absolute gain of **+36.70 percentage points**.
- **Entailment F1:** Surged from **15.57%** to **69.70%** (+54.13 percentage points).
- **Contradiction F1:** Surged from **16.77%** to **59.54%** (+42.77 percentage points).
- **Early Stopping:** Triggered after Epoch 2 when training loss fell to 0.2709 but validation Macro-F1 degraded to 56.33% due to neutral overfitting. Epoch 1 was selected as the optimal production checkpoint.

---

## 2. Dataset Composition & Data Hygiene

### 2.1 Split Integrity and Source Grouping
FinFact contains 3,369 claims collected from Snopes and PolitiFact. To ensure zero test leakage and prevent memorization of repetitive financial fact-check phrases, claims were split strictly by `source_url` domain grouping matching the master project splits:
- **Training Set:** 2,357 claims (70.0%) $\to$ **7,481 premise-hypothesis pairs**
- **Validation Set:** 337 claims (10.0%) $\to$ **1,155 premise-hypothesis pairs**
- **Test Set (Held-Out):** 675 claims (20.0%) $\to$ **2,361 premise-hypothesis pairs** (frozen, evaluated exclusively end-to-end)

Zero claim ID overlap exists across the splits:
$$\text{Claim ID Overlap}(\text{Train}, \text{Val}) = \emptyset, \quad \text{Claim ID Overlap}(\text{Train}, \text{Test}) = \emptyset, \quad \text{Claim ID Overlap}(\text{Val}, \text{Test}) = \emptyset$$

### 2.2 Premise-Hypothesis Construction Rules
Pair extraction avoided the *False Contradiction Fallacy* (assuming all evidence under a false claim contradicts it) and the *False Neutral Fallacy* (treating related but non-decisive context as neutral without verification):
1. **Entailment Generation:**
   - Gold `true` claims paired with their core justification / explanation sentences ($\ge 0.40$ semantic/lexical overlap).
   - High-overlap passages confirming the factual assertions of the claim.
2. **Contradiction Generation:**
   - Gold `false` claims paired with refutation sentences containing explicit negating lexical cues (*"false", "incorrect", "debunked", "no evidence", "refuted"*, antonymic shifts, numerical mismatches).
   - Weakly supervised refutations quarantined and audited.
3. **Neutral Generation:**
   - Cross-claim topical hard negatives: Premise retrieved from the corpus by BM25 using claim entities (e.g. Federal Reserve, inflation rates) from an entirely unrelated claim.
   - Gold `NEI` claims paired with inconclusive context sentences.
4. **Quarantine Queue:**
   - Ambiguous pairs where neither refutation nor confirmation could be established with high confidence were directed to [`phase2/data/financial_nli/human_review_queue.jsonl`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/phase2/data/financial_nli/human_review_queue.jsonl) (4,633 pairs) and excluded from training.

### 2.3 Training & Validation Distributions

| Dataset Split | Entailment | Contradiction | Neutral | Total Pairs | High-Confidence | Weakly Supervised |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2,456 (32.8%) | 1,160 (15.5%) | 3,865 (51.7%) | **7,481** | 6,717 (89.8%) | 764 (10.2%) |
| **Validation** | 421 (36.5%) | 190 (16.5%) | 544 (47.1%) | **1,155** | 1,043 (90.3%) | 112 (9.7%) |
| **Test (Frozen)** | 823 (34.9%) | 382 (16.2%) | 1,156 (49.0%) | **2,361** | 2,128 (90.1%) | 233 (9.9%) |

To prevent majority-class domination during gradient updates, training applied balanced class subsampling capping each class at 800 high-confidence samples per epoch (2,400 active training samples per epoch, shuffled with seed 42).

---

## 3. Model Architecture & Training Configuration

### 3.1 Architecture & Label Mapping
- **Base Checkpoint:** `cross-encoder/nli-distilroberta-base`
- **Backbone:** DistilRoBERTa (6 layers, 768 hidden dimensions, 12 attention heads, ~82 million parameters).
- **Classification Head:** Linear projection $W \in \mathbb{R}^{768 \times 3}$ over the `[CLS]` token representation followed by Cross-Entropy Loss.
- **Dynamic Label Alignment:**
  - Index 0: `contradiction`
  - Index 1: `entailment`
  - Index 2: `neutral`
  - Confirmed matching Hugging Face checkpoint config:
    ```json
    "id2label": {
      "0": "contradiction",
      "1": "entailment",
      "2": "neutral"
    }
    ```

### 3.2 Hyperparameter Specification
The fine-tuning configuration was preserved in [`models/finetuned_nli/training_config.json`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/models/finetuned_nli/training_config.json):

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Standard decoupled weight decay optimization for transformers |
| **Learning Rate** | $2 \times 10^{-5}$ | Conservative rate to prevent catastrophic forgetting of base NLI |
| **LR Scheduler** | Linear warmup (10% steps) + Linear decay | Smooth convergence and stability |
| **Weight Decay** | $0.01$ | $L_2$ regularization on non-bias/layer-norm weights |
| **Batch Size** | 16 | Optimal CPU cache utilization and gradient stability |
| **Max Sequence Length** | 160 tokens | Covers 98.4% of financial premise-hypothesis concatenations |
| **Epochs** | 2 | CPU constraint + early stopping boundary |
| **Early Stopping Metric** | Validation Macro-F1 | Prevents majority-neutral bias from dominating checkpoint selection |
| **Seed** | 42 | Full deterministic reproducibility across data loader and weights |

### 3.3 Hardware Environment & Execution Profile
- **Processor:** Intel/AMD multi-core host CPU (`PyTorch 2.12.1+cpu`, CUDA unavailable).
- **Thread Allocation:** PyTorch intra-op and inter-op threads set to 8.
- **Total Training Duration:** 2,098.77 seconds (~34.98 minutes).
- **Average Batch Speed:** 6.94 seconds per batch of 16 pairs on CPU.
- **Memory Footprint:** Peak system RAM utilization ~2.4 GB; model checkpoint on disk: 328.7 MB (`model.safetensors`).

---

## 4. Training Dynamics & Checkpoint Selection

### 4.1 Epoch-by-Epoch Progress

```
Initial Zero-Shot Baseline:
  Val Loss: 2.2810 | Val Acc: 36.54% | Val Macro-F1: 27.94%
  Class F1s: Contradiction: 16.77% | Entailment: 15.57% | Neutral: 51.46%

Epoch 1 / 2:
  Train Loss: 0.7950 (Batches 1-150: 0.9954 -> 0.4862) | Duration: 1,041.14s
  Val Loss: 0.8776 | Val Acc: 65.28% | Val Macro-F1: 64.64%  <-- [BEST CHECKPOINT SAVED]
  Class F1s: Contradiction: 59.54% | Entailment: 69.70% | Neutral: 64.68%

Epoch 2 / 2:
  Train Loss: 0.2709 (Batches 1-150: 0.3521 -> 0.1844) | Duration: 866.61s
  Val Loss: 1.3881 | Val Acc: 56.62% | Val Macro-F1: 56.33%  <-- [DEGRADATION: OVERFITTING]
  Class F1s: Contradiction: 65.76% | Entailment: 64.97% | Neutral: 38.26%
```

### 4.2 Training vs Validation Loss Curve

```
Loss
2.50 +-- [Zero-Shot Val: 2.281]
2.00 |
1.50 |                                 [Epoch 2 Val: 1.3881]
1.00 +-- [Epoch 1 Train: 0.7950] ----- [Epoch 1 Val: 0.8776] (Optimal)
0.50 |
0.00 +--------------------------------- [Epoch 2 Train: 0.2709]
     +---------------------------------+---------------------->
     Epoch 0 (Zero-Shot)            Epoch 1                 Epoch 2
```

### 4.3 Validation Confusion Matrix Transition

#### Zero-Shot Baseline Confusion Matrix ($N = 1,155$)
```
                   Predicted Contradiction   Predicted Entailment   Predicted Neutral
Gold Contradiction           41                       12                   137   (Recall: 21.58%)
Gold Entailment              74                       38                   309   (Recall:  9.03%)
Gold Neutral                184                       17                   343   (Recall: 63.05%)
```
*Diagnosis:* Severe neutral collapse. 73.4% of Gold Entailments (309/421) and 72.1% of Gold Contradictions (137/190) misclassified as Neutral.

#### Epoch 1 Fine-Tuned Confusion Matrix ($N = 1,155$) — [Selected Model]
```
                   Predicted Contradiction   Predicted Entailment   Predicted Neutral
Gold Contradiction          156                        9                    25   (Recall: 82.11%)
Gold Entailment              48                      283                    90   (Recall: 67.22%)
Gold Neutral                130                       99                   315   (Recall: 57.90%)
```
*Diagnosis:* Symmetrical discrimination across all three classes. Entailment recall multiplied by $7.4\times$ (from 9.03% to 67.22%); Contradiction recall multiplied by $3.8\times$ (from 21.58% to 82.11%).

#### Epoch 2 Confusion Matrix ($N = 1,155$)
```
                   Predicted Contradiction   Predicted Entailment   Predicted Neutral
Gold Contradiction          169                       16                     5   (Recall: 88.95%)
Gold Entailment              30                      344                    47   (Recall: 81.71%)
Gold Neutral                125                      278                   141   (Recall: 25.92%)
```
*Diagnosis:* Overfitting. The model began assigning polarity to neutral statements, causing Neutral recall to crash from 57.90% to 25.92%.

---

## 5. Early Stopping & Checkpoint Integrity

In accordance with empirical machine learning best practices:
1. **Checkpointing Rule:** Checkpoints are evaluated strictly on validation Macro-F1 at the completion of each epoch.
2. **Patience Trigger:** Epoch 2 showed a validation loss increase (+0.5105) and a Macro-F1 decrease (-8.31 percentage points). The early stopping controller restored the weights from **Epoch 1**.
3. **Artifact Integrity:** The saved checkpoint in [`models/finetuned_nli/`](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/models/finetuned_nli/) contains:
   - `model.safetensors` (328,707,316 bytes)
   - `config.json` (dynamic labels verified: 0=contradiction, 1=entailment, 2=neutral)
   - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`
   - `training_config.json`, `training_history.json`

This model serves as the verifier engine for **System C** in FinGuard Phase 2.

