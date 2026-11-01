# FinGuard Root-Cause Investigation: Evidence-Grounded Verification

## Executive findings

On the aligned source-grouped test set, the failure is **not explained by an
NLI label-index reversal or malformed premise/hypothesis ordering**. The cached
`cross-encoder/nli-distilroberta-base` configuration labels index 0
contradiction, 1 entailment, and 2 neutral; the verifier reads this configuration
dynamically, sends evidence as premise and claim as hypothesis, applies softmax,
and maps entailment to true, contradiction to false, and neutral to NEI. Six
controlled probes behaved as expected for direct entailment, contradiction,
neutrality, negation, wrong metric, and wrong period. This verifies basic
implementation mechanics, not broad domain reliability.

The strongest evidence from the held-out traces points to **evidence selection and
specificity**: the top four were operationally available on all 675 claims, but
the recorded NLI outputs were 1,489 neutral, 1,145 contradiction, and only 66
entailment decisions over 2,700 passages. No support-threshold passage appeared
for 243/255 gold-true claims. A retrieval-text heuristic flagged entity mismatch
on 304 claims, but this extractor is noisy on non-financial claims; it is a
screening signal, not a human relevance judgment.

Removing the cross-encoder while keeping BM25, the same test claims, the same
safe corpus, the same NLI checkpoint, and the same production aggregation reduced
accuracy from 40.30% to 39.41% and macro-F1 from 0.3107 to 0.2958. This modest
change is not evidence that reranking improves *evidence relevance*: the review
sheet is awaiting human labels, and the cross-encoder was not independently
validated against human judgments. The largest observed limitation remains
irrelevant or insufficient evidence, followed by NLI decisions on the selected
passages.

## 1. NLI implementation audit

### Configuration and mapping

The locally cached model configuration was read directly (not inferred from a
model card):

| Output index | `id2label` | FinGuard semantic class | Final verdict |
|---:|---|---|---|
| 0 | contradiction | contradiction | false |
| 1 | entailment | entailment | true |
| 2 | neutral | neutral | NEI |

`label2id` agrees with `id2label`. The artifact is a three-label
`RobertaForSequenceClassification`; the tokenizer is `RobertaTokenizer`.
`src/verifier.py` loads `config.id2label` and maps by label names. It applies
softmax over the raw logits and sums probabilities into the corresponding
semantic classes. No hard-coded class order is needed for this checkpoint.

### Input construction, tokenization, and truncation

The implementation calls the tokenizer positionally as
`tokenizer(evidence_sentence, claim, ...)`, which is the Hugging Face paired-text
form: premise first, hypothesis second. It uses pair special tokens, padded
tensor output, truncation enabled, and `max_length=160`. The model is called under
`torch.no_grad()`. The sanity-test JSON records each test's raw logits, probabilities
by index and semantic label, final interpreted label, and token counts.

Across the 2,700 saved production NLI pairs, **142 (5.26%)** exceeded 160 tokens
before truncation; median length was 81, 95th percentile 163, and maximum 1,466.
The tokenizer truncation side is right, with Hugging Face's longest-first pair
truncation. Thus some long premises are truncated, but truncation is not pervasive.
It is a plausible contributor for a small subset, not an explanation for the
overall collapse.

### Controlled sanity probes

The values below are actual model outputs from the cached checkpoint. Probabilities
are shown in the semantic order false / true / NEI.

| Probe | Model argmax | P(false) | P(true) | P(NEI) | Assessment |
|---|---|---:|---:|---:|---|
| A. Identical revenue claim and evidence | entailment | 0.0034 | 0.9913 | 0.0053 | Correct direct entailment |
| B. ₹100 crore premise vs ₹115 crore claim | contradiction | 0.9860 | 0.0028 | 0.0112 | Correct direct contradiction |
| C. Offices premise, revenue hypothesis | neutral | 0.0146 | 0.0014 | 0.9841 | Correct neutral |
| D. “Did not increase” premise vs “increased” claim | contradiction | 0.9947 | 0.0013 | 0.0040 | Correct polarity handling |
| E. Operating expenses vs revenue | contradiction | 0.6419 | 0.3199 | 0.0382 | Correctly rejects entailment; strong contradiction |
| F. FY2024 premise vs FY2025 claim | contradiction | 0.5485 | 0.2047 | 0.2468 | Contradiction is top label, but below the verifier's 0.60 decisive threshold |

The output mapping and semantic direction are therefore technically consistent in
these tests. The wrong-metric result also cautions that this NLI model may treat
related financial assertions as contradiction rather than neutral; that is model
behavior, not a label-index bug. Six hand-built probes are a smoke test, not a
reliability benchmark.

### Implementation verdict

Basic loading, tokenization, pair ordering, softmax, and label interpretation are
correct for the inspected checkpoint. Entailment maps to true, contradiction to
false, and neutral to NEI. The unresolved issue is whether this general-purpose
MNLI checkpoint reliably interprets noisy, retrieved financial/article snippets.
The aligned evaluation shows that it does not do so reliably in this pipeline.

## 2. Retrieval-quality analysis

`retrieval_error_analysis.csv` contains one row per claim and retains the original
claim, extracted organizations/metrics/amounts/percentages/periods/direction/
negations, generated expanded query, all 15 BM25 passages with BM25 scores and
source provenance, their cross-encoder rank/scores, and the selected top four
with NLI outputs. Gold is included only as a diagnostic field in this offline
analysis; it was not supplied to query construction, retrieval, reranking, or NLI.

### Corpus and passage availability

The analysis reuses the saved experiment corpus and outputs: the sanitized corpus
was created from grouped train/validation records and excluded test records and
test-linked citation URLs. The prior run verified zero test-source URL overlap.
All 675 claims had 15 BM25 candidates, 15 reranked candidates, and four selected
passages. Thus “no usable evidence” in the mechanical sense (at least one passage
with three or more words) occurred zero times. This says nothing by itself about
factual relevance.

### Structured screening flags

The analysis uses deterministic text/extractor heuristics to prioritize human
review. These are **not** gold relevance judgments and are especially imperfect
for this dataset, which contains many non-financial claims. Counts among the 675
claims:

| Screening signal | Claims flagged |
|---|---:|
| Extracted entity absent from the selected top-four text | 304 |
| Extracted metric absent (only one claim had a detected metric mismatch) | 1 |
| Extracted quantity/number not found literally in selected text | 47 |
| Extracted date/period not found literally in selected text | 16 |
| Extracted direction/negation not found literally in selected text | 11 |
| Selected top-four contains an exact duplicate sentence | 1 |
| Top-15 BM25 candidates contain normalized-text duplicates | 4 duplicate instances across 4 claims |
| Low claim-term overlap (“related topic” screen) | 65 |
| Gold-true claim with no support-threshold passage | 243 / 255 |
| Gold-false claim with no contradiction-threshold passage | 67 / 299 |

The entity extractor also marked 203 claims as having no extracted organization,
and its proper-noun heuristic can label descriptive phrases as entities. Similarly,
financial metric extraction returned no target for 629 claims. The very low metric
flag count is therefore not evidence of metric correctness. These signals can
identify candidate failure modes but cannot establish them without annotators.

The 65 low-overlap claims include demonstrable topical mismatches. For example,
the claim about a man from Ohio dying after calling COVID a political scheme
received passages about Joe Diffie, Andrew Cuomo, and unrelated COVID statistics.
The Lukoil ownership claim received evidence about Citgo, another company.
This is consistent with the earlier manual error review and shows why BM25 term
overlap and a cross-encoder relevance score are not sufficient evidence-quality
measures.

### Reranking behavior

Cross-encoder reranking changed the membership of the BM25 top-four set in
**588/675** claims. The average overlap between BM25 top four and reranked top four
was **2.20 passages**. This confirms that reranking is materially changing passage
selection, but not whether those changes are improvements. The reranker score is a
relevance score trained on a different task; it is not a truth score or a human
relevance label.

The no-reranker ablation used the first four passages in the saved BM25 top 15.
The reranked configuration used the same BM25 top 15 and the cross-encoder's top
four. Retrieval results and source provenance are available in the CSV for
claim-level inspection. Duplicate selected evidence was rare under an exact-text
check. In the full top-15 BM25 lists, a case-folded, punctuation/whitespace-
normalized text comparison found four duplicate passage instances across four
claims; the corresponding per-claim counts and flags are in the retrieval CSV.
This exact normalized-text check will not catch paraphrases or near-duplicates,
and does not establish that the repeated passage is relevant.

## 3. Aggregation and conflict-gating logic

The unchanged production behavior is traced and illustrated in
`aggregation_analysis.md`; exact synthetic outputs are also stored in
`ablation_results.json`.

The verifier drops passages under three words. For each remaining passage, it
assigns contradiction if P(contradiction) ≥ 0.50; otherwise support if
P(entailment) ≥ 0.50; otherwise neutral. Across passages it computes the maximum
entailment and contradiction probabilities, not sums or averages. If both support
and contradiction groups are nonempty, it returns NEI at confidence 0.55.
Otherwise, a contradiction maximum ≥ 0.60 returns false; else an entailment maximum
≥ 0.60 returns true; all other cases return NEI at confidence 0.50.

Consequences:

- The strongest one-sided passage can control a verdict, but one opposing
  threshold-crossing passage forces NEI even if another passage is much stronger.
- A lone contradiction at 0.55 is not enough for false; one at 0.61 is. The
  entailment threshold is symmetrically 0.60.
- Repeated weak evidence does not accumulate. Passage count, reranker score, source
  reliability, and source independence are not considered.
- NEI is a thresholded abstention, but not a calibrated selective-prediction
  mechanism. The neutral probability itself is not a decision threshold.
- Reported class probabilities use independent maxima that may come from different
  passages. They are not pooled calibrated probabilities; the NEI remainder can
  become negative in a synthetic but valid softmax-vector example.

This aggregation is not intrinsically false-directional: false and true use the
same 0.60 decisive threshold, and an opposing support/contradiction signal produces
NEI. But its “any contradiction plus max” structure is sensitive to individual
spurious contradiction passages; the false-heavy outcome is already present in
the NLI passage outputs. Among the 2,700 selected passages there were 1,145
contradiction-group outputs, 66 support-group outputs, and 1,489 neutral outputs.
Final decisions were 402 false, 16 true, 257 NEI, including 10 conflict-gated NEI
cases.

## 4. Controlled component ablations

All configurations use the exact same 675 claims, safe retrieval corpus,
`cross-encoder/nli-distilroberta-base`, evidence-as-premise/claim-as-hypothesis
ordering, and 160-token truncation. No thresholds or parameters were tuned on
the test set.

- **A — BM25 top four + NLI:** first four of the saved 15 BM25 candidates; existing
  verifier aggregation applied.
- **B — BM25 top 15 + reranker top four + NLI:** reranker applied to the same 15;
  existing verifier aggregation applied. This is recomputed using the saved
  per-claim candidate list and the same NLI checkpoint.
- **C — Pre-aggregation diagnostic:** report all four selected-passage NLI outputs
  in the saved per-claim trace; for aligned claim-level metrics, use only the top
  reranked passage's NLI argmax and do not apply production aggregation. This is an
  explicit diagnostic, not a proposed production method.
- **D — Actual complete default pipeline:** saved actual `FinGuardPipeline.run()`
  output, including query expansion, BM25, reranking, NLI, and aggregation.
  B and D are the same execution and must not be interpreted as independent
  systems.

| Configuration | Accuracy | Macro-F1 | Weighted-F1 | false F1 | true F1 | NEI F1 | Predictions false / true / NEI | NEI rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. BM25 top 4 + NLI, no reranker | 0.3941 | 0.2958 | 0.3238 | 0.6022 | 0.0299 | 0.2554 | 415 / 13 / 247 | 36.59% |
| B. BM25 + cross-encoder top 4 + NLI | 0.4030 | 0.3107 | 0.3383 | 0.6106 | 0.0517 | 0.2698 | 402 / 16 / 257 | 38.07% |
| C. Top reranked passage argmax diagnostic | 0.3319 | 0.2833 | 0.2981 | 0.5043 | 0.0641 | 0.2816 | 280 / 26 / 369 | 54.67% |
| D. Actual default pipeline | 0.4030 | 0.3107 | 0.3383 | 0.6106 | 0.0517 | 0.2698 | 402 / 16 / 257 | 38.07% |

**Confusion matrices** (rows gold, columns prediction; order false / true / NEI):

| Configuration | Matrix |
|---|---|
| A | `[[215, 5, 79], [130, 4, 121], [70, 4, 47]]` |
| B and D | `[[214, 7, 78], [120, 7, 128], [68, 2, 51]]` |
| C | `[[146, 12, 141], [87, 9, 159], [47, 5, 69]]` |

The reranker adds 0.89 percentage points of accuracy and 0.0148 macro-F1 over
no-reranker A under this exact downstream decision rule; true-class F1 rises from
0.0299 to 0.0517. It remains very poor at identifying true claims. Since there are
no human relevance judgments yet, the experiment supports only a **small
verification-metric improvement**, not a claim that the reranker improves evidence
quality. Configuration C also shows that taking one passage's raw argmax is not a
replacement for the current multi-passage rule.

## 5. Human relevance assessment

`human_relevance_review.csv` contains 500 rows: five top reranked passages for each
of 100 claims, stratified as 34 false, 33 true, and 33 NEI. Each row has claim ID,
claim, passage, source URL, BM25 score, reranker score, heuristic entity/metric/
number/period match fields, and blank human relevance, support/contradiction/neutral,
and reviewer-notes fields.

The gold label is present in its separate `gold_label_reviewer_only` column, as
requested for reviewer access. It was never exposed to the retrieval pipeline.
Neither human relevance nor human stance labels were automatically filled in. The
sheet is ready for annotation; consequently no human relevance rate or causal
reranker-quality conclusion is claimed in this report.

## 6. Answers and recommended next action

1. **Is the NLI implementation technically correct?** The checkpoint's config
   mapping, pair input order, tokenization, softmax, and semantic-to-verdict
   mapping are technically consistent. Truncation occurs for 5.26% of the selected
   pairs. Correct wiring does not establish that this NLI model is reliable on
   retrieved financial/article text.
2. **Is the model unsuitable, or is evidence poor?** Evidence quality is a
   substantial demonstrated problem: the manual error review found 44/50 reviewed
   errors primarily about irrelevant evidence, wrong entities, or insufficient
   evidence; 243/255 true claims had no support-threshold passage in the saved
   top four. The NLI checkpoint also makes questionable contradiction decisions
   on related-but-not-equivalent financial assertions. The evidence supports a
   combined retrieval + domain-inference failure, not a claim that either component
   alone is universally unsuitable.
3. **Does reranking improve evidence quality?** Not established. It changes the
   top-four membership on 588 claims and slightly improves downstream verification
   metrics, but the human review sheet is unannotated. Only human judgments can
   establish relevance/stance quality independently of the evaluated cross-encoder.
4. **Does aggregation introduce false-verdict bias?** Its decisive thresholds are
   symmetric; it is not directly false-biased. It can allow one strong contradiction
   to dominate a one-sided decision and has a broad conflict-to-NEI gate. The large
   false/true imbalance is visible upstream in passage predictions (1,145
   contradiction vs 66 entailment), so do not attribute the collapse to aggregation
   alone.
5. **Which stage contributes most to the performance decline?** The strongest
   evidence points to retrieval/reranking producing irrelevant or insufficient
   passages, followed by NLI misinterpreting some selected passages. Aggregation
   affects borderline/conflicting cases but cannot make missing relevant evidence
   appear. The diagnostics are consistent with that ordering, while exact causal
   attribution still requires labeled evidence and component-level judgments.
6. **Smallest defensible next improvement?** Complete and adjudicate the
   500-passage human relevance/stance sheet, blind annotators to the gold claim
   label where practical, and use disagreements for adjudication. This measures
   whether retrieval or the NLI decision is the bottleneck without changing
   production thresholds or fine-tuning. After that, a narrow experiment-only
   retrieval filter or abstention rule can be tested using grouped validation only,
   then evaluated once on the held-out test set. Do not select a new threshold
   using these test labels.

## Artifacts

Investigation outputs: `outputs/experiments/root_cause_investigation/`

- `nli_sanity_tests.json` — raw logits, labels, probabilities, and token lengths for
  six controlled probes.
- `nli_label_mapping_audit.json` — checkpoint config and semantic mapping.
- `retrieval_error_analysis.csv` — 675 claims with all BM25/reranker passages,
  provenance, extracted claim fields, and screening flags.
- `aggregation_analysis.md` — exact decision trace and synthetic examples.
- `ablation_results.json` — metrics, per-class results, confusion matrices, and
  configuration definitions.
- `human_relevance_review.csv` — 500 passage rows awaiting human annotation.
- `experiment.log` — execution and artifact summary.

No production code, existing models, original datasets, or frontend files were
modified. No model was fine-tuned.
