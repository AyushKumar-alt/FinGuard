# FinGuard Aligned Grouped Verification Benchmark

## Executive result

On the same 675 source-grouped test claims, the claim-only TF-IDF/logistic-regression
baseline performed best: **55.56% accuracy, 0.4897 macro-F1**. The evidence-grounded
BM25 → cross-encoder → zero-shot NLI path achieved **40.30% accuracy, 0.3107
macro-F1**. The evidence pipeline did not improve verification over the claim-only
baseline in this experiment.

Systems B and C are **functionally identical**, not distinct systems: the specified
System B settings match every `FinGuardPipeline.run()` default. One actual default
pipeline call per test claim was therefore recorded once and reused under both
labels. This prevents presenting duplicated executions as separate evidence.

## Protocol and integrity

The runner consumed the stored grouped splits without regenerating them: 2,357 train,
337 validation, and 675 test rows. It verified no claim-ID, normalized claim-text,
or source-URL overlap between splits. The held-out labels were 299 false, 255 true,
and 121 NEI.

For retrieval, an experiment-only JSON corpus was built from grouped train and
validation records only. The pipeline received only claim text and evidence
sentences/source links: no labels, justifications, summaries, or test records were
included. The 3,853 normalized URLs associated with held-out test records and their
citations were excluded from the corpus; the remaining corpus had **zero** overlap
with those URLs. After sentence deduplication, the actual pipeline indexed 9,653
evidence passages from 13,088 retained evidence sentences and 11,976 citation URLs.
The sanitized corpus SHA-256 is recorded in `results.json`.

All 675 claims completed in each system with **zero failed or missing predictions**.
The per-claim file has 2,025 rows (675 per label), including all retrieved passages,
BM25 scores, reranker scores, selected evidence/provenance, NLI outputs, verdicts,
confidence, and status. B/C rows share an execution ID and are explicitly marked
as reused.

System A used claim text only. The vectorizer and class-balanced logistic regression
were fit on grouped training data. `C` was selected by grouped-validation macro-F1
from `[0.1, 0.5, 1, 2, 5]`; **C=0.5** was selected (validation accuracy 0.5905,
macro-F1 0.5211). The test set was not used for tuning.

System C invoked the actual `FinGuardPipeline.run(claim)` without overrides:
BM25, query expansion, top 15 retrieval, cross-encoder reranking to top 4, and the
pretrained `cross-encoder/nli-distilroberta-base` verifier. The actual reranker
and verifier were loaded. The unused optional dense-retrieval object was disabled
only in the experiment harness during construction; the default BM25 `run()` branch
does not consult it. The BM25 corpus, production pipeline method, reranker, and NLI
implementation were not modified. The reranker was instrumented to retain all 15
scores while returning the same top four; an ID/order assertion verified the
pipeline's selected passages were unchanged.

## Verification metrics

| System | Accuracy | Macro-F1 | Weighted-F1 | Coverage | NEI outputs |
|---|---:|---:|---:|---:|---:|
| A — claim-only TF-IDF + Logistic Regression | **0.5556** | **0.4897** | **0.5498** | 1.000 | 102 |
| B — evidence-grounded zero-shot NLI | 0.4030 | 0.3107 | 0.3383 | 1.000 | 257 |
| C — actual default FinGuard pipeline | 0.4030 | 0.3107 | 0.3383 | 1.000 | 257 |

Coverage is successful predictions divided by 675; selective accuracy equals
accuracy because there were no failed calls. NEI output counts are abstentions.
The confusion matrices below use rows as gold labels and columns as predictions,
ordered `false, true, NEI`; the explicit `FAILURE` column in the CSV matrices is
zero for all systems.

**Per-class precision / recall / F1**

| System | false P / R / F1 | true P / R / F1 | NEI P / R / F1 |
|---|---|---|---|
| A | 0.6018 / 0.6622 / 0.6306 | 0.6189 / 0.5922 / 0.6052 | 0.2549 / 0.2149 / 0.2332 |
| B and C | 0.5323 / 0.7157 / 0.6106 | 0.4375 / 0.0275 / 0.0517 | 0.1984 / 0.4215 / 0.2698 |

**Confusion matrices**

System A:

| Gold \ Predicted | false | true | NEI |
|---|---:|---:|---:|
| false | 198 | 59 | 42 |
| true | 70 | 151 | 34 |
| NEI | 61 | 34 | 26 |

Systems B and C (identical):

| Gold \ Predicted | false | true | NEI |
|---|---:|---:|---:|
| false | 214 | 7 | 78 |
| true | 120 | 7 | 128 |
| NEI | 68 | 2 | 51 |

Prediction distributions were A: 329 false, 244 true, 102 NEI; B/C: 402 false,
16 true, 257 NEI. In particular, B/C recovered only **7 of 255 true claims**
(true-class recall 2.75%).

## Evidence diagnostics

These are evidence availability and NLI-group metrics—not gold relevance or
retrieval-ranking metrics:

| Measure (675-claim denominator) | B and C |
|---|---:|
| At least one usable selected passage (3+ tokens) | 675 / 675 (100.00%) |
| At least one supporting passage (existing verifier threshold) | 34 / 675 (5.04%) |
| At least one contradicting passage | 447 / 675 (66.22%) |
| Conflicting support and contradiction groups | 10 / 675 (1.48%) |
| No usable selected evidence | 0 / 675 (0.00%) |

“Retrieval success” here means that a selected passage was available for verifier
input; it **does not** mean the passage was relevant or supported the gold label.
Gold-evidence relevance was not evaluated because no independent human relevance
labels were available and test evidence was excluded from the retrieval corpus.

## Error analysis

All incorrect or failed rows are retained in `error_analysis.csv`: 300 errors for A
and 403 each for B and C. Fifty unique incorrect B/C executions were manually
reviewed (17 false, 17 true, 16 NEI gold cases); duplicate C rows were not counted as
additional cases. The reviewed sample was categorized as:

| Primary category | Reviewed cases |
|---|---:|
| Irrelevant retrieval | 23 |
| Wrong entity | 11 |
| Insufficient evidence | 10 |
| Aggregation or threshold error | 1 |
| Wrong fiscal period | 2 |
| NLI neutral collapse | 1 |
| Wrong financial metric | 1 |
| Other (NLI interpretation) | 1 |

Thus, **44/50 reviewed errors** were principally evidence-selection/availability
problems (irrelevant passages, missing claim-specific evidence, or wrong entity).
For example, the “30 Marines” claim received passages about unrelated Marine deaths
and helicopter news; the Lukoil ownership claim received a Citgo ownership passage;
and several financial claims received superficially related budget/tax passages
that did not contain the claimed amount or period. In a separate case, asteroid
passages explicitly describing a safe pass were assigned to the NLI neutral group.
The selected evidence, source URLs, reranker scores, and sentence-level NLI outputs
are preserved for inspection.

This manually reviewed sample points to retrieval/reranking as the largest error
source; the NLI stage then sometimes assigns contradiction to semantically
misleading passages or neutral to directly relevant ones. The remaining rows have
rule-based diagnostic categories in the CSV/results and are not presented as
human-adjudicated classifications.

## Answers to the research questions

1. **Best on the unseen-source test:** System A, with 55.56% accuracy and 0.4897
   macro-F1, versus 40.30% and 0.3107 for B/C.
2. **Did evidence retrieval improve on claim-only classification?** No, not in this
   end-to-end comparison; accuracy fell by 15.26 percentage points and macro-F1 by
   0.1790. This comparison changes both the evidence input and the model/decision
   process, so it is not a causal isolation of retrieval alone.
3. **Did NLI contribute useful verification?** Not as configured here: true-class
   recall was 2.75%, only 16 predictions were true, and macro-F1 was 0.3107. This
   does not establish that every NLI model is ineffective; it shows this
   pretrained model plus current retrieval/aggregation did not help on this test.
4. **Largest error source:** Retrieval/reranking. Irrelevant or insufficient
   passages and entity mismatches accounted for 44 of the 50 manually reviewed
   errors.
5. **What is scientifically supported?** On this fixed, source-grouped,
   leakage-controlled 675-claim test, the tuned claim-only logistic-regression
   baseline outperformed the default evidence-grounded NLI pipeline. The default
   pipeline execution is operational across all claims, but its evidence did not
   yield stronger verification. B and C are one functionally identical method, not
   two independent systems.

**What remains unproven:** These results do not show that retrieval itself caused
the performance gap, that another evidence corpus or NLI model would perform
similarly, or that the system generalizes beyond this stored split. The evidence
retrieval success rate is not a relevance score; no independent human gold
retrieval benchmark was used. Component-level ablations, adjudicated evidence
relevance, confidence calibration, and repeated source-grouped splits remain
necessary before making broader claims about evidence-grounded financial
verification.

## Artifacts

Experiment outputs: `outputs/experiments/aligned_grouped_verification/`

- `results.json`
- `per_claim_predictions.csv`
- `confusion_matrices/`
- `error_analysis.csv`
- `experiment.log`
- `methodology.md`
- `retrieval_corpus_train_val.json` (sanitized corpus input)

Runner: `src/run_aligned_grouped_verification.py`
