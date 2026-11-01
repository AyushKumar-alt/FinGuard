# Aligned Grouped FinGuard Verification Benchmark: Methodology

## Study design

This is a fixed-population, source-grouped comparison using `data/grouped_train.csv`,
`data/grouped_val.csv`, and `data/grouped_test.csv`. The test set contains 675 claims.
The runner verifies disjoint claim IDs, normalized source URLs, and normalized claim
texts across all three splits before training or inference. All three systems use the
same test IDs and gold labels; failures remain explicit rows and are not omitted.

Random seeds are set to 42 for Python, NumPy, PyTorch, and the logistic-regression
estimator. The grouped split is consumed as stored and is not regenerated.

## Systems

**A — Claim-only classical baseline.** A scikit-learn pipeline fits a TF-IDF
word-unigram/bigram vectorizer (maximum 10,000 features, English stop words,
sublinear term frequency) and class-balanced logistic regression on grouped training
claims only. `C` is selected by validation macro-F1 over `[0.1, 0.5, 1, 2, 5]`.
The validation set is not added to model fitting, and the test set is not used for
selection.

**B — Evidence-grounded pretrained NLI.** The stated method is BM25 retrieval of 15
candidate evidence sentences, query expansion, cross-encoder reranking to four
passages, then the existing pretrained NLI verifier and its existing aggregation.
The runner records every retrieved candidate and BM25 score, every reranked passage
and reranker score, per-passage NLI scores/groups, the verdict, and confidence.

**C — Default FinGuard pipeline.** Each claim is passed to the actual
`FinGuardPipeline.run(claim)` without overrides. In `src/pipeline.py`, the method
defaults are BM25, query expansion enabled, reranking enabled, NLI enabled, 15
retrieved candidates, and four reranked passages. The same BM25 branch, reranker,
and verifier are called as specified for System B. Therefore B and C are
functionally identical: one actual default pipeline execution per claim is recorded
under both labels, with C marked as a reused result. They are not independent
systems or independent trials.

The reranker is instrumented only to retain scores for all 15 inputs: its existing
`rerank` implementation scores and sorts the same candidates, and the wrapper returns
the requested top four to `FinGuardPipeline.run()`. An assertion checks that the
selected IDs and order are identical. No verdict or selected passage is recomputed
outside the pipeline.

## Retrieval corpus isolation

The production pipeline normally builds its corpus from all records in
`finfact.json`. For this experiment, an experiment-only JSON corpus is generated from
grouped training and validation record IDs. It contains only claim text and evidence
sentences/source links; labels, justifications, summaries, and all test records are
excluded. Evidence linked to any grouped-test article URL or citation URL is also
excluded. The runner asserts zero overlap between remaining corpus citation URLs and
the held-out source set. The production corpus, pipeline source, model artifacts,
datasets, and frontend are not modified.

The experiment harness disables only the optional dense-retriever object during
construction because the specified default BM25 branch never consults it. This
avoids building an unused dense index or touching the pre-existing
`outputs/dense_evidence_embeddings.npz`. The production `FinGuardPipeline.run()`
method, BM25 retriever, reranker, NLI verifier, and default arguments remain
unchanged.

## Metrics and missing outputs

Classification metrics use all 675 claims. Failed inferences are represented as a
fourth `FAILURE` prediction column in the confusion matrix and count as incorrect
for accuracy; they are not converted to NEI. Per-class metrics cover false, true,
and NEI. Coverage is successful predictions divided by 675; selective accuracy is
accuracy among successful calls. NEI output count is reported as abstentions.

For evidence systems, retrieval success means at least one selected passage meets
the verifier's existing three-token usability rule. This is an operational
availability measure, **not** evidence relevance or gold-evidence retrieval
accuracy. Support, contradiction, and conflict rates use the verifier's existing
per-passage score thresholds. The experiment does not use gold evidence to score
retrieval relevance.

The per-claim CSV preserves the complete evidence trace and any error. The
error-analysis CSV contains all incorrect/failed predictions and an explicit
triage category plus its basis. Fifty unique incorrect pipeline cases were manually
reviewed with their claims, labels, selected passages, and NLI outputs; matching B/C
rows are one execution. Remaining error rows use rule-based triage and are not
represented as human adjudications.

## Outputs

- `results.json` — split integrity, tuning, metrics, pipeline execution status, and
  equivalence trace.
- `per_claim_predictions.csv` — one row per claim per system.
- `confusion_matrices/` — CSV matrices with a failure column.
- `error_analysis.csv` — all wrong/failed predictions with captured evidence.
- `experiment.log` — execution progress, model initialization, and failures.
- `retrieval_corpus_train_val.json` — the sanitized, reproducible corpus input.
