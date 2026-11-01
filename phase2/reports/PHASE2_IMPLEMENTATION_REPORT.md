# FinGuard Phase 2: Implementation and Demonstration Report

**Status:** Core Python claim-verification path executed locally; optional retrieval routing fixes covered by regression tests. This is an implementation report, not a new benchmark.

## 1. Entry points and recommended demonstration

**Reliable demonstration path:** From repository root, pipe one JSON payload to `python -m src.api_bridge verify`. The API bridge calls `handle_verify()` → `get_pipeline()` → `FinGuardPipeline.run()`. The verified default uses BM25, query expansion, cross-encoder reranking, and pretrained NLI with 15 candidates and 4 selected passages.

The tested environment has the required Hugging Face model weights cached and a valid `outputs/dense_evidence_embeddings.npz`; on a clean environment, the first model load needs internet access to download the checkpoints. Running on CPU is supported but slower.

Other entry points:

| Entry point | Purpose | Status |
|---|---|---|
| `python -m src.api_bridge verify` | One-shot JSON/CLI verification | **Operational**; tested end-to-end. |
| `python -m src.api_bridge serve --port 8000` | Python HTTP API | Implemented; not separately load-tested as a service in this pass. |
| `frontend/app/api/verify/route.ts` | Tries local Python HTTP API, then Python child-process bridge | The misleading archive-label fallback is removed; on bridge failure it returns HTTP 503. TypeScript type-check and Next production build passed; route runtime not verified. |
| `app.py` | Streamlit research dashboard | **Unsupported for tomorrow's demo:** initialization/call sites mismatch current positional-index/parser APIs and expected claim/query fields. It was not repaired because a reliable alternative is the CLI, and additional UI issues remain. |
| `src/run_aligned_grouped_verification.py` | Saved grouped evaluation protocol | Research runner; not run during this release task. Do not use tomorrow as an interactive demo. |

## 2. Phase 2 components and actual call chain

Default flow:

`API bridge handle_verify()` → `FinGuardPipeline.run()` → `extract_financial_entities()` → `FinancialQueryBuilder.build_queries()` → `EvidenceRetrievalBenchmark.retrieve_bm25()` → `PretrainedCrossEncoderReranker.rerank()` → `FinancialClaimVerifier.verify_claim()` → `predict_nli_single_pair()` for each selected passage → threshold/conflict aggregation → response with evidence provenance.

| Component | Purpose / algorithm | Input → output | Implementation and caller | Status / why used / limitation |
|---|---|---|---|---|
| Claim understanding | Regexes, finance vocabulary, capitalization and organization patterns; no learned NER. | Claim → entities, metrics, amounts, periods, direction and negation cues. | `src/claim_understanding.py::extract_financial_entities`; called by `src/pipeline.py::FinGuardPipeline.run`. | **Operational heuristic.** Lightweight and interpretable; can misidentify organizations or miss scope. |
| Query construction | Deterministic metric synonym and entity/value/period expansion; retains the original claim. | Parsed claim → combined lexical/semantic search string. | `src/query_construction.py::FinancialQueryBuilder.build_queries`; called by `run`. | **Operational default.** No learned query rewriting; expansion benefit is not isolated on an uncontaminated retrieval evaluation. |
| BM25 | Okapi term-frequency scoring with document-length normalization and RSJ IDF; inverted postings are constructed in repository code. | Claim query → ranked evidence IDs and scores. | `src/retrieval.py::BM25Okapi`, `EvidenceRetrievalBenchmark.retrieve_bm25`; default branch in `run`. | **Default and operational.** Fast and lexical; can miss paraphrases and the production corpus includes all FinFact evidence. |
| TF-IDF | Sparse unigram/bigram vectors with cosine similarity. | Query/corpus → ranked evidence IDs. | `src/retrieval.py::retrieve_tfidf`; `run` optional `tfidf` mode. | **Optional, operational.** Lexical comparator; vocabulary/word overlap limitations. |
| Dense retrieval | Mean-pooled, L2-normalized MiniLM embeddings; cosine is computed by inner product, with cached corpus vectors. | Query → ranked evidence IDs. | `src/dense_retrieval.py::PretrainedDenseRetriever`; optional `dense` and hybrid branches in `run`. | **Optional; real path smoke-tested.** Semantic matching can retrieve related but non-probative passages; pretrained, not FinFact fine-tuned. |
| BM25 + dense RRF | Reciprocal Rank Fusion adds `1/(k + rank)` from BM25 and dense lists (`k=60`). | Two ranked lists → fused candidate ranking. | `src/pipeline.py::_reciprocal_rank_fusion`; `run(retrieval_mode="hybrid_bm25_dense")`. | **Routing bug fixed; operational path smoke-tested.** It fuses ranks, not raw scores; no relevance guarantee. |
| BM25 + TF-IDF RRF | Rank fusion of the two lexical retrievers. | Ranked lists → fused ranking. | `src/retrieval.py::retrieve_hybrid_rrf`; `run` accepts `hybrid`, `rrf`, `hybrid_rrf`. | **Optional.** Does not use dense vectors. |
| SVD / LSA | Truncated SVD projection of TF-IDF matrix (128 components), with normalized query/document projections. | Query/corpus → ranked evidence IDs. | `src/retrieval.py::retrieve_dense`; now reached by `run` for `svd`, `dense_svd`, `lsa`. | **Optional; routing fixed and unit-tested.** This is not the MiniLM dense model; historical retrieval scores were weak. |
| P1 lemmatized retrieval | Separate financial tokenizer, custom stopwords, POS/WordNet lemmatization and TF-IDF retrieval over P1 index. | Claim query → ranked P1 passages. | `src/p1_pipeline.py::Pipeline1.retrieve`; optional `p1_lemma` / `lemma_p1` branch in `run`. | **Optional/experimental.** Separate corpus and index; failure falls back to BM25. |
| Cross-encoder reranker | Jointly encodes claim-passage pairs and applies sigmoid to one relevance logit. | Top retrieved candidates → top reranked passages with scores. | `src/reranker.py::PretrainedCrossEncoderReranker.rerank`; called by `run`. | **Default when available.** MS MARCO model is a relevance ranker, not truth classifier; unavailable mode is passthrough. |
| NLI verifier | Pretrained cross-encoder classifies evidence premise vs. claim hypothesis into contradiction, entailment, neutral using model `id2label`. | Each selected passage + claim → NLI scores and support/contradiction/neutral group. | `src/verifier.py::FinancialClaimVerifier.predict_nli_single_pair`, called by `verify_claim`. | **Default when available.** Zero-shot checkpoint, not FinFact fine-tuned; journalistic phrasing/domain mismatch and poor true-class recall. |
| Evidence aggregation | Per-passage thresholds (0.50 to group support/contradiction; 0.60 for decisive verdict); mixed support and contradiction gates to NEI. | Passage-level NLI scores → True/False/NEI, confidence and evidence lists. | `src/verifier.py::FinancialClaimVerifier.verify_claim`. | **Operational heuristic.** Thresholds are uncalibrated. Reported class scores now normalize maxima of per-class evidence scores and are not calibrated probabilities. |
| TF-IDF + Logistic Regression | Supervised sparse-text baseline trained on FinFact. | Claim/evidence text → class label. | `src/baseline_models.py`; saved model `outputs/best_baseline_model.pkl`; fallback when NLI is disabled/unavailable. | **Implemented trained baseline.** It is not necessarily evidence-grounded; performance depends on split and possible source/label cues. |
| Positional Boolean/phrase retrieval | Term postings with token offsets; Boolean set operations and exact adjacent phrase matching. | Boolean/quoted query → matching evidence IDs. | `src/positional_index.py`; UI caller in `app.py`. | **Implemented separately, not default verification.** The current Streamlit constructor/parser calls do not match these APIs; unsupported UI. |

## 3. Model and artifact provenance

| Component | Model/checkpoint | Provenance |
|---|---|---|
| Dense encoder | `sentence-transformers/all-MiniLM-L6-v2` | Pretrained Hugging Face model; no repository evidence of FinFact fine-tuning. Cached embeddings at `outputs/dense_evidence_embeddings.npz` are vectors, not model weights. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pretrained relevance model; no FinFact fine-tuning established. |
| NLI verifier | `cross-encoder/nli-distilroberta-base` | Pretrained NLI model; runtime reads checkpoint label mapping. No FinFact fine-tuning established. |
| Classical baseline | TF-IDF + balanced Logistic Regression (saved pickle) | Supervised FinFact baseline; `outputs/best_baseline_model.pkl`. The pickle's provenance is not fully tied to every historical metrics artifact. |
| BM25, query builder, claim parser, aggregation | Repository implementations | Hand-coded/standard-library and scikit-learn components; not pretrained neural models. |

## 4. Fixes made in this release

1. **Hybrid routing:** the former broad `"dense"` test captured `"hybrid_bm25_dense"` first, so the dense-only branch ran. Exact mode dispatch now reaches BM25+dense RRF; if dense retrieval is unavailable, the mode reports an explicit BM25 fallback in `retrieval_engine`.
2. **SVD routing:** `svd`, `dense_svd`, and `lsa` now dispatch to the existing TF-IDF/SVD `retrieve_dense()` implementation. These names do not invoke MiniLM.
3. **Unknown modes:** unsupported names now raise `ValueError` rather than silently appearing to run BM25.
4. **Probability integrity:** per-class NLI maxima could exceed 1 in combination, producing a negative residual NEI score. Decisive verdict outputs now normalize the three non-negative class-wise maxima. This is a descriptive heuristic score, **not calibration**; thresholds and verdict logic were not tuned.
5. **Frontend archive fallback:** the frontend no longer returns archived FinFact labels, fabricated confidence, or synthetic evidence when Python fails; it returns HTTP 503 with an error message.

Targeted tests in `tests/test_pipeline_retrieval_modes.py` cover fused retrieval, unavailable-dense fallback, SVD dispatch, and rejection of unknown modes. `tests/test_verifier.py` covers the negative-probability regression.

## 5. Verification performed

- Before implementation edits: `python -m unittest discover tests` — **95 tests passed**.
- After routing edits and before probability fix: full suite — **99 tests passed**.
- After probability fix: `python -m unittest discover -s tests -p 'test_verifier.py'` — **4 tests passed**.
- Final `python -m unittest discover tests` after all Python changes — **100 tests passed** in 112.3 seconds.
- `python -m py_compile src\pipeline.py src\verifier.py tests\test_pipeline_retrieval_modes.py tests\test_verifier.py` — passed.
- `frontend\node_modules\.bin\tsc.cmd --noEmit --pretty false` — passed.
- `npm --prefix frontend run build` — passed; Next.js compiled and generated production route/page output. The build itself reports that type validation and linting were skipped; TypeScript was checked separately.
- `npm --prefix frontend run lint` was attempted but opens an interactive ESLint configuration prompt and exits without linting; no config was created.
- `npx tsc` / `npm exec --prefix frontend -- tsc` were attempted from the wrong invocation context and did not type-check. The project-local compiler command above is the successful check.
- Final CLI command tested (PowerShell here-string piped to `python -m src.api_bridge verify`) with training claim ID 1351, BM25, top-15/top-4: loaded BM25, cached MiniLM vectors, MS MARCO reranker, and NLI; retrieved 15, reranked 4, used NLI, retained source URLs, returned `true`. `class_probabilities` were `{false: 0.1619, true: 0.4304, NEI: 0.4076}`; all were in `[0,1]` and summed to 0.9999. The decisive `confidence` (0.8741) is the maximum evidence score and is distinct from this normalized uncalibrated distribution.
- Separate real `hybrid_bm25_dense` run on the same training claim observed `retrieval_engine=hybrid_bm25_dense_rrf`, 15 candidates, 4 reranked passages and active NLI.
- Real SVD-mode smoke with heavy models disabled returned `retrieval_engine=tfidf_svd` and five candidates; the saved logistic-regression verifier was used as fallback.
- Three default-pipeline smoke claims from the training split returned `true`, `false`, and `false`; the third source record is NEI. These are demonstrations only, not evaluation, and expose an observed classification error. No thresholds were changed.

The first PowerShell invocation using `--input` failed JSON parsing due to native argument quoting; piping JSON through stdin succeeded and is the documented command. No new dataset, model, or benchmark was created, and no historical metric file was edited.

## 6. Historical evaluation (not reproduced here)

The source-grouped aligned artifact `outputs/experiments/aligned_grouped_verification/results.json` evaluates 675 held-out grouped claims (299 false, 255 true, 121 NEI) on the same split:

| System | Accuracy | Macro-F1 | True recall |
|---|---:|---:|---:|
| Claim-only TF-IDF + Logistic Regression | 0.5556 | 0.4897 | 0.5922 |
| BM25 → cross-encoder → pretrained NLI | 0.4030 | 0.3107 | 0.0275 |

The evidence pipeline predicted only 16 of 675 claims as true. This is the strongest aligned comparison in the repository and does **not** show an improvement over the claim-only classifier.

Other historical artifacts are not interchangeable:

- `outputs/retrieval_metrics.json`: 150-query reference-evidence ranking suite; it uses a corpus built from the full FinFact archive, including held-out claims' own reference sentences. It is not an unbiased held-out retrieval result.
- `outputs/verification_metrics.json`: includes older 100-claim zero-shot NLI and pipeline results and 506-claim classifier results with different protocols. Do not compare unlike sample sizes/splits as one model comparison.
- Random-split classifier results have article/citation source overlap; raw evidence may include verdict-related phrasing. Do not use the random split as proof of out-of-source generalization.

No experiment or historical metrics were rerun/edited during this release task. The smoke claims are not a benchmark.

## 7. Known limitations and presentation wording

- The default corpus is a closed FinFact evidence archive, not live web search; its all-record construction can expose held-out reference evidence in naïve evaluation.
- The NLI model has not been FinFact-fine-tuned, and the 675-claim grouped result has only 2.75% recall for true claims.
- NLI aggregation thresholds are heuristics, confidence is not calibrated, and numerical arithmetic/currency normalization is absent.
- Query/entity parsing is rule-based and can produce bad spans, as observed in the smoke claim.
- Streamlit is not a reliable presentation entry point until its positional-index constructor, Boolean parser, and parsed/query result field mismatches are resolved and tested.
- The frontend route is type-checked and the Next.js production build passed, but the route was not runtime-tested against a live Python daemon. If Python cannot execute, it now reports service unavailability rather than a stored FinFact verdict.
- Repository metadata has no `.git` directory at the supplied project path. `git status` could not run, so no complete Git baseline/diff can be certified.
- Files intentionally changed in this task: `README.md`, `src/pipeline.py`, `src/verifier.py`, `frontend/app/api/verify/route.ts`, `tests/test_pipeline_retrieval_modes.py`, `tests/test_verifier.py`, and this report. Validation refreshed Python bytecode under `src/__pycache__/` and `tests/__pycache__/`, `frontend/tsconfig.tsbuildinfo`, and generated/updated frontend build artifacts under `frontend/.next/`. These are generated runtime/build outputs; no experiment result, dataset, split, or model artifact was intentionally edited.

Suggested presentation claim: “FinGuard is a research prototype that combines lexical/dense evidence retrieval, neural reranking, and pretrained NLI; components execute locally, but the current grouped evaluation does not show evidence-pipeline gains over the claim-only baseline.”

## 8. Likely viva questions and technically accurate answers

1. **What is the default pipeline?** Rule/regex claim understanding, deterministic expansion, BM25 over an evidence-sentence archive, a pretrained cross-encoder reranker, then passage–claim NLI and threshold/conflict aggregation.
2. **Is BM25 dense retrieval?** No. BM25 is lexical term-frequency ranking with IDF and length normalization. MiniLM separately embeds query/passages for cosine similarity.
3. **What does RRF do?** It sums reciprocal rank contributions, `1/(k+r)`, across two ranked lists; it does not combine raw BM25 and cosine scores or guarantee relevant evidence.
4. **How is the cross-encoder different from retrieval?** Retrieval cheaply generates candidates; the cross-encoder sees claim and passage jointly and reorders a limited candidate list. Its relevance score is not a truth score.
5. **How does verification work?** Evidence is the NLI premise and the claim is the hypothesis. Per-passage entailment/contradiction/neutral scores are grouped; conflicting support and contradiction yield NEI, and thresholds select decisive labels.
6. **Were the transformer models fine-tuned on FinFact?** No such training is established. The three checkpoint names are pretrained Hugging Face models; the saved TF-IDF/logistic-regression baseline is the supervised FinFact component.
7. **Does a retrieval hit prove the claim?** No. Relevance ranking, evidential stance, and final verification are distinct tasks.
8. **Does FinGuard search the web?** No. The production corpus is built from stored FinFact evidence sentences and source metadata.
9. **What did the grouped evaluation show?** On the same 675 claims, claim-only LR scored 55.56% accuracy / 0.4897 macro-F1, while the evidence pipeline scored 40.30% / 0.3107 and 2.75% true recall.
10. **Why is the random split weaker evidence?** Claims can share fact-check article/citation sources across splits; the grouped protocol is more demanding, though it still does not guarantee every possible source or semantic duplicate is isolated.
11. **What is the status of dense hybrid?** The routing bug is fixed; a real local run confirmed BM25+dense RRF, then reranking and NLI. The ranking method remains optional, not default.
12. **What is SVD mode?** It projects TF-IDF vectors into a 128-dimensional truncated-SVD space and ranks by cosine; it is separate from transformer embeddings.
13. **What are the class probabilities?** They are normalized heuristic aggregate scores, not calibrated posterior probabilities. Calibration and threshold tuning are not part of this release.
14. **Why not demo Streamlit?** Its current calls do not match positional-index/parser APIs and module result fields; it needs separate repair and runtime validation. Use the Python bridge CLI.
15. **What happens when the backend fails in the frontend?** The route now returns HTTP 503; it no longer looks up a benchmark claim and presents its stored label as a fresh model prediction.
