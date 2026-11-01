# FinGuard NLP System: Technical Methodology Audit

**Scope:** Repository-grounded review of the Python system, user-facing request paths, data, saved evaluations, and research documentation.  
**Integrity note:** No implementation, model, dataset, configuration, or frontend file was changed. This report replaces a pre-existing audit whose descriptions did not consistently match the code or current saved experiments.

## 1. Executive summary

FinGuard contains two different kinds of system:

1. **The default Python verification pipeline** in `src/pipeline.py`: deterministic claim parsing and query expansion → BM25 retrieval → optional cross-encoder reranking → pretrained NLI classification and hand-written evidence aggregation. It is an evidence-grounded prototype over the FinFact evidence archive, not open-web retrieval.
2. **Alternative paths and experiments:** TF-IDF, dense retrieval, RRF, a lemmatized Pipeline 1, positional Boolean/phrase search, and trained TF-IDF classifiers. These do not all participate in the default verification call. The API/UI has additional fallbacks; notably, the Next.js `/api/verify` route can return a stored FinFact label after a claim-text match if the Python calls fail. That is an archive lookup, not independent verification.

The pretrained models are used for inference, not fine-tuned on FinFact: `sentence-transformers/all-MiniLM-L6-v2` (dense encoder), `cross-encoder/ms-marco-MiniLM-L-6-v2` (relevance reranker), and `cross-encoder/nli-distilroberta-base` (NLI). A scikit-learn TF-IDF/logistic-regression classifier was trained on FinFact; it is a separate baseline and a fallback engine, not a transformer verifier.

There are no authoritative release records defining historical Phase 1 and Phase 2 boundaries. The mapping below is a defensible organization of existing code, not a claim about chronology. The most useful current end-to-end result is the source-grouped 675-claim run: the claim-only logistic-regression system scored **55.56% accuracy / 0.4897 macro-F1**, while the default evidence pipeline scored **40.30% / 0.3107**. The latter predicted only 16 of 675 claims as true and had **2.75% recall on the 255 true claims**. This experiment does not establish that retrieval alone caused the gap, but it does not support a claim that the current evidence pipeline improves verification.

The older 150-query retrieval comparison and 75-claim pooled benchmark report retrieval ranking, not truth verification. The former searches a corpus built from the full dataset, including held-out claims’ annotated evidence. The latter has mostly silver reference evidence and automated negative judgments, not independent human gold relevance labels. Results from these protocols must not be mixed.

**Finding labels used throughout:** **VERIFIED** = directly supported by inspected code or a saved artifact; **INFERRED** = interpretation of verified behavior/results, not a direct repository assertion; **UNRESOLVED** = evidence is missing, conflicting, or insufficient. Metrics copied from saved outputs are historical results, not independently reproduced here. “Audit calculation” identifies a read-only count/overlap calculation made from repository data.

## 2. End-to-end architecture and request flow

### User-facing request paths

- **VERIFIED — Next.js frontend:** `frontend/lib/api.ts::verifyClaim()` POSTs to `frontend/app/api/verify/route.ts::POST()`. The route first tries the local Python HTTP service at `127.0.0.1:8000/verify`, then invokes `python -m src.api_bridge verify --file …`. If both fail, it searches `finfact.json` for a claim that contains or is contained by the submitted claim. A match returns the record’s stored label, selected evidence, and hard-coded confidence `0.91`; it is not model inference or independent verification. If no match exists, it returns NEI. Evidence: `frontend/lib/api.ts#L18-L26`; `frontend/app/api/verify/route.ts#L63-L73`, `#L127-L158`.
- **VERIFIED — Python API bridge:** `src/api_bridge.py` routes `/verify` and its CLI equivalent into `FinGuardPipeline.run()`. Default arguments are BM25, query expansion on, reranking on, NLI on, 15 retrieved candidates and 4 selected passages. See `src/pipeline.py#L68-L74` and `src/api_bridge.py` request handlers.
- **VERIFIED — Streamlit `app.py`:** Several displayed controls call incompatible interfaces: it passes `index_path=` to `PositionalInvertedIndex`, calls `parse_and_execute()` where `BooleanQueryParser` exposes `evaluate()`, and reads parsed/query keys not returned by the called modules. Evidence: `app.py#L120`, `#L418`; `src/positional_index.py::PositionalInvertedIndex`, `BooleanQueryParser.evaluate()`; returned keys in `src/claim_understanding.py::extract_financial_entities()` and `src/query_construction.py::FinancialQueryBuilder.build_queries()`. **INFERRED:** These paths are not operational as written; no UI runtime test was performed.

### Default Python pipeline

| Stage | Problem / input | VERIFIED algorithm and call chain | Output / next consumer | Status |
|---|---|---|---|---|
| Claim input | User supplies a claim string. | `src/api_bridge.py::handle_verify()` → `FinGuardPipeline.run()`; empty input returns NEI (`src/api_bridge.py#L28-L45`; `src/pipeline.py#L68-L91`). | Nonempty string to claim understanding. | VERIFIED: implemented. |
| Claim understanding | Identify lexical anchors for retrieval. | `run()` → `extract_financial_entities()` → `finance_aware_tokenize()`; regexes, vocabularies, capitalization and finite organization lexicon; no learned NER (`src/pipeline.py#L92-L93`; `src/claim_understanding.py#L69-L130`). | Structured fields to query builder. | VERIFIED: implemented heuristic. UNRESOLVED: extraction accuracy not benchmarked. |
| Query construction | Improve retrieval wording while retaining claim terms. | `run()` → `FinancialQueryBuilder.build_queries()`; fixed synonym/anchor rules, not generative rewriting (`src/pipeline.py#L95-L98`; `src/query_construction.py#L44-L130`). | Combined query to retriever. | VERIFIED: implemented. UNRESOLVED: benefit not isolated on aligned test. |
| Candidate retrieval | Find archive passages. | `run()` → `EvidenceRetrievalBenchmark.retrieve_bm25()` over corpus built by `_build_corpus()` (`src/pipeline.py#L99-L131`; `src/retrieval.py#L79-L186`). | Up to 15 `(doc_id, score)` items; text, URLs, parent IDs attached. | VERIFIED: default-executed; full-data corpus contaminates old held-out retrieval tests. |
| Reranking | Order a small candidate set using joint claim/passage semantics. | `run()` → `PretrainedCrossEncoderReranker.rerank()` (`src/pipeline.py#L158-L164`; `src/reranker.py#L17-L75`). | Top 4 to verifier; passthrough first four if unavailable/disabled. | VERIFIED: implemented. INFERRED: score is not a truth score or calibrated relevance. |
| Verification | Determine passage relation to claim. | `run()` → `FinancialClaimVerifier.verify_claim()` → `predict_nli_single_pair()` with evidence premise and claim hypothesis, 160-token truncation (`src/pipeline.py#L166-L172`; `src/verifier.py#L70-L108`, `#L125-L230`). | Per-passage NLI outputs; baseline fallback when NLI disabled/unavailable. | VERIFIED: implemented; pretrained, not FinFact fine-tuned. |
| Aggregation | Convert passage outputs to verdict. | `verify_claim()` groups contradiction at ≥.50, else support at ≥.50; mixed support/contradiction → NEI; otherwise max contradiction ≥.60 → false, max entailment ≥.60 → true, else NEI (`src/verifier.py#L173-L230`). | Verdict, confidence, evidence groups, uncertainty/conflict flags. | VERIFIED: implemented heuristic. UNRESOLVED: thresholds/confidences uncalibrated. |

`src/pipeline.py` builds its main corpus/index at initialization. Evidence provenance is carried as metadata; the default run does **not** exclude passages whose `parent_claim_ids` include the current claim. There is no live external-source search, independent source-quality scoring, or arithmetic verification stage.

### Alternative retrieval and fallback branches

- `retrieval_mode="tfidf"` uses the optional TF-IDF retriever.
- `retrieval_mode="dense"` uses dense retrieval when the encoder is available.
- The code intends `"hybrid_bm25_dense"` to fuse BM25 and dense rankings using RRF with `k=60`. However, `if "dense" in used_mode` precedes that condition, so `"hybrid_bm25_dense"` enters the dense-only branch. Generic `"hybrid"`/`"rrf"` invokes `retrieve_hybrid_rrf()`, which fuses BM25 and TF-IDF, not BM25 and dense.
- `p1_lemma`/`lemma_p1` invokes `src.p1_pipeline.Pipeline1`; failures are caught and fall back to BM25. `"svd"` has no pipeline branch and falls through to BM25.
- `use_reranker=False` or unavailable reranker selects the first `top_k_rerank` candidates without reranking.
- `use_nli=False` or unavailable NLI selects the baseline classifier, if loaded. Missing evidence produces NEI before either verifier is invoked.
- The positional Boolean and exact-phrase search in `src/positional_index.py` is a separate search interface, not a stage in the default claim-verification chain. General proximity (`NEAR/k`) search is absent.

## 3. Phase 1 methodology — defensible code-based mapping

No phase manifest or source-control history was used to establish a historical boundary. Sections 3 and 4 are therefore a **methodology mapping**, not a claim about chronology.

**VERIFIED:** This is a code-based grouping, not a historically recorded milestone.

### Classical and positional retrieval foundation

- **Objective:** Represent evidence lexically and provide traditional retrieval/classification baselines.
- **Methods/files:** Finance-aware tokenization in `src/claim_understanding.py`; custom positional postings, Boolean operators and exact phrase matching in `src/positional_index.py`; the separate lemmatized path in `src/p1_tokenizer.py`, `p1_stopwords.py`, `p1_lemmatizer.py`, `p1_corpus_builder.py`, `p1_index.py`, `p1_retriever.py` and `p1_pipeline.py`; TF-IDF/BM25/SVD implementations in `src/retrieval.py`; classification baselines in `src/baseline_models.py`.
- **Input/output:** Claim or Boolean query plus a corpus derived from evidence; returns ranked or matching evidence IDs, or a class prediction from concatenated claim/evidence text.
- **Evidence of execution/evaluation:** The P1 positional-index artifact reports 23,568 indexed units, 32,845 vocabulary terms and 393,323 positional postings. Baselines have saved random and source-grouped metrics. Boolean/phrase tests establish functional behavior, not retrieval effectiveness on a judged query set.
- **Limitations:** The P1 lemmatized retriever is optional, not default. Index construction and corpus definition differ from the main retrieval corpus. The positional UI calls are inconsistent with the exposed constructors/methods.

## 4. Phase 2 methodology — defensible code-based mapping

**VERIFIED:** This is a code-based grouping, not a historically recorded milestone.

### Expanded retrieval, neural ranking and NLI verification

- **Objective:** Move from lexical retrieval toward semantic retrieval and evidence-conditioned classification.
- **Methods/files:** Rule-based claim understanding and query expansion; BM25 default retrieval; optional dense encoder and RRF; MS MARCO cross-encoder reranker; pretrained NLI verifier and multi-passage gating in `src/pipeline.py`, `dense_retrieval.py`, `reranker.py`, and `verifier.py`.
- **Input/output:** Claim text → expanded query → candidate evidence → reranked passages → True/False/NEI verdict with provenance.
- **Experiments:** Older 150-query retrieval suite; 100-example historical NLI/full-pipeline evaluation; later leakage-controlled 675-claim aligned grouped verification and component ablations.
- **Assessment:** Phase 2 adds substantial implemented machinery, but it has not demonstrated better claim verification than the claim-only baseline on the aligned grouped test. Retrieval metrics on silver references cannot establish evidence-grounded verification quality.

## 5. Text preprocessing and linguistic processing

**VERIFIED:** Concrete implementations are in `src/claim_understanding.py::finance_aware_tokenize()` / `extract_financial_entities()`, `src/preprocess.py::strip_verdict_cues()`, and the optional `src/p1_tokenizer.py::FinancialAwareTokenizer`, `src/p1_stopwords.py::filter_tokens()`, and `src/p1_lemmatizer.py::POSGuidedLemmatizer.lemmatize_tokens()`. **UNRESOLVED:** No held-out entity-extraction or preprocessing quality benchmark was found.

| Technique | Implementation and effect | What it preserves/removes; status |
|---|---|---|
| Normalization/cleaning | `src/preprocess.py` creates raw and “cleaned” evidence text; the cue-removal regex strips a limited list of verdict phrases. Retrieval tokenization lowercases/tokenizes for vectorization; the P1 tokenizer is a separate path. | Cue removal is incomplete; raw evidence remains available and is used in the raw baseline. Cleaning does not prove all verdict leakage is removed. |
| Finance-aware tokenization | `finance_aware_tokenize()` uses regex patterns for currency/amounts, percentages, fiscal periods and selected symbols/financial forms. | Preserves useful tokens such as `$`, percentages, `FY`/quarter patterns and forms like `S&P 500`; it is a tokenizer, not financial semantic parsing. No empirical downstream ablation isolates its value. |
| Stopword removal | The optional P1 path has a custom stopword list with protected financial polarity/direction terms. Other lexical paths use standard vectorizer/English-stopword behavior; the default BM25 path does not use P1 filtering. | P1 explicitly preserves terms such as negations, directions and scale words. Standard English stopword removal can erase meaningful tokens such as “up”; no blanket finance-aware filter is used everywhere. |
| Stemming | No stemming implementation found. | Not implemented. |
| Lemmatization/POS | `POSGuidedLemmatizer` in `src/p1_lemmatizer.py` uses NLTK POS tagging and WordNet lemmatization with protected-token passthrough. | Optional P1 only. WordNet’s general vocabulary/irregular forms may not normalize finance-specific terms reliably; no head-to-head validated gain is established. |
| NER | Entity extraction uses known-organization matching and capitalization/regex heuristics. | No learned statistical NER model. Proper nouns can be missed or descriptive phrases falsely labeled as organizations. |
| Numbers, units, dates and fiscal periods | Regex patterns extract monetary amounts, percentages, periods and directional/negation cues for query construction. | Extraction may preserve surface forms, but no deterministic currency conversion, scale arithmetic, date normalization or numerical consistency checker is implemented. |
| Negation/polarity | Claim understanding extracts negation/directional language; P1 stopword filter protects selected polarity terms; query expansion applies fixed rules. | No general scope/parsing model. The default lexical BM25 uses ordinary word tokens, so preserving a token does not guarantee polarity-sensitive ranking. |

`outputs/dataset_audit.json` flags verdict-related cues in 1,348/3,369 records (40.01%). The flag is a screening count, not proof every feature leaks a label. Cleaning changes the random-split classifier result only modestly and does not remove all ways evidence can encode a fact-check outcome.

## 6. Information retrieval methods

**VERIFIED:** Main implementations/callers are `src/retrieval.py::EvidenceRetrievalBenchmark._build_corpus()`, `fit_indices()`, `retrieve_bm25()`, `retrieve_tfidf()`, `retrieve_hybrid_rrf()`, `src/dense_retrieval.py::PretrainedDenseRetriever`, `src/positional_index.py::PositionalInvertedIndex` / `BooleanQueryParser.evaluate()`, and `src/p1_retriever.py::NaturalLanguageEvidenceRetriever.retrieve_for_claim()`.

### Algorithms actually present

- **Inverted/positional index:** `PositionalInvertedIndex` maps term → document ID → token positions. Position lists permit exact adjacency checks and Boolean set operations. It supports AND/OR/NOT and quoted exact phrases. This is a genuine implementation, but it is not the default verifier retriever and has no judged IR effectiveness result. The indexed P1 artifact is another related index with a different corpus and processing.
- **Boolean retrieval:** The parser’s `evaluate()` returns matching document IDs using the postings. It is not a relevance-ranked method; no probabilistic ranking is implied. Functional unit tests do not establish precision/recall on user queries.
- **Phrase retrieval:** Exact adjacent token positions are checked for a quoted phrase. Arbitrary ordered/unordered proximity windows are not implemented.
- **TF-IDF:** Scikit-learn sparse vectors and cosine similarity in `EvidenceRetrievalBenchmark.retrieve_tfidf()` (`src/retrieval.py#L153-L158`). A typical term weight is `tf(t,d) × idf(t)`; cosine ranks the angle between query and document vectors. It emphasizes discriminative lexical terms; exact-form dependent and weak on paraphrase.
- **BM25:** `rank_bm25.BM25Okapi`-style scoring in `retrieve_bm25()` (`src/retrieval.py#L35-L66`, `#L159-L172`). For query term `t`, score contributions use IDF times `tf(t,d)(k1+1) / [tf(t,d)+k1(1-b+b·|d|/avgdl)]`; implementation `k1=1.5`, `b=.75`. Saturating term frequency and document-length normalization are useful for lexical ranking; it misses paraphrase and depends on query/corpus term matching.
- **SVD/LSA:** `TruncatedSVD` reduces TF-IDF document vectors to 128 latent dimensions; query and document similarity is calculated in the reduced space. It is an implemented optional experiment, not part of the default pipeline; saved results are weak. Low-rank projection can blur rare financial entities, figures and polarity.
- **Dense retrieval / sentence embeddings:** `src/dense_retrieval.py` loads `sentence-transformers/all-MiniLM-L6-v2` through Transformers, mean-pools token embeddings, normalizes vectors and uses cosine similarity. Embeddings are 384-dimensional and a cache file exists. It can recover paraphrases but may conflate related entities, metrics, periods and negation. This is pretrained inference, not FinFact fine-tuning.
- **Hybrid retrieval / RRF:** `retrieve_hybrid_rrf()` combines BM25 and TF-IDF rankings via reciprocal-rank contributions `1/(k+r)`, with `k=60` and one-based rank `r` (`src/retrieval.py#L173-L185`). This avoids directly combining incomparable raw scores, but generic fusion is BM25+TF-IDF. The intended BM25+dense branch in `src/pipeline.py#L103-L121` is shadowed by the earlier dense condition. The 150-query offline benchmark independently evaluates BM25+dense fusion.
- **Evidence corpus indexing:** Main `src/retrieval.py` indexes unique evidence sentences and metadata. No claim/justification/digest text is deliberately mixed into that corpus. However, the evidence itself includes fact-checker-provided reference sentences, including held-out claims’ evidence in the default full-data corpus.

### What is not implemented

No general proximity retrieval, learned query rewriting, neural sparse retrieval, or LSA component in the default pipeline. There is no external document fetcher/search engine. Library imports or descriptive docs are not evidence of an executed branch; in particular, SVD mode falls through to BM25 and `hybrid_bm25_dense` selects dense-only in `FinGuardPipeline.run()`.

## 7. Reranking

**VERIFIED:** `src/reranker.py::PretrainedCrossEncoderReranker` loads **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (`RERANKER_MODEL_NAME`, `#L17-L42`). Its `rerank()` (`#L44-L75`) forms a claim–passage pair per candidate, applies joint cross-attention, emits a scalar logit, transforms it with sigmoid, sorts by score and truncates to `top_k` (4 by default). `src/pipeline.py::run()` passes at most 15 retrieved candidates, not the whole corpus.

Retrieval finds a broad candidate set efficiently using independent query/document representations; reranking jointly reads a claim and each candidate to improve ordering. The cross-encoder score is not a probability that the claim is true, nor an independent relevance judgment. It is not trained on FinFact in this repository. In the 675-claim ablation it changed top-four membership for 588 claims, yet no human relevance labels were available to show the changes improved evidence relevance.

## 8. Verification models and outputs

**VERIFIED:** The default call chain is `FinGuardPipeline.run()` → `FinancialClaimVerifier.verify_claim()` → `predict_nli_single_pair()`; fallback is `predict_baseline()`. Sources: `src/pipeline.py#L166-L172`, `src/verifier.py#L70-L108`, `#L125-L230`.

### Pretrained NLI

- **Exact checkpoint:** `cross-encoder/nli-distilroberta-base`, loaded using Hugging Face `AutoTokenizer` and `AutoModelForSequenceClassification` in `src/verifier.py`.
- **Training status:** Repository evidence supports pretrained inference only; no FinFact fine-tuning script, training run, or saved fine-tuned NLI checkpoint was found. The cached config maps contradiction, entailment, neutral; the verifier reads `id2label` dynamically.
- **Input:** Evidence sentence as premise, claim as hypothesis; pair tokenization, truncation and 160-token maximum.
- **Output:** Three softmax scores mapped to contradiction→false, entailment→true, neutral→NEI. Each passage is evaluated independently.
- **Verified checkpoint mapping:** Cached config audited in `outputs/experiments/root_cause_investigation/nli_label_mapping_audit.json`: output index 0 contradiction, 1 entailment, 2 neutral; `src/verifier.py` reads `config.id2label` and maps by label name.
- **Training/evaluation:** No local transformer training script or transformer checkpoint in project outputs. A 100-example historical NLI result exists; the later 675-claim grouped end-to-end result is the stronger test. Six hand-authored probes check basic label/pair wiring but are not an evaluation dataset.

### TF-IDF + Logistic Regression and other baselines

- `src/baseline_models.py::build_models()` trains TF-IDF + Logistic Regression, Linear SVM, Multinomial Naive Bayes and Random Forest configurations (`#L34-L54`). Baseline LR uses `max_features=10000`, word unigrams+bigrams, English stopwords, sublinear TF, balanced classes, `C=1.0`, `max_iter=1000`, `random_state=42`. Experiments in `run_all_baseline_experiments()` use claim-only, raw claim+evidence, and cleaned claim+evidence settings. This evidence is dataset annotation text, not independently retrieved evidence in those baseline experiments.
- `outputs/best_baseline_model.pkl` is a saved scikit-learn classifier loaded by `FinancialClaimVerifier`. It is not a transformer and its use as a runtime fallback is separate from the default NLI path. The model’s `predict_proba` output is treated as confidence; no calibration curve/ECE or calibration procedure is evidenced by the inspected training code. “Calibrated” in comments/docs should not be read as proof of empirical calibration.
- Training/evaluation entry points include `src/baseline_models.py` and `src/run_source_grouped_baseline.py`; `src/evaluate_verifiers.py` and `src/run_aligned_grouped_verification.py` run verification evaluations. Saved classical weights include `outputs/best_baseline_model.pkl` and source-grouped experiment joblibs. The saved model artifact does not itself establish that its training protocol matches every reported result.

### Transformer class/template

`src/transformer_model.py` contains a transformer-based classification component/template. No FinFact-trained transformer weights, completed fine-tuning run, or valid held-out result attributable to it was found. The repository outputs contain dense embedding vectors but not fine-tuned NLI, reranker or transformer-classifier weights; referenced pretrained checkpoints are loaded from the Hugging Face model cache. This is not evidence that the default NLI checkpoint was fine-tuned.

### Aggregation and confidence caveats

The verifier partitions each sentence using .50 thresholds, then considers maximum entailment and contradiction scores across sentences; it does not weight by retrieval rank, source independence, source credibility or repeated evidence. Any support-plus-contradiction mix gates to NEI. Otherwise a single maximum ≥.60 can decide true/false. In the decisive branch, class probabilities are built from independent maxima that can come from different passages; the calculated NEI remainder can be negative before normalization. The conflict branch instead uses fixed class values. These outputs are not calibrated posterior probabilities, and no calibration or selective-risk metric is reported.

## 9. FinFact data and corpus audit

**VERIFIED:** Counts below are read from `finfact.json`, split CSVs and saved audits; corpus constructors are `src/retrieval.py::EvidenceRetrievalBenchmark._build_corpus()` and `src/p1_corpus_builder.py::build_p1_corpus()`. The count differences are representation differences, not silently harmonized.

### Dataset schema and counts

`finfact.json` contains **3,369 claims** with `claim`, `label`, `evidence` (sentence and source `hrefs`), `justification`, `sci_digest`, fact-check article `url`, and metadata such as author, posted date, issues and image fields. Labels: **1,492 false, 1,275 true, 602 NEI**.

| Unit | Count and definition |
|---|---:|
| Claim records | 3,369 |
| Evidence array entries | 18,357 |
| Evidence entries with truthy sentence text | 18,316 |
| Nonempty evidence sentences after whitespace strip | 18,270 |
| Nonempty justifications | 3,367 |
| Records with a digest | 2,661 |
| Nonempty digest text units after strip | 2,640 |
| Distinct fact-check article URLs | About 2,662 in the audit outputs |

These are field counts, not a single deduplicated corpus size. Across normalized nonempty fields the three text categories total 24,277 units, but they are not mixed by the production retriever.

### Corpus variants

- **Production `src/retrieval.py`:** Exact-text sentence deduplication produces **12,943** evidence passages in the saved corpus/provenance artifacts. Entries carry text, source URL(s), parent claim ID(s) and a type.
- **Reference-evidence helper:** `src/reference_evidence.py` applies its own normalization/deduplication and reports **12,941** units. The small discrepancy is a corpus-construction/normalization difference; it is not a change in claims.
- **Pipeline 1:** `src/p1_corpus_builder.py` starts from processed `cleaned_evidence_text`, splits into sentence-like units, filters short text and deduplicates differently. `outputs/p1_positional_index.json` reports 23,568 indexed units from 36,590 extracted sentence candidates. This is a different representation and is not the production BM25 corpus.
- **Aligned grouped test corpus:** The saved experiment corpus uses grouped train+validation data, excludes test records and test-linked source URLs, and contains 9,653 deduplicated passages from 13,088 retained evidence sentence entries. It reports no overlap with the held-out test’s 3,853 normalized URLs.

Claims, justifications and digests are not concatenated into the default retrieval index. Nevertheless, **gold/reference evidence sentences are present in the full-data index** because those sentences are the evidence field in FinFact.

## 10. Metrics and evaluation protocols

### Retrieval metrics

- **Precision@K:** Relevant retrieved items divided by K; measures top-K purity, not whether all needed facts were found.
- **Recall@K:** Relevant retrieved items among top K divided by known relevant items; with only reference sentences annotated, this is reference retrieval, not complete evidence recall.
- **MAP:** Mean of per-query average precision; rewards finding multiple relevant items early. It is sensitive to incomplete relevance judgments.
- **MRR:** Mean reciprocal rank of the first known relevant item; ignores additional relevant evidence after the first.
- **NDCG@K:** Graded, rank-discounted gain; implemented in `src/benchmark_evaluator.py` for the pooled graded benchmark, not in the older A–H retrieval output.
- **Coverage@K / bpref / RefRate:** The gold benchmark also reports judged coverage, bpref and reference-evidence retrieval rate. “Coverage” only means judged fraction under the evaluator’s current non-null grade handling; it does not imply human judgment. In v1, only silver references are graded relevant; in v2, most new grades are automated grade-zero judgments. RefRate equals Recall when the positive set is exactly the reference-evidence set.

Retrieval metrics do not establish a correct True/False/NEI verdict, source reliability, sufficiency of evidence, or calibration. The 150-query scores and 75-claim gold-benchmark scores use different query sets and judgment definitions and are not directly comparable.

### Classification and verification metrics

Accuracy is the overall correct fraction. Per-class precision/recall/F1 and confusion matrices show error direction; macro-F1 averages class F1 equally; weighted-F1 weights by class support. Macro-F1 is important for True/False/NEI because the class frequencies differ and a model can perform well on the majority class while missing NEI or True. No calibration metric (ECE, Brier score, reliability plot), selective risk/coverage curve, or NLI-specific entailment calibration metric was found.

### Splitting

- Standard split: stratified random 70/15/15, approximately 2,358/505/506; preserves label proportions but allows source overlap.
- Grouped split: `StratifiedGroupKFold` keyed on fact-check article URL; saved counts 2,357/337/675 (roughly 70/10/20). It is more defensible for unseen article-source generalization, but cited URLs require separate filtering and near duplicates still need audit.
- Best next protocol: source-grouped or temporal split plus explicit citation-source exclusion, exact and fuzzy duplicate quarantine, train-only corpus construction and a held-out test not used for thresholds/model selection. Repeated grouped folds or an additional temporal set are needed for robustness.

**VERIFIED metric-definition conflict:** `src/evaluate_all_retrieval.py::compute_ir_metrics()` and `src/retrieval.py::evaluate_retrieval()` divide AP by `min(number_of_relevant_items, K)`. `docs/metric_definitions.md` describes the pooled benchmark’s MAP denominator as the total number of relevant items. When relevant items exceed K these conventions differ; do not directly compare A–H MAP to pooled-benchmark MAP without reconciling definitions.

## 11. Actual experimental results

All rows below identify their own evaluation set. Do not interpret metrics on different sample sizes/splits as a direct system comparison.

### Classical classifier results

| Result artifact / protocol | Model and input | N / label supports | Accuracy | Macro-F1 | Interpretation |
|---|---|---:|---:|---:|---|
| `outputs/baseline_metrics.json`, standard random raw | TF-IDF + Logistic Regression, claim + raw dataset evidence | 506 (224 false / 191 true / 91 NEI) | .7470 | .7195 | Evidence text may contain verdict cues; article/citation overlap. |
| Same file, standard claim-only | TF-IDF + Logistic Regression, claim only | 506 | .6403 | .6028 | Not evidence-grounded verification. |
| Same file, standard cleaned evidence | TF-IDF + Logistic Regression, claim + limited cue-stripped evidence | 506 | .7372 | .7064 | Cleaning lowers accuracy by about 1 point, not a leakage proof. |
| Same file, source-grouped | TF-IDF + Logistic Regression, raw claim+evidence | 675 (299/255/121) | .5970 | .5528 | Same broad grouped protocol; older untuned run. |
| `outputs/experiments/source_grouped_baseline/results.json` | Tuned TF-IDF + Logistic Regression; raw / cleaned evidence | 675 | .5763 / .5778 | .5400 / .5411 | Validation-tuned configuration. Do not silently substitute for the older metric; experiment configurations differ. |
| `outputs/verification_metrics.json`, older standard result | Logistic Regression, claim + gold evidence | 506 | .7451 | .7178 | Similar, but not identical to baseline artifact result. |

`baseline_metrics.json` also reports standard-split Linear SVM (accuracy .7253 / macro-F1 .6822), Multinomial NB (.6877/.5915) and Random Forest (.7194/.6784); grouped variants score .6015/.5560, .5852/.4356 and .5822/.5003 respectively. These are supervised FinFact baselines, not evidence retrieval systems.

### Source-isolated aligned verification

`outputs/experiments/aligned_grouped_verification/results.json` uses the same held-out **675 source-grouped claims** for both systems. The pipeline corpus contains only train+validation evidence and filters the held-out claims’ cited URLs. The claim-only LR chooses C on validation; test is not used for tuning.

| System | Accuracy | Macro-F1 | Weighted-F1 | True-class recall | Notes |
|---|---:|---:|---:|---:|---|
| A — claim-only TF-IDF + Logistic Regression | .5556 | .4897 | .5498 | .5922 | Best of the two systems; no evidence input. |
| B/C — default BM25 → cross-encoder → pretrained NLI | .4030 | .3107 | .3383 | .0275 | B and C are the same actual default execution, not independent systems. 402 false / 16 true / 257 NEI predictions. |

The root-cause ablation holds the same 675 claims, safe corpus, NLI model and aggregation constant: BM25 top four without reranking is .3941 accuracy/.2958 macro-F1; BM25 plus reranker top four is .4030/.3107; top-passage argmax diagnostic is .3319/.2833. The reranker therefore gives a small downstream verification increase in this ablation; it does **not** establish better relevance because the 500-row `human_relevance_review.csv` is blank and awaiting labels. A separate 50-error review is qualitative failure analysis, not a relevance benchmark. NLI predicted 1,145 contradiction, 66 entailment and 1,489 neutral passage outcomes over 2,700 selected passages; 243/255 gold-true claims had no support-threshold passage.

### Historical NLI and pipeline results

`outputs/verification_metrics.json` reports the pretrained NLI model on **100** claims at accuracy .3400 / macro-F1 .2985 and a full pipeline on **100** claims at .3400/.3210. The LR baseline there is on **506** claims at .7451/.7178. Different N, evidence input and execution protocols prohibit a direct model comparison. These older values are not replacements for the aligned 675-claim experiment.

### Retrieval A–H results (`outputs/retrieval_metrics.json`)

This saved output reports **150 queries**, not the 506 claims sometimes claimed in top-level project prose. It compares ranking against dataset reference evidence in a corpus constructed from all FinFact evidence, including held-out claim evidence.

| Configuration | P@10 | R@10 | MRR | MAP |
|---|---:|---:|---:|---:|
| BM25 | .2253 | .4775 | .7680 | .3969 |
| TF-IDF cosine | .1893 | .3834 | .5800 | .2716 |
| 128-D SVD/LSA | .0407 | .0707 | .1291 | .0394 |
| RRF BM25 + TF-IDF | .2173 | .4350 | .6711 | .3304 |
| all-MiniLM dense | .2707 | .5302 | .8046 | .4673 |
| RRF BM25 + dense | .2767 | .5646 | .8267 | .4711 |
| BM25 + cross-encoder reranking | .2547 | .5076 | .8713 | .4803 |
| Expanded-query BM25 | .2167 | .4500 | .7592 | .3710 |

Within this artifact, BM25+dense RRF has the highest Recall@10 and cross-encoder reranking the highest MRR/MAP. These are ranking results on a contaminated, silver-reference setup, not evidence that those systems verify more claims correctly. `src/evaluate_all_retrieval.py` defaults to 150 queries; its nested `List`/`Dict` annotations lack the corresponding typing import in the inspected source, so saved output is reported as an artifact and current script reproducibility is not assumed.

### Pooled graded benchmark (`outputs/gold_benchmark_results*.json`)

The independent protocol is not fully achieved: v1 has 75 pilot claims, 4,891 pooled pairs, 410 grade-3 dataset-reference pairs and 4,480 unjudged pairs (plus one non-reference reviewed/grade-1 pair). In v2, 4,374 pairs receive automated grade 0, 410 remain reference positives, and 107 remain unjudged. The MS MARCO reranker and rule rubric participate in automated judgments; these are not independent human labels. The 500-row human review sheet in the root-cause experiment remains unfilled.

| System | R@10 | MRR | MAP | nDCG@10 | bpref | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | .3636 | .5661 | .2723 | .3719 | .2634 | .9373 |
| TF-IDF | .2841 | .4653 | .2102 | .2925 | .1999 | .9600 |
| SVD | .0547 | .0928 | .0322 | .0504 | .0340 | .9813 |
| Dense transformer | .4100 | .6469 | .3315 | .4375 | .3248 | .9293 |
| BM25+dense RRF | .4212 | .6445 | .3385 | .4405 | .3296 | .9013 |
| BM25+TF-IDF RRF | .3315 | .5136 | .2477 | .3406 | .2457 | .9480 |
| Cross-encoder rerank | .3920 | .6686 | .3332 | .4366 | .3396 | .9227 |
| Query expansion BM25 | .3411 | .5462 | .2564 | .3534 | .2510 | .9400 |

These are 75 pilot queries, not the 150-query A–H set. Positive relevance remains primarily the 410 original reference sentences; the new automated grade-zero assessments change judged coverage/bpref but are not independent gold. The output’s bootstrap intervals quantify query resampling under that qrels definition; they do not cure corpus leakage or label dependence.

## 12. Data leakage and methodological risks

- **VERIFIED — held-out gold evidence in old retrieval corpus:** `src/retrieval.py::EvidenceRetrievalBenchmark._build_corpus()` iterates all FinFact records and stores `parent_claim_ids`; `evaluate_retrieval()` uses each query’s evidence document IDs as positives. The held-out claim’s own annotated evidence remains in the index, and retrieval does not mask candidates by parent ID (`src/retrieval.py#L79-L135`, `#L210-L270`). This is direct target-evidence contamination for the old retrieval evaluation; it does not mean every hit was leaked.
- **VERIFIED — fact-check article URL overlap:** Standard split is stratified random 2,358/505/506. Pairwise article URL overlaps are train–validation **125**, train–test **117**, validation–test **48**. Grouped splits use `StratifiedGroupKFold` keyed on article `url` (`src/preprocess.py#L118-L132`).
- **VERIFIED — citation URL overlap:** Read-only audit over normalized evidence `hrefs` found standard train–validation **1,268**, train–test **1,084**, validation–test **517** shared URLs; grouped splits still have **120**, **175**, **58**, respectively. Grouping by fact-check article URL alone does not ensure cited-source disjointness.
- **VERIFIED — exact/fuzzy duplicate screen:** Exact lowercased claim overlap: 1 train–validation, 1 train–test, 0 validation–test in random splits; none across grouped splits. Character TF-IDF 3–5-gram cosine ≥.90 found **28 non-identical cross-split pairs** in random splits and none grouped. This is a screening heuristic, not human-adjudicated duplicates; examples include near-identical debt claims and a polarity-changing donation question.
- **VERIFIED — verdict cues:** `outputs/dataset_audit.json` flags 1,348/3,369 records (40.01%). `src/preprocess.py::strip_verdict_cues()` removes a limited configured list; raw evidence remains in baseline features.
- **VERIFIED — leakage test scope:** `tests/test_leakage_and_reranker.py::test_candidate_masking_exclusion_rule()` filters a small inline synthetic list; it does not call `EvidenceRetrievalBenchmark._build_corpus()` or the production retrieval evaluator. The test therefore does not show production held-out-parent exclusion.
- **VERIFIED — UI label lookup fallback:** `frontend/app/api/verify/route.ts#L127-L158` may return the dataset’s stored verdict after a partial claim-text match.
- **VERIFIED — stronger aligned corpus:** `outputs/experiments/aligned_grouped_verification/results.json` and `experiment.log` record train+validation-only corpus construction, exclusion of 1,253 test-linked evidence entries, 3,853 test-linked URLs and zero remaining test-source overlap. This safeguard belongs to that saved experiment, not default full-data production construction.
- **INFERRED — deployment contamination:** FinFact evidence and justifications are retrospective fact-checking artifacts. Using held-out claims’ own reference sentences does not simulate a contemporaneous real-world evidence search.
- **UNRESOLVED — complete near-duplicate risk:** No semantic human adjudication of all split pairs exists; the character cosine threshold can miss paraphrases or flag benign similarities.

## 13. Implemented, experimental, planned and unsupported

**VERIFIED status taxonomy:** “Default” means reachable from the default `FinGuardPipeline.run()` settings; “optional” means a code branch or separate tool exists; “experimental” means a saved experiment exists but not a production path; “not established” means no implementation/artifact or no adequate validation was found.

| Status | Components |
|---|---|
| **Implemented and default-executed** | Regex claim parsing, deterministic query expansion, BM25 retrieval, optional/available cross-encoder reranking, pretrained NLI, heuristic aggregation; default BM25 and top-15/top-4 settings. |
| **Implemented, optional or separate UI** | TF-IDF; SVD/LSA; dense retrieval; BM25+TF-IDF RRF; intended BM25+dense RRF (pipeline routing bug); P1 lemmatized TF-IDF; positional index; Boolean and exact phrase search; baseline classifier fallback. |
| **Evaluated but weakly validated** | 150-query full-corpus reference retrieval; 75-claim pooled benchmark with silver/automated judgments; standard random classifier split with source overlap; 100-example NLI/pipeline run. |
| **Evaluated on stronger, limited protocol** | Source-grouped classifier and aligned 675-claim test with a train/validation-only evidence corpus and test citation exclusion. Supports a result for this fixed split, not broad generalization or causal attribution. |
| **Not established / planned or absent** | FinFact-fine-tuned transformer/NLI; learned NER; deterministic financial arithmetic/currency-period verifier; arbitrary proximity search; calibrated confidence; independently adjudicated large relevance set; open-web evidence retrieval. |

## 14. Methodology comparison table

| Approach | Category | Core idea | Input | Output | Implemented status | Evaluation status | Strength | Limitation |
|---|---|---|---|---|---|---|---|---|
| Regex finance tokenizer (`src/claim_understanding.py`) | Preprocessing | Preserve financial token forms | Claim text | Token list | VERIFIED: used by parsing | Unit tests; no isolated efficacy test | Retains amounts/periods | Regex coverage and boundary errors |
| Rule/lexicon claim extraction (`src/claim_understanding.py`) | Claim understanding | Extract entities, metrics, polarity, units | Claim | Structured fields | VERIFIED: default | No held-out extraction score | Cheap, explainable | Not learned NER; misses/false positives |
| Custom stopwords (`src/p1_stopwords.py`) | Preprocessing | Remove common words but protect polarity/finance | P1 tokens | Filtered tokens | VERIFIED: P1 only | Unit tests | Protects selected negation/direction | Not used in default BM25 |
| POS-guided WordNet lemma (`src/p1_lemmatizer.py`) | Preprocessing | Normalize inflected forms | P1 tokens | Lemmas | VERIFIED: P1 only | Unit tests | Reduces lexical variation | General-purpose vocabulary; not proven better |
| Positional index (`src/positional_index.py`) | IR | Store term locations per document | Evidence corpus | Postings/positions | VERIFIED: implemented, separate path | Index statistics/tests | Boolean and phrase matching | Not default; UI caller mismatch |
| Boolean AND/OR/NOT (`BooleanQueryParser.evaluate`) | IR | Set operations over postings | Boolean query | Matching IDs | VERIFIED: implemented | Functional tests only | Precise operator matching | No ranked relevance |
| Exact phrase (`PositionalInvertedIndex.search_phrase`) | IR | Require adjacent token positions | Quoted phrase | Matching IDs | VERIFIED: implemented | Functional tests only | Enforces exact local order | No NEAR/k proximity |
| TF-IDF + cosine (`src/retrieval.py`) | IR | Weight discriminative lexical terms | Query/corpus | Ranked passages | VERIFIED: optional | 150 and 75 query artifacts | Simple lexical baseline | Surface overlap; corpus contamination |
| BM25 (`src/retrieval.py`) | IR | Saturated term-frequency with length normalization | Query/corpus | Ranked passages | VERIFIED: default | 150 and 75 query artifacts | Strong lexical ranking | Misses paraphrase/entails |
| 128-D SVD/LSA (`src/retrieval.py`) | IR | Compare low-rank TF-IDF semantics | Query/corpus | Ranked passages | VERIFIED: optional; pipeline mode falls through | 150 and 75 query artifacts | Latent term association | Poor saved scores; loses rare detail |
| MiniLM dense vectors (`src/dense_retrieval.py`) | IR | Cosine similarity in pretrained embedding space | Query/passages | Ranked passages | VERIFIED: optional | 150 and 75 query artifacts | Semantic matching | May conflate related financial facts |
| RRF (`src/retrieval.py`, `src/pipeline.py`) | IR fusion | Sum reciprocal rank contributions | Ranked lists | Fused ranking | VERIFIED: BM25+TF-IDF; dense branch ordering bug | Offline 150/75 artifacts | Avoids raw-score calibration | No guarantee evidence is relevant |
| MS MARCO cross-encoder (`src/reranker.py`) | Reranking | Jointly score each claim/passage pair | Top 15 pairs | Top 4 | VERIFIED: default if available | Retrieval artifacts; no independent human relevance | Reorders candidates semantically | Not a truth score; domain mismatch |
| DistilRoBERTa NLI (`src/verifier.py`) | Verification | Classify premise/hypothesis relation | Passage + claim | Entail/contradict/neutral | VERIFIED: default if available; pretrained | N=100 historical, N=675 aligned | Evidence-conditioned decisions | Low true recall; noisy evidence; no tuning |
| TF-IDF + Logistic Regression (`src/baseline_models.py`) | Classification | Supervised sparse-text class prediction | Claim, optionally evidence | True/False/NEI | VERIFIED: trained on FinFact; fallback | Random and grouped tests | Strong simple baseline | Can learn label/source cues; not intrinsically evidence-grounded |
| SVM / NB / Random Forest (`src/baseline_models.py`) | Classification | Classical supervised baselines | Claim + evidence variants | Class label | VERIFIED: experimental | Saved random/grouped metrics | Comparators | Mixed class performance; protocol dependence |
| Pooled grade automation (`src/judgment_automation.py`) | Relevance labeling | Cross-encoder plus rules assign grades | Claim/passage pair | Grade/status | VERIFIED: implemented for v2 | Automated qrels only | Scales annotation triage | Not independent human gold; model overlap |
| Evidence aggregation (`src/verifier.py`) | Verification decision | Threshold sentence scores and gate conflicts | NLI outputs | Verdict/confidence | VERIFIED: default | Ablations on one grouped split | Simple explicit abstention | Maxima uncalibrated; source/rank ignored |

## 15. Research limitations and recommended next experiments

1. **Replace archive leakage with a deployment-matched corpus.** Build the index only from training/allowed sources; exclude every test claim’s parent IDs and citations. Keep provenance checks as executable assertions.
2. **Complete independent human qrels.** Judge a stratified sample of claim–passage pairs for relevance and stance, double-annotate and adjudicate disagreements. Keep the evaluating cross-encoder out of the gold labels. Report judged coverage and an unjudged-aware analysis.
3. **Audit sources and duplicates before splitting.** Group by fact-check article and cited source URLs, quarantine exact and near-duplicate claims, and test a temporal split. Publish counts and criteria.
4. **Run controlled ablations on one fixed grouped protocol.** Compare BM25, dense, correctly routed hybrid, reranking, and query expansion on the same corpus and queries; evaluate relevance and final verification separately. Tune only on validation.
5. **Improve class-conditional verification.** Report confusion matrices/per-class recall, especially True and NEI. Test domain-adapted NLI only as a distinct trained method with saved checkpoint, data lineage and unseen-source evaluation.
6. **Calibrate abstention and confidence.** Fit thresholds/calibration on grouped validation; report Brier/ECE and risk–coverage, never call max-score outputs probabilities without calibration.
7. **Add deterministic financial checks as a separate component experiment.** Normalize currencies/units, fiscal periods, percentages and directional comparisons, and evaluate against annotated numerical claims. No such verifier currently exists.
8. **Separate archive lookup from verification in product paths.** Label the Next.js archive fallback as lookup-only (or disable it for evaluation); never merge its stored label outputs with model metrics.
9. **Fix method routing and caller contracts before claiming UI execution.** Test the `hybrid_bm25_dense` ordering, SVD route, positional-index constructor and Boolean parser names, and parsed/query result keys. These are implementation concerns, not a reason to reinterpret saved metrics.

**VERIFIED test scope:** The tests cover schemas, token preservation, parsing, synthetic retrieval, model fallbacks, corpus metadata and hand-calculated metrics. **INFERRED:** Passing these unit tests would establish those tested behaviors only, not unbiased evidence retrieval or fact-verification accuracy; no such tests were executed in this audit.

### Unresolved questions and missing artifacts

- **UNRESOLVED — historical phases:** No changelog/phase manifest/source history inspected establishes when Phase 1 or Phase 2 was developed; Sections 3–4 are explicitly code-based groupings.
- **UNRESOLVED — transformer training:** No FinFact-fine-tuned transformer/NLI/reranker checkpoints, training logs or training data lineage were found. The model names are loaded from the external Hugging Face cache; `outputs/dense_evidence_embeddings.npz` is an embedding cache, not fine-tuned weights.
- **UNRESOLVED — baseline pickle provenance:** The training/configuration lineage for `outputs/best_baseline_model.pkl` is not fully linked to each saved metrics file; do not assume the runtime pickle is the same fitted model as every historical baseline result.
- **UNRESOLVED — relevance truth:** The 500-row human review sheet is blank. No independent adjudicated relevance/stance score can be reported.
- **UNRESOLVED — reproducibility:** No evaluation or training script was rerun, so agreement between current source and historical saved metrics remains unverified. In particular, the local `List`/`Dict` annotation issue in `src/evaluate_all_retrieval.py` appears to make the current 150-query script fail during function definition on Python versions that eagerly evaluate annotations; this was not executed.
- **UNRESOLVED — pretrained model revision:** Repository outputs identify checkpoint names but do not freeze the exact revision/hash of every pretrained snapshot used.

## Appendix A. Files and evidence inspected

The following inventory distinguishes source coverage from artifacts used for results. All 32 Python modules under `src/` and all 18 test files listed below were read. Root README/project report, listed methodology docs/reports/configs, raw data/splits, benchmark judgment artifacts and saved result files were inspected. The listed app/API files were read; the entire frontend page/component tree was not reviewed. Evaluation/training scripts were not run because they may regenerate outputs or perform expensive model loading.

### Source modules (`src/`, all 32)

`adjudication.py`, `api_bridge.py`, `baseline_models.py`, `benchmark_evaluator.py`, `claim_understanding.py`, `dataset_audit.py`, `dense_retrieval.py`, `eda.py`, `evaluate_all_retrieval.py`, `evaluate_verifiers.py`, `freeze_benchmark.py`, `gold_benchmark_schema.py`, `judgment_automation.py`, `p1_corpus_builder.py`, `p1_index.py`, `p1_lemmatizer.py`, `p1_pipeline.py`, `p1_retriever.py`, `p1_stopwords.py`, `p1_tokenizer.py`, `pipeline.py`, `pooling.py`, `positional_index.py`, `preprocess.py`, `query_construction.py`, `reference_evidence.py`, `reranker.py`, `retrieval.py`, `run_aligned_grouped_verification.py`, `run_source_grouped_baseline.py`, `transformer_model.py`, `verifier.py`.

### Tests (all 18 source files read)

`test_audit_and_splits.py`, `test_claim_understanding.py`, `test_gold_benchmark.py`, `test_judgment_automation.py`, `test_leakage_and_reranker.py`, `test_p1_corpus.py`, `test_p1_index.py`, `test_p1_lemmatizer.py`, `test_p1_pipeline.py`, `test_p1_retriever.py`, `test_p1_stopwords.py`, `test_p1_tokenizer.py`, `test_pipeline_and_nli.py`, `test_positional_index.py`, `test_query_expansion.py`, `test_retrieval.py`, `test_source_grouped_baseline.py`, `test_verifier.py`.

### Documentation, configs, UI and reports

- Root: `README.md`, `PROJECT_REPORT.md`.
- `docs/`: `annotation_guidelines.md`, `gold_evidence_benchmark.md`, `independent_evaluation_audit.md`, `metric_definitions.md`, `relevance_automation_research.md`, `relevance_judgment_protocol.md`.
- `reports/`: `FINGuard_ALIGNED_GROUPED_VERIFICATION.md`, `FINGuard_NLP_METHODOLOGY_AUDIT.md` (pre-existing version replaced), `FINGuard_RESEARCH_VALIDATION_AUDIT.md`, `FINGuard_ROOT_CAUSE_INVESTIGATION.md`.
- `configs/`: `gold_evidence_benchmark.json`, `gold_evidence_benchmark_v2.json`.
- UI: root `app.py`; `frontend/app/api/verify/route.ts`; `frontend/lib/api.ts`.
- Additional audit material: `.agents/tasks/codebase-investigation.md`; `outputs/implementation_audit.md`; `outputs/gold_benchmark_audit.md`; `outputs/gold_benchmark_audit_v2.md`; gold benchmark result markdown files.

### Data and benchmark files

- `finfact.json`; `data/processed_finfact.csv`; `data/train.csv`, `val.csv`, `test.csv`, `grouped_train.csv`, `grouped_val.csv`, `grouped_test.csv`.
- `data/benchmark/v1/`: `benchmark_manifest.json`, `claims_manifest.json`, `corpus_manifest.json`, `pooled_candidates_annotation.csv`, `pooled_candidates_annotation.jsonl`, `reference_evidence.json`, `relevance_judgments.csv`, `relevance_judgments.json`.
- `data/benchmark/v2/`: the same manifest, candidate-pool, reference and judgment files, plus `human_review_queue.csv` and `human_review_queue.jsonl`.

### Saved result and experiment artifacts

- Root outputs: `dataset_audit.json`, `baseline_metrics.json`, `retrieval_metrics.json`, `verification_metrics.json`, `best_baseline_model.pkl`, `dense_evidence_embeddings.npz`, `p1_positional_index.json`, `corpus_provenance_sample.json`, `gold_benchmark_results.json`, `gold_benchmark_results_v1.json`, `gold_benchmark_results_v2.json`, and their corresponding result/audit markdown files.
- `outputs/experiments/aligned_grouped_verification/`: `results.json`, `methodology.md`, `experiment.log`, `per_claim_predictions.csv`, `error_analysis.csv`, `retrieval_corpus_train_val.json`, and three confusion-matrix CSVs.
- `outputs/experiments/root_cause_investigation/`: `ablation_results.json`, `aggregation_analysis.md`, `experiment.log`, `human_relevance_review.csv`, `nli_label_mapping_audit.json`, `nli_sanity_tests.json`, `retrieval_error_analysis.csv`.
- `outputs/experiments/source_grouped_baseline/`: `results.json`, `experiment.log`, raw/cleaned model joblibs and confusion/per-class plots.

The raw dataset and saved outputs were inspected read-only. The Python and frontend test suites, benchmark scripts and training scripts were **not run**; reading test source is not equivalent to executing the tests. Metrics in this report are historical artifacts; dataset counts/source overlaps are read-only audit calculations, not model experiment reproductions.
