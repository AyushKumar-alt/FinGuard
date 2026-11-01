# FinGuard: Project Defense Speaker Notes

**Project Title:** FinGuard: Financial Claim Verification Using NLP and Evidence Retrieval  
**Degree / Context:** Final Year B.Tech Capstone Project Defense  
**Department:** Computer Science & Engineering  
**Pacing Target:** 60 to 90 seconds per slide (~12–14 minutes total presentation time)

---

## Slide 1: Title and Project Overview

**Visual Focus:** Title card, project metadata, and problem overview statement.

### Spoken Script (~75 seconds)
"Good morning, respected committee members and faculty evaluators. Today, I am defending our final-year capstone project: **FinGuard: Financial Claim Verification Using Natural Language Processing and Evidence Retrieval**.

In the modern financial ecosystem, automated claim verification is both a critical necessity and a formidable computational challenge. Unlike general-domain fact-checking—such as verifying historical dates or celebrity trivia—financial discourse revolves around subtle numerical assertions, corporate reporting periods, stock tickers, and strict directional polarity. A single word change, such as 'revenue increased 14%' versus 'operating profit fell 14%', can be the difference between compliance and multi-million-dollar fraud.

The core motivation of FinGuard is to build an automated, evidence-grounded verification system capable of ingesting raw financial claims, querying an extensive evidence base, and outputting an objective veracity verdict—True, False, or Not Enough Information—supported by cited documentation and an output confidence score.

Throughout this project, we executed a two-phase research methodology: Phase 1 engineered domain-aware linguistic preprocessing and custom indexing structures, while Phase 2 designed a multi-stage neural retrieval, reranking, and Natural Language Inference pipeline. Crucially, we evaluated both systems under a strict, leakage-resistant benchmark.

Let us now look closely at why financial text resists standard NLP tools and the dataset we used to ground our investigation."

### Slide Transition
*"To understand our technical architecture, let us examine the domain-specific linguistic hurdles of financial text and the FinFact benchmark dataset."*

---

## Slide 2: Problem Statement and Dataset

**Visual Focus:** Five domain challenges on the left; FinFact benchmark statistics (3,369 claims) and class breakdown on the right.

### Spoken Script (~85 seconds)
"Why is financial fact-checking uniquely difficult for standard NLP pipelines?

First, **financial terminology** is dense and non-standard. We encounter ticker symbols like `$AAPL` or `$TSLA`, fiscal quarters like `Q2 FY2024`, and domain-specific acronyms like `EBITDA` or `basis points`. Off-the-shelf tokenizers often fragment these into nonsensical sub-tokens.

Second, **numerical precision** is paramount. General text models treat numbers as interchangeable tokens, but in finance, magnitudes, currency signs, and percentages completely define the claim's truth value.

Third, **directional polarity and negation**: words like 'profit', 'loss', 'up', 'down', 'exceeded', and 'underperformed' carry the entire polarity of the statement. Standard stopword lists discard many of these words, inverting the claim's meaning.

Fourth, **temporal context**: financial claims are strictly time-bound. A claim stating 'Apple revenue reached $90 billion' may be true for Q2 2024 but false for Q1 2023.

Fifth, **evidence relevance**: fact-checking cannot rely on generic press releases; it requires retrieving verifiable evidence sentences.

To evaluate our systems, we utilized the **FinFact benchmark**, consisting of **3,369 fact-checked financial claims**. The class distribution reflects real-world fact-checking conditions: **1,492 False claims (44.3%)**, **1,275 True claims (37.8%)**, and **602 NEI claims (17.9%)**.

Our task is formulated as 3-way classification: predicting whether a claim is True, False, or NEI given the evidence corpus, returning a predicted confidence score based on the highest-scoring decisive passage."

### Slide Transition
*"Faced with these challenges, we structured our research into two distinct phases, contrasting classical information retrieval with neural architectures."*

---

## Slide 3: Two-Phase Architecture

**Visual Focus:** Structural progression arrow connecting Phase 1 (Classical Foundations) to Phase 2 (Neural Verification Pipeline).

### Spoken Script (~80 seconds)
"Slide 3 outlines our architectural roadmap and the progression from Phase 1 to Phase 2.

In **Phase 1**, our goal was to build robust classical and linguistic foundations from the ground up:
- We designed a specialized regex tokenizer that preserves tickers, currency amounts, and percentages.
- We curated a domain-adapted stopword filter that explicitly retains 58 polarity and directional words.
- We implemented a POS-guided WordNet lemmatizer that protects financial named entities and acronyms from morphological corruption.
- We built a memory-resident **positional inverted index** tracking token offsets across 23,568 corpus passages, supporting exact phrase search and Shunting-Yard Boolean queries.
- Finally, we trained a supervised claim-only baseline using TF-IDF and tuned Logistic Regression.

In **Phase 2**, we transitioned from lexical indexing to an end-to-end, multi-stage neural fact-checking system:
- We added rule-based financial entity extraction and structured query expansion.
- We implemented BM25 candidate retrieval, complemented by optional dense vector search and Reciprocal Rank Fusion.
- We deployed a Cross-Encoder reranker using `ms-marco-MiniLM-L-6-v2` to score joint claim-passage cross-attention.
- We integrated a zero-shot Natural Language Inference verifier using `nli-distilroberta-base`.
- And finally, we engineered a conflict-aware evidence aggregator that evaluates multi-passage entailment and contradiction threshold signals to emit the final verdict.

Let us inspect the implementation details of Phase 1."

### Slide Transition
*"Let us dive into the concrete engineering of Phase 1, focusing on our custom tokenizer, preserved stopwords, and positional index."*

---

## Slide 4: Phase 1 Implementation

**Visual Focus:** Top pipeline diagram; bottom-left positional index statistics table (23,568 units, 32,845 terms, 393,323 postings); bottom-right search engine capabilities and the proximity scope note.

### Spoken Script (~85 seconds)
"In Phase 1, our primary engineering objective was solving the tokenization and retrieval errors that plague off-the-shelf NLP toolkits when applied to financial text.

As shown in the top pipeline flow:
1. Raw financial text enters our **custom regex tokenizer**. Instead of treating punctuation uniformly, our patterns recognize currency symbols like `$90.8B` or `₹16,811 crore`, decimal percentages like `14.5%`, and fiscal notations like `Q2 FY2024` as single, indivisible lexical units.
2. Next, our **polarity-preserving stopword filter** strips 214 high-frequency grammatical tokens but explicitly protects **58 domain-critical polarity terms**, such as 'profit', 'loss', 'surged', 'fell', 'above', 'below', and negation markers like 'not' and 'never'.
3. Our **POS-guided lemmatizer** converts Penn Treebank POS tags to WordNet categories, lemmatizing verbs and nouns accurately while bypassing capitalized corporate tickers and fiscal acronyms.

Using this normalized vocabulary, we constructed an in-memory **positional inverted index** from scratch. As verified in our audit:
- The index covers **23,568 segmented units**—combining all 3,369 claims, 18,270 evidence sentences, and 3,367 fact-checker justifications.
- It contains **32,845 unique vocabulary terms** and **393,323 positional postings**.
- Each posting stores not just document IDs, but the exact integer word offsets of each occurrence.

This positional structure powers our **exact phrase search engine**, which verifies word adjacency—meaning position $i+1$ equals position $i$ plus 1. We also built a Shunting-Yard Boolean query engine supporting `AND`, `OR`, `NOT`, and parenthetical precedence.

*I want to state a clear architectural boundary for the committee:* our Phase 1 engine implements strict adjacent phrase search ($k=1$); arbitrary variable-distance proximity search—such as term A within $k$ words of term B—was not implemented and remains part of our future roadmap."

### Slide Transition
*"Now, let us examine how Phase 2 builds upon this foundation to construct an end-to-end neural verification pipeline."*

---

## Slide 5: Phase 2 Architecture and Pipeline Flow

**Visual Focus:** Left-hand 7-stage verified default execution trace; right-hand architecture distinctions and modular options.

### Spoken Script (~85 seconds)
"Slide 5 illustrates the complete architecture and runtime execution path of FinGuard Phase 2.

When a claim enters `FinGuardPipeline.run()`, it follows a verified 7-stage call chain:
1. **Input Claim**: For instance, *'Apple Q2 FY2024 revenue surged 14% to $90.8 billion.'*
2. **Claim Understanding**: We execute **rule-based financial entity extraction**. I want to highlight an important technical clarification: this is implemented using deterministic regex patterns and curated financial gazetteers—extracting organizations, financial metrics, currency amounts, percentages, and fiscal periods. *It is not a fine-tuned neural NER model.*
3. **Query Construction**: The system binds extracted entities, metric aliases, and ticker symbols into an expanded, high-specificity retrieval query.
4. **Candidate Retrieval**: The query searches an inverted Okapi BM25 index over our 12,943 evidence sentence pool, retrieving the Top-15 candidate sentences.
5. **Cross-Encoder Reranking**: Because BM25 only captures lexical overlap, we pass the claim and candidate passages to a neural Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), which performs joint cross-attention across all tokens and selects the Top-4 most contextually relevant passages.
6. **Zero-Shot NLI Verification**: Each of the Top-4 passages is paired with the claim as (Premise=Evidence, Hypothesis=Claim) and evaluated by `nli-distilroberta-base`, yielding 3-way softmax probabilities for Entailment, Contradiction, and Neutral.
7. **Evidence Aggregation**: A decision thresholding engine aggregates these passage-level probabilities into a final verdict: True, False, or NEI, accompanied by cited evidence.

On the right side of the slide, we note our modular extensions: while BM25 is the configured default, our codebase also implements an optional 384-dimensional dense semantic vector search and Reciprocal Rank Fusion for hybrid retrieval."

### Slide Transition
*"Let us inspect the exact models, dimensionality, parameter counts, and zero-shot weight status deployed in this pipeline."*

---

## Slide 6: Technical Implementation

**Visual Focus:** Model specifications table comparing BM25, SVD (128-dim), MiniLM (384-dim), Cross-Encoder (22.7M), and NLI DistilRoBERTa (82M); bottom decision threshold cards and the zero-shot integrity guarantee.

### Spoken Script (~80 seconds)
"Slide 6 provides the rigorous technical specifications of every model and algorithm in FinGuard.

In our candidate retrieval layer, we deploy **Okapi BM25** with standard parameters $k_1 = 1.5$ and $b = 0.75$, computing inverse document frequencies across the evidence corpus.

Regarding dense representations, our repository implements two distinct approaches that must not be confused:
- In Phase 1, we implemented Latent Semantic Analysis using **TF-IDF with Truncated SVD**, producing a **128-dimensional** dense semantic space fit directly on our corpus.
- In Phase 2, we implemented dense sentence embeddings using **`all-MiniLM-L6-v2`**, which produces **384-dimensional** dense vectors via mean-pooled transformer hidden states.

For reranking, we utilize **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, a 22.7-million parameter model trained on MS-MARCO passage ranking. Unlike dual-encoder bi-encoders that encode queries and passages independently, the Cross-Encoder feeds the claim and passage together through 6 transformer layers, capturing token-to-token semantic interactions.

For verification, we utilize **`cross-encoder/nli-distilroberta-base`**, an 82-million parameter model pretrained on Multi-Genre NLI and Stanford NLI.

In our aggregator, we apply heuristic decision thresholds: an entailment probability of at least 0.65 flags supporting evidence, while a contradiction probability of at least 0.60 flags refuting evidence. The reported confidence is simply the highest probability among decisive passages.

*Finally, I want to emphasize an essential scientific integrity guarantee:* all transformer models in our evaluation operate strictly in **pretrained zero-shot mode without fine-tuning on FinFact in the configured evaluation pipeline**. While PyTorch training scaffolding exists in our codebase, no fine-tuned weights were deployed. This experimental control allows us to measure how well general-purpose neural models generalize to financial verification."

### Slide Transition
*"Now, let us turn to the empirical evaluation of these systems under our strict source-grouped benchmark."*

---

## Slide 7: Experimental Evaluation and Results

**Visual Focus:** Clustered bar chart comparing System A vs System B across metrics; benchmark comparison table; notes on System C equivalence and the True Recall anomaly.

### Spoken Script (~90 seconds)
"Slide 7 presents the central empirical findings of our defense, evaluated on a strictly held-out, **source URL-grouped test set of 675 claims**.

To guarantee that our results measure true generalization rather than data leakage, no test set claim shares a source URL with any training or validation record. All 531 source URLs in the test set are completely held out. Furthermore, fact-checker verdict cues were stripped from all evidence text.

We evaluated three systems:
- **System A**: Our Phase 1 claim-only baseline (TF-IDF + Logistic Regression, tuned to $C=0.5$).
- **System B**: Our Phase 2 evidence-grounded neural pipeline (BM25 + Cross-Encoder + Zero-Shot NLI).
- **System C**: Our configured pipeline, which executed identical default parameters (BM25 top-15, Cross-Encoder top-4, zero-shot NLI) and therefore produced identical predictions to System B.

Here is the key finding: **System A, the classical baseline, substantially outperformed the evidence-grounded neural pipeline.**
- System A achieved **55.56% Test Accuracy** and **48.97% Macro-F1**.
- System B achieved **40.30% Test Accuracy** and **31.07% Macro-F1**.
- While System B achieved high recall on False claims (**71.57%**), its **True Recall collapsed to just 2.75%**—correctly identifying only 7 out of 255 gold-true claims.
- In contrast, System A achieved a balanced **59.22% True Recall** and **66.22% False Recall**.

As computer science researchers, we do not conceal or apologize for this negative result. In scientific research, an honest negative result that uncovers fundamental domain limitations is far more valuable than an artificially inflated metric.

In the next slide, we present the root-cause post-mortem that explains exactly why this occurred."

### Slide Transition
*"Let us break down the exact mathematical and empirical reasons behind this performance gap in our error analysis."*

---

## Slide 8: Error Analysis and Limitations

**Visual Focus:** Top cards showing NLI output distribution (1,489 Neutral, 1,145 Contradiction, 66 Entailment), the 243/255 True claim evidence threshold gap, and the 50-case error taxonomy; bottom cards contrasting observed facts from hypotheses.

### Spoken Script (~90 seconds)
"Slide 8 presents our root-cause investigation into why the neural evidence-based pipeline struggled. Through quantitative log analysis and a 50-case manual triage, we isolated three distinct mechanisms:

First, **Neutral Collapse in General NLI**:
We audited all 2,700 claim-passage pairs evaluated by the NLI verifier during testing (675 claims multiplied by 4 candidate passages).
- **1,489 pairs (55.1%)** were classified as **Neutral**.
- **1,145 pairs (42.4%)** were classified as **Contradiction**.
- **Only 66 pairs (2.4%)** were classified as **Entailment**.
Pretrained NLI models trained on conversational benchmarks like MNLI are calibrated for literal, deductive reasoning. When confronted with corporate guidance, forward-looking statements, or qualified financial metrics, the model defaults to Neutral because the evidence does not provide absolute mathematical proof.

Second, **The Evidence Aggregation Threshold Gap**:
When we inspected the 255 gold-true claims, we discovered that **243 of 255 gold-true claims (95.3%) did not receive a supporting passage above the evidence aggregation threshold ($p_{\text{entail}} \ge 0.65$)**. Only 12 claims had at least one passage meeting the entailment threshold, and after conflict resolution, only 7 survived as final True verdicts.

Third, our **50-case audited error taxonomy** revealed:
- **Irrelevant Retrieval** accounted for 23 cases (46.0%).
- **Wrong Entity Mismatch** accounted for 11 cases (22.0%).
- **Insufficient Evidence Pool** accounted for 10 cases (20.0%).
Crucially, these first three categories account for **44 out of the 50 audited cases (88.0%)**! The remaining 6 cases (12.0%) consisted of specific issues: 2 wrong fiscal period errors, 1 wrong metric error, 1 aggregation error, 1 NLI neutral collapse, and 1 miscellaneous error.

We carefully distinguish between **observed facts**—such as the 2.4% entailment rate and the evidence threshold gap—and **working hypotheses**, such as domain vocabulary shift between news releases and corporate filings."

### Slide Transition
*"This brings us to our concluding takeaways, technical contributions, and future research roadmap."*

---

## Slide 9: Conclusion and Future Work

**Visual Focus:** Summary of achievements on the left; five actionable future research directions on the right.

### Spoken Script (~75 seconds)
"To conclude our presentation:

FinGuard successfully demonstrated a complete, end-to-end journey in financial NLP:
1. In **Phase 1**, we engineered domain-adapted preprocessing, preserved 58 directional stopwords, and built a custom positional inverted index over 23,568 units.
2. In **Phase 2**, we architected an operational multi-stage neural fact-checking pipeline integrating financial entity extraction, query expansion, BM25 retrieval, Cross-Encoder reranking, and NLI verification.
3. Methodologically, our project maintained absolute evaluation integrity. By enforcing a source-grouped benchmark without data leakage, we uncovered the critical truth that **evidence-grounded verification is strictly bounded by retrieval quality and verifier domain adaptation**. A complex neural pipeline cannot outperform a simple baseline if the retrieved passages fail to exceed the entailment threshold, and if the verifier suffers from zero-shot domain mismatch.

For future work, we propose five concrete technical enhancements:
1. **Domain-Specific NLI Fine-Tuning**: Fine-tuning compact NLI backbones on financial claim-evidence pairs with contrastive loss to eliminate neutral collapse.
2. **Tabular and Numerical Reasoning**: Incorporating structured table-parsing models like TAPAS to verify numerical claims directly against financial balance sheets.
3. **Variable-Distance Proximity Retrieval**: Extending our Phase 1 positional index to support $k$-word proximity windows.
4. **Hybrid Retrieval Calibration**: Dynamically weighting BM25 and dense embeddings based on query entity density.
5. **Bayesian Multi-Source Aggregation**: Replacing heuristic decision thresholds with Bayesian evidence fusion and uncertainty estimation to produce calibrated output probabilities.

Thank you for your time and attention. I am now ready for your questions and viva voce evaluation."

---

*(End of Presentation Deck — Ready for Committee Q&A)*
