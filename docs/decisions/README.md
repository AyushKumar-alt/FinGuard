# FinGuard Architectural & Empirical Decision Records (ADRs)

This directory documents key architectural decisions, design choices, and scientific trade-offs made across the development of FinGuard.

---

## Key Decision Records

### ADR-001: Polarity Preservation in Stopword Filtering
- **Decision**: Reject standard NLTK / Scikit-learn stopword lists.
- **Rationale**: Standard lists strip words like *up, down, above, below, loss, profit, not*. In financial fact-checking, removing these reverses the polarity of the claim.
- **Outcome**: Preserved 58 critical financial terms in `phase1/src/tokenization/stopwords.py`.

### ADR-002: WordNet Lemmatization with Entity Bypass (Pipeline A)
- **Decision**: Use POS-guided WordNet lemmatization with strict regex-based protection for financial entities.
- **Rationale**: Standard lemmatizers distort currency quantities, percentages, ticker symbols, and fiscal markers (e.g. `FY2025` or `401(k)`).
- **Outcome**: Protected tokens bypass morphological alteration in `phase1/src/lemmatization/lemmatizer.py`.

### ADR-003: Inverted-Index Okapi BM25 as Default Retrieval Engine
- **Decision**: Standardize on BM25Okapi over candidate evidence sentences for default production verification.
- **Rationale**: Lexical matching ensures exact entity and numerical grounding (e.g., exact dollar amounts and corporate names). Dense retrieval is available in hybrid mode via RRF ($k=60$).

### ADR-004: Rigorous Source-Grouped Benchmark Evaluation
- **Decision**: Reject random train/val/test splits as the sole measure of model quality; adopt strictly held-out source-grouped splits.
- **Rationale**: Random splits suffer from publisher leakage, where claims from the same financial news outlet appear in both train and test.
- **Finding**: Revealed that while the baseline achieved 55.56% accuracy on held-out source URLs, the evidence-grounded neural system achieved 40.30% due to retrieval domain shift and zero-shot NLI neutral collapse.

