# FinGuard Architecture Documentation

This directory details the architectural flow, component relationships, and multi-stage pipeline of the FinGuard NLP system.

## Key Architecture Sections
1. **Phase 1: NLP Foundations & Classical Retrieval** (`phase1/README.md`)
   - Preprocessing & Cleaning $\to$ Finance Tokenizer $\to$ Polarity-Preserving Stopwords $\to$ POS-guided Lemmatization $\to$ Positional Inverted Index $\to$ Boolean / Phrase / TF-IDF Retrieval.
2. **Phase 2: Neural Retrieval, Reranking & Verification** (`phase2/README.md`)
   - Financial Claim Understanding $\to$ Structured Query Expansion $\to$ Candidate Evidence Retrieval (BM25 / Dense) $\to$ Neural Cross-Encoder Reranking $\to$ Zero-Shot NLI Verification $\to$ Verdict Aggregation.
3. **Faculty Presentation Guide** (`docs/PROFESSOR_DEMO_GUIDE.md`)
4. **Phase Completion Matrix** (`docs/PHASE_COMPLETION_MATRIX.md`)

