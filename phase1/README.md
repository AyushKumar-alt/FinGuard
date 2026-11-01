# FinGuard — Phase 1: Natural Language Processing Foundations & Classical Retrieval

## Executive Overview

**Phase 1** establishes the core linguistic and information retrieval foundations for the FinGuard financial fact-checking architecture. It focuses on domain-adapted text preprocessing, finance-aware tokenization, polarity-preserving stopword filtering, part-of-speech (POS)-guided lemmatization, inverted positional indexing with term-offset tracking, boolean/phrase retrieval, and supervised classical machine learning baselines.

---

## Directory Structure

```
phase1/
├── README.md                           # This comprehensive guide
├── src/                                # Core Phase 1 implementations
│   ├── preprocessing/                  # Text normalization & cue stripping
│   │   └── text_cleaner.py             # HTML removal, whitespace, quote normalization
│   ├── tokenization/                   # Domain-aware tokenization & stopwords
│   │   ├── tokenizer.py                # Regex tokenizer preserving financial entities
│   │   └── stopwords.py                # 214 stopwords preserving 58 polarity cues
│   ├── lemmatization/                  # Morphological normalization
│   │   └── lemmatizer.py               # POS-guided WordNet lemmatizer (Pipeline A)
│   ├── indexing/                       # Inverted indexing engine
│   │   ├── positional_index.py         # Positional inverted index & postings list
│   │   ├── corpus_builder.py           # Segments FinFact into 23,568 retrieval units
│   │   └── p1_index.py                 # Persistent JSON index serialization wrapper
│   ├── retrieval/                      # Lexical & syntactic query engines
│   │   ├── boolean/                    # Boolean search parser (Shunting-Yard, AND/OR/NOT)
│   │   ├── phrase/                     # Exact phrase & proximity search algorithms
│   │   └── tfidf/                      # Sublinear TF-IDF retrieval scoring
│   ├── classical_ml/                   # Supervised classification baselines
│   │   ├── baseline_models.py          # TF-IDF + Logistic Regression, SVM, RF, Naive Bayes
│   │   └── source_grouped_baseline.py  # Leakage-resistant source-grouped baseline trainer
│   └── pipeline.py                     # Pipeline1 standalone orchestrator
├── scripts/
│   └── run_demo.py                     # Self-contained Phase 1 demonstration script
├── experiments/                        # Phase 1 experimental configurations
├── results/                            # Artifacts, evaluation outputs & visual figures
│   ├── metrics/                        # JSON performance metrics
│   │   ├── baseline_metrics.json
│   │   └── source_grouped_baseline_results.json
│   ├── figures/                        # Confusion matrices & distribution plots
│   │   ├── baseline_comparison.png
│   │   ├── best_baseline_confusion_matrix.png
│   │   ├── grouped_split_confusion_matrix.png
│   │   └── label_distribution.png
│   └── logs/                           # Execution logs
│       └── experiment.log
├── reports/                            # Technical methodology reports
│   └── FINGuard_NLP_METHODOLOGY_AUDIT.md
└── tests/                              # Dedicated Phase 1 unit test suite
    ├── test_p1_stopwords.py
    ├── test_p1_tokenizer.py
    ├── test_p1_lemmatizer.py
    ├── test_p1_index.py
    ├── test_p1_corpus.py
    ├── test_p1_retriever.py
    ├── test_p1_pipeline.py
    ├── test_positional_index.py
    └── test_source_grouped_baseline.py
```

---

## Core Components & Methodologies

### 1. Finance-Aware Preprocessing & Tokenization
- **Text Cleaner** (`src/preprocessing/text_cleaner.py`):
  Strips noisy HTML tags, normalizes Unicode punctuation, fixes quotation anomalies, and provides `strip_verdict_cues()` to prevent data leakage in claim verification.
- **Financial Tokenizer** (`src/tokenization/tokenizer.py`):
  Standard NLP tokenizers fragment financial constructs (e.g., splitting `₹16,811` into `₹`, `16`, `,`, `811`). FinGuard implements custom regex patterns preserving:
  - Currencies & Amounts: `$90.8B`, `₹16,811 crore`, `€50M`, `£1.2B`
  - Percentages: `33.5%`, `0.25%`, `-4.2%`
  - Fiscal Periods: `FY2024`, `Q2 2025`, `FY2025-26`
  - Financial Ratios & Identifiers: `401(k)`, `P/E`, `EV/EBITDA`

### 2. Polarity-Preserving Stopword Filtering (`src/tokenization/stopwords.py`)
Standard stopword lists remove directional indicators like *up, down, above, below, loss, not*. In finance, removing *"loss"* changes *"company reported a net loss"* to *"company reported net"*, reversing polarity.
- FinGuard preserves **58 critical financial polarity terms** (including negation, fiscal units, directionality).
- Retains 214 high-frequency non-informative domain stopwords.

### 3. POS-Guided WordNet Lemmatization (`src/lemmatization/lemmatizer.py`)
- Maps Penn Treebank POS tags (`NN*` $\to$ noun, `VB*` $\to$ verb, `JJ*` $\to$ adjective, `RB*` $\to$ adverb).
- Resolves contextual inflections (e.g., *"shares"* as noun stays *"share"*, *"falling"* as verb becomes *"fall"*).
- Protected tokens (tickers, currencies, fiscal markers) strictly bypass lemmatization.

### 4. Positional Inverted Index (`src/indexing/positional_index.py`)
- Stores positional offsets: `term -> {doc_id: [pos_0, pos_1, ...]}`.
- Enables exact phrase matching ($pos_{i+1} = pos_i + 1$) and proximity queries.
- Built across **23,568 segmented units** from FinFact:
  - **Indexed Units**: 23,568 (3,369 claims + 18,270 evidence sentences + 3,367 justifications)
  - **Vocabulary Size**: 32,845 unique lemmas
  - **Total Positional Postings**: 393,323

### 5. Query Engines (`src/retrieval/`)
- **Exact Phrase Search**: Deterministic positional verification.
- **Boolean Search Parser**: Full Shunting-Yard algorithm supporting `AND`, `OR`, `NOT`, and parenthetical grouping.
- **TF-IDF Retrieval**: Sublinear term-frequency weighting with cosine similarity ranking.

### 6. Supervised Classical ML Baselines (`src/classical_ml/`)
- Models: Logistic Regression (balanced class weights), Linear SVM, Random Forest, Multinomial Naive Bayes.
- Features: Word & character n-grams ($1 \le n \le 3$), sublinear TF scaling.
- Evaluated on both random 70/15/15 splits and strictly held-out **source-grouped splits** ($N=675$ test claims) to prevent publisher leakage.

---

## Experimental Results Summary

| Model / Architecture | Split Strategy | Accuracy | Macro-F1 | False F1 | True F1 | NEI F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **Random Split** | **67.00%** | **59.98%** | **78.71%** | **65.00%** | **36.23%** |
| Linear SVM | Random Split | 64.62% | 58.74% | 75.31% | 63.89% | 37.01% |
| Random Forest | Random Split | 61.46% | 46.10% | 74.45% | 60.10% | 03.74% |
| Multinomial Naive Bayes | Random Split | 61.46% | 45.47% | 74.59% | 59.88% | 01.94% |
| **Logistic Regression (Raw Text)** | **Source-Grouped** | **55.56%** | **48.97%** | **66.19%** | **52.27%** | **28.44%** |
| Logistic Regression (Cleaned Text) | Source-Grouped | 54.96% | 48.43% | 65.75% | 51.52% | 28.02% |

---

## Quickstart & Execution

### 1. Run the Complete Phase 1 Demonstration
```powershell
python phase1/scripts/run_demo.py
```

### 2. Run Phase 1 Unit Tests (58 Tests)
```powershell
python -m unittest discover -s phase1/tests -p "test_*.py"
```

### 3. Inspect Pre-Computed Positional Index
```powershell
python -c "import json; d=json.load(open('outputs/phase1/p1_positional_index.json')); print(d['stats'])"
```

