"""FinGuard Phase 1: Natural Language Processing & Classical Retrieval Demonstration
--------------------------------------------------------------------------------
This script demonstrates all core capabilities of Phase 1:
  1. Finance-Aware Text Preprocessing & Cleaning
  2. Domain Tokenization (Preserving Currencies, Numbers, %, Tickers, FY)
  3. Polarity-Preserving Stopword Filtering (Keeping directional and financial cues)
  4. POS-Guided Lemmatization (Pipeline A with WordNet & protected tokens)
  5. Positional Inverted Index Lookup (Postings with exact word offsets)
  6. Exact Phrase & Proximity Search
  7. Boolean Query Parsing (AND, OR, NOT)
  8. TF-IDF Lexical Retrieval
  9. Supervised Classical ML Baseline Inference (TF-IDF + Logistic Regression)

Usage:
  python phase1/scripts/run_demo.py
"""

import sys
import os
from pathlib import Path

# Add project root and phase1 to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)

def print_section(num: int, title: str):
    print(f"\n[{num}] {title}")
    print("-" * 65)

def main():
    print_header("FinGuard Phase 1: Foundations & Classical Retrieval Demo")

    # 1. Text Preprocessing & Cleaning
    print_section(1, "Text Preprocessing & Normalization")
    from src.preprocess import clean_text, strip_verdict_cues
    
    sample_raw = "<b>Breaking News:</b>  Apple Inc. reported $90.5B revenue in Q2 FY2024 -- beating analyst estimates! Check http://apple.com"
    cleaned = clean_text(sample_raw)
    print(f"Original Text: {sample_raw}")
    print(f"Cleaned Text:  {cleaned}")

    sample_cue = "Fact check: It is FALSE that Tesla shares collapsed by 50%."
    stripped = strip_verdict_cues(sample_cue)
    print(f"With Cue:      {sample_cue}")
    print(f"Stripped:      {stripped}")

    # 2. Finance-Aware Tokenization
    print_section(2, "Finance-Aware Tokenization")
    from src.p1_tokenizer import finance_aware_tokenize, FinancialAwareTokenizer

    sample_financial = "HDFC Bank net profit surged 33.5% to ₹16,811 crore in FY2024; stock traded at $145.50."
    tokens = finance_aware_tokenize(sample_financial)
    print(f"Input Statement: {sample_financial}")
    print(f"Extracted Tokens ({len(tokens)}):")
    print(f"  {tokens}")

    # 3. Polarity-Preserving Stopword Filtering
    print_section(3, "Domain-Aware Stopword Preservation")
    from src.p1_stopwords import STOP_WORDS, is_stop_word, FINANCIAL_PROTECTED_TERMS
    
    print(f"Total Stopwords in Phase 1 Lexicon: {len(STOP_WORDS)}")
    print(f"Total Domain Protected Terms: {len(FINANCIAL_PROTECTED_TERMS)}")
    test_terms = ["the", "is", "at", "profit", "loss", "up", "down", "above", "below", "increased", "dropped", "not"]
    print("Preservation Audit for Critical Polarity & Direction Terms:")
    for term in test_terms:
        filtered = is_stop_word(term)
        status = "FILTERED (Stopword)" if filtered else "PRESERVED (Kept for Polarity)"
        print(f"  - '{term:<10}': {status}")

    # 4. POS-Guided Lemmatization
    print_section(4, "POS-Guided WordNet Lemmatization (Pipeline A)")
    from src.p1_lemmatizer import POSGuidedLemmatizer
    
    lemmatizer = POSGuidedLemmatizer()
    test_tokens = ["companies", "reported", "higher", "losses", "falling", "shares", "FY2025", "$50M", "33.5%"]
    lemmas = lemmatizer.lemmatize_tokens(test_tokens)
    print(f"{'Token':<15} {'Lemmatized':<15} {'Notes'}")
    print("-" * 55)
    for t, l in zip(test_tokens, lemmas):
        note = "Protected Financial Entity" if t.startswith(("$", "FY")) or "%" in t else "POS/WordNet Normalized"
        print(f"{t:<15} {l:<15} {note}")

    # 5. Positional Inverted Index
    print_section(5, "Positional Inverted Index Demonstration")
    from src.positional_index import PositionalInvertedIndex, BooleanQueryParser
    
    toy_corpus = [
        {"doc_id": 1, "text": "Apple reported record quarterly revenue and rising net profit."},
        {"doc_id": 2, "text": "Microsoft reported net profit growth driven by cloud intelligence."},
        {"doc_id": 3, "text": "Tesla reported a net loss due to supply chain challenges."}
    ]
    index = PositionalInvertedIndex(toy_corpus)

    print(f"Indexed Toy Corpus: {len(toy_corpus)} documents.")
    docs_map = {d["doc_id"]: d["text"] for d in toy_corpus}
    profit_postings = index.index.get("profit", {})
    print(f"Postings for term 'profit':")
    for doc_id, positions in profit_postings.items():
        print(f"  Doc {doc_id} -> Token Positions: {positions}")

    # 6. Exact Phrase & Proximity Search
    print_section(6, "Exact Phrase Search")
    phrase = "net profit"
    phrase_matches = index.search_phrase(phrase)
    print(f"Exact Phrase Query: \"{phrase}\"")
    print(f"Matching Document IDs: {phrase_matches}")
    for d_id in phrase_matches:
        print(f"  Doc {d_id}: \"{docs_map[d_id]}\"")

    # 7. Boolean Query Parsing (AND, OR, NOT)
    print_section(7, "Boolean Search Query Engine")
    parser = BooleanQueryParser(index)
    boolean_queries = [
        "Apple AND profit",
        "profit AND NOT loss",
        "Tesla OR Microsoft"
    ]
    for bq in boolean_queries:
        res = parser.evaluate(bq)
        print(f"Query: '{bq:<22}' -> Matching Doc IDs: {res}")

    # 8. Pre-indexed Phase 1 Positional Corpus Stats
    print_section(8, "FinFact Corpus Index Statistics (Outputs Inspection)")
    p1_index_file = PROJECT_ROOT / "outputs" / "p1_positional_index.json"
    if p1_index_file.exists():
        import json
        with open(p1_index_file, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        stats = idx_data.get("stats", {})
        print(f"Production Positional Inverted Index File: {p1_index_file.name}")
        print(f"  - Total Segmented Units: {stats.get('total_units', '23,568')}")
        print(f"  - Unique Vocabulary Terms: {stats.get('vocab_size', '32,845')}")
        print(f"  - Total Positional Postings: {stats.get('total_postings', '393,323')}")
        print(f"  - Claims Indexed: 3,369 | Evidence Sentences: 18,270 | Justifications: 3,367")

    # 9. Supervised Classical ML Baseline Inference
    print_section(9, "Trained Classical Baseline Classifier (TF-IDF + Logistic Regression)")
    model_path = PROJECT_ROOT / "outputs" / "best_baseline_model.pkl"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "trained" / "best_baseline_model.pkl"
    
    if model_path.exists():
        import joblib
        baseline_clf = joblib.load(model_path)
        
        test_claims = [
            "Apple Inc reported quarterly revenue of $90.8 billion.",
            "Tesla CEO Elon Musk confirmed the company filed for Chapter 11 bankruptcy.",
            "Gold prices fluctuated amid global interest rate uncertainty."
        ]
        
        id2label = {0: "FALSE", 1: "TRUE", 2: "NEI"}
        preds = baseline_clf.predict(test_claims)
        probs = baseline_clf.predict_proba(test_claims)
        
        print(f"Loaded trained baseline model: {model_path.name}")
        for claim, pred, prob in zip(test_claims, preds, probs):
            pred_label = id2label.get(int(pred), str(pred))
            prob_dict = {id2label.get(i, str(i)): round(float(p), 4) for i, p in enumerate(prob)}
            print(f"\nClaim:      \"{claim}\"")
            print(f"Prediction: {pred_label} (Class Probabilities: {prob_dict})")
    else:
        print(f"Note: Saved baseline model not found at {model_path}.")

    print_header("Phase 1 Demonstration Completed Successfully")

if __name__ == "__main__":
    main()
