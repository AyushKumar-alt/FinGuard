"""
Pipeline 1: Corpus Builder from Existing FinFact Dataset.
Loads from data/processed_finfact.csv (already built from finfact.json).
Does NOT regenerate splits or modify existing data files.
Applies Pipeline 1 text cleaning and filtering.
"""
import os
import sys
import json
import re
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocess import clean_text  # REUSE existing clean_text
from src.p1_tokenizer import FinancialAwareTokenizer

DEFAULT_CSV = os.path.join(REPO_ROOT, "data", "processed_finfact.csv")
MIN_TOKEN_COUNT = 4  # consistent with existing retrieval.py

def build_p1_corpus(
    csv_path: str = DEFAULT_CSV,
    tokenizer: FinancialAwareTokenizer = None,
) -> dict:
    """
    Builds the Pipeline 1 evidence corpus from the existing processed dataset.
    
    Returns:
        {
          "corpus": List[Dict],          # doc_id, text, lemmatized_tokens, source_url, record_ids, label, ...
          "stats": Dict,                 # audit statistics
          "tokenizer_config": Dict,
        }
    """
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        resolved_csv = os.path.join(REPO_ROOT, csv_path)
        if os.path.exists(resolved_csv):
            csv_path = resolved_csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    if tokenizer is None:
        tokenizer = FinancialAwareTokenizer()
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    # Audit counters
    total_input_records = len(df)
    total_raw_sentences = 0
    empty_excluded = 0
    short_excluded = 0
    duplicate_excluded = 0
    
    seen_texts = {}   # normalized_text -> doc_id (deduplication)
    corpus = []
    
    for _, row in df.iterrows():
        record_id = int(row["id"]) if "id" in row and pd.notna(row["id"]) else -1
        source_url = str(row.get("url", "")) if pd.notna(row.get("url", "")) else ""
        label = str(row.get("label", "")) if pd.notna(row.get("label", "")) else ""
        
        # Use cleaned_evidence_text (already has strip_verdict_cues applied)
        # This reuses the existing preprocessing output — no leakage of justification or labels
        evidence_text = str(row.get("cleaned_evidence_text", "")) if pd.notna(row.get("cleaned_evidence_text", "")) else ""
        
        if not evidence_text.strip():
            empty_excluded += 1
            continue
        
        # Split into sentences (simple sentence splitting; consistent with existing retrieval.py approach)
        raw_sentences = re.split(r'(?<=[.!?])\s+', evidence_text.strip())
        
        for sent in raw_sentences:
            total_raw_sentences += 1
            sent = clean_text(sent)
            
            if not sent.strip():
                empty_excluded += 1
                continue
            
            tokens = tokenizer.get_index_tokens(sent)
            if len(tokens) < MIN_TOKEN_COUNT:
                short_excluded += 1
                continue
            
            # Deduplication by normalized text (lowercase stripped)
            norm_key = sent.lower().strip()
            if norm_key in seen_texts:
                duplicate_excluded += 1
                existing_doc_id = seen_texts[norm_key]
                # Merge parent record IDs (same sentence from multiple claims)
                if record_id not in corpus[existing_doc_id]["record_ids"]:
                    corpus[existing_doc_id]["record_ids"].append(record_id)
                continue
            
            doc_id = len(corpus)
            seen_texts[norm_key] = doc_id
            
            corpus.append({
                "doc_id": doc_id,
                "text": sent,                        # original cleaned text (for display/attribution)
                "lemmatized_tokens": tokens,          # for indexing
                "source_url": source_url,
                "record_ids": [record_id],
                "label": label,                       # record-level label (NOT used for retrieval)
                # Positional index compatibility fields:
                "source_urls": [source_url] if source_url else [],
                "parent_claim_ids": [record_id],
            })
    
    stats = {
        "input_csv": csv_path,
        "input_record_count": total_input_records,
        "total_raw_sentences_extracted": total_raw_sentences,
        "empty_excluded": empty_excluded,
        "short_excluded_min_tokens": short_excluded,
        "duplicate_excluded": duplicate_excluded,
        "final_corpus_size": len(corpus),
        "distinct_source_urls": len(set(d["source_url"] for d in corpus if d["source_url"])),
        "min_token_threshold": MIN_TOKEN_COUNT,
    }
    
    return {
        "corpus": corpus,
        "stats": stats,
        "tokenizer_config": tokenizer.config,
    }

