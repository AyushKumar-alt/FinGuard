"""
Pipeline 1: Positional Inverted Index built from lemmatized tokens.
WRAPS the existing PositionalInvertedIndex — does not modify it.
Stores a separate artifact: outputs/p1_positional_index.json

Index version: "1.0-p1-lemmatized"
This version string distinguishes Pipeline 1 index from the existing "2.0-positional" index.
"""
import os
import sys
import json
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.positional_index import PositionalInvertedIndex  # REUSE existing class

P1_INDEX_VERSION = "1.0-p1-lemmatized"
DEFAULT_P1_INDEX_PATH = os.path.join(REPO_ROOT, "outputs", "p1_positional_index.json")

class Pipeline1Index:
    """
    Builds and serializes a positional inverted index from Pipeline 1 lemmatized corpus.
    Uses the existing PositionalInvertedIndex internally.
    Stores separate metadata in a wrapper JSON structure.
    """
    def __init__(self):
        self._index = PositionalInvertedIndex()
        self.metadata = {}
        self.corpus = []
        self._corpus_texts = {}

    def build(self, corpus_result: dict):
        """
        corpus_result: output from p1_corpus_builder.build_p1_corpus()
        """
        corpus = corpus_result["corpus"]
        self.corpus = corpus
        self._corpus_texts = {doc["doc_id"]: doc["text"] for doc in corpus}
        
        # Build index from lemmatized tokens (NOT raw text)
        index_corpus = []
        for doc in corpus:
            index_corpus.append({
                "doc_id": doc["doc_id"],
                "text": " ".join(doc["lemmatized_tokens"]),  # join lemmas as text for position indexing
                "source_urls": doc.get("source_urls", []),
                "parent_claim_ids": doc.get("parent_claim_ids", []),
            })
        
        self._index.build_index(index_corpus)
        
        self.metadata = {
            "pipeline_name": "Pipeline1_FinancialLemmatized",
            "pipeline_version": P1_INDEX_VERSION,
            "base_index_class": "src.positional_index.PositionalInvertedIndex",
            "corpus_source": corpus_result["stats"].get("input_csv", "data/processed_finfact.csv"),
            "corpus_fingerprint": str(corpus_result["stats"].get("final_corpus_size", 0)),
            "tokenizer_config": corpus_result.get("tokenizer_config", {}),
            "stopword_config": corpus_result.get("tokenizer_config", {}).get("stopword_config", {}),
            "lemmatizer": corpus_result.get("tokenizer_config", {}).get("lemmatizer", ""),
            "pos_tagger": corpus_result.get("tokenizer_config", {}).get("pos_tagger", ""),
            "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "indexed_evidence_units": len(corpus),
            "vocabulary_size": len(self._index.index),
            "total_postings": sum(len(v) for v in self._index.index.values()),
            "corpus_stats": corpus_result["stats"],
        }

    def save(self, filepath: str = DEFAULT_P1_INDEX_PATH):
        """Saves Pipeline 1 index with metadata wrapper. Separate from existing index."""
        if not os.path.isabs(filepath):
            filepath = os.path.join(REPO_ROOT, filepath)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Build the save structure: metadata wrapper + base index data
        corpus_texts_dict = self._corpus_texts or {doc["doc_id"]: doc["text"] for doc in self.corpus}
        corpus_lemmas_dict = self._corpus_lemmas if hasattr(self, "_corpus_lemmas") and self._corpus_lemmas else {
            doc["doc_id"]: " ".join(doc["lemmatized_tokens"]) for doc in self.corpus
        }
        data = {
            "p1_metadata": self.metadata,
            "index_version": P1_INDEX_VERSION,
            "doc_count": self._index.doc_count,
            "all_doc_ids": list(self._index.all_doc_ids),
            "doc_metadata": {str(k): v for k, v in self._index.doc_metadata.items()},
            "index": {k: {str(did): pos for did, pos in postings.items()} 
                      for k, postings in self._index.index.items()},
            # Store original text for display (not in base index)
            "corpus_texts": {str(did): text for did, text in corpus_texts_dict.items()},
            "corpus_lemmas": {str(did): lemmas for did, lemmas in corpus_lemmas_dict.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Pipeline 1 index saved: {filepath} ({len(self._index.index):,} unique lemmas, {self._index.doc_count:,} docs)")

    def load(self, filepath: str = DEFAULT_P1_INDEX_PATH):
        """Loads and validates Pipeline 1 index."""
        if not os.path.isabs(filepath) and not os.path.exists(filepath):
            resolved = os.path.join(REPO_ROOT, filepath)
            if os.path.exists(resolved):
                filepath = resolved

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate version
        loaded_version = data.get("index_version", "")
        if loaded_version != P1_INDEX_VERSION:
            raise ValueError(f"Index version mismatch: expected {P1_INDEX_VERSION}, got {loaded_version}. "
                             f"Rebuild index with current Pipeline 1 configuration.")
        
        self.metadata = data.get("p1_metadata", {})
        
        # Restore base index fields
        self._index.version = loaded_version
        self._index.doc_count = data.get("doc_count", 0)
        self._index.all_doc_ids = set(data.get("all_doc_ids", []))
        self._index.doc_metadata = {int(k): v for k, v in data.get("doc_metadata", {}).items()}
        self._index.index = {k: {int(did): pos for did, pos in postings.items()} 
                             for k, postings in data.get("index", {}).items()}
        
        # Restore corpus texts for display
        self._corpus_texts = {int(k): v for k, v in data.get("corpus_texts", {}).items()}
        self._corpus_lemmas = {int(k): v for k, v in data.get("corpus_lemmas", {}).items()}
        
        print(f"Pipeline 1 index loaded: {filepath}")
        print(f"  Indexed units: {self._index.doc_count:,}, Vocab: {len(self._index.index):,}")
        return self

    def get_term_freq(self, term: str, doc_id: int) -> int:
        return self._index.get_term_freq(term, doc_id)

    def get_doc_freq(self, term: str) -> int:
        return self._index.get_doc_freq(term)

    def get_original_text(self, doc_id: int) -> str:
        if self._corpus_texts:
            return self._corpus_texts.get(doc_id, "")
        for doc in self.corpus:
            if doc["doc_id"] == doc_id:
                return doc["text"]
        return ""

    @property
    def inner_index(self) -> PositionalInvertedIndex:
        return self._index
