"""
Pipeline 1: Financial-Aware Tokenizer with POS-Guided Lemmatization.
Standalone, reusable component. Does not modify any existing module.

Processing order (documented):
  1. Finance-aware tokenization (reuses claim_understanding.finance_aware_tokenize)
  2. Lowercase normalization (for non-protected tokens)
  3. Stop-word removal (scoped to Pipeline 1)
  4. POS-guided lemmatization (NLTK WordNet + averaged_perceptron_tagger_eng)
  5. Returns (raw_tokens, filtered_tokens, lemmatized_tokens) for full traceability
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.claim_understanding import finance_aware_tokenize  # REUSE existing tokenizer
from src.p1_stopwords import filter_tokens, STOPWORD_CONFIG
from src.p1_lemmatizer import POSGuidedLemmatizer, _is_protected

TOKENIZER_VERSION = "1.0-p1-financial-lemma"

class FinancialAwareTokenizer:
    """
    Pipeline 1 tokenizer: finance-aware -> stopword removal -> POS lemmatization.
    Deterministic. Reuses existing finance_aware_tokenize() from claim_understanding.py.
    """
    def __init__(self):
        self.lemmatizer = POSGuidedLemmatizer()
        self.version = TOKENIZER_VERSION
        self._config = {
            "version": TOKENIZER_VERSION,
            "base_tokenizer": "src.claim_understanding.finance_aware_tokenize",
            "stopword_config": STOPWORD_CONFIG,
            "lemmatizer": self.lemmatizer.library_info,
            "pos_tagger": "nltk.averaged_perceptron_tagger_eng",
            "lemmatization_available": self.lemmatizer.is_available,
        }

    @property
    def config(self) -> dict:
        return dict(self._config)

    def tokenize(self, text: str) -> dict:
        """
        Full Pipeline 1 preprocessing.
        Returns dict with all intermediate stages for traceability:
          raw_tokens        - output of finance_aware_tokenize (original casing)
          filtered_tokens   - after stop-word removal (lowercase non-protected)
          lemmatized_tokens - after POS-guided lemmatization
        """
        if not text or not isinstance(text, str):
            return {"raw_tokens": [], "filtered_tokens": [], "lemmatized_tokens": []}
        
        # Stage 1: Finance-aware tokenization (reuse existing)
        raw_tokens = finance_aware_tokenize(text)
        
        # Stage 2: Lowercase non-protected tokens
        lowercased = [t if _is_protected(t) else t.lower() for t in raw_tokens]
        
        # Stage 3: Stop-word removal (Pipeline 1 scoped)
        filtered = filter_tokens(lowercased)
        
        # Stage 4: POS-guided lemmatization
        lemmatized = self.lemmatizer.lemmatize_tokens(filtered)
        
        return {
            "raw_tokens": raw_tokens,
            "filtered_tokens": filtered,
            "lemmatized_tokens": lemmatized,
        }

    def get_index_tokens(self, text: str) -> list:
        """Returns lemmatized tokens for indexing (the final stage output)."""
        return self.tokenize(text)["lemmatized_tokens"]

    def get_query_tokens(self, text: str) -> list:
        """Applies same preprocessing to query as to corpus. Consistent."""
        return self.get_index_tokens(text)

