"""
Pipeline 1: Natural-Language Evidence Retriever.
Accepts full natural-language financial claims (not Boolean queries).
Applies Pipeline 1 preprocessing to both query and corpus, then retrieves
using TF-IDF cosine similarity over lemmatized tokens.
Primary interface: retrieve_for_claim(claim, top_k)

NOT a Boolean query system — Boolean search from positional_index.py remains intact.
"""
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.p1_tokenizer import FinancialAwareTokenizer
from src.p1_index import Pipeline1Index

class NaturalLanguageEvidenceRetriever:
    """
    Pipeline 1 retriever: TF-IDF over lemmatized tokens for natural-language claims.
    Returns evidence in same (doc_id, score) format as existing retrieval.py methods.
    """
    def __init__(self, p1_index: Pipeline1Index, tokenizer: FinancialAwareTokenizer = None):
        self.p1_index = p1_index
        self.tokenizer = tokenizer or FinancialAwareTokenizer()
        self._tfidf = None
        self._tfidf_matrix = None
        self._corpus_texts = {}   # doc_id -> original text
        self._fitted = False

    def fit(self, corpus: list):
        """
        Fits TF-IDF vectorizer over lemmatized corpus tokens.
        corpus: list of dicts with doc_id, text, lemmatized_tokens
        """
        self._corpus_texts = {doc["doc_id"]: doc["text"] for doc in corpus}
        
        # Use pre-lemmatized tokens joined as string (consistent with index)
        lemmatized_docs = [" ".join(doc["lemmatized_tokens"]) for doc in corpus]
        
        self._tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=1,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(lemmatized_docs)
        self._fitted = True
        print(f"Pipeline 1 TF-IDF fitted: {len(corpus):,} docs, vocab={len(self._tfidf.vocabulary_):,}")

    def retrieve_for_claim(self, claim: str, top_k: int = 10) -> list:
        """
        Accept natural-language claim. Apply Pipeline 1 preprocessing to query.
        Returns list of (doc_id, score) tuples — same format as existing retrieval.py.
        """
        if not claim or not isinstance(claim, str) or not claim.strip():
            return []
        if not self._fitted:
            raise RuntimeError("Retriever not fitted. Call fit(corpus) first.")
        
        # Apply SAME preprocessing as corpus indexing (consistent)
        query_lemmas = self.tokenizer.get_query_tokens(claim)
        if not query_lemmas:
            return []
        
        query_text = " ".join(query_lemmas)
        q_vec = self._tfidf.transform([query_text])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0.0]
        return results

    def retrieve(self, claim: str, top_k: int = 10) -> list:
        """Alias for retrieve_for_claim."""
        return self.retrieve_for_claim(claim, top_k=top_k)

    def retrieve_with_provenance(self, claim: str, top_k: int = 10) -> list:
        """
        Returns full evidence dicts compatible with pipeline.py candidate_docs format:
        [{doc_id, text, score, source_urls, parent_claim_ids}]
        """
        raw_results = self.retrieve_for_claim(claim, top_k=top_k)
        meta = self.p1_index.inner_index.doc_metadata
        
        candidates = []
        for doc_id, score in raw_results:
            doc_meta = meta.get(doc_id, {})
            text = self._corpus_texts.get(doc_id, "") or self.p1_index.get_original_text(doc_id)
            candidates.append({
                "doc_id": doc_id,
                "text": text,
                "score": round(score, 4),
                "source_urls": doc_meta.get("source_urls", []),
                "parent_claim_ids": doc_meta.get("parent_claim_ids", []),
                "retrieval_pipeline": "pipeline1_lemmatized_tfidf",
            })
        return candidates

