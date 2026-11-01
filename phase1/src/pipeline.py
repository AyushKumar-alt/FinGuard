"""
Pipeline 1: Standalone orchestrator for Financial-Aware Lemmatized Evidence Retrieval.
Entry point: Pipeline1.retrieve(claim, top_k) or Pipeline1.run(claim)

Does NOT replace FinGuardPipeline. Can be used standalone or as retrieval source
fed into existing verifier (via retrieval_with_provenance()).
"""
import os
import sys
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.p1_tokenizer import FinancialAwareTokenizer
from src.p1_corpus_builder import build_p1_corpus, DEFAULT_CSV
from src.p1_index import Pipeline1Index, DEFAULT_P1_INDEX_PATH
from src.p1_retriever import NaturalLanguageEvidenceRetriever

class Pipeline1:
    """
    Pipeline 1: Financial-Aware Lemmatization + Positional Index + NL Evidence Retrieval.
    Standalone. Preserves existing FinGuardPipeline behavior unchanged.
    """
    def __init__(self, csv_path: str = DEFAULT_CSV, 
                 index_path: str = DEFAULT_P1_INDEX_PATH,
                 force_rebuild: bool = False):
        self.csv_path = csv_path
        self.index_path = index_path
        self.tokenizer = FinancialAwareTokenizer()
        self.p1_index = Pipeline1Index()
        self.retriever = None
        self._corpus = None
        
        if not force_rebuild and os.path.exists(index_path):
            try:
                self.p1_index.load(index_path)
                print("Pipeline 1 initialized from existing index.", flush=True)
                self._rebuild_corpus_for_retriever(csv_path)
            except Exception as e:
                print(f"Warning: Failed loading index ({e}), rebuilding fresh...", flush=True)
                self._build(csv_path, index_path)
        else:
            self._build(csv_path, index_path)

    def _build(self, csv_path: str, index_path: str):
        print("Building Pipeline 1 corpus and index...", flush=True)
        result = build_p1_corpus(csv_path=csv_path, tokenizer=self.tokenizer)
        self._corpus = result["corpus"]
        print(f"Corpus stats: {json.dumps(result['stats'], indent=2)}", flush=True)
        
        self.p1_index.build(result)
        self.p1_index.save(index_path)
        
        self.retriever = NaturalLanguageEvidenceRetriever(self.p1_index, self.tokenizer)
        self.retriever.fit(self._corpus)

    def _rebuild_corpus_for_retriever(self, csv_path: str):
        """Reconstructs corpus tokens for retriever without re-tokenizing if cached."""
        if hasattr(self.p1_index, "_corpus_lemmas") and self.p1_index._corpus_lemmas:
            doc_count = len(self.p1_index._corpus_lemmas)
            lemmatized_docs = [self.p1_index._corpus_lemmas.get(i, "") for i in range(doc_count)]
            self.retriever = NaturalLanguageEvidenceRetriever(self.p1_index, self.tokenizer)
            self.retriever._corpus_texts = self.p1_index._corpus_texts
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.retriever._tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
                min_df=1,
            )
            self.retriever._tfidf_matrix = self.retriever._tfidf.fit_transform(lemmatized_docs)
            self.retriever._fitted = True
            print(f"Pipeline 1 TF-IDF fitted from cached index: {doc_count:,} docs, vocab={len(self.retriever._tfidf.vocabulary_):,}", flush=True)
            return

        result = build_p1_corpus(csv_path=csv_path, tokenizer=self.tokenizer)
        self._corpus = result["corpus"]
        self.retriever = NaturalLanguageEvidenceRetriever(self.p1_index, self.tokenizer)
        self.retriever.fit(self._corpus)

    def retrieve(self, claim: str, top_k: int = 10) -> list:
        """Natural-language claim -> top-K evidence candidates. Returns (doc_id, score) list."""
        if not self.retriever:
            return []
        return self.retriever.retrieve_for_claim(claim, top_k=top_k)

    def retrieve_with_provenance(self, claim: str, top_k: int = 10) -> list:
        """Returns full evidence dicts compatible with existing pipeline candidate_docs format."""
        if not self.retriever:
            return []
        return self.retriever.retrieve_with_provenance(claim, top_k=top_k)

    def get_index_metadata(self) -> dict:
        return self.p1_index.metadata

    def run(self, claim: str, top_k: int = 10) -> dict:
        """
        Standalone run: returns retrieval results without verification.
        Verification should be done via the existing FinGuardPipeline or FinancialClaimVerifier.
        """
        candidates = self.retrieve_with_provenance(claim, top_k=top_k)
        query_info = self.tokenizer.tokenize(claim)
        return {
            "claim": claim,
            "query_lemmas": query_info["lemmatized_tokens"],
            "query_filtered_tokens": query_info["filtered_tokens"],
            "candidates": candidates,
            "total_retrieved": len(candidates),
            "pipeline": "Pipeline1_FinancialLemmatized",
        }











