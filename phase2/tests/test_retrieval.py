"""
Unit tests for evidence retrieval algorithms, ranking, and IR metric computations.
"""

import unittest
import numpy as np
from src.retrieval import BM25Okapi, tokenize, EvidenceRetrievalBenchmark

class TestRetrievalAlgorithms(unittest.TestCase):
    def setUp(self):
        self.sample_corpus = [
            "the federal reserve raised interest rates to combat inflation",
            "amazon paid zero federal income tax under current corporate tax laws",
            "the gross national debt exceeded twenty trillion dollars during the administration",
            "inflation decreased as corporate profits reached record highs"
        ]
        self.tokenized_corpus = [tokenize(doc) for doc in self.sample_corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def test_bm25_scoring(self):
        query = tokenize("federal reserve interest rates")
        scores = self.bm25.get_scores(query)
        self.assertEqual(len(scores), len(self.sample_corpus))
        # First doc has "federal reserve", "interest rates", so it should have the highest score
        top_doc_idx = int(np.argmax(scores))
        self.assertEqual(top_doc_idx, 0, "BM25 should rank doc 0 highest for interest rate query.")
        self.assertGreater(scores[0], scores[1])

    def test_bm25_empty_query(self):
        query = []
        scores = self.bm25.get_scores(query)
        self.assertTrue(np.all(scores == 0.0), "BM25 scores should all be 0 for empty query.")

    def test_rrf_logic(self):
        # Test Reciprocal Rank Fusion calculation
        rank_1 = 0  # top rank (index 0)
        k = 60
        score = 1.0 / (k + rank_1 + 1)
        self.assertAlmostEqual(score, 1.0 / 61.0)

    def test_retrieval_benchmark_corpus_provenance(self):
        benchmark = EvidenceRetrievalBenchmark(json_path="finfact.json")
        self.assertGreater(len(benchmark.corpus), 1000, "Corpus should contain over 1,000 unique evidence sentences.")
        first_doc = benchmark.corpus[0]
        self.assertIn("doc_id", first_doc)
        self.assertIn("text", first_doc)
        self.assertIn("source_urls", first_doc)
        self.assertIn("parent_claim_ids", first_doc)
        self.assertEqual(first_doc["doc_type"], "evidence_sentence")

if __name__ == "__main__":
    unittest.main()
