import unittest
import os
from src.p1_pipeline import Pipeline1
from src.p1_index import DEFAULT_P1_INDEX_PATH
from src.pipeline import FinGuardPipeline

class TestP1Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline without rebuilding index if already saved
        cls.p1 = Pipeline1(force_rebuild=False)

    def test_pipeline1_index_separate(self):
        self.assertTrue(DEFAULT_P1_INDEX_PATH.endswith("p1_positional_index.json"))
        self.assertNotEqual(DEFAULT_P1_INDEX_PATH, os.path.join("outputs", "positional_index.json"))
        self.assertTrue(os.path.exists(DEFAULT_P1_INDEX_PATH), "P1 index artifact should exist on disk")

    def test_standalone_run(self):
        claim = "Apple reported record iPhone revenue in FY2025."
        result = self.p1.run(claim, top_k=5)
        self.assertIn("claim", result)
        self.assertIn("candidates", result)
        self.assertIn("query_lemmas", result)
        self.assertEqual(result["pipeline"], "Pipeline1_FinancialLemmatized")
        self.assertGreater(len(result["candidates"]), 0)

    def test_retrieval_mode_integration(self):
        pipeline = FinGuardPipeline(load_heavy_models=False)
        claim = "Tesla automotive revenue declined significantly."
        result = pipeline.run(claim, retrieval_mode="p1_lemma", use_reranker=False, use_nli=False, top_k_retrieve=3)
        self.assertEqual(result["retrieval_mode"], "p1_lemma")
        self.assertIn("selected_evidence_docs", result)
        self.assertGreater(len(result["selected_evidence_docs"]), 0)
        self.assertEqual(result["selected_evidence_docs"][0]["retrieval_pipeline"], "pipeline1_lemmatized_tfidf")

if __name__ == "__main__":
    unittest.main()
