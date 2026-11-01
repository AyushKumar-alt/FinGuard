"""
Unit tests for End-to-End FinGuard Pipeline, NLI inference, multi-evidence conflict aggregation,
and graceful offline/unavailable model fallback.
"""

import unittest
from src.verifier import FinancialClaimVerifier
from src.pipeline import FinGuardPipeline
from src.dense_retrieval import PretrainedDenseRetriever
from src.reranker import PretrainedCrossEncoderReranker

class TestPipelineAndNLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline without heavy models for fast CI unit tests
        cls.light_pipeline = FinGuardPipeline(json_path="finfact.json", load_heavy_models=False)
        cls.verifier = FinancialClaimVerifier(
            baseline_model_path="outputs/best_baseline_model.pkl",
            use_transformer=False
        )

    def test_pipeline_empty_claim(self):
        res = self.light_pipeline.run("")
        self.assertEqual(res["verdict"], "NEI")
        self.assertTrue(res["is_uncertain"])
        self.assertIn("Empty Claim", res["verdict_display"])

    def test_pipeline_light_run(self):
        claim = "Did Biden call for an end to shareholder capitalism?"
        res = self.light_pipeline.run(
            claim=claim,
            retrieval_mode="bm25",
            use_query_expansion=True,
            use_reranker=False,
            use_nli=False,
            top_k_retrieve=5,
            top_k_rerank=3
        )
        self.assertEqual(res["claim"], claim)
        self.assertIn(res["verdict"], ["true", "false", "NEI"])
        self.assertGreaterEqual(len(res["selected_evidence_docs"]), 1)
        # Check provenance
        first_doc = res["selected_evidence_docs"][0]
        self.assertIn("source_urls", first_doc)
        self.assertIn("parent_claim_ids", first_doc)
        self.assertIn("score", first_doc)

    def test_conflicting_evidence_aggregation(self):
        # Mock candidate sentences: one clearly supporting, one clearly contradicting
        claim = "The company reported profit growth of 20% in 2022."
        # Directly mock NLI predictions via verify_claim
        # When engine_choice='auto' without transformer, fallback occurs
        evidence = [
            {"text": "The company reported profit growth of 20% in 2022.", "source_urls": ["url1"], "doc_id": 1, "score": 0.9},
            {"text": "The company suffered a net loss with profit declining 15% in 2022.", "source_urls": ["url2"], "doc_id": 2, "score": 0.85}
        ]
        res = self.verifier.verify_claim(claim, evidence_docs=evidence)
        self.assertIn("verdict", res)
        self.assertIn("class_probabilities", res)

    def test_dense_retrieval_unavailable_fallback(self):
        retriever = PretrainedDenseRetriever(model_name="nonexistent_mock_model_12345")
        self.assertFalse(retriever.is_available)
        results = retriever.retrieve("test query", top_k=5)
        self.assertEqual(results, [])

    def test_reranker_unavailable_fallback(self):
        reranker = PretrainedCrossEncoderReranker(model_name="nonexistent_mock_model_12345")
        self.assertFalse(reranker.is_available)
        candidates = [{"text": "doc 1", "score": 0.8}, {"text": "doc 2", "score": 0.5}]
        reranked = reranker.rerank("test query", candidates, top_k=2)
        # Should return passthrough unchanged
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["text"], "doc 1")

if __name__ == "__main__":
    unittest.main()
