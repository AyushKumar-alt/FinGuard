import unittest

from src.pipeline import FinGuardPipeline
from src.query_construction import FinancialQueryBuilder


class FakeRetriever:
    def __init__(self):
        self.corpus = [
            {"doc_id": i, "text": f"evidence passage {i}", "source_urls": [], "parent_claim_ids": []}
            for i in range(4)
        ]
        self.calls = []

    def retrieve_bm25(self, query, top_k):
        self.calls.append(("bm25", top_k))
        return [(0, 5.0), (1, 4.0)]

    def retrieve_dense(self, query, top_k):
        self.calls.append(("svd", top_k))
        return [(3, 0.9)]

    def retrieve_tfidf(self, query, top_k):
        self.calls.append(("tfidf", top_k))
        return [(2, 0.8)]

    def retrieve_hybrid_rrf(self, query, top_k):
        self.calls.append(("hybrid_tfidf", top_k))
        return [(1, 0.03)]


class FakeDenseRetriever:
    is_available = True

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append(top_k)
        return [(1, 0.99), (2, 0.8)]


class FakeVerifier:
    has_transformer = False
    baseline_model = None

    def verify_claim(self, claim, evidence_docs, engine_choice):
        return {
            "verdict": "NEI",
            "verdict_display": "NEI",
            "confidence": 0.5,
            "class_probabilities": {"false": 0.25, "true": 0.25, "NEI": 0.5},
            "is_uncertain": True,
            "is_conflicting": False,
            "uncertainty_reason": None,
            "explanation": "",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "neutral_evidence": [],
            "active_engine": "test",
        }


def make_pipeline(dense_retriever=None):
    pipeline = FinGuardPipeline.__new__(FinGuardPipeline)
    pipeline.query_builder = FinancialQueryBuilder()
    pipeline.retriever = FakeRetriever()
    pipeline.corpus = pipeline.retriever.corpus
    pipeline.dense_retriever = dense_retriever
    pipeline.reranker = None
    pipeline.verifier = FakeVerifier()
    return pipeline


class TestPipelineRetrievalModes(unittest.TestCase):
    def test_bm25_dense_hybrid_uses_rrf_from_both_retrievers(self):
        dense = FakeDenseRetriever()
        pipeline = make_pipeline(dense)

        result = pipeline.run(
            "The company revenue increased by 10 percent.",
            retrieval_mode="hybrid_bm25_dense",
            use_reranker=False,
            use_nli=False,
            top_k_retrieve=3,
            top_k_rerank=3,
        )

        self.assertEqual(pipeline.retriever.calls, [("bm25", 6)])
        self.assertEqual(dense.calls, [6])
        self.assertEqual(result["retrieval_engine"], "hybrid_bm25_dense_rrf")
        self.assertEqual([doc["doc_id"] for doc in result["selected_evidence_docs"]], [1, 0, 2])

    def test_bm25_dense_hybrid_falls_back_to_bm25_when_dense_unavailable(self):
        pipeline = make_pipeline(dense_retriever=None)

        result = pipeline.run(
            "The company revenue increased by 10 percent.",
            retrieval_mode="hybrid_bm25_dense",
            use_reranker=False,
            use_nli=False,
            top_k_retrieve=2,
            top_k_rerank=2,
        )

        self.assertEqual(pipeline.retriever.calls, [("bm25", 2)])
        self.assertEqual(result["retrieval_engine"], "bm25_fallback_dense_unavailable")

    def test_svd_mode_uses_truncated_svd_retriever(self):
        pipeline = make_pipeline()

        result = pipeline.run(
            "The company revenue increased by 10 percent.",
            retrieval_mode="svd",
            use_reranker=False,
            use_nli=False,
            top_k_retrieve=2,
            top_k_rerank=2,
        )

        self.assertEqual(pipeline.retriever.calls, [("svd", 2)])
        self.assertEqual(result["retrieval_engine"], "tfidf_svd")
        self.assertEqual([doc["doc_id"] for doc in result["selected_evidence_docs"]], [3])

    def test_unknown_retrieval_mode_is_rejected(self):
        pipeline = make_pipeline()
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval_mode"):
            pipeline.run("A financial claim.", retrieval_mode="not-a-mode")


if __name__ == "__main__":
    unittest.main()
