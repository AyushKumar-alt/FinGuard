import unittest
from src.p1_index import Pipeline1Index
from src.p1_retriever import NaturalLanguageEvidenceRetriever
from src.p1_tokenizer import FinancialAwareTokenizer

class TestP1Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_corpus = [
            {
                "doc_id": 0,
                "text": "Apple reported higher revenue and strong investment policies in FY2025.",
                "lemmatized_tokens": ["apple", "higher", "revenue", "strong", "investment", "policy", "FY2025"],
                "source_urls": ["https://apple.com/ir"],
                "parent_claim_ids": [101]
            },
            {
                "doc_id": 1,
                "text": "Tesla vehicle deliveries declined by 8% in the third quarter.",
                "lemmatized_tokens": ["tesla", "vehicle", "delivery", "decline", "8%", "third", "quarter"],
                "source_urls": ["https://tesla.com/ir"],
                "parent_claim_ids": [102]
            },
            {
                "doc_id": 2,
                "text": "Microsoft cloud revenue increased significantly over the previous year.",
                "lemmatized_tokens": ["microsoft", "cloud", "revenue", "increase", "significantly", "previous", "year"],
                "source_urls": ["https://microsoft.com/ir"],
                "parent_claim_ids": [103]
            }
        ]
        cls.mock_result = {
            "corpus": cls.mock_corpus,
            "stats": {"input_csv": "data/processed_finfact.csv", "final_corpus_size": 3},
            "tokenizer_config": {}
        }
        cls.index = Pipeline1Index()
        cls.index.build(cls.mock_result)
        
        cls.tokenizer = FinancialAwareTokenizer()
        cls.retriever = NaturalLanguageEvidenceRetriever(cls.index, cls.tokenizer)
        cls.retriever.fit(cls.mock_corpus)

    def test_returns_doc_id_score_tuples(self):
        results = self.retriever.retrieve_for_claim("What was Apple's investment policy?", top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        doc_id, score = results[0]
        self.assertIsInstance(doc_id, int)
        self.assertIsInstance(score, float)
        self.assertEqual(doc_id, 0)  # Apple should rank 1st

    def test_top_k_respected(self):
        results = self.retriever.retrieve_for_claim("revenue investments", top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.retriever.retrieve_for_claim(""), [])
        self.assertEqual(self.retriever.retrieve_for_claim("   "), [])

    def test_oov_query_returns_empty(self):
        results = self.retriever.retrieve_for_claim("xyzabc unknownnonexistentword123")
        self.assertEqual(results, [])

    def test_provenance_format(self):
        candidates = self.retriever.retrieve_with_provenance("Tesla vehicle deliveries", top_k=1)
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertIn("doc_id", cand)
        self.assertIn("text", cand)
        self.assertIn("score", cand)
        self.assertIn("source_urls", cand)
        self.assertIn("parent_claim_ids", cand)
        self.assertIn("retrieval_pipeline", cand)
        self.assertEqual(cand["doc_id"], 1)

    def test_no_label_leakage(self):
        candidates = self.retriever.retrieve_with_provenance("Microsoft cloud", top_k=1)
        cand = candidates[0]
        self.assertNotIn("label", cand)

if __name__ == "__main__":
    unittest.main()

