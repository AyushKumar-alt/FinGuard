import unittest
import os
from src.p1_corpus_builder import build_p1_corpus, DEFAULT_CSV, MIN_TOKEN_COUNT

class TestP1Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build_p1_corpus(csv_path=DEFAULT_CSV)
        cls.corpus = cls.result["corpus"]
        cls.stats = cls.result["stats"]

    def test_loads_from_processed_csv(self):
        self.assertIsNotNone(self.corpus)
        self.assertGreater(len(self.corpus), 0)

    def test_input_record_count(self):
        self.assertEqual(self.stats["input_record_count"], 3369)

    def test_corpus_size_positive(self):
        self.assertGreater(self.stats["final_corpus_size"], 1000)

    def test_short_filter(self):
        for doc in self.corpus[:500]:
            self.assertGreaterEqual(
                len(doc["lemmatized_tokens"]),
                MIN_TOKEN_COUNT,
                f"Document {doc['doc_id']} has fewer than {MIN_TOKEN_COUNT} tokens: {doc['lemmatized_tokens']}"
            )

    def test_deduplication(self):
        seen = set()
        for doc in self.corpus:
            norm = doc["text"].lower().strip()
            self.assertNotIn(norm, seen, f"Duplicate document found: {norm}")
            seen.add(norm)

    def test_provenance_preserved(self):
        for doc in self.corpus[:200]:
            self.assertIn("source_url", doc)
            self.assertIn("record_ids", doc)
            self.assertIsInstance(doc["record_ids"], list)
            self.assertGreater(len(doc["record_ids"]), 0)

    def test_no_labels_in_searchable_text(self):
        # Verify that record-level labels are kept as separate metadata, not baked into text
        for doc in self.corpus[:200]:
            self.assertIn("label", doc)
            self.assertFalse(doc["text"].startswith(f"label:{doc['label']}"))
            self.assertFalse(doc["text"].startswith(f"verdict:{doc['label']}"))

if __name__ == "__main__":
    unittest.main()
