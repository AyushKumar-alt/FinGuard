import unittest
import os
import json
import tempfile
from src.p1_index import Pipeline1Index, P1_INDEX_VERSION, DEFAULT_P1_INDEX_PATH

class TestP1Index(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_corpus_result = {
            "corpus": [
                {
                    "doc_id": 0,
                    "text": "Apple reported higher revenue and strong investment policies in FY2025.",
                    "lemmatized_tokens": ["apple", "higher", "revenue", "strong", "investment", "policy", "FY2025"],
                    "source_urls": ["https://apple.com/ir"],
                    "parent_claim_ids": [101]
                },
                {
                    "doc_id": 1,
                    "text": "Tesla investments declined despite higher vehicle production.",
                    "lemmatized_tokens": ["tesla", "investment", "decline", "higher", "vehicle", "production"],
                    "source_urls": ["https://tesla.com/ir"],
                    "parent_claim_ids": [102]
                }
            ],
            "stats": {"input_csv": "data/processed_finfact.csv", "final_corpus_size": 2},
            "tokenizer_config": {
                "lemmatizer": "WordNet",
                "pos_tagger": "averaged_perceptron_tagger_eng",
                "stopword_config": {"total": 50}
            }
        }
        cls.index = Pipeline1Index()
        cls.index.build(cls.mock_corpus_result)

    def test_build_from_corpus(self):
        self.assertGreater(self.index.inner_index.doc_count, 0)
        self.assertGreater(len(self.index.inner_index.index), 0)

    def test_vocab_positive(self):
        self.assertGreater(len(self.index.inner_index.index), 5)
        self.assertIn("investment", self.index.inner_index.index)

    def test_term_frequency(self):
        tf = self.index.get_term_freq("investment", 0)
        self.assertEqual(tf, 1)

    def test_doc_freq(self):
        df = self.index.get_doc_freq("investment")
        self.assertEqual(df, 2)

    def test_save_load_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name
        try:
            self.index.save(temp_path)
            loaded = Pipeline1Index().load(temp_path)
            self.assertEqual(loaded.inner_index.doc_count, self.index.inner_index.doc_count)
            self.assertEqual(loaded.get_doc_freq("investment"), 2)
            self.assertEqual(loaded.get_original_text(0), self.mock_corpus_result["corpus"][0]["text"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_version_mismatch_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
            json.dump({"index_version": "invalid-v0.1"}, tf)
            temp_path = tf.name
        try:
            loaded = Pipeline1Index()
            with self.assertRaises(ValueError):
                loaded.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_separate_from_existing(self):
        self.assertNotEqual(DEFAULT_P1_INDEX_PATH, "outputs/positional_index.json")
        self.assertTrue(DEFAULT_P1_INDEX_PATH.endswith("p1_positional_index.json"))

    def test_metadata_complete(self):
        meta = self.index.metadata
        for key in ["pipeline_name", "pipeline_version", "corpus_source", "build_timestamp", "lemmatizer", "pos_tagger"]:
            self.assertIn(key, meta, f"Metadata key '{key}' should be present")

if __name__ == "__main__":
    unittest.main()

