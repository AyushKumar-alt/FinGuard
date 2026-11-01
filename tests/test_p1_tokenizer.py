import unittest
from src.p1_tokenizer import FinancialAwareTokenizer

class TestP1Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FinancialAwareTokenizer()

    def test_currency_preserved(self):
        res = self.tokenizer.tokenize("Apple made $11B in revenue.")
        self.assertIn("$11B", res["raw_tokens"])
        self.assertIn("$11B", res["lemmatized_tokens"])

    def test_percentage_preserved(self):
        res = self.tokenizer.tokenize("Interest rate rose by 6.50%.")
        self.assertIn("6.50%", res["raw_tokens"])
        self.assertIn("6.50%", res["lemmatized_tokens"])

    def test_fiscal_period_preserved(self):
        res = self.tokenizer.tokenize("Revenue in FY2025 was high.")
        self.assertIn("FY2025", res["raw_tokens"])
        self.assertIn("FY2025", res["lemmatized_tokens"])

    def test_acronym_protected(self):
        res = self.tokenizer.tokenize("RBI and GDP trends.")
        self.assertIn("RBI", res["lemmatized_tokens"])
        self.assertIn("GDP", res["lemmatized_tokens"])

    def test_negation_not_removed(self):
        res = self.tokenizer.tokenize("Tesla did not achieve the target.")
        self.assertIn("not", res["filtered_tokens"])
        self.assertIn("not", res["lemmatized_tokens"])

    def test_directional_preserved(self):
        res = self.tokenizer.tokenize("Profits increased while margins declined.")
        self.assertIn("increased", res["filtered_tokens"])
        self.assertIn("declined", res["filtered_tokens"])

    def test_lemmatization_noun(self):
        res = self.tokenizer.tokenize("Various investments and policies.")
        self.assertIn("investment", res["lemmatized_tokens"])
        self.assertIn("policy", res["lemmatized_tokens"])

    def test_lemmatization_verb(self):
        res = self.tokenizer.tokenize("Revenues are declining.")
        self.assertIn("decline", res["lemmatized_tokens"])

    def test_deterministic(self):
        t1 = self.tokenizer.tokenize("Microsoft reported $50B in Q4 2025.")
        t2 = self.tokenizer.tokenize("Microsoft reported $50B in Q4 2025.")
        self.assertEqual(t1, t2)

    def test_empty_text(self):
        res = self.tokenizer.tokenize("")
        self.assertEqual(res["raw_tokens"], [])
        self.assertEqual(res["filtered_tokens"], [])
        self.assertEqual(res["lemmatized_tokens"], [])

    def test_unicode_safe(self):
        res = self.tokenizer.tokenize("Profit was ₹5,000 and cost was €40M.")
        self.assertIn("₹5,000", res["lemmatized_tokens"])
        self.assertIn("€40M", res["lemmatized_tokens"])

    def test_query_corpus_consistency(self):
        query_tokens = self.tokenizer.get_query_tokens("Banks with declining investments")
        corpus_tokens = self.tokenizer.get_index_tokens("Banks with declining investments")
        self.assertEqual(query_tokens, corpus_tokens)

if __name__ == "__main__":
    unittest.main()

