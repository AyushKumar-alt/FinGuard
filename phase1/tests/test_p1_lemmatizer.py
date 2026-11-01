import unittest
from src.p1_lemmatizer import POSGuidedLemmatizer

class TestP1Lemmatizer(unittest.TestCase):
    def setUp(self):
        self.lemmatizer = POSGuidedLemmatizer()

    def test_available_flag(self):
        self.assertTrue(self.lemmatizer.is_available)
        self.assertIn("WordNetLemmatizer", self.lemmatizer.library_info)

    def test_pos_guided_noun(self):
        res = self.lemmatizer.lemmatize_tokens(["investments", "policies", "banks"])
        self.assertEqual(res, ["investment", "policy", "bank"])

    def test_pos_guided_verb(self):
        res = self.lemmatizer.lemmatize_tokens(["declining", "increasing", "running"])
        self.assertEqual(res, ["decline", "increase", "run"])

    def test_protected_currency_passthrough(self):
        tokens = ["$11B", "₹5,000", "€40M", "$100"]
        res = self.lemmatizer.lemmatize_tokens(tokens)
        self.assertEqual(res, tokens)

    def test_protected_percentage_passthrough(self):
        tokens = ["10.5%", "6.50%", "100%"]
        res = self.lemmatizer.lemmatize_tokens(tokens)
        self.assertEqual(res, tokens)

    def test_protected_acronym_passthrough(self):
        tokens = ["GDP", "RBI", "EPS", "NPA"]
        res = self.lemmatizer.lemmatize_tokens(tokens)
        self.assertEqual(res, tokens)

    def test_protected_fiscal_passthrough(self):
        tokens = ["FY2025", "FY2025-26", "Q4 2025"]
        res = self.lemmatizer.lemmatize_tokens(tokens)
        self.assertEqual(res, tokens)

    def test_fallback_irregular_verb(self):
        # Documented limitation of WordNet: irregular verb 'fell' remains 'fell'
        res = self.lemmatizer.lemmatize_tokens(["fell"])
        self.assertEqual(res, ["fell"])

    def test_adjective_unchanged(self):
        res = self.lemmatizer.lemmatize_tokens(["profitable", "financial"])
        self.assertEqual(res, ["profitable", "financial"])

if __name__ == "__main__":
    unittest.main()
