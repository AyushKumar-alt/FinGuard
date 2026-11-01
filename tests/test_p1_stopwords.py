import unittest
from src.p1_stopwords import STOP_WORDS, STOPWORD_CONFIG, is_stop_word, filter_tokens

class TestP1Stopwords(unittest.TestCase):
    def test_negation_not_stopped(self):
        for word in ["not", "no", "never", "without", "nor", "neither"]:
            self.assertNotIn(word, STOP_WORDS, f"Negation word '{word}' should not be in STOP_WORDS")
            self.assertFalse(is_stop_word(word))

    def test_directional_not_stopped(self):
        for word in ["increased", "decreased", "declined", "rose", "fell", "grew"]:
            self.assertNotIn(word, STOP_WORDS, f"Directional word '{word}' should not be in STOP_WORDS")
            self.assertFalse(is_stop_word(word))

    def test_acronyms_not_stopped(self):
        for word in ["gdp", "rbi", "eps", "npa", "sec", "ipo"]:
            self.assertNotIn(word, STOP_WORDS, f"Acronym '{word}' should not be in STOP_WORDS")
            self.assertFalse(is_stop_word(word))

    def test_common_words_stopped(self):
        for word in ["the", "a", "an", "is", "was", "said", "reported"]:
            self.assertIn(word, STOP_WORDS, f"Common word '{word}' should be in STOP_WORDS")
            self.assertTrue(is_stop_word(word))

    def test_filter_preserves_order(self):
        tokens = ["apple", "the", "revenue", "is", "not", "declined"]
        filtered = filter_tokens(tokens)
        self.assertEqual(filtered, ["apple", "revenue", "not", "declined"])

    def test_configurable(self):
        self.assertIsInstance(STOP_WORDS, set)
        self.assertIn("base_list", STOPWORD_CONFIG)
        self.assertIn("domain_additions", STOPWORD_CONFIG)
        self.assertIn("protected_terms_excluded", STOPWORD_CONFIG)
        self.assertIn("total_stop_words", STOPWORD_CONFIG)

if __name__ == "__main__":
    unittest.main()

