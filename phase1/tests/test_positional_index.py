"""
Unit tests for positional inverted index and Boolean/phrase search query parser.
"""

import unittest
from src.positional_index import PositionalInvertedIndex, BooleanQueryParser, QuerySyntaxError

class TestPositionalIndexAndBooleanSearch(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"doc_id": 1, "text": "Joe Biden called for an end to shareholder capitalism."},
            {"doc_id": 2, "text": "Amazon paid zero dollars in federal corporate income tax despite record high profits."},
            {"doc_id": 3, "text": "The national debt reached twenty trillion dollars in record time."},
            {"doc_id": 4, "text": "Corporate profit should serve stakeholders rather than pure shareholder capitalism."}
        ]
        self.index = PositionalInvertedIndex(self.corpus)
        self.parser = BooleanQueryParser(self.index)

    def test_phrase_search_consecutive_positions(self):
        # "shareholder capitalism" occurs in doc 1 and doc 4
        matching = self.index.search_phrase("shareholder capitalism")
        self.assertEqual(matching, {1, 4})
        
        # Out-of-order phrase should NOT match
        reversed_match = self.index.search_phrase("capitalism shareholder")
        self.assertEqual(reversed_match, set(), "Reversed phrase must not match.")

    def test_boolean_and_or_not_precedence(self):
        # Operator precedence: NOT > AND > OR
        # "Amazon OR (debt AND NOT Biden)"
        # Doc 2 has Amazon. Doc 3 has debt, no Biden.
        res = self.parser.evaluate('Amazon OR (debt AND NOT Biden)')
        self.assertEqual(res, {2, 3})

    def test_boolean_quoted_phrase(self):
        # '"shareholder capitalism" AND Biden' -> only Doc 1 has both
        res = self.parser.evaluate('"shareholder capitalism" AND Biden')
        self.assertEqual(res, {1})

    def test_syntax_errors_raised(self):
        # Mismatched parentheses
        with self.assertRaises(QuerySyntaxError):
            self.parser.evaluate('(Amazon OR debt')
            
        # Unclosed quotes
        with self.assertRaises(QuerySyntaxError):
            self.parser.evaluate('"shareholder capitalism AND Biden')

if __name__ == "__main__":
    unittest.main()
