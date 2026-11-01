"""
Unit tests for query construction, expansion deduplication,
and direction-preservation safeguards.
"""

import unittest
from src.query_construction import FinancialQueryBuilder

class TestQueryExpansion(unittest.TestCase):
    def setUp(self):
        self.builder = FinancialQueryBuilder(max_expansions=3)

    def test_direction_preservation_in_expansion(self):
        claim = "ABC Bank net profit increased by 30% in FY2025."
        res = self.builder.build_queries(claim)
        
        self.assertEqual(res["original_query"], claim)
        # Check that 'increased' is preserved and never flipped to 'decreased'
        for exp in res["expanded_queries"]:
            self.assertNotIn("decreased", exp.lower(), "Query expansion must never flip direction to decreased!")
            self.assertNotIn("fell", exp.lower())

    def test_deduplication_and_cap(self):
        claim = "Amazon paid $0 in federal corporate income tax in FY2018."
        res = self.builder.build_queries(claim)
        
        self.assertLessEqual(len(res["expanded_queries"]), 3, "Expansions must be capped to max_expansions.")
        # Ensure all expansions are unique
        self.assertEqual(len(res["expanded_queries"]), len(set(res["expanded_queries"])))

if __name__ == "__main__":
    unittest.main()
