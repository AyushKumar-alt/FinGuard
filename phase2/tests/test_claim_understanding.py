"""
Unit tests for finance-aware claim processing, token preservation,
and structured financial entity/metric extraction.
"""

import unittest
from src.claim_understanding import finance_aware_tokenize, extract_financial_entities

class TestClaimUnderstanding(unittest.TestCase):
    def test_finance_token_preservation(self):
        sample = "The company raised $11B in debt, while ₹5,000 was allocated. Rates rose by 10.5% in FY2025 (Q4 2025) affecting 401(k) and S&P 500 accounts."
        tokens = finance_aware_tokenize(sample)
        
        self.assertIn("$11B", tokens, "$11B currency amount must be preserved as single token.")
        self.assertIn("₹5,000", tokens, "₹5,000 currency amount must be preserved.")
        self.assertIn("10.5%", tokens, "10.5% percentage must be preserved.")
        self.assertIn("FY2025", tokens, "FY2025 fiscal period must be preserved.")
        self.assertIn("Q4 2025", tokens, "Q4 2025 fiscal period must be preserved.")
        self.assertIn("401(k)", tokens, "401(k) retirement token must be preserved.")
        self.assertIn("S&P 500", tokens, "S&P 500 market index must be preserved.")

    def test_structured_metric_extraction(self):
        claim = "Amazon paid $0 in federal corporate income tax despite billions in profit in FY2018."
        res = extract_financial_entities(claim)
        
        self.assertIn("Amazon", res["organizations"])
        self.assertTrue(any("income tax" in m or "profit" in m for m in res["financial_metrics"]))
        self.assertIn("$0", res["monetary_amounts"])
        self.assertIn("FY2018", res["periods"])

    def test_directional_language_and_negation(self):
        claim = "Did Biden call for an end to shareholder capitalism, and did debt not increase?"
        res = extract_financial_entities(claim)
        
        # Negation check
        negations = res["negations"]
        self.assertTrue(any("not" in n or "end to" in n for n in negations))
        # Direction check
        directions = [d["term"] for d in res["directional_terms"]]
        self.assertIn("increase", directions)

    def test_incomplete_or_ambiguous_claim_uncertainty(self):
        claim = "The executive gave an unverified speech on Tuesday."
        res = extract_financial_entities(claim)
        self.assertTrue(res["is_uncertain"], "Claims lacking financial figures/metrics must be flagged uncertain.")
        self.assertEqual(len(res["monetary_amounts"]), 0)
        self.assertEqual(len(res["percentages"]), 0)

if __name__ == "__main__":
    unittest.main()
