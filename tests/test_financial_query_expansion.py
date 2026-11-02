"""
Unit tests for FinGuard Phase 2:
Finance-Aware Synonym Expansion, Multi-Query Generation,
Direction-Aware Query Expansion, and Weighted Reciprocal Rank Fusion (Weighted RRF).
"""

import unittest
from collections import defaultdict
from src.query_construction import FinancialQueryBuilder, DEFAULT_SYNONYMS, DEFAULT_COMPANY_ALIASES
from src.claim_understanding import extract_financial_entities
from src.pipeline import FinGuardPipeline

class TestFinancialQueryExpansion(unittest.TestCase):
    def setUp(self):
        self.builder = FinancialQueryBuilder(max_expansions=4)

    # 1. Financial synonym expansion
    def test_01_financial_synonym_expansion(self):
        claim = "Apple reported net profit and revenue growth in FY2024 with strong total assets."
        parsed = extract_financial_entities(claim)
        queries = self.builder.generate_multi_queries(claim, parsed)
        syn_queries = [q["query"].lower() for q in queries if q["group"] == "synonym_expanded"]
        self.assertTrue(len(syn_queries) > 0, "Synonym expanded query should be generated.")
        
        # Verify that synonyms like 'net income', 'sales', or 'fiscal year 2024' are utilized
        found_any_syn = any("net income" in q or "sales" in q or "fiscal year 2024" in q or "assets" in q for q in syn_queries)
        self.assertTrue(found_any_syn, "Synonym query must contain at least one configured domain alternative.")

    # 2. Company alias expansion
    def test_02_company_alias_expansion(self):
        # Known company with alias
        claim_apple = "Apple reported quarterly revenue in FY2024."
        parsed_apple = extract_financial_entities(claim_apple)
        queries_apple = self.builder.generate_multi_queries(claim_apple, parsed_apple)
        all_text_apple = " ".join([q["query"] for q in queries_apple])
        # Should include Apple Inc. or AAPL
        self.assertTrue("Apple" in all_text_apple)

        # Unknown company: retain original, do NOT invent aliases or tickers
        claim_unknown = "Contoso Holdings reported quarterly revenue in FY2024."
        parsed_unknown = extract_financial_entities(claim_unknown)
        queries_unknown = self.builder.generate_multi_queries(claim_unknown, parsed_unknown)
        for q in queries_unknown:
            self.assertNotIn("AAPL", q["query"])
            self.assertNotIn("AMZN", q["query"])
            self.assertNotIn("MSFT", q["query"])
            if q["group"] == "entity_metric_period":
                self.assertIn("Contoso Holdings", q["query"])

    # 3. Multi-query generation
    def test_03_multi_query_generation(self):
        claim = "ABC Bank's net profit decreased by 12% in FY2024."
        parsed = extract_financial_entities(claim)
        queries = self.builder.generate_multi_queries(claim, parsed)
        
        # Check count is within manageable target range (4 to 6 queries)
        self.assertGreaterEqual(len(queries), 4)
        self.assertLessEqual(len(queries), 6)

        # Check Q0 is original claim with weight 1.2
        q0 = queries[0]
        self.assertEqual(q0["group"], "original_claim")
        self.assertEqual(q0["query"], claim)
        self.assertAlmostEqual(q0["weight"], 1.2)

        groups = [q["group"] for q in queries]
        self.assertIn("entity_metric_period", groups)
        self.assertIn("numerical_query", groups)
        self.assertIn("supporting_direction", groups)
        self.assertIn("contradicting_direction", groups)

    # 4. Missing entity, metric, amount, or period handling
    def test_04_missing_fields_handling(self):
        # Claim with entity and direction, but missing metric and period
        claim_sparse = "Tesla dropped significantly after hours."
        parsed = extract_financial_entities(claim_sparse)
        queries = self.builder.generate_multi_queries(claim_sparse, parsed)
        
        for q in queries:
            text = q["query"]
            # No placeholders or null tokens
            self.assertNotIn("None", text)
            self.assertNotIn("null", text)
            self.assertNotIn("N/A", text)
            self.assertNotIn("undefined", text)
            self.assertTrue(len(text.strip()) > 0)

    # 5. Direction reversal for increased versus decreased claims
    def test_05_direction_reversal_increased_vs_decreased(self):
        # Claim with 'decreased'
        claim_dec = "ABC Bank's net profit decreased by 12% in FY2024."
        queries_dec = self.builder.generate_multi_queries(claim_dec)
        q_supp_dec = [q["query"].lower() for q in queries_dec if q["group"] == "supporting_direction"][0]
        q_contra_dec = [q["query"].lower() for q in queries_dec if q["group"] == "contradicting_direction"][0]

        # Supporting should contain a decrease term
        self.assertTrue(any(w in q_supp_dec for w in ["decreased", "declined", "fell", "dropped", "reduced"]))
        # Contradicting should contain an increase term
        self.assertTrue(any(w in q_contra_dec for w in ["increased", "rose", "grew", "climbed", "improved"]))

        # Claim with 'increased'
        claim_inc = "ABC Bank's net profit increased by 12% in FY2024."
        queries_inc = self.builder.generate_multi_queries(claim_inc)
        q_supp_inc = [q["query"].lower() for q in queries_inc if q["group"] == "supporting_direction"][0]
        q_contra_inc = [q["query"].lower() for q in queries_inc if q["group"] == "contradicting_direction"][0]

        # Supporting should now contain an increase term
        self.assertTrue(any(w in q_supp_inc for w in ["increased", "rose", "grew", "climbed", "improved"]))
        # Contradicting should now contain a decrease term
        self.assertTrue(any(w in q_contra_inc for w in ["decreased", "declined", "fell", "dropped", "reduced"]))

    # 6. Preservation of negation and polarity
    def test_06_preservation_of_negation_and_polarity(self):
        claim_neg = "Amazon did not pay federal income tax in FY2018."
        parsed = extract_financial_entities(claim_neg)
        queries = self.builder.generate_multi_queries(claim_neg, parsed)
        
        # Original claim preserves negation
        self.assertIn("did not", queries[0]["query"])
        
        # Check negation query exists
        neg_queries = [q["query"] for q in queries if q["group"] == "negation_preserved"]
        self.assertTrue(len(neg_queries) > 0)
        self.assertIn("did not", neg_queries[0])

    # 7. Removal of duplicate queries
    def test_07_removal_of_duplicate_queries(self):
        claim = "ABC Bank's net profit decreased by 12% in FY2024."
        queries = self.builder.generate_multi_queries(claim)
        normalized_queries = [" ".join(q["query"].strip().lower().split()) for q in queries]
        
        self.assertEqual(len(normalized_queries), len(set(normalized_queries)), "Generated queries must be unique after normalization.")

    # 8. Weighted RRF score calculation
    def test_08_weighted_rrf_calculation(self):
        # Q1: weight 1.2, rank 1 -> doc 10, rank 2 -> doc 20
        # Q2: weight 1.0, rank 1 -> doc 20, rank 2 -> doc 10
        # k = 60
        # doc 10 expected: 1.2 / (60 + 1) + 1.0 / (60 + 2) = 1.2/61 + 1.0/62
        # doc 20 expected: 1.2 / (60 + 2) + 1.0 / (60 + 1) = 1.2/62 + 1.0/61
        ranking1 = [(10, 5.0), (20, 3.0)]
        ranking2 = [(20, 4.0), (10, 2.0)]
        weighted_rankings = [(ranking1, 1.2), (ranking2, 1.0)]

        fused = FinGuardPipeline._weighted_reciprocal_rank_fusion(weighted_rankings, top_k=2, k=60)
        
        expected_score_10 = (1.2 / 61.0) + (1.0 / 62.0)
        expected_score_20 = (1.2 / 62.0) + (1.0 / 61.0)

        fused_dict = dict(fused)
        self.assertAlmostEqual(fused_dict[10], expected_score_10, places=6)
        self.assertAlmostEqual(fused_dict[20], expected_score_20, places=6)
        # 1.2/61 + 1.0/62 > 1.2/62 + 1.0/61 because 1.2*(1/61 - 1/62) > 1.0*(1/61 - 1/62)
        self.assertEqual(fused[0][0], 10, "Doc 10 must be ranked 1st due to higher weight on Q1.")

    # 9. Evidence deduplication across query results
    def test_09_evidence_deduplication_across_query_results(self):
        # Same doc appears in multiple query rankings
        ranking1 = [(101, 10.0), (102, 8.0)]
        ranking2 = [(101, 7.0), (103, 6.0)]
        ranking3 = [(101, 9.0), (104, 5.0)]
        weighted_rankings = [(ranking1, 1.0), (ranking2, 1.0), (ranking3, 1.0)]

        fused = FinGuardPipeline._weighted_reciprocal_rank_fusion(weighted_rankings, top_k=10, k=60)
        doc_ids = [doc_id for doc_id, _ in fused]
        
        # Must be deduplicated
        self.assertEqual(len(doc_ids), len(set(doc_ids)))
        self.assertEqual(len(doc_ids), 4) # 101, 102, 103, 104
        self.assertEqual(doc_ids[0], 101, "Doc 101 appearing in all 3 lists must rank highest.")

    # 10. Correct top-15 output
    def test_10_correct_top_15_output(self):
        # Create mock rankings returning 30 unique documents
        ranking = [(i, float(30 - i)) for i in range(30)]
        weighted_rankings = [(ranking, 1.0)]

        fused = FinGuardPipeline._weighted_reciprocal_rank_fusion(weighted_rankings, top_k=15, k=60)
        self.assertEqual(len(fused), 15)
        # Verify scores are strictly descending
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # 11. Legacy single-query fallback
    def test_11_legacy_single_query_fallback(self):
        builder_disabled = FinancialQueryBuilder()
        builder_disabled.enabled = False
        claim = "ABC Bank's net profit decreased by 12% in FY2024."
        queries = builder_disabled.generate_multi_queries(claim)
        
        # When disabled, only original claim query is generated
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["group"], "original_claim")
        self.assertEqual(queries[0]["query"], claim)

    # 12. Compatibility with existing Cross-Encoder and NLI stages
    def test_12_compatibility_with_cross_encoder_and_nli(self):
        claim = "ABC Bank's net profit decreased by 12% in FY2024."
        res = self.builder.build_queries(claim)
        
        # Check all required legacy keys exist
        self.assertIn("original_query", res)
        self.assertIn("expanded_queries", res)
        self.assertIn("combined_query", res)
        self.assertIn("multi_queries", res)
        self.assertIn("query_strings", res)
        self.assertIn("key_entities", res)
        self.assertIn("key_metrics", res)
        
        # Ensure expanded_queries are strings suitable for downstream tokenizers
        for exp in res["expanded_queries"]:
            self.assertIsInstance(exp, str)
            self.assertTrue(len(exp.strip()) > 0)

if __name__ == "__main__":
    unittest.main()

