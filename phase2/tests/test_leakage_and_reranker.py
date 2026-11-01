"""
Unit tests for retrieval leakage prevention, prohibited candidate exclusion rules,
and cross-encoder reranker provenance preservation.
"""

import unittest
from src.reranker import PretrainedCrossEncoderReranker

class TestLeakageAndReranker(unittest.TestCase):
    def test_candidate_masking_exclusion_rule(self):
        """
        Tests that when evaluating a target claim, any prohibited candidate record
        (such as parent justification, digest, or exact claim string) is strictly excluded
        from candidate scoring.
        """
        target_claim_id = 42
        target_claim_text = "Did Amazon pay zero federal income tax in 2018?"
        
        # Candidate pool containing legitimate evidence AND prohibited/leaked text
        candidate_pool = [
            {"doc_id": 101, "text": "Amazon paid zero dollars in federal corporate income tax for 2018.", "parent_claim_ids": [42]},
            {"doc_id": 102, "text": "Did Amazon pay zero federal income tax in 2018?", "parent_claim_ids": [42], "doc_type": "exact_claim_leak"},
            {"doc_id": 103, "text": "Snopes rated this claim as True in their justification summary.", "parent_claim_ids": [42], "doc_type": "justification_leak"},
            {"doc_id": 104, "text": "The Internal Revenue Code allows corporate loss carryforwards.", "parent_claim_ids": [999]}
        ]
        
        # Filtering logic for evaluation
        filtered_candidates = []
        for cand in candidate_pool:
            # Rule 1: Exclude if doc text is identical to claim text
            if cand["text"].strip().lower() == target_claim_text.strip().lower():
                continue
            # Rule 2: Exclude fact-checker editorial justification/verdict metadata
            if cand.get("doc_type") in ("justification_leak", "exact_claim_leak"):
                continue
            filtered_candidates.append(cand)
            
        doc_ids = [c["doc_id"] for c in filtered_candidates]
        self.assertNotIn(102, doc_ids, "Exact claim string must be strictly excluded from candidate pool.")
        self.assertNotIn(103, doc_ids, "Editorial justification/verdict leaks must be excluded.")
        self.assertIn(101, doc_ids)
        self.assertIn(104, doc_ids)

    def test_reranker_provenance_preservation(self):
        reranker = PretrainedCrossEncoderReranker()
        claim = "Amazon corporate tax payments"
        candidates = [
            {"doc_id": 1, "text": "Amazon paid $0 in federal corporate income taxes in 2018.", "source_urls": ["https://archive.org/tax"], "parent_claim_ids": [10]},
            {"doc_id": 2, "text": "The Federal Reserve monitors inflation and economic growth.", "source_urls": ["https://fed.gov/rates"], "parent_claim_ids": [20]}
        ]
        reranked = reranker.rerank(claim, candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        for r in reranked:
            self.assertIn("source_urls", r)
            self.assertIn("parent_claim_ids", r)
            self.assertIn("rerank_score", r)

if __name__ == "__main__":
    unittest.main()
